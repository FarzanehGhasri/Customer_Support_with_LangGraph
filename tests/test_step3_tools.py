"""Step 3 tests -- billing tools and the RAG retriever. All offline."""

from __future__ import annotations

import json

import pytest

from support_system.config import Settings
from support_system.domain import RetrievalResult
from support_system.infrastructure.billing import JsonSubscriptionRepository, MockRefundGateway
from support_system.infrastructure.retrieval import (
    Chunk,
    EmbeddingKnowledgeRetriever,
    KeywordKnowledgeRetriever,
    load_chunks,
    split_document,
)
from support_system.interfaces import EmbeddingProvider, KnowledgeRetriever
from support_system.tools import make_billing_tools, make_technical_tools


@pytest.fixture()
def settings() -> Settings:
    return Settings.from_env()


@pytest.fixture()
def repo(settings) -> JsonSubscriptionRepository:
    return JsonSubscriptionRepository(settings.subscriptions_file)


@pytest.fixture()
def gateway(settings) -> MockRefundGateway:
    return MockRefundGateway(settings.transactions_file)


@pytest.fixture()
def retriever(settings) -> KeywordKnowledgeRetriever:
    return KeywordKnowledgeRetriever(settings.knowledge_base_dir)


# --------------------------------------------------------------------------- #
# Subscription repository
# --------------------------------------------------------------------------- #


def test_scenario_two_user_is_expired(repo):
    """The assignment's scenario 2 must answer 'your subscription has expired'."""
    status = repo.get_status("12345")
    assert status.found and status.status == "expired" and not status.is_active


def test_unknown_user_is_reported_not_raised(repo):
    assert repo.get_status("does-not-exist").found is False


def test_blank_user_id_is_handled(repo):
    assert repo.get_status("").found is False


def test_repository_does_not_touch_disk_until_used(tmp_path):
    """Construction must be cheap and side-effect free."""
    JsonSubscriptionRepository(tmp_path / "missing.json")  # must not raise


def test_missing_file_degrades_to_not_found(tmp_path):
    assert JsonSubscriptionRepository(tmp_path / "missing.json").get_status("1").found is False


def test_unparseable_date_does_not_crash(tmp_path):
    path = tmp_path / "subs.json"
    path.write_text(json.dumps({"customers": [
        {"user_id": "1", "plan": "P", "status": "active", "expires_on": "not-a-date"}
    ]}), encoding="utf-8")
    status = JsonSubscriptionRepository(path).get_status("1")
    assert status.found and status.expires_on is None


# --------------------------------------------------------------------------- #
# Refund gateway -- the dangerous tool, so policy lives in code, not in a prompt
# --------------------------------------------------------------------------- #


def test_refundable_transaction_is_approved(gateway):
    receipt = gateway.process_refund("TXN-1001")
    assert receipt.approved and receipt.amount == 19.99 and receipt.reference


def test_non_refundable_transaction_is_refused(gateway):
    assert gateway.process_refund("TXN-1002").approved is False


def test_already_refunded_transaction_is_refused(gateway):
    assert gateway.process_refund("TXN-1003").approved is False


def test_unknown_transaction_is_never_invented(gateway):
    receipt = gateway.process_refund("TXN-DOES-NOT-EXIST")
    assert receipt.approved is False and "No such transaction" in receipt.reason


def test_blank_transaction_id_is_refused(gateway):
    assert gateway.process_refund("  ").approved is False


def test_the_same_transaction_cannot_be_refunded_twice(gateway):
    assert gateway.process_refund("TXN-1001").approved is True
    second = gateway.process_refund("TXN-1001")
    assert second.approved is False and "already refunded" in second.reason.lower()


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def test_document_is_split_on_level_two_headings():
    chunks = split_document("# Title\n\nintro text\n\n## One\nbody one\n\n## Two\nbody two",
                            source="d.md")
    assert [c.heading for c in chunks] == ["Title", "Title — One", "Title — Two"]
    # The preamble keeps the document's title line: it carries the keyword hints
    # that make the article findable.
    assert chunks[0].content == "# Title\n\nintro text"
    assert chunks[1].content.startswith("## One")


def test_document_without_headings_is_kept_whole():
    chunks = split_document("just a paragraph", source="d.md")
    assert len(chunks) == 1 and chunks[0].source == "d.md"


def test_empty_document_yields_no_chunks():
    assert split_document("   ", source="d.md") == []


def test_searchable_text_includes_the_heading():
    chunk = Chunk(content="body", source="s.md", heading="Reset password")
    assert "Reset password" in chunk.searchable_text


def test_knowledge_base_loads(settings):
    chunks = load_chunks(settings.knowledge_base_dir)
    assert len(chunks) >= 20
    assert {c.source for c in chunks} >= {"password_reset.md", "sync_and_offline.md"}


def test_missing_directory_yields_no_chunks(tmp_path):
    assert load_chunks(tmp_path / "nope") == []


# --------------------------------------------------------------------------- #
# Keyword retriever
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query,expected_source",
    [
        ("How can I reset my password?", "password_reset.md"),   # scenario 1
        ("I forgot my password", "password_reset.md"),
        ("the app crashes on launch", "app_crashes_and_errors.md"),
        ("error E-204", "app_crashes_and_errors.md"),
        ("my data is not syncing", "sync_and_offline.md"),
        ("how do I enable dark mode", "notifications_and_settings.md"),
        ("system requirements for macOS", "installation_and_updates.md"),
    ],
)
def test_retriever_finds_the_right_article(retriever, query, expected_source):
    result = retriever.search(query)
    assert result.has_grounding
    assert result.snippets[0].source == expected_source


@pytest.mark.parametrize(
    "query",
    [
        "What is the capital of France?",
        "how do I train a neural network",
        "tell me a joke",
        "what is your CEO name",
    ],
)
def test_off_topic_questions_are_not_grounded(retriever, query):
    """This is the spec's anti-hallucination requirement."""
    assert retriever.search(query).has_grounding is False


def test_empty_query_is_not_grounded(retriever):
    assert retriever.search("").has_grounding is False


def test_results_are_ordered_best_first(retriever):
    scores = [s.score for s in retriever.search("reset password", top_k=3).snippets]
    assert scores == sorted(scores, reverse=True)


def test_top_k_is_respected(retriever):
    assert len(retriever.search("password", top_k=2).snippets) <= 2


def test_retriever_satisfies_the_protocol(retriever):
    assert isinstance(retriever, KnowledgeRetriever)


def test_stopwords_alone_are_not_grounding(retriever):
    """A query of pure filler words must not match anything."""
    assert retriever.search("what is the of and to").has_grounding is False


# --------------------------------------------------------------------------- #
# Embedding retriever -- exercised with a deterministic fake embedder
# --------------------------------------------------------------------------- #


class _FakeEmbeddings:
    """Maps text to a 3-D vector by counting three marker words.

    Deterministic and offline, but still a genuine vector space, so cosine
    ranking and the min_score threshold are really exercised.
    """

    MARKERS = ("password", "refund", "sync")

    def _vector(self, text: str) -> list[float]:
        """Three marker axes plus a fourth 'unrelated' axis.

        The fourth axis matters: without it, text matching no marker would get a
        vector pointing equally at all three and appear similar to everything,
        which would test the fake rather than the retriever.
        """
        low = text.lower()
        vector = [float(low.count(m)) for m in self.MARKERS]
        return vector + [0.0] if any(vector) else [0.0, 0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


class _ExplodingEmbeddings:
    def embed_documents(self, texts):
        raise RuntimeError("embedding endpoint down")

    def embed_query(self, text):
        raise RuntimeError("embedding endpoint down")


@pytest.fixture()
def vector_chunks() -> list[Chunk]:
    return [
        Chunk(content="how to reset your password", source="p.md", heading="password"),
        Chunk(content="how to request a refund", source="r.md", heading="refund"),
        Chunk(content="fixing sync problems", source="s.md", heading="sync"),
    ]


def test_fake_embedder_satisfies_the_protocol():
    assert isinstance(_FakeEmbeddings(), EmbeddingProvider)


def test_vector_search_ranks_the_right_chunk_first(vector_chunks):
    retriever = EmbeddingKnowledgeRetriever(_FakeEmbeddings(), chunks=vector_chunks)
    result = retriever.search("I forgot my password")
    assert result.has_grounding and result.snippets[0].source == "p.md"


def test_vector_search_reports_no_grounding_below_threshold(vector_chunks):
    retriever = EmbeddingKnowledgeRetriever(_FakeEmbeddings(), chunks=vector_chunks)
    assert retriever.search("something entirely unrelated").has_grounding is False


def test_corpus_is_embedded_only_once(vector_chunks):
    class _Counting(_FakeEmbeddings):
        calls = 0

        def embed_documents(self, texts):
            _Counting.calls += 1
            return super().embed_documents(texts)

    retriever = EmbeddingKnowledgeRetriever(_Counting(), chunks=vector_chunks)
    retriever.search("password")
    retriever.search("refund")
    assert _Counting.calls == 1


def test_embedding_failure_falls_back_to_keyword_search(settings, vector_chunks):
    fallback = KeywordKnowledgeRetriever(settings.knowledge_base_dir)
    retriever = EmbeddingKnowledgeRetriever(
        _ExplodingEmbeddings(), chunks=vector_chunks, fallback=fallback
    )
    result = retriever.search("How can I reset my password?")
    assert result.has_grounding and result.snippets[0].source == "password_reset.md"


def test_embedding_failure_without_fallback_degrades_quietly(vector_chunks):
    retriever = EmbeddingKnowledgeRetriever(_ExplodingEmbeddings(), chunks=vector_chunks)
    result = retriever.search("password")
    assert isinstance(result, RetrievalResult) and result.has_grounding is False


def test_both_retrievers_are_interchangeable(settings, vector_chunks):
    """Liskov: the agent must not be able to tell them apart."""
    for candidate in (
        KeywordKnowledgeRetriever(settings.knowledge_base_dir),
        EmbeddingKnowledgeRetriever(_FakeEmbeddings(), chunks=vector_chunks),
    ):
        assert isinstance(candidate, KnowledgeRetriever)
        assert isinstance(candidate.search("password"), RetrievalResult)


# --------------------------------------------------------------------------- #
# Tool objects
# --------------------------------------------------------------------------- #


def test_billing_tools_have_the_names_the_spec_requires(repo, gateway):
    names = [t.name for t in make_billing_tools(repo, gateway)]
    assert names == ["check_subscription_status", "process_refund"]


def test_technical_tools_have_the_name_the_spec_requires(retriever):
    assert [t.name for t in make_technical_tools(retriever)] == ["search_knowledge_base"]


def test_technical_agent_has_no_access_to_billing_tools(retriever):
    """Scope is enforced by the tool list, not only by the prompt."""
    names = {t.name for t in make_technical_tools(retriever)}
    assert "process_refund" not in names and "check_subscription_status" not in names


def test_tools_expose_a_single_clean_argument(repo, gateway, retriever):
    tools = make_billing_tools(repo, gateway) + make_technical_tools(retriever)
    expected = {
        "check_subscription_status": ["user_id"],
        "process_refund": ["transaction_id"],
        "search_knowledge_base": ["query"],
    }
    for tool in tools:
        assert list(tool.args.keys()) == expected[tool.name]
        assert tool.description  # sent to the model; must not be empty


def test_subscription_tool_returns_a_quotable_sentence(repo, gateway):
    output = make_billing_tools(repo, gateway)[0].invoke({"user_id": "12345"})
    assert "expired" in output


def test_knowledge_tool_signals_no_results_explicitly(retriever):
    output = make_technical_tools(retriever)[0].invoke({"query": "capital of France"})
    assert output.startswith("NO_RESULTS")
