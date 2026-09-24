"""
Step 1 regression tests.

These run with **no API key and no LangChain installed** -- that is the point.
If they ever start needing a network call, a layering rule has been broken.
"""

from __future__ import annotations

import operator
from typing import get_args, get_type_hints

import pytest

from support_system.config import Settings
from support_system.domain import (
    Department,
    HumanDecision,
    KnowledgeSnippet,
    NextStep,
    RefundReceipt,
    RetrievalResult,
    Sentiment,
    SentimentAssessment,
    SubscriptionStatus,
    SupportState,
    TriageDecision,
    initial_state,
    transcript,
)
from support_system.interfaces import (
    ChatModelProvider,
    HumanReviewer,
    IntentClassifier,
    KnowledgeRetriever,
    RefundGateway,
    SentimentAnalyzer,
    SubscriptionRepository,
    SupportNode,
)

# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


def test_enum_values_match_the_spec_strings():
    """The state stores plain strings; the enums must produce exactly those."""
    assert Department.BILLING == "Billing"
    assert Department.TECHNICAL == "Technical"
    assert Sentiment.NEGATIVE == "Negative"
    assert NextStep.HUMAN_REVIEW == "human_review"


def test_only_negative_sentiment_escalates():
    assert Sentiment.NEGATIVE.needs_human
    assert not Sentiment.NEUTRAL.needs_human
    assert not Sentiment.POSITIVE.needs_human


@pytest.mark.parametrize(
    "department,expected",
    [
        (Department.BILLING, NextStep.BILLING),
        (Department.TECHNICAL, NextStep.TECHNICAL),
        (Department.GENERAL, NextStep.GENERAL),
        (Department.TRIAGE, NextStep.TRIAGE),
    ],
)
def test_department_maps_to_a_node(department, expected):
    """Every department must have a destination -- no unroutable states."""
    assert NextStep.for_department(department) is expected


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


def test_state_has_the_five_keys_required_by_the_assignment():
    hints = get_type_hints(SupportState, include_extras=True)
    for key in ("messages", "user_id", "sentiment", "department", "next_step"):
        assert key in hints


def test_messages_channel_uses_the_operator_add_reducer():
    """LangGraph reads this annotation to append instead of overwrite."""
    hints = get_type_hints(SupportState, include_extras=True)
    assert operator.add in get_args(hints["messages"])


def test_initial_state_starts_at_reception():
    state = initial_state("My subscription is not working. My id is 12345", "12345")
    assert state["department"] == Department.TRIAGE
    assert state["next_step"] == NextStep.TRIAGE
    assert state["sentiment"] == Sentiment.NEUTRAL
    assert state["escalated"] is False
    assert transcript(state).startswith("user: My subscription")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


def test_triage_decision_rejects_an_unknown_department():
    with pytest.raises(ValueError):
        TriageDecision(department="Marketing")


def test_triage_schema_exposes_the_allowed_departments_to_the_llm():
    """with_structured_output sends this schema to the model."""
    schema = TriageDecision.model_json_schema()
    enum_values = schema["$defs"]["Department"]["enum"]
    assert set(enum_values) == {"Billing", "Technical", "General", "Triage"}


def test_sentiment_assessment_drives_the_interrupt():
    assert SentimentAssessment(sentiment=Sentiment.NEGATIVE).requires_human
    assert not SentimentAssessment(sentiment=Sentiment.POSITIVE).requires_human


def test_expired_subscription_is_not_active():
    status = SubscriptionStatus(
        user_id="12345", found=True, plan="Pro Monthly",
        status="expired", expires_on="2025-08-14",
    )
    assert not status.is_active
    assert "expired" in status.summary()


def test_missing_customer_is_reported_not_raised():
    status = SubscriptionStatus(user_id="00000", found=False)
    assert "No subscription record" in status.summary()


def test_refund_receipt_summarises_both_outcomes():
    ok = RefundReceipt(transaction_id="TXN-1001", approved=True, amount=19.99, reference="RF-1")
    assert "approved" in ok.summary()
    nope = RefundReceipt(transaction_id="TXN-1002", approved=False, reason="outside the window")
    assert "refused" in nope.summary()


def test_retrieval_sorts_snippets_by_score():
    result = RetrievalResult(
        query="password",
        snippets=[
            KnowledgeSnippet(content="low", source="a.md", score=0.2),
            KnowledgeSnippet(content="high", source="b.md", score=0.9),
        ],
        has_grounding=True,
    )
    assert [s.content for s in result.snippets] == ["high", "low"]
    assert "b.md" in result.as_context()


def test_empty_retrieval_signals_no_grounding():
    """This is what stops the technical agent from hallucinating."""
    result = RetrievalResult(query="quantum toaster")
    assert not result.has_grounding
    assert "no relevant documentation" in result.as_context()


def test_human_decision_can_approve_or_override():
    draft = "Have you tried turning it off and on again?"
    assert HumanDecision(approved=True).resolve(draft) == draft
    override = HumanDecision(approved=False, replacement_response="I am the senior manager.")
    assert override.resolve(draft) == "I am the senior manager."


# --------------------------------------------------------------------------- #
# Interfaces -- runtime_checkable Protocols, verified with minimal fakes
# --------------------------------------------------------------------------- #


class _FakeRetriever:
    def search(self, query, *, top_k=3):  # noqa: D102
        return RetrievalResult(query=query)


class _FakeRepo:
    def get_status(self, user_id):  # noqa: D102
        return SubscriptionStatus(user_id=user_id, found=False)


class _FakeGateway:
    def process_refund(self, transaction_id):  # noqa: D102
        return RefundReceipt(transaction_id=transaction_id, approved=False, reason="fake")


class _FakeClassifier:
    def classify(self, user_message, *, history=""):  # noqa: D102
        return TriageDecision(department=Department.GENERAL)


class _FakeSentiment:
    def analyze(self, user_message, *, history=""):  # noqa: D102
        return SentimentAssessment(sentiment=Sentiment.NEUTRAL)


class _FakeReviewer:
    def review(self, state):  # noqa: D102
        return HumanDecision(approved=True)


class _FakeNode:
    name = "fake"

    def __call__(self, state):  # noqa: D102
        return {}


@pytest.mark.parametrize(
    "instance,protocol",
    [
        (_FakeRetriever(), KnowledgeRetriever),
        (_FakeRepo(), SubscriptionRepository),
        (_FakeGateway(), RefundGateway),
        (_FakeClassifier(), IntentClassifier),
        (_FakeSentiment(), SentimentAnalyzer),
        (_FakeReviewer(), HumanReviewer),
        (_FakeNode(), SupportNode),
    ],
)
def test_a_plain_object_can_satisfy_each_protocol(instance, protocol):
    """No inheritance required -- this is what keeps the layers decoupled."""
    assert isinstance(instance, protocol)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_settings_pick_the_default_model_for_the_provider(monkeypatch):
    monkeypatch.delenv("SUPPORT_MODEL", raising=False)
    assert Settings.from_env(provider="anthropic").model == "claude-opus-5"
    assert Settings.from_env(provider="openai").model == "gpt-4o-mini"


def test_settings_never_leak_the_api_key():
    described = Settings(provider="openai", api_key="sk-supersecretvalue").describe()
    assert "supersecretvalue" not in described


def test_missing_key_raises_a_helpful_error():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        Settings(provider="openai", api_key="").require_credentials()


def test_provider_object_satisfies_the_llm_protocol():
    """Imported here so the rest of the suite runs without the infra layer."""
    from support_system.infrastructure.llm import LangChainModelProvider, available_providers

    provider = LangChainModelProvider(Settings(api_key="dummy"))
    assert isinstance(provider, ChatModelProvider)
    assert {"openai", "anthropic", "google"} <= set(available_providers())


def test_unknown_provider_fails_with_a_useful_message():
    from support_system.infrastructure.llm import LangChainModelProvider

    with pytest.raises(ValueError, match="Unknown provider"):
        LangChainModelProvider(Settings(provider="acme", api_key="dummy")).get_chat_model()


# --------------------------------------------------------------------------- #
# Data assets
# --------------------------------------------------------------------------- #


def test_knowledge_base_exists_and_covers_the_first_scenario():
    settings = Settings.from_env()
    docs = list(settings.knowledge_base_dir.glob("*.md"))
    assert len(docs) >= 5
    combined = "\n".join(d.read_text(encoding="utf-8").lower() for d in docs)
    assert "forgot password" in combined  # scenario 1 must be answerable from RAG


def test_mock_subscription_for_user_12345_is_expired():
    """Scenario 2 of the assignment expects exactly this answer."""
    import json

    settings = Settings.from_env()
    data = json.loads(settings.subscriptions_file.read_text(encoding="utf-8"))
    record = next(c for c in data["customers"] if c["user_id"] == "12345")
    assert record["status"] == "expired"


def test_support_model_does_not_leak_across_an_explicit_provider_override(monkeypatch):
    """A model id set for one provider must not be applied to another.

    Regression: with SUPPORT_MODEL=gpt-4o-mini in .env, asking for the Anthropic
    provider used to hand Anthropic an OpenAI model name.
    """
    monkeypatch.setenv("SUPPORT_PROVIDER", "openai")
    monkeypatch.setenv("SUPPORT_MODEL", "gpt-4o-mini")
    assert Settings.from_env().model == "gpt-4o-mini"
    assert Settings.from_env(provider="anthropic").model == "claude-opus-5"
    assert Settings.from_env(provider="google").model == "gemini-2.0-flash"


def test_explicit_model_override_always_wins(monkeypatch):
    monkeypatch.setenv("SUPPORT_MODEL", "gpt-4o-mini")
    assert Settings.from_env(provider="anthropic", model="claude-sonnet-5").model == "claude-sonnet-5"


def test_dotenv_survives_a_utf8_byte_order_mark(tmp_path, monkeypatch):
    """PowerShell's `-Encoding utf8` writes a BOM; it must not join the first key.

    Without utf-8-sig the first variable is named "﻿OPENAI_API_KEY" and the
    real one silently looks unset.
    """
    from support_system.config import load_dotenv

    env_file = tmp_path / ".env"
    env_file.write_bytes(
        b"\xef\xbb\xbfOPENAI_API_KEY=sk-from-powershell\r\nSUPPORT_MODEL=gpt-4o-mini\r\n"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SUPPORT_MODEL", raising=False)

    applied = load_dotenv(env_file, override=True)
    assert applied["OPENAI_API_KEY"] == "sk-from-powershell"   # not "﻿OPENAI_API_KEY"
    assert applied["SUPPORT_MODEL"] == "gpt-4o-mini"           # CRLF stripped
