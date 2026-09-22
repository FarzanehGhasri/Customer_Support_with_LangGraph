"""
Step 2 tests -- the triage agent.

Runs entirely offline. The LLM path is exercised with a fake provider that
mimics ``with_structured_output``, so we test *our* logic (prompting, fallback,
coercion, state updates) without paying for or depending on a model.
"""

from __future__ import annotations

import pytest

from support_system.agents import TriageNode
from support_system.domain import Department, NextStep, TriageDecision, initial_state
from support_system.infrastructure.classification import (
    KeywordIntentClassifier,
    LLMIntentClassifier,
)
from support_system.interfaces import IntentClassifier
from support_system.prompts.triage import TRIAGE_SYSTEM_PROMPT, build_history_block

# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class _FakeStructuredRunnable:
    """Stands in for ``model.with_structured_output(TriageDecision)``."""

    def __init__(self, result, record: list) -> None:
        self._result = result
        self._record = record

    def invoke(self, messages):
        self._record.append(messages)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeProvider:
    """Implements the ChatModelProvider Protocol without any network."""

    def __init__(self, result) -> None:
        self._result = result
        self.calls: list = []

    def get_chat_model(self, *, temperature=None):
        raise AssertionError("triage must use the structured model, not raw chat")

    def get_structured_model(self, schema, *, temperature=None):
        assert schema is TriageDecision
        self.recorded_temperature = temperature
        return _FakeStructuredRunnable(self._result, self.calls)


# --------------------------------------------------------------------------- #
# Keyword classifier -- also the offline substitute used elsewhere
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "message,expected",
    [
        # The three delivery scenarios from the assignment:
        ("How can I reset my password?", Department.TECHNICAL),
        ("My subscription is not working. My id is 12345", Department.BILLING),
        ("You stole my money! This service is useless! I want to talk to a manager",
         Department.BILLING),
        # Plus the ordinary cases:
        ("Hello there!", Department.GENERAL),
        ("The app crashes with error E-204", Department.TECHNICAL),
        ("I want a refund for transaction TXN-1001", Department.BILLING),
    ],
)
def test_keyword_classifier_routes_the_scenarios(message, expected):
    assert KeywordIntentClassifier().classify(message).department is expected


def test_subscription_wording_beats_technical_wording():
    """'my subscription is not working' matches both lists; Billing must win."""
    decision = KeywordIntentClassifier().classify("my subscription is not working")
    assert decision.department is Department.BILLING


def test_unmatched_message_is_general_with_low_confidence():
    decision = KeywordIntentClassifier().classify("qwertyuiop")
    assert decision.department is Department.GENERAL
    assert decision.confidence < 0.5


def test_keyword_matching_is_whole_word():
    """'app' must not match inside 'happy'."""
    assert not KeywordIntentClassifier._contains(" happy customer ", "app")
    assert KeywordIntentClassifier._contains(" the app crashed ", "app")


def test_keyword_classifier_satisfies_the_protocol():
    assert isinstance(KeywordIntentClassifier(), IntentClassifier)


# --------------------------------------------------------------------------- #
# LLM classifier
# --------------------------------------------------------------------------- #


def test_llm_classifier_returns_the_models_decision():
    expected = TriageDecision(department=Department.BILLING, reasoning="refund", confidence=0.9)
    classifier = LLMIntentClassifier(_FakeProvider(expected))
    assert classifier.classify("I want my money back") == expected


def test_llm_classifier_forces_temperature_zero():
    """Routing must be reproducible across runs."""
    provider = _FakeProvider(TriageDecision(department=Department.GENERAL))
    LLMIntentClassifier(provider).classify("hi")
    assert provider.recorded_temperature == 0.0


def test_llm_classifier_sends_system_prompt_and_message():
    provider = _FakeProvider(TriageDecision(department=Department.GENERAL))
    LLMIntentClassifier(provider).classify("Where is my invoice?")
    (system_role, system_text), (human_role, human_text) = provider.calls[0]
    assert system_role == "system" and system_text == TRIAGE_SYSTEM_PROMPT
    assert human_role == "human" and "Where is my invoice?" in human_text


def test_llm_classifier_includes_history_only_when_present():
    provider = _FakeProvider(TriageDecision(department=Department.GENERAL))
    classifier = LLMIntentClassifier(provider)
    classifier.classify("hi", history="")
    assert "Conversation so far" not in provider.calls[0][1][1]
    classifier.classify("hi", history="user: earlier message")
    assert "Conversation so far" in provider.calls[1][1][1]


def test_llm_classifier_accepts_a_dict_from_the_provider():
    """Some gateways return a dict rather than the Pydantic object."""
    provider = _FakeProvider({"department": "Technical", "reasoning": "bug", "confidence": 0.8})
    decision = LLMIntentClassifier(provider).classify("it crashes")
    assert isinstance(decision, TriageDecision)
    assert decision.department is Department.TECHNICAL


def test_llm_classifier_degrades_instead_of_crashing():
    """A dead endpoint must not kill the conversation."""
    classifier = LLMIntentClassifier(_FakeProvider(RuntimeError("502 Bad Gateway")))
    decision = classifier.classify("anything")
    assert decision.department is Department.GENERAL
    assert decision.confidence == 0.0
    assert "RuntimeError" in decision.reasoning


def test_llm_classifier_never_routes_back_to_triage():
    """TRIAGE is reserved for specialists bouncing a request back."""
    provider = _FakeProvider(TriageDecision(department=Department.TRIAGE))
    assert LLMIntentClassifier(provider).classify("hello").department is Department.GENERAL


def test_empty_message_skips_the_api_call():
    provider = _FakeProvider(TriageDecision(department=Department.BILLING))
    decision = LLMIntentClassifier(provider).classify("   ")
    assert provider.calls == []           # no paid request was made
    assert decision.confidence == 0.0


def test_building_the_classifier_needs_no_credentials():
    """Construction must be side-effect free; the model is built on first use."""
    from support_system.config import Settings
    from support_system.infrastructure.llm import LangChainModelProvider

    LLMIntentClassifier(LangChainModelProvider(Settings(api_key="")))  # must not raise


# --------------------------------------------------------------------------- #
# Prompt helper
# --------------------------------------------------------------------------- #


def test_history_block_is_empty_for_a_fresh_conversation():
    assert build_history_block("") == ""
    assert build_history_block("   ") == ""
    assert "earlier" in build_history_block("user: earlier")


# --------------------------------------------------------------------------- #
# Triage node (state handling)
# --------------------------------------------------------------------------- #


@pytest.fixture()
def node() -> TriageNode:
    return TriageNode(KeywordIntentClassifier(), max_attempts=2)


def test_node_sets_department_and_next_step(node):
    update = node(initial_state("How can I reset my password?", "12345"))
    assert update["department"] == Department.TECHNICAL
    assert update["next_step"] == NextStep.TECHNICAL


def test_node_returns_a_partial_update_only(node):
    """Returning the whole state would break the append-only messages reducer."""
    update = node(initial_state("hello", "1"))
    assert "sentiment" not in update
    assert "final_response" not in update


def test_node_appends_exactly_one_transcript_line(node):
    update = node(initial_state("I need a refund", "1"))
    assert len(update["messages"]) == 1
    assert update["messages"][0].startswith("triage:")


def test_node_counts_attempts(node):
    state = initial_state("I need a refund", "1")
    assert node(state)["triage_attempts"] == 1
    state["triage_attempts"] = 1
    assert node(state)["triage_attempts"] == 2


def test_node_stops_bouncing_after_max_attempts(node):
    """Guards against specialist -> triage -> specialist ping-pong."""
    state = initial_state("something ambiguous about a refund", "1")
    state["triage_attempts"] = 2
    update = node(state)
    assert update["next_step"] == NextStep.GENERAL
    assert "unsuccessful routing attempts" in update["messages"][0]


def test_node_passes_history_to_the_classifier_on_a_retry():
    seen: list[str] = []

    class _Recorder:
        def classify(self, user_message, *, history=""):
            seen.append(history)
            return TriageDecision(department=Department.GENERAL)

    node = TriageNode(_Recorder(), max_attempts=3)
    state = initial_state("first try", "1")
    node(state)                       # attempt 1 -> no history
    state["triage_attempts"] = 1
    node(state)                       # attempt 2 -> history supplied
    assert seen[0] == ""
    assert "first try" in seen[1]


def test_node_falls_back_to_the_transcript_when_user_query_is_absent(node):
    state = initial_state("I need a refund", "1")
    del state["user_query"]
    assert node(state)["department"] == Department.BILLING


def test_node_is_reusable_across_conversations(node):
    """No per-conversation state may be kept on the node itself."""
    first = node(initial_state("reset my password", "1"))
    second = node(initial_state("I need a refund", "2"))
    assert first["department"] == Department.TECHNICAL
    assert second["department"] == Department.BILLING
