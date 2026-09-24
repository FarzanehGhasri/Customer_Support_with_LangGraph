"""
Regression tests for a silent mis-routing bug reported from a live run.

Symptom: every message classified as General, no tool ever ran, an angry
customer was never escalated -- yet the replies were fluent model-written prose,
so the system looked healthy.

Cause: the gateway answered ordinary chat but not ``with_structured_output``.
The start-up probe only tested chat, so the system ran "live"; triage then hit
an exception on every message and fell back to its default department.

Fix: probe the two capabilities separately and, when only chat works, classify
with the deterministic components while the model keeps phrasing the answers.
"""

from __future__ import annotations

import pytest

from support_system.config import Settings
from support_system.domain import Department, Sentiment, TriageDecision
from support_system.graph import application as application_module
from support_system.graph import build_application
from support_system.infrastructure.classification import LLMIntentClassifier
from support_system.infrastructure.llm import ProviderCapabilities

ANGRY = "You stole my money! This service is useless! I want to talk to a manager"


class _ChatOnlyProvider:
    """A gateway that answers chat but rejects JSON-schema output."""

    def __init__(self) -> None:
        self.chat_calls = 0

    def get_chat_model(self, *, temperature=None):
        provider = self

        class _Model:
            def invoke(self, messages, **kwargs):
                provider.chat_calls += 1

                class _Response:
                    content = "Hello! How can I assist you today?"

                return _Response()

        return _Model()

    def get_structured_model(self, schema, *, temperature=None):
        class _Runnable:
            def invoke(self, messages):
                raise RuntimeError("400 Unsupported parameter: 'tools'")

        return _Runnable()


@pytest.fixture()
def chat_only(monkeypatch):
    """An application built against a chat-only endpoint."""
    monkeypatch.setattr(
        application_module,
        "probe_capabilities",
        lambda settings, **kwargs: ProviderCapabilities(
            chat=True, structured_error="RuntimeError: 400 Unsupported parameter: 'tools'"
        ),
    )
    # The composer would really call out, so keep it deterministic here.
    monkeypatch.setattr(
        application_module,
        "LLMResponseComposer",
        lambda provider, **kwargs: application_module.TemplateResponseComposer(),
    )
    return build_application(Settings(provider="openai", api_key="sk-looks-real"))


# --------------------------------------------------------------------------- #
# The underlying failure
# --------------------------------------------------------------------------- #


def test_a_chat_only_endpoint_collapses_every_message_to_the_fallback():
    """This is the bug, reproduced at the component level."""
    classifier = LLMIntentClassifier(_ChatOnlyProvider())
    for message in ("I have a billing problem", "My subscription is broken", ANGRY):
        decision = classifier.classify(message)
        assert decision.department is Department.GENERAL
        assert decision.confidence == 0.0       # the tell-tale sign


def test_confidence_zero_marks_a_fallback_not_a_judgement():
    """A real classification is never reported at zero confidence."""
    classifier = LLMIntentClassifier(_ChatOnlyProvider())
    assert classifier.classify("I want a refund").confidence == 0.0


# --------------------------------------------------------------------------- #
# The probe now catches it
# --------------------------------------------------------------------------- #


def test_capabilities_are_reported_separately():
    caps = ProviderCapabilities(chat=True, structured_error="tools unsupported")
    assert caps.chat and not caps.structured_output
    assert not caps.fully_usable
    assert "structured output unavailable" in caps.summary


def test_a_chat_only_endpoint_selects_degraded_mode(chat_only):
    assert chat_only.mode == "degraded"
    assert chat_only.degraded is True
    assert chat_only.offline is False           # the model is still used for wording
    assert "Unsupported parameter" in chat_only.offline_reason
    assert "DEGRADED" in chat_only.describe_mode()


def test_degraded_mode_classifies_with_the_deterministic_components(chat_only):
    """The whole point: routing must work even without structured output."""
    state = chat_only.run("I have a billing problem", thread_id="d1")
    assert state["department"] == Department.BILLING


def test_degraded_mode_still_detects_anger(chat_only):
    """Without this fix an angry customer was never escalated."""
    chat_only.run(ANGRY, thread_id="d2")
    assert chat_only.state("d2")["sentiment"] == Sentiment.NEGATIVE
    assert chat_only.is_interrupted("d2")


def test_a_fully_capable_endpoint_still_uses_the_model(monkeypatch):
    monkeypatch.setattr(
        application_module, "probe_capabilities",
        lambda settings, **kwargs: ProviderCapabilities(chat=True, structured_output=True),
    )
    app = build_application(Settings(provider="openai", api_key="sk-x"))
    assert app.mode == "live" and not app.degraded


def test_structured_output_that_answers_wrongly_is_treated_as_unusable():
    """Valid JSON is not enough -- the answer has to be right.

    A gateway that accepts the schema and returns a nonsense department would
    otherwise pass the probe and mis-route every conversation.
    """
    from support_system.infrastructure.llm import factory

    class _WrongAnswer:
        def get_chat_model(self, *, temperature=None):
            class _M:
                def invoke(self, messages, **kwargs):
                    class R:
                        content = "ok"
                    return R()
            return _M()

        def get_structured_model(self, schema, *, temperature=None):
            class _R:
                def invoke(self, messages):
                    # A refund request is unambiguously Billing.
                    return TriageDecision(department=Department.GENERAL)
            return _R()

    with pytest.MonkeyPatch.context() as monkey:
        monkey.setattr(factory, "LangChainModelProvider", lambda settings: _WrongAnswer())
        caps = factory.probe_capabilities(Settings(provider="openai", api_key="sk-x"))

    assert caps.chat is True
    assert caps.structured_output is False
    assert "wrong" in caps.structured_error


# --------------------------------------------------------------------------- #
# The conversation from the bug report, end to end
# --------------------------------------------------------------------------- #


def test_the_reported_conversation_now_routes_and_asks(chat_only):
    """'hello' -> 'I have a billing problem' must reach billing and ask for an id."""
    greeting = chat_only.run("hello", thread_id="report")
    assert greeting["department"] == Department.GENERAL      # a greeting really is General

    billing = chat_only.run("I have a billing problem", thread_id="report")
    assert billing["department"] == Department.BILLING       # was General before the fix
    assert billing["awaiting"] == "account_id"               # and it asks
    assert "account id" in billing["final_response"].lower()

    answered = chat_only.run("12345", thread_id="report")
    assert answered["tool_calls"][-1] == "check_subscription_status(12345) -> expired"
    assert "expired" in answered["final_response"]


def test_a_general_billing_question_is_not_turned_into_a_lookup(chat_only):
    """Asking what payment methods exist must not demand an account id."""
    state = chat_only.run("What payment methods do you accept?", thread_id="general-q")
    assert state["department"] == Department.BILLING
    assert state["awaiting"] == ""
    assert state["tool_calls"] == []
