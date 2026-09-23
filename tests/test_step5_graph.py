"""
Step 5 tests -- guardrail, human-in-the-loop, and the assembled graph.

Everything runs offline against the deterministic components, so these tests
cover the same code paths a graded run would take, minus the model.
"""

from __future__ import annotations

import pytest

from support_system.agents import HumanReviewNode, SentimentGuardrail
from support_system.domain import (
    Department,
    HumanDecision,
    NextStep,
    Sentiment,
    SentimentAssessment,
    initial_state,
)
from support_system.graph import ROUTES, build_application, node_names
from support_system.infrastructure.classification import KeywordSentimentAnalyzer
from support_system.infrastructure.human import (
    AutoApproveReviewer,
    ScriptedReviewer,
)

ANGRY = "You stole my money! This service is useless! I want to talk to a manager"
MANAGER_REPLY = (
    "I am the senior manager. I apologise; I will personally follow up on your issue."
)


@pytest.fixture()
def guardrail() -> SentimentGuardrail:
    return SentimentGuardrail(KeywordSentimentAnalyzer())


@pytest.fixture()
def app():
    return build_application(force_offline=True)


# --------------------------------------------------------------------------- #
# Guardrail
# --------------------------------------------------------------------------- #


def test_calm_conversation_releases_the_draft(guardrail):
    state = initial_state("How can I reset my password?", "1")
    state["draft_response"] = "Click the reset link."
    update = guardrail(state)
    assert update["final_response"] == "Click the reset link."
    assert update["next_step"] == NextStep.FINISH
    assert update["escalated"] is False


def test_angry_conversation_withholds_the_draft(guardrail):
    """The core requirement: an angry customer gets no automated reply."""
    state = initial_state(ANGRY, "1")
    state["draft_response"] = "Your subscription has expired."
    update = guardrail(state)
    assert update["next_step"] == NextStep.HUMAN_REVIEW
    assert update["escalated"] is True
    assert "final_response" not in update


def test_guardrail_judges_the_customer_not_the_agent(guardrail):
    """Tone is read from the customer's words, never from the draft."""
    state = initial_state("Could you check my invoice please?", "1")
    state["draft_response"] = "This is terrible and useless and a scam."
    assert guardrail(state)["next_step"] == NextStep.FINISH


def test_guardrail_records_sentiment_even_when_disabled():
    disabled = SentimentGuardrail(KeywordSentimentAnalyzer(), enabled=False)
    state = initial_state(ANGRY, "1")
    state["draft_response"] = "draft"
    update = disabled(state)
    assert update["sentiment"] == Sentiment.NEGATIVE     # still measured
    assert update["next_step"] == NextStep.FINISH        # but not escalated


def test_analyzer_failure_does_not_escalate_everyone():
    """An outage must not send every conversation to a human."""

    class _Broken:
        def analyze(self, user_message, *, history=""):
            return SentimentAssessment(sentiment=Sentiment.NEUTRAL, reasoning="unavailable")

    state = initial_state(ANGRY, "1")
    state["draft_response"] = "draft"
    assert SentimentGuardrail(_Broken())(state)["next_step"] == NextStep.FINISH


# --------------------------------------------------------------------------- #
# Human review node
# --------------------------------------------------------------------------- #


def test_injected_manager_reply_wins():
    """This is the update_state path the assignment requires."""
    state = initial_state(ANGRY, "1")
    state["draft_response"] = "automated draft"
    state["final_response"] = MANAGER_REPLY
    update = HumanReviewNode()(state)
    assert update["next_step"] == NextStep.FINISH
    assert "final_response" not in update  # the injected value is left untouched


def test_reviewer_can_override_the_draft():
    state = initial_state(ANGRY, "1")
    state["draft_response"] = "automated draft"
    reviewer = ScriptedReviewer(
        HumanDecision(approved=False, replacement_response=MANAGER_REPLY, note="handled")
    )
    assert HumanReviewNode(reviewer)(state)["final_response"] == MANAGER_REPLY


def test_reviewer_can_approve_the_draft():
    state = initial_state(ANGRY, "1")
    state["draft_response"] = "automated draft"
    update = HumanReviewNode(AutoApproveReviewer())(state)
    assert update["final_response"] == "automated draft"


def test_escalation_without_a_manager_holds_the_conversation():
    """No reviewer and no injected reply: send a holding message, not the draft."""
    state = initial_state(ANGRY, "1")
    state["draft_response"] = "automated draft"
    final = HumanReviewNode()(state)["final_response"]
    assert "escalated" in final.lower() and final != "automated draft"


# --------------------------------------------------------------------------- #
# Graph wiring
# --------------------------------------------------------------------------- #


def test_every_next_step_value_has_a_route():
    """A control value with no destination would strand a conversation."""
    for step in NextStep:
        assert step.value in ROUTES


def test_graph_registers_every_node(app):
    registered = set(app.graph.get_graph().nodes)
    assert set(node_names()) <= registered


def test_graph_declares_the_interrupt(app):
    """The spec's 'stop and wait for a human' is a property of the graph."""
    assert "human_review" in app.graph.get_graph().draw_mermaid()
    assert "__interrupt = before" in app.graph.get_graph().draw_mermaid()


# --------------------------------------------------------------------------- #
# The three delivery scenarios, end to end
# --------------------------------------------------------------------------- #


def test_scenario_one_technical_happy_path(app):
    state = app.run("How can I reset my password?", user_id="12345", thread_id="s1")
    assert state["department"] == Department.TECHNICAL
    assert state["escalated"] is False
    assert "search_knowledge_base" in state["tool_calls"][0]
    assert "Forgot password" in state["final_response"]
    assert not app.is_interrupted("s1")


def test_scenario_two_sensitive_billing_operation(app):
    state = app.run(
        "My subscription is not working. My id is 12345", user_id="12345", thread_id="s2"
    )
    assert state["department"] == Department.BILLING
    assert state["tool_calls"] == ["check_subscription_status(12345) -> expired"]
    assert "expired" in state["final_response"]
    assert not app.is_interrupted("s2")


def test_scenario_three_escalates_and_accepts_a_manager_reply(app):
    # 1. Triage says Billing, but the guardrail detects anger and halts.
    state = app.run(ANGRY, user_id="12345", thread_id="s3")
    assert state["department"] == Department.BILLING
    assert state["sentiment"] == Sentiment.NEGATIVE
    assert state["escalated"] is True
    assert app.is_interrupted("s3")
    assert app.next_nodes("s3") == ("human_review",)
    assert not state.get("final_response")      # nothing was sent

    # 2. The manager writes a reply with update_state, then the graph resumes.
    app.inject_manager_reply(MANAGER_REPLY, thread_id="s3")
    final = app.resume("s3")
    assert final["final_response"] == MANAGER_REPLY
    assert not app.is_interrupted("s3")
    assert any("manager:" in m for m in final["messages"])


def test_conversations_do_not_leak_between_threads(app):
    app.run("How can I reset my password?", user_id="1", thread_id="a")
    app.run("My subscription is not working. My id is 12345", user_id="12345", thread_id="b")
    assert app.state("a")["department"] == Department.TECHNICAL
    assert app.state("b")["department"] == Department.BILLING


def test_billing_bounce_back_reroutes_and_terminates(app):
    """A technical question that reaches billing must not loop forever."""
    state = app.run("How do I reset my password?", user_id="12345", thread_id="bounce")
    assert state["final_response"]
    assert state["triage_attempts"] <= app.settings.max_triage_attempts + 1


def test_unattended_mode_never_interrupts():
    unattended = build_application(
        force_offline=True, interrupt_before_human=False, reviewer=AutoApproveReviewer()
    )
    state = unattended.run(ANGRY, user_id="12345", thread_id="u")
    assert state["escalated"] is True
    assert not unattended.is_interrupted("u")
    assert state["final_response"]      # answered by the reviewer, not left hanging


def test_application_runs_offline_without_credentials():
    from support_system.config import Settings

    app = build_application(Settings(provider="openai", api_key=""))
    assert app.offline is True
    assert app.run("hello", thread_id="nokey")["final_response"]


# --------------------------------------------------------------------------- #
# Mode detection -- regression for a bug found by executing the notebook
# --------------------------------------------------------------------------- #


def test_probe_reports_a_missing_key():
    from support_system.config import Settings
    from support_system.infrastructure.llm import probe_provider

    ok, reason = probe_provider(Settings(provider="openai", api_key=""))
    assert ok is False and "OPENAI_API_KEY" in reason


def test_probe_reports_an_unknown_provider():
    from support_system.config import Settings
    from support_system.infrastructure.llm import probe_provider

    ok, reason = probe_provider(Settings(provider="acme", api_key="dummy"))
    assert ok is False and reason


def test_an_unreachable_endpoint_forces_offline_mode(monkeypatch):
    """Holding a key must not be mistaken for being able to use the model.

    Regression: the notebook ran "live" against an unreachable endpoint. Every
    component degraded quietly, so the graph completed and answered all three
    scenarios wrongly instead of failing honestly.
    """
    from support_system.config import Settings
    from support_system.graph import application as application_module

    monkeypatch.setattr(
        application_module, "probe_provider",
        lambda settings, **kwargs: (False, "ConnectionError: unreachable"),
    )
    app = build_application(Settings(provider="openai", api_key="sk-looks-real"))
    assert app.offline is True
    assert "unreachable" in app.offline_reason
    # ...and the fallback must actually work, not merely be selected.
    state = app.run("How can I reset my password?", user_id="1", thread_id="probe")
    assert state["department"] == Department.TECHNICAL
    assert "Forgot password" in state["final_response"]


def test_a_reachable_endpoint_selects_live_components(monkeypatch):
    from support_system.config import Settings
    from support_system.graph import application as application_module
    from support_system.infrastructure.classification import LLMIntentClassifier

    monkeypatch.setattr(
        application_module, "probe_provider", lambda settings, **kwargs: (True, "")
    )
    app = build_application(Settings(provider="openai", api_key="sk-looks-real"))
    assert app.offline is False and app.offline_reason == ""


def test_probe_can_be_skipped(monkeypatch):
    """probe=False must not make a call -- used when the mode is already known."""
    from support_system.config import Settings
    from support_system.graph import application as application_module

    def _boom(*args, **kwargs):
        raise AssertionError("probe_provider must not be called when probe=False")

    monkeypatch.setattr(application_module, "probe_provider", _boom)
    app = build_application(Settings(provider="openai", api_key="sk-x"), probe=False)
    assert app.offline is False


def test_forced_offline_skips_the_probe(monkeypatch):
    from support_system.config import Settings
    from support_system.graph import application as application_module

    def _boom(*args, **kwargs):
        raise AssertionError("probe_provider must not be called when force_offline=True")

    monkeypatch.setattr(application_module, "probe_provider", _boom)
    app = build_application(Settings(api_key="sk-x"), force_offline=True)
    assert app.offline is True and app.offline_reason == "forced by the caller"
