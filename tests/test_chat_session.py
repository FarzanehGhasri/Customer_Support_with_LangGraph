"""Tests for the interactive chat session. No stdin, no network."""

from __future__ import annotations

import pytest

from support_system.cli import ChatSession
from support_system.graph import build_application

ANGRY = "You stole my money! This service is useless! I want to talk to a manager"


@pytest.fixture()
def session() -> ChatSession:
    return ChatSession(build_application(force_offline=True), user_id="12345")


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #


def test_a_message_produces_a_reply(session):
    turn = session.handle("How can I reset my password?")
    assert turn.kind == "reply"
    assert "Forgot password" in turn.text
    assert turn.state["department"] == "Technical"


def test_the_trace_names_the_department_and_the_tool(session):
    turn = session.handle("My subscription is not working")
    assert "Billing" in turn.trace
    assert "check_subscription_status(12345)" in turn.trace


def test_the_trace_shows_only_this_turns_tools(session):
    """Regression: tool_calls is append-only, so it used to replay every turn."""
    session.handle("How can I reset my password?")       # uses search_knowledge_base
    turn = session.handle("My subscription is not working")
    assert "check_subscription_status" in turn.trace
    assert "search_knowledge_base" not in turn.trace


def test_history_accumulates_within_one_conversation(session):
    session.handle("Hello there")
    turn = session.handle("How can I reset my password?")
    transcript = turn.state["messages"]
    assert any("Hello there" in m for m in transcript)
    assert any("reset my password" in m for m in transcript)


def test_each_turn_is_routed_afresh(session):
    """A greeting followed by a billing question must not stay in General."""
    assert session.handle("Hello there").state["department"] == "General"
    assert session.handle("My subscription is not working").state["department"] == "Billing"


def test_blank_input_does_nothing(session):
    turn = session.handle("   ")
    assert turn.kind == "info" and turn.text == ""


def test_a_failing_graph_does_not_kill_the_session(session, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("graph exploded")

    monkeypatch.setattr(session.app, "run", _boom)
    turn = session.handle("hello")
    assert turn.kind == "error" and "graph exploded" in turn.text


# --------------------------------------------------------------------------- #
# Escalation
# --------------------------------------------------------------------------- #


def test_an_angry_message_halts_and_asks_for_a_manager(session):
    turn = session.handle(ANGRY)
    assert turn.kind == "escalated"
    assert session.awaiting_manager
    assert not turn.state.get("final_response")      # nothing was sent


def test_the_manager_reply_finishes_the_conversation(session):
    session.handle(ANGRY)
    turn = session.resolve_escalation("I am the senior manager, I will follow up.")
    assert turn.kind == "reply"
    assert turn.text == "I am the senior manager, I will follow up."
    assert not session.awaiting_manager


def test_an_empty_manager_reply_is_refused(session):
    session.handle(ANGRY)
    assert session.resolve_escalation("   ").kind == "error"
    assert session.awaiting_manager          # still halted, nothing was sent


def test_escalation_works_in_dynamic_mode():
    app = build_application(force_offline=True, hitl_mode="dynamic")
    session = ChatSession(app, user_id="12345")
    turn = session.handle(ANGRY)
    assert turn.kind == "escalated"
    # The dynamic form also hands over the payload a review UI would render.
    assert turn.interrupt_payload["sentiment"] == "Negative"
    assert "agent_draft" in turn.interrupt_payload
    assert session.resolve_escalation("Manager here.").text == "Manager here."


def test_the_conversation_continues_after_an_escalation(session):
    session.handle(ANGRY)
    session.resolve_escalation("Manager here.")
    turn = session.handle("How can I reset my password?")
    assert turn.kind == "reply" and "Forgot password" in turn.text


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def test_help_lists_the_commands(session):
    text = session.handle("/help").text
    for command in ("/new", "/user", "/state", "/kb", "/quit"):
        assert command in text


def test_quit_is_reported(session):
    assert session.handle("/quit").kind == "quit"
    assert session.handle("/exit").kind == "quit"


def test_new_starts_a_clean_conversation(session):
    session.handle("Hello there")
    old_thread = session.thread_id
    session.handle("/new")
    assert session.thread_id != old_thread
    assert session.app.state(session.thread_id).get("messages", []) == []


def test_user_command_switches_the_account(session):
    turn = session.handle("/user 67890")
    assert session.user_id == "67890"
    assert "active" in turn.text                      # looked the account up
    reply = session.handle("is my subscription active?")
    assert "check_subscription_status(67890)" in reply.trace


def test_user_command_without_an_argument_reports_the_current_id(session):
    assert "12345" in session.handle("/user").text


def test_state_command_dumps_the_graph_state(session):
    session.handle("How can I reset my password?")
    text = session.handle("/state").text
    assert "department" in text and "sentiment" in text


def test_history_command_prints_the_transcript(session):
    session.handle("Hello there")
    assert "Hello there" in session.handle("/history").text


def test_history_is_empty_on_a_fresh_conversation(session):
    assert "nothing said yet" in session.handle("/history").text


def test_tools_command_lists_the_three_tools(session):
    text = session.handle("/tools").text
    for name in ("check_subscription_status", "process_refund", "search_knowledge_base"):
        assert name in text


def test_kb_command_reports_a_grounded_match(session):
    assert "password_reset.md" in session.handle("/kb reset password").text


def test_kb_command_reports_when_nothing_is_grounded(session):
    assert "No grounded match" in session.handle("/kb capital of France").text


def test_kb_command_without_an_argument_shows_usage(session):
    assert "Usage" in session.handle("/kb").text


def test_an_unknown_command_is_reported(session):
    turn = session.handle("/banana")
    assert turn.kind == "error" and "/help" in turn.text


def test_the_documented_examples_actually_behave_as_described(session):
    """The /help 'Try' list must not promise behaviour the system lacks."""
    assert session.handle("How can I reset my password?").state["department"] == "Technical"
    assert session.handle("The app crashes with error E-204").state["department"] == "Technical"

    ignorant = session.handle("How do I connect the app to my smart fridge?")
    assert ignorant.state["department"] == "Technical"
    assert "no grounding" in ignorant.trace
    assert "could not find" in ignorant.text.lower()

    assert session.handle("I want a refund for TXN-1001").trace.count("approved") == 1
    assert "refused" in session.handle("I want a refund for TXN-1002").trace
    assert session.handle(ANGRY).kind == "escalated"
