"""
Ask -> look the id up in the customer records -> answer.

When a billing request needs an id the customer has not given, the agent must
ask for it, treat the next message as the answer, search the records, and say
plainly when no such customer exists.
"""

from __future__ import annotations

import pytest

from support_system.cli import ChatSession
from support_system.domain import Awaiting, Department
from support_system.graph import build_application


@pytest.fixture()
def app():
    return build_application(force_offline=True)


def say(app, message, thread="t"):
    return app.run(message, thread_id=thread)


# --------------------------------------------------------------------------- #
# Asking
# --------------------------------------------------------------------------- #


def test_a_subscription_question_without_an_id_asks_for_one(app):
    state = say(app, "My subscription is not working")
    assert state["awaiting"] == Awaiting.ACCOUNT_ID
    assert "account id" in state["final_response"].lower()
    assert state["tool_calls"] == []          # nothing was looked up yet


def test_the_question_names_the_expected_format(app):
    """A customer who does not know what an 'account ID' is needs an example."""
    assert "12345" in say(app, "is my subscription active?")["final_response"]


def test_a_refund_without_an_id_asks_for_the_transaction_id(app):
    state = say(app, "I want a refund")
    assert state["awaiting"] == Awaiting.TRANSACTION_ID
    assert "transaction id" in state["final_response"].lower()
    assert state["tool_calls"] == []


def test_an_id_given_up_front_is_not_asked_for(app):
    state = say(app, "My subscription is not working. My id is 12345")
    assert state["awaiting"] == Awaiting.NOTHING
    assert state["tool_calls"] == ["check_subscription_status(12345) -> expired"]


# --------------------------------------------------------------------------- #
# Looking the id up
# --------------------------------------------------------------------------- #


def test_a_known_id_is_looked_up_and_answered(app):
    say(app, "My subscription is not working")
    state = say(app, "12345")
    assert state["tool_calls"] == ["check_subscription_status(12345) -> expired"]
    assert "expired" in state["final_response"]
    assert state["awaiting"] == Awaiting.NOTHING       # the question is closed


@pytest.mark.parametrize(
    "user_id,expected",
    [("12345", "expired"), ("67890", "active"), ("11111", "trial"), ("22222", "cancelled")],
)
def test_every_id_in_the_records_resolves(app, user_id, expected):
    say(app, "is my subscription active?", thread=user_id)
    state = say(app, user_id, thread=user_id)
    assert expected in state["final_response"]


def test_an_id_that_is_not_in_the_records_is_reported_not_invented(app):
    say(app, "My subscription is not working")
    state = say(app, "99999")
    assert state["tool_calls"][-1] == "check_subscription_status(99999) -> not found"
    assert "99999" in state["final_response"]
    assert "could not find" in state["final_response"].lower()
    # No account details may appear for a customer that does not exist.
    for leak in ("Pro Monthly", "Team Annual", "expired", "active"):
        assert leak not in state["final_response"]


def test_after_an_unknown_id_a_corrected_one_still_works(app):
    say(app, "My subscription is not working")
    say(app, "99999")
    assert app.awaiting("t") is Awaiting.ACCOUNT_ID     # still waiting
    state = say(app, "12345")
    assert "expired" in state["final_response"]
    assert state["awaiting"] == Awaiting.NOTHING


def test_the_id_is_extracted_from_a_sentence_not_just_a_bare_number(app):
    say(app, "My subscription is not working")
    state = say(app, "sure, it's 12345 I think")
    assert state["tool_calls"][-1].startswith("check_subscription_status(12345)")


def test_a_looked_up_id_is_remembered_for_the_rest_of_the_conversation(app):
    say(app, "My subscription is not working")
    say(app, "12345")
    state = say(app, "and is my plan renewing?")
    assert state["user_id"] == "12345"
    assert "check_subscription_status(12345)" in state["tool_calls"][-1]


# --------------------------------------------------------------------------- #
# Refunds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "transaction,outcome",
    [("TXN-1001", "approved"), ("TXN-1002", "refused"), ("TXN-9999", "refused")],
)
def test_the_refund_id_is_checked_against_the_records(app, transaction, outcome):
    say(app, "I want a refund", thread=transaction)
    state = say(app, transaction, thread=transaction)
    assert state["tool_calls"][-1] == f"process_refund({transaction}) -> {outcome}"


def test_an_unknown_transaction_is_never_invented(app):
    say(app, "I want a refund")
    state = say(app, "TXN-9999")
    assert "no such transaction" in state["final_response"].lower()


# --------------------------------------------------------------------------- #
# Routing of the follow-up
# --------------------------------------------------------------------------- #


def test_a_bare_id_goes_back_to_billing_not_through_triage(app):
    """Sent through triage, '12345' would be classified as small talk."""
    say(app, "My subscription is not working")
    state = say(app, "12345")
    assert state["department"] == Department.BILLING


def test_the_original_question_is_remembered_while_asking(app):
    state = say(app, "My subscription is not working")
    assert state["pending_query"] == "My subscription is not working"
    assert state["pending_department"] == Department.BILLING


def test_the_follow_up_state_is_cleared_once_answered(app):
    say(app, "My subscription is not working")
    state = say(app, "12345")
    assert state["awaiting"] == ""
    assert state["pending_action"] == ""
    assert state["pending_department"] == ""
    assert state["ask_attempts"] == 0


# --------------------------------------------------------------------------- #
# The customer does not answer
# --------------------------------------------------------------------------- #


def test_a_non_answer_is_asked_about_once_more(app):
    say(app, "is my subscription active?")
    state = say(app, "hmm let me think")
    assert state["awaiting"] == Awaiting.ACCOUNT_ID
    assert state["ask_attempts"] == 2


def test_the_system_stops_asking_after_the_limit(app):
    """Nobody should be asked the same question forever."""
    say(app, "is my subscription active?")
    say(app, "hold on")
    state = say(app, "still looking")
    assert state["awaiting"] == Awaiting.NOTHING       # gave up
    assert "cannot check" in state["final_response"].lower()


def test_after_giving_up_the_conversation_continues_normally(app):
    say(app, "is my subscription active?")
    say(app, "hold on")
    say(app, "still looking")
    state = say(app, "How can I reset my password?")
    assert state["department"] == Department.TECHNICAL
    assert "Forgot password" in state["final_response"]


# --------------------------------------------------------------------------- #
# Through the chat REPL
# --------------------------------------------------------------------------- #


def test_the_repl_starts_without_an_account_id(app):
    """With an id pre-set, the bot would never ask -- the flow would be invisible."""
    assert ChatSession(app).user_id == ""


def test_the_repl_walks_the_whole_flow(app):
    session = ChatSession(app)

    asked = session.handle("My subscription is not working")
    assert "account id" in asked.text.lower()

    unknown = session.handle("99999")
    assert "could not find" in unknown.text.lower()
    assert "not found" in unknown.trace

    answered = session.handle("12345")
    assert "expired" in answered.text
    assert "check_subscription_status(12345) -> expired" in answered.trace


def test_setting_the_account_id_up_front_skips_the_question(app):
    session = ChatSession(app)
    session.handle("/user 67890")
    turn = session.handle("is my subscription active?")
    assert "check_subscription_status(67890)" in turn.trace
    assert "active" in turn.text
