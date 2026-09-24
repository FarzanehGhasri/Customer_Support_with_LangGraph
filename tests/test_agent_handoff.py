"""
The hand-off between agents must be visible to the customer.

The spec describes a company where a receptionist examines requests and refers
them to the relevant specialist. A customer who is silently re-routed sees none
of that, so two things are made explicit:

1. reception says which team is taking over, and
2. that specialist introduces itself and refers to what was asked.
"""

from __future__ import annotations

import pytest

from support_system.agents import BillingAgent, GeneralAgent, TechnicalAgent, TriageNode
from support_system.domain import Department, initial_state
from support_system.graph import build_application
from support_system.infrastructure.classification import KeywordIntentClassifier
from support_system.infrastructure.composition import TemplateResponseComposer
from support_system.infrastructure.retrieval import KeywordKnowledgeRetriever


@pytest.fixture()
def app():
    return build_application(force_offline=True)


# --------------------------------------------------------------------------- #
# Reception announces the hand-off
# --------------------------------------------------------------------------- #


def test_a_billing_problem_announces_the_billing_team(app):
    """The exact complaint from the bug report."""
    state = app.run("I have a billing problem", thread_id="t")
    assert state["department"] == Department.BILLING
    assert "Billing team" in state["final_response"]


def test_a_technical_question_announces_the_technical_team(app):
    state = app.run("How can I reset my password?", thread_id="t")
    assert "Technical Support team" in state["final_response"]


def test_the_notice_names_the_department_that_will_handle_it(app):
    for message, expected in [
        ("I have a billing problem", "Billing team"),
        ("the app crashes with error E-204", "Technical Support team"),
    ]:
        state = app.run(message, thread_id=expected)
        assert expected in state["final_response"]


def test_the_notice_appears_only_when_the_department_changes(app):
    """Repeating it every turn would be noise; a move is what matters."""
    first = app.run("I have a billing problem", thread_id="t")
    assert "Connecting you to" in first["final_response"]

    app.run("12345", thread_id="t")
    again = app.run("is my plan renewing?", thread_id="t")
    assert "Connecting you to" not in again["final_response"]


def test_moving_between_departments_announces_each_move(app):
    app.run("I have a billing problem", thread_id="t")
    app.run("12345", thread_id="t")
    technical = app.run("How can I reset my password?", thread_id="t")
    assert "Technical Support team" in technical["final_response"]

    billing_again = app.run("is my subscription active?", thread_id="t")
    assert "Billing team" in billing_again["final_response"]


def test_the_notice_is_consumed_and_not_repeated(app):
    app.run("I have a billing problem", thread_id="t")
    assert app.state("t")["routing_notice"] == ""


# --------------------------------------------------------------------------- #
# The specialist introduces itself
# --------------------------------------------------------------------------- #


def test_the_billing_specialist_introduces_itself(app):
    state = app.run("I have a billing problem", thread_id="t")
    assert "Billing Specialist" in state["final_response"]


def test_the_technical_specialist_introduces_itself(app):
    state = app.run("How can I reset my password?", thread_id="t")
    assert "Technical Support Specialist" in state["final_response"]


def test_an_agent_introduces_itself_only_once(app):
    first = app.run("I have a billing problem", thread_id="t")
    assert "Hello, I am the" in first["final_response"]

    app.run("12345", thread_id="t")
    later = app.run("is my plan renewing?", thread_id="t")
    assert "Hello, I am the" not in later["final_response"]


def test_each_agent_introduces_itself_the_first_time_it_speaks(app):
    billing = app.run("I have a billing problem", thread_id="t")
    assert "Billing Specialist" in billing["final_response"]

    app.run("12345", thread_id="t")
    technical = app.run("How can I reset my password?", thread_id="t")
    assert "Technical Support Specialist" in technical["final_response"]


def test_a_multi_line_answer_is_separated_from_the_introduction(app):
    """Without this the intro runs into a markdown heading."""
    state = app.run("How can I reset my password?", thread_id="t")
    body = state["final_response"]
    assert "Specialist.\n" in body
    assert "Specialist. ##" not in body


# --------------------------------------------------------------------------- #
# Node-level behaviour
# --------------------------------------------------------------------------- #


def test_has_spoken_reads_the_transcript():
    agent = GeneralAgent(TemplateResponseComposer())
    state = initial_state("hi", "1")
    assert not agent.has_spoken(state)
    state["messages"].append("general: hello there")
    assert agent.has_spoken(state)


def test_the_introduction_is_empty_after_the_agent_has_spoken():
    agent = GeneralAgent(TemplateResponseComposer())
    state = initial_state("hi", "1")
    assert agent.introduction(state)
    state["messages"].append("general: hello there")
    assert agent.introduction(state) == ""


def test_the_prompt_asks_the_model_to_introduce_itself():
    """In live mode the wording is the model's, so the instruction must be sent."""
    agent = GeneralAgent(TemplateResponseComposer())
    instruction = agent.intro_instruction(initial_state("hi", "1"))
    assert "introducing yourself" in instruction
    assert agent.display_name in instruction


def test_every_agent_has_a_customer_facing_name(app, settings=None):
    from support_system.config import Settings

    cfg = Settings.from_env()
    agents = [
        TriageNode(KeywordIntentClassifier()),
        BillingAgent(
            __import__(
                "support_system.infrastructure.planning", fromlist=["x"]
            ).RuleBasedBillingPlanner(),
            app.repository, app.gateway, TemplateResponseComposer(),
        ),
        TechnicalAgent(KeywordKnowledgeRetriever(cfg.knowledge_base_dir),
                       TemplateResponseComposer()),
        GeneralAgent(TemplateResponseComposer()),
    ]
    for agent in agents:
        assert agent.display_name and agent.display_name != "Support"
        # A customer should hear a job title, not a graph node name.
        assert agent.display_name != agent.name


def test_triage_sets_no_notice_when_the_department_is_unchanged():
    node = TriageNode(KeywordIntentClassifier())
    state = initial_state("I want a refund", "1")
    state["department"] = Department.BILLING.value
    assert node(state)["routing_notice"] == ""


# --------------------------------------------------------------------------- #
# The spec's scenarios must still hold
# --------------------------------------------------------------------------- #


def test_scenario_one_still_returns_the_documented_steps(app):
    state = app.run("How can I reset my password?", user_id="12345", thread_id="s1")
    assert state["department"] == Department.TECHNICAL
    assert "Forgot password" in state["final_response"]


def test_scenario_two_still_reports_the_expired_subscription(app):
    state = app.run(
        "My subscription is not working. My id is 12345", user_id="12345", thread_id="s2"
    )
    assert state["tool_calls"] == ["check_subscription_status(12345) -> expired"]
    assert "expired" in state["final_response"]


def test_scenario_three_still_withholds_the_reply_and_escalates(app):
    state = app.run(
        "You stole my money! This service is useless! I want to talk to a manager",
        user_id="12345", thread_id="s3",
    )
    assert state["sentiment"] == "Negative"
    assert app.is_interrupted("s3")
    # An angry customer gets no automated reply -- not even a routing notice.
    assert not state.get("final_response")
