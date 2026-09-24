"""Step 4 tests -- the billing, technical and general specialists. Offline."""

from __future__ import annotations

import pytest

from support_system.agents import BillingAgent, GeneralAgent, TechnicalAgent
from support_system.config import Settings
from support_system.domain import (
    BillingAction,
    BillingPlan,
    Department,
    KnowledgeSnippet,
    NextStep,
    RetrievalResult,
    initial_state,
)
from support_system.infrastructure.billing import JsonSubscriptionRepository, MockRefundGateway
from support_system.infrastructure.composition import TemplateResponseComposer
from support_system.infrastructure.planning import RuleBasedBillingPlanner
from support_system.infrastructure.retrieval import KeywordKnowledgeRetriever
from support_system.interfaces import SupportNode


@pytest.fixture()
def settings() -> Settings:
    return Settings.from_env()


@pytest.fixture()
def composer() -> TemplateResponseComposer:
    return TemplateResponseComposer()


@pytest.fixture()
def billing(settings, composer) -> BillingAgent:
    return BillingAgent(
        RuleBasedBillingPlanner(),
        JsonSubscriptionRepository(settings.subscriptions_file),
        MockRefundGateway(settings.transactions_file),
        composer,
    )


@pytest.fixture()
def technical(settings, composer) -> TechnicalAgent:
    return TechnicalAgent(KeywordKnowledgeRetriever(settings.knowledge_base_dir), composer)


class _FixedPlanner:
    def __init__(self, plan: BillingPlan) -> None:
        self._plan = plan

    def plan(self, user_message: str) -> BillingPlan:
        return self._plan


# --------------------------------------------------------------------------- #
# Billing specialist
# --------------------------------------------------------------------------- #


def test_billing_is_a_support_node(billing):
    assert isinstance(billing, SupportNode) and billing.name == "billing"


def test_scenario_two_reports_the_expired_subscription(billing):
    """Assignment scenario 2 end to end through the node."""
    update = billing(initial_state("My subscription is not working. My id is 12345", "12345"))
    assert "expired" in update["draft_response"]
    assert update["tool_calls"] == ["check_subscription_status(12345) -> expired"]
    assert update["next_step"] == NextStep.GUARDRAIL


def test_billing_returns_technical_questions_to_triage(billing):
    """The spec's explicit requirement for this agent."""
    update = billing(initial_state("How can I reset my password?", "12345"))
    assert update["department"] == Department.TRIAGE
    assert update["next_step"] == NextStep.TRIAGE
    assert "draft_response" not in update  # it must not answer at all


def test_billing_processes_a_refund_with_a_valid_id(billing):
    update = billing(initial_state("I want a refund for TXN-1001", "12345"))
    assert "approved" in update["draft_response"]
    assert update["tool_calls"] == ["process_refund(TXN-1001) -> approved"]


def test_billing_refuses_a_refund_the_gateway_rejects(billing):
    update = billing(initial_state("Please refund TXN-1002", "67890"))
    assert "refused" in update["draft_response"]
    assert update["tool_calls"] == ["process_refund(TXN-1002) -> refused"]


def test_billing_never_refunds_a_transaction_id_the_customer_did_not_write(settings, composer):
    """A hallucinated id must never reach the gateway -- refunds are irreversible."""
    agent = BillingAgent(
        _FixedPlanner(BillingPlan(action=BillingAction.PROCESS_REFUND, argument="TXN-1001")),
        JsonSubscriptionRepository(settings.subscriptions_file),
        MockRefundGateway(settings.transactions_file),
        composer,
    )
    update = agent(initial_state("I am unhappy with my purchase", "12345"))
    # The important part: the gateway was never called with the invented id.
    assert update.get("tool_calls", []) == []
    # Instead of acting on it, the agent asks the customer for a real one.
    assert update["awaiting"] == "transaction_id"
    assert "transaction id" in update["draft_response"].lower()


def test_billing_uses_the_session_account_id_over_the_planners_guess(settings, composer):
    agent = BillingAgent(
        _FixedPlanner(BillingPlan(action=BillingAction.CHECK_SUBSCRIPTION, argument="99999")),
        JsonSubscriptionRepository(settings.subscriptions_file),
        MockRefundGateway(settings.transactions_file),
        composer,
    )
    update = agent(initial_state("what is my plan?", "12345"))
    assert "check_subscription_status(12345)" in update["tool_calls"][0]


def test_billing_asks_for_an_account_id_when_none_exists(settings, composer):
    agent = BillingAgent(
        _FixedPlanner(BillingPlan(action=BillingAction.CHECK_SUBSCRIPTION)),
        JsonSubscriptionRepository(settings.subscriptions_file),
        MockRefundGateway(settings.transactions_file),
        composer,
    )
    update = agent(initial_state("is my subscription active?", ""))
    assert update.get("tool_calls", []) == []
    assert "account id" in update["draft_response"].lower()


def test_billing_answer_directly_calls_no_tool(billing):
    update = billing(initial_state("What payment methods do you accept?", "12345"))
    assert update.get("tool_calls", []) == []
    assert update["next_step"] == NextStep.GUARDRAIL


# --------------------------------------------------------------------------- #
# Technical specialist
# --------------------------------------------------------------------------- #


def test_scenario_one_answers_from_the_knowledge_base(technical):
    """Assignment scenario 1: triage -> technical -> RAG."""
    update = technical(initial_state("How can I reset my password?", "12345"))
    assert "Forgot password" in update["draft_response"]
    assert "grounded" in update["tool_calls"][0]
    assert "password_reset.md" in update["tool_calls"][0]


def test_technical_admits_ignorance_when_nothing_is_retrieved(technical):
    """The spec's anti-hallucination rule, at the node level."""
    update = technical(initial_state("What is the capital of France?", "1"))
    assert "no grounding" in update["tool_calls"][0]
    assert "could not find" in update["draft_response"].lower()


def test_technical_always_records_a_tool_call(technical):
    """The audit trail must show the RAG search ran, grounded or not."""
    for question in ("how do I reset my password", "unrelated nonsense query"):
        assert len(technical(initial_state(question, "1"))["tool_calls"]) == 1


def test_technical_uses_the_best_snippet_as_its_fallback(composer):
    """Composer failure must still yield the documented steps, not a context dump."""

    class _Retriever:
        def search(self, query, *, top_k=3):
            return RetrievalResult(
                query=query,
                snippets=[
                    KnowledgeSnippet(content="BEST ANSWER", source="a.md", score=0.9),
                    KnowledgeSnippet(content="worse answer", source="b.md", score=0.4),
                ],
                has_grounding=True,
            )

    update = TechnicalAgent(_Retriever(), composer)(initial_state("q", "1"))
    assert update["draft_response"] == "BEST ANSWER"


def test_technical_never_routes_back_to_triage(technical):
    """Unlike billing, this agent has no bounce-back rule; it always answers."""
    update = technical(initial_state("I want a refund", "1"))
    assert update["next_step"] == NextStep.GUARDRAIL


# --------------------------------------------------------------------------- #
# General agent
# --------------------------------------------------------------------------- #


def test_general_agent_answers_and_moves_to_the_guardrail(composer):
    update = GeneralAgent(composer)(initial_state("Hello there!", "1"))
    assert update["draft_response"]
    assert update["next_step"] == NextStep.GUARDRAIL


def test_general_agent_calls_no_tools(composer):
    assert "tool_calls" not in GeneralAgent(composer)(initial_state("hi", "1"))


# --------------------------------------------------------------------------- #
# Planner rules
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "message,expected",
    [
        ("My subscription is not working. My id is 12345", BillingAction.CHECK_SUBSCRIPTION),
        ("I want a refund for TXN-1001", BillingAction.PROCESS_REFUND),
        ("How can I reset my password?", BillingAction.RETURN_TO_TRIAGE),
        ("my login is broken", BillingAction.RETURN_TO_TRIAGE),
        ("What payment methods do you accept?", BillingAction.ANSWER_DIRECTLY),
        ("how much is the Pro plan?", BillingAction.ANSWER_DIRECTLY),
        # Mentions a fault AND money -> stays with billing, does not bounce.
        # It is a problem with this customer's own billing, so identify them
        # first: the agent asks for an account id rather than guessing.
        ("I was charged twice but also the app crashes", BillingAction.CHECK_SUBSCRIPTION),
        # A billing complaint with no detail at all still needs an id.
        ("I have a billing problem", BillingAction.CHECK_SUBSCRIPTION),
    ],
)
def test_rule_planner_decisions(message, expected):
    assert RuleBasedBillingPlanner().plan(message).action is expected


def test_refund_without_a_transaction_id_is_still_a_refund_intent():
    """The planner states intent; a missing id is the agent's problem to solve.

    This reverses an earlier rule. Previously a refund request without an id fell
    through to answer_directly, which meant the system never asked for the id and
    the customer was stuck. Now the action is kept and `argument` is left empty,
    so the agent asks and resumes once the id arrives.
    """
    plan = RuleBasedBillingPlanner().plan("I want my money back")
    assert plan.action is BillingAction.PROCESS_REFUND
    assert plan.argument == ""


def test_template_composer_returns_the_verified_fallback(composer):
    assert composer.compose("ignored prompt", "hi", fallback="the truth") == "the truth"
