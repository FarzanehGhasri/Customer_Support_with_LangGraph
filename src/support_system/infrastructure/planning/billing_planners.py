"""
Concrete :class:`BillingPlanner` implementations.

* :class:`LLMBillingPlanner` -- ``with_structured_output(BillingPlan)``.
* :class:`RuleBasedBillingPlanner` -- keyword rules; keeps the billing agent
  working (and testable) with no credentials.

Both return the same :class:`BillingPlan`, so the agent cannot tell them apart.
"""

from __future__ import annotations

import logging
import re

from ...domain.enums import BillingAction
from ...domain.schemas import BillingPlan
from ...interfaces.llm import ChatModelProvider
from ...prompts.specialists import BILLING_PLAN_PROMPT

logger = logging.getLogger(__name__)


class LLMBillingPlanner:
    """Picks the billing action with a structured-output call."""

    def __init__(self, provider: ChatModelProvider, *, temperature: float = 0.0) -> None:
        self._provider = provider
        self._temperature = temperature
        self._runnable = None

    def plan(self, user_message: str) -> BillingPlan:
        if self._runnable is None:
            self._runnable = self._provider.get_structured_model(
                BillingPlan, temperature=self._temperature
            )
        try:
            plan = self._runnable.invoke(
                [("system", BILLING_PLAN_PROMPT), ("human", user_message)]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Billing planning failed (%s); answering directly.", exc)
            return BillingPlan(
                action=BillingAction.ANSWER_DIRECTLY,
                reasoning=f"Planner unavailable ({type(exc).__name__}).",
            )
        if isinstance(plan, dict):
            plan = BillingPlan.model_validate(plan)
        return plan


#: Words that mean the customer is asking about their plan/subscription.
SUBSCRIPTION_MARKERS = (
    "subscription", "subscribe", "plan", "renew", "renewal", "expire", "expired",
    "membership", "account status",
)
#: Words that mean the customer wants money back.
REFUND_MARKERS = ("refund", "money back", "reimburse", "chargeback", "charge back")
#: Any other wording that makes a message financial. Used by the bounce-back
#: guard: a message mentioning money is ours even if it also mentions a fault.
MONEY_MARKERS = (
    "charge", "charged", "charges", "invoice", "billing", "bill", "payment",
    "paid", "price", "pricing", "money", "credit card", "transaction", "receipt",
)
#: Words that mean this is not a billing matter at all.
TECHNICAL_MARKERS = (
    "password", "log in", "login", "sign in", "crash", "crashes", "bug", "error",
    "install", "update", "sync", "2fa", "two-factor",
)

_TRANSACTION_RE = re.compile(r"\bTXN-\d+\b", re.IGNORECASE)
_ACCOUNT_RE = re.compile(r"\b\d{4,}\b")


class RuleBasedBillingPlanner:
    """Deterministic stand-in for :class:`LLMBillingPlanner`."""

    def plan(self, user_message: str) -> BillingPlan:
        text = user_message.lower()

        # A refund request is a refund request whether or not the id is there.
        # The planner states the *intent*; the agent asks for a missing id.
        transaction = _TRANSACTION_RE.search(user_message)
        if any(m in text for m in REFUND_MARKERS):
            return BillingPlan(
                action=BillingAction.PROCESS_REFUND,
                argument=transaction.group(0).upper() if transaction else "",
                reasoning=(
                    "Refund requested with a transaction id."
                    if transaction
                    else "Refund requested; the transaction id still has to be asked for."
                ),
            )

        if any(m in text for m in SUBSCRIPTION_MARKERS):
            account = _ACCOUNT_RE.search(user_message)
            return BillingPlan(
                action=BillingAction.CHECK_SUBSCRIPTION,
                argument=account.group(0) if account else "",
                reasoning="Message is about the customer's subscription.",
            )

        # Only bounce back when the message is technical AND shows no sign of
        # being about money -- "I was charged for an app that crashes" is ours.
        if any(m in text for m in TECHNICAL_MARKERS) and not any(
            m in text for m in REFUND_MARKERS + SUBSCRIPTION_MARKERS + MONEY_MARKERS
        ):
            return BillingPlan(
                action=BillingAction.RETURN_TO_TRIAGE,
                reasoning="Technical issue, not a billing matter.",
            )

        return BillingPlan(
            action=BillingAction.ANSWER_DIRECTLY,
            reasoning="Billing question needing no lookup.",
        )
