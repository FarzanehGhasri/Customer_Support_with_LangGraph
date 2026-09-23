"""
Node 2 -- Billing Specialist ("متخصص امور مالی").

Handles money: subscriptions and refunds. Two behaviours the spec singles out:

* it owns the two billing tools, and
* **it must refuse work that is not billing** and send the request back to
  triage ("اگر سوال نامرتبطی دریافت کرد، باید درخواست را دوباره به تریاژ
  برگرداند").

Structure: plan -> execute -> phrase.
The LLM picks an action (constrained by :class:`BillingPlan`), the *node*
executes the corresponding tool, and the LLM then phrases the verified result.
The model never decides whether money moves -- the gateway does.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping

from ..domain.enums import BillingAction, Department, NextStep
from ..domain.schemas import BillingPlan
from ..interfaces.billing import RefundGateway, SubscriptionRepository
from ..interfaces.composition import ResponseComposer
from ..interfaces.planning import BillingPlanner
from ..prompts.specialists import BILLING_ANSWER_PROMPT
from .base import BaseSupportNode

logger = logging.getLogger(__name__)

#: Transaction ids look like TXN-1001. Used to sanity-check what the model
#: claims to have found, so a hallucinated id never reaches the gateway.
_TRANSACTION_RE = re.compile(r"\bTXN-\d+\b", re.IGNORECASE)
#: A bare number in the message, used as the account id fallback.
_ACCOUNT_RE = re.compile(r"\b\d{4,}\b")


class BillingAgent(BaseSupportNode):
    """Answers billing questions using the subscription and refund tools.

    Args:
        planner: Chooses the action (LLM-backed or rule-based).
        repository: Subscription lookups.
        gateway: Refund processing.
        composer: Phrases the final answer.

    Note what this agent does *not* depend on: there is no reference to a chat
    model here. Both model-using steps sit behind their own interface, so the
    whole agent runs unchanged with no credentials.
    """

    node_name = "billing"

    def __init__(
        self,
        planner: BillingPlanner,
        repository: SubscriptionRepository,
        gateway: RefundGateway,
        composer: ResponseComposer,
    ) -> None:
        self._planner = planner
        self._repository = repository
        self._gateway = gateway
        self._composer = composer

    # ------------------------------------------------------------------ #
    def handle(self, state) -> Mapping[str, Any]:
        message = self.current_message(state)
        plan = self._planner.plan(message)
        logger.info("Billing plan: %s(%r) -- %s", plan.action.value, plan.argument, plan.reasoning)

        # The spec's bounce-back rule. Nothing else in this method runs.
        if plan.action is BillingAction.RETURN_TO_TRIAGE:
            return {
                "department": Department.TRIAGE.value,
                "next_step": NextStep.TRIAGE.value,
                "messages": [
                    self.say(f"Not a billing matter; returning to triage. {plan.reasoning}".strip())
                ],
            }

        tool_result, tool_log = self._execute(plan, state, message)
        draft = self._compose(message, tool_result)

        update: dict[str, Any] = {
            "draft_response": draft,
            "next_step": NextStep.GUARDRAIL.value,
            "messages": [self.say(draft)],
        }
        if tool_log:
            update["tool_calls"] = [tool_log]
        return update

    # ------------------------------------------------------------------ #
    # execute
    # ------------------------------------------------------------------ #
    def _execute(self, plan: BillingPlan, state, message: str) -> tuple[str, str]:
        """Run the planned tool and return (result text, audit-log line)."""
        if plan.action is BillingAction.CHECK_SUBSCRIPTION:
            user_id = self._resolve_account_id(plan.argument, state, message)
            if not user_id:
                return (
                    "The customer has not provided an account id, so no lookup was possible.",
                    "",
                )
            status = self._repository.get_status(user_id)
            return status.summary(), self.tool_log(
                "check_subscription_status", user_id,
                status.status if status.found else "not found",
            )

        if plan.action is BillingAction.PROCESS_REFUND:
            transaction_id = self._resolve_transaction_id(plan.argument, state, message)
            if not transaction_id:
                return (
                    "No transaction id was supplied, so no refund was attempted. "
                    "The customer must provide one (for example TXN-1001).",
                    "",
                )
            receipt = self._gateway.process_refund(transaction_id)
            return receipt.summary(), self.tool_log(
                "process_refund", transaction_id,
                "approved" if receipt.approved else "refused",
            )

        # ANSWER_DIRECTLY: no tool, and therefore no facts to quote.
        return (
            "No account lookup was needed for this question. Answer from general "
            "billing knowledge only, and ask for an account or transaction id if "
            "one is required.",
            "",
        )

    def _resolve_account_id(self, argument: str, state, message: str) -> str:
        """Prefer the id already in the state, then the message, then the model.

        Order matters: the session's own ``user_id`` is authoritative, and the
        model's suggestion is trusted last because it is the only source that
        can invent one.
        """
        from_state = str(state.get("user_id", "")).strip()
        if from_state:
            return from_state
        found = _ACCOUNT_RE.search(message)
        if found:
            return found.group(0)
        argument = argument.strip()
        return argument if argument.isdigit() else ""

    def _resolve_transaction_id(self, argument: str, state, message: str) -> str:
        """Only accept a transaction id that really appears in the conversation.

        A refund is irreversible, so the id must be traceable to something the
        customer actually wrote -- never to the model's imagination.
        """
        stated = str(state.get("transaction_id", "")).strip()
        if stated:
            return stated
        found = _TRANSACTION_RE.search(message)
        if found:
            return found.group(0).upper()
        argument = argument.strip().upper()
        if argument and _TRANSACTION_RE.fullmatch(argument) and argument in message.upper():
            return argument
        if argument:
            logger.warning("Discarding transaction id %r: not present in the message.", argument)
        return ""

    # ------------------------------------------------------------------ #
    # phrase
    # ------------------------------------------------------------------ #
    def _compose(self, message: str, tool_result: str) -> str:
        """Turn the verified tool output into the customer-facing reply.

        The tool result doubles as the fallback, so a failed wording step
        degrades to an unpolished but still *true* answer.
        """
        prompt = BILLING_ANSWER_PROMPT.format(tool_result=tool_result, user_message=message)
        return self._composer.compose(prompt, message, fallback=tool_result)
