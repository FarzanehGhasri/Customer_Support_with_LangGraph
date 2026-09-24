"""
Node 2 -- Billing Specialist ("متخصص امور مالی").

Handles money: subscriptions and refunds. Three behaviours matter here.

1. **It owns the two billing tools** (``check_subscription_status``,
   ``process_refund``).
2. **It refuses work that is not billing** and sends the request back to triage
   -- the spec's "اگر سوال نامرتبطی دریافت کرد، باید درخواست را دوباره به تریاژ
   برگرداند".
3. **It asks for what it is missing instead of guessing.** A subscription
   lookup needs an account id and a refund needs a transaction id; when the
   customer has not given one, the agent asks, remembers that it asked
   (``awaiting``), and treats the next message as the answer. The id is then
   looked up in the customer records and, if no such customer exists, the agent
   says so rather than inventing an account.

Structure: plan -> execute -> phrase.
The LLM picks an action (constrained by :class:`BillingPlan`), the *node*
executes the corresponding tool, and the LLM then phrases the verified result.
The model never decides whether money moves -- the gateway does.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping

from ..domain.enums import Awaiting, BillingAction, Department, NextStep
from ..domain.schemas import BillingPlan
from ..interfaces.billing import RefundGateway, SubscriptionRepository
from ..interfaces.composition import ResponseComposer
from ..interfaces.planning import BillingPlanner
from ..prompts.specialists import (
    BILLING_ANSWER_PROMPT,
    BILLING_ASK_ID_PROMPT,
    BILLING_GAVE_UP_PROMPT,
    BILLING_NOT_FOUND_PROMPT,
)
from .base import BaseSupportNode

logger = logging.getLogger(__name__)

#: Transaction ids look like TXN-1001. Used to sanity-check what the model
#: claims to have found, so a hallucinated id never reaches the gateway.
_TRANSACTION_RE = re.compile(r"\bTXN-\d+\b", re.IGNORECASE)
#: An account id: a run of digits long enough not to be a stray number.
_ACCOUNT_RE = re.compile(r"\b\d{4,}\b")

#: Format hints shown to the customer when asking.
_EXAMPLES = {
    Awaiting.ACCOUNT_ID: "a number such as 12345",
    Awaiting.TRANSACTION_ID: "something like TXN-1001",
}


class BillingAgent(BaseSupportNode):
    """Answers billing questions using the subscription and refund tools.

    Args:
        planner: Chooses the action (LLM-backed or rule-based).
        repository: Subscription lookups.
        gateway: Refund processing.
        composer: Phrases the final answer.
        max_asks: How many times to ask for the same missing id before giving
            up. Without a ceiling a customer who ignores the question would be
            asked forever.

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
        *,
        max_asks: int = 2,
    ) -> None:
        self._planner = planner
        self._repository = repository
        self._gateway = gateway
        self._composer = composer
        self._max_asks = max(1, max_asks)

    # ------------------------------------------------------------------ #
    def handle(self, state) -> Mapping[str, Any]:
        message = self.current_message(state)
        awaiting = Awaiting(state.get("awaiting", "") or "")

        # A question is outstanding: this message is the answer to it, not a
        # new request. Do not re-plan -- resume what we were already doing.
        if awaiting.is_pending:
            return self._resume(state, message, awaiting)

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

        return self._act(state, message, plan)

    # ------------------------------------------------------------------ #
    # First pass: work out what we need, ask for it if it is missing
    # ------------------------------------------------------------------ #
    def _act(self, state, message: str, plan: BillingPlan) -> Mapping[str, Any]:
        if plan.action is BillingAction.CHECK_SUBSCRIPTION:
            user_id = self._resolve_account_id(plan.argument, state, message)
            if not user_id:
                return self._ask_for(Awaiting.ACCOUNT_ID, plan.action, message, state)
            return self._lookup_subscription(user_id, message)

        if plan.action is BillingAction.PROCESS_REFUND:
            transaction_id = self._resolve_transaction_id(plan.argument, state, message)
            if not transaction_id:
                return self._ask_for(Awaiting.TRANSACTION_ID, plan.action, message, state)
            return self._run_refund(transaction_id, message)

        # ANSWER_DIRECTLY: no tool, and therefore no facts to quote.
        return self._finish(
            message,
            (
                "No account lookup was needed for this question. Answer from general "
                "billing knowledge only, and ask for an account or transaction id if "
                "one is required."
            ),
        )

    # ------------------------------------------------------------------ #
    # Second pass: the customer has replied to our question
    # ------------------------------------------------------------------ #
    def _resume(self, state, message: str, awaiting: Awaiting) -> Mapping[str, Any]:
        """Interpret this message as the id we asked for."""
        value = self._extract(message, awaiting)
        attempts = int(state.get("ask_attempts", 0))
        action = BillingAction(state.get("pending_action") or BillingAction.ANSWER_DIRECTLY.value)
        original = state.get("pending_query", "") or message

        if not value:
            # Nothing id-shaped in the reply. Ask once more, then let it go --
            # the customer may simply have moved on to something else.
            if attempts >= self._max_asks:
                logger.info("Giving up asking for %s after %d attempts.", awaiting.value, attempts)
                return {
                    **self._clear_pending(),
                    "draft_response": self._compose(
                        BILLING_GAVE_UP_PROMPT.format(what=awaiting.label, user_message=message),
                        message,
                        fallback=(
                            f"I still do not have your {awaiting.label}, so I cannot check "
                            "your account. Send it whenever you are ready and I will look "
                            "it up straight away."
                        ),
                    ),
                    "next_step": NextStep.GUARDRAIL.value,
                    "messages": [self.say("Gave up asking for the missing id.")],
                }
            return self._ask_again(awaiting, action, original, message, attempts)

        # We have a value. Look it up for real.
        if awaiting is Awaiting.ACCOUNT_ID:
            return self._lookup_subscription(value, original, extra_clear=True)
        return self._run_refund(value, original, extra_clear=True)

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #
    def _lookup_subscription(
        self, user_id: str, message: str, *, extra_clear: bool = False
    ) -> Mapping[str, Any]:
        """Search the customer records, then answer according to what was found."""
        status = self._repository.get_status(user_id)
        tool_log = self.tool_log(
            "check_subscription_status", user_id,
            status.status if status.found else "not found",
        )

        if not status.found:
            # The id does not exist. Say so plainly -- never invent an account.
            logger.info("Account %r is not in the customer records.", user_id)
            draft = self._compose(
                BILLING_NOT_FOUND_PROMPT.format(
                    what=Awaiting.ACCOUNT_ID.label, value=user_id, user_message=message
                ),
                message,
                fallback=(
                    f"I could not find any account with the ID '{user_id}'. "
                    "Could you double-check the number and send it again?"
                ),
            )
            return {
                # Stay in the asking state so a corrected id is understood.
                "awaiting": Awaiting.ACCOUNT_ID.value,
                "pending_department": Department.BILLING.value,
                "pending_action": BillingAction.CHECK_SUBSCRIPTION.value,
                "pending_query": message,
                "ask_attempts": 1,
                "user_id": "",
                "draft_response": draft,
                "next_step": NextStep.GUARDRAIL.value,
                "tool_calls": [tool_log],
                "messages": [self.say(draft)],
            }

        draft = self._compose(
            BILLING_ANSWER_PROMPT.format(tool_result=status.summary(), user_message=message),
            message,
            fallback=status.summary(),
        )
        update = {
            **self._clear_pending(),
            "user_id": user_id,          # remember it for the rest of the conversation
            "draft_response": draft,
            "next_step": NextStep.GUARDRAIL.value,
            "tool_calls": [tool_log],
            "messages": [self.say(draft)],
        }
        return update

    def _run_refund(
        self, transaction_id: str, message: str, *, extra_clear: bool = False
    ) -> Mapping[str, Any]:
        receipt = self._gateway.process_refund(transaction_id)
        tool_log = self.tool_log(
            "process_refund", transaction_id, "approved" if receipt.approved else "refused"
        )
        draft = self._compose(
            BILLING_ANSWER_PROMPT.format(tool_result=receipt.summary(), user_message=message),
            message,
            fallback=receipt.summary(),
        )
        return {
            **self._clear_pending(),
            "transaction_id": transaction_id,
            "draft_response": draft,
            "next_step": NextStep.GUARDRAIL.value,
            "tool_calls": [tool_log],
            "messages": [self.say(draft)],
        }

    # ------------------------------------------------------------------ #
    # Asking
    # ------------------------------------------------------------------ #
    def _ask_for(
        self, awaiting: Awaiting, action: BillingAction, message: str, state
    ) -> Mapping[str, Any]:
        return self._ask_again(awaiting, action, message, message, int(state.get("ask_attempts", 0)))

    def _ask_again(
        self,
        awaiting: Awaiting,
        action: BillingAction,
        original_query: str,
        message: str,
        attempts: int,
    ) -> Mapping[str, Any]:
        """Ask the customer for the missing id and remember that we asked."""
        logger.info("Asking the customer for their %s (attempt %d).", awaiting.value, attempts + 1)
        draft = self._compose(
            BILLING_ASK_ID_PROMPT.format(
                what=awaiting.label,
                example=_EXAMPLES.get(awaiting, ""),
                user_message=message,
            ),
            message,
            fallback=(
                f"Of course -- could you tell me your {awaiting.label} so I can look "
                f"that up? It is {_EXAMPLES.get(awaiting, '')}."
            ),
        )
        return {
            "awaiting": awaiting.value,
            "pending_department": Department.BILLING.value,
            "pending_action": action.value,
            "pending_query": original_query,
            "ask_attempts": attempts + 1,
            "draft_response": draft,
            "next_step": NextStep.GUARDRAIL.value,
            "messages": [self.say(draft)],
        }

    @staticmethod
    def _clear_pending() -> dict[str, Any]:
        """Reset the follow-up bookkeeping once the question is answered."""
        return {
            "awaiting": Awaiting.NOTHING.value,
            "pending_department": "",
            "pending_action": "",
            "pending_query": "",
            "ask_attempts": 0,
        }

    # ------------------------------------------------------------------ #
    # Identifier extraction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract(message: str, awaiting: Awaiting) -> str:
        """Pull the id out of a reply like "12345" or "it's TXN-1001"."""
        if awaiting is Awaiting.TRANSACTION_ID:
            found = _TRANSACTION_RE.search(message)
            return found.group(0).upper() if found else ""
        found = _ACCOUNT_RE.search(message)
        return found.group(0) if found else ""

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
    # Phrasing
    # ------------------------------------------------------------------ #
    def _compose(self, prompt: str, message: str, *, fallback: str) -> str:
        return self._composer.compose(prompt, message, fallback=fallback)

    def _finish(self, message: str, tool_result: str) -> Mapping[str, Any]:
        """Answer with no tool involved."""
        draft = self._compose(
            BILLING_ANSWER_PROMPT.format(tool_result=tool_result, user_message=message),
            message,
            fallback=tool_result,
        )
        return {
            **self._clear_pending(),
            "draft_response": draft,
            "next_step": NextStep.GUARDRAIL.value,
            "messages": [self.say(draft)],
        }
