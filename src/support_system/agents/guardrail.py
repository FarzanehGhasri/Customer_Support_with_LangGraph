"""
Node 4 -- Sentiment Guardrail ("محافظ تحلیل احساسات").

Sits between every specialist and the customer. It inspects the tone of the
*customer's* message (not the agent's draft) and decides whether the draft may
be sent:

* Negative  -> stop. Control passes to a human (``next_step = human_review``).
* Otherwise -> the draft becomes the final response.

Where the interrupt actually happens
------------------------------------
This node does **not** call ``interrupt()`` itself. It only records the verdict
in the state; the graph is compiled with ``interrupt_before=["human_review"]``,
so LangGraph halts on the edge into the human node. Keeping the decision and
the halt separate means the node stays a pure function of the state -- fully
testable without a checkpointer -- and the graph keeps ownership of control
flow. Both satisfy the spec's requirement to stop and wait for a human.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from ..domain.enums import NextStep, Sentiment
from ..domain.state import transcript
from ..interfaces.classification import SentimentAnalyzer
from .base import BaseSupportNode

logger = logging.getLogger(__name__)


class SentimentGuardrail(BaseSupportNode):
    """Quality gate in front of the customer.

    Args:
        analyzer: Any :class:`SentimentAnalyzer`.
        enabled: When False the guardrail still records the sentiment but never
            escalates. Useful for a demo run without a human in the loop.
    """

    node_name = "guardrail"

    def __init__(self, analyzer: SentimentAnalyzer, *, enabled: bool = True) -> None:
        self._analyzer = analyzer
        self._enabled = enabled

    # ------------------------------------------------------------------ #
    def handle(self, state) -> Mapping[str, Any]:
        message = self.current_message(state)
        draft = state.get("draft_response", "")

        assessment = self._analyzer.analyze(message, history=transcript(state))
        sentiment = assessment.sentiment
        logger.info("Guardrail: %s -- %s", sentiment.value, assessment.reasoning)

        if self._enabled and assessment.requires_human:
            # Stop here. The draft is deliberately NOT promoted to
            # final_response: an angry customer must not receive an automated
            # reply, which is the entire point of this node.
            return {
                "sentiment": sentiment.value,
                "escalated": True,
                "next_step": NextStep.HUMAN_REVIEW.value,
                "messages": [
                    self.say(
                        f"Negative sentiment detected ({assessment.reasoning}) "
                        "-- holding the reply and escalating to a human agent."
                    )
                ],
            }

        return {
            "sentiment": sentiment.value,
            "escalated": False,
            "final_response": draft,
            "next_step": NextStep.FINISH.value,
            "messages": [
                self.say(f"Sentiment {sentiment.value}; releasing the reply to the customer.")
            ],
        }


class HumanReviewNode(BaseSupportNode):
    """Applies the support manager's decision after the interrupt.

    In the assignment's scenario 3 the manager's reply is injected from outside
    with ``graph.update_state(...)`` while execution is paused. This node runs
    *after* the resume and turns whatever the manager left behind into the final
    answer. If a reviewer was injected it is consulted instead, which is what
    lets the notebook demonstrate both styles.
    """

    node_name = "human_review"

    def __init__(self, reviewer=None) -> None:
        self._reviewer = reviewer

    def handle(self, state) -> Mapping[str, Any]:
        # 1. A manager reply written straight into the state wins: this is the
        #    update_state path the spec asks for.
        injected = str(state.get("final_response", "")).strip()
        if injected:
            logger.info("Using the manager's injected reply.")
            return {
                "escalated": True,
                "next_step": NextStep.FINISH.value,
                "messages": [self.say("Manager's reply approved and sent.")],
            }

        # 2. Otherwise ask the reviewer object, if one was provided.
        if self._reviewer is not None:
            decision = self._reviewer.review(state)
            final = decision.resolve(state.get("draft_response", ""))
            return {
                "escalated": True,
                "final_response": final,
                "next_step": NextStep.FINISH.value,
                "messages": [
                    self.say(
                        f"Human decision: {'approved' if decision.approved else 'overridden'}. "
                        f"{decision.note}".strip()
                    )
                ],
            }

        # 3. Nothing available: hold the conversation rather than sending the
        #    automated draft an angry customer must not receive.
        logger.warning("Escalated with no manager reply available.")
        return {
            "escalated": True,
            "final_response": (
                "Your message has been escalated to a support manager, who will "
                "contact you shortly."
            ),
            "next_step": NextStep.FINISH.value,
            "messages": [self.say("Escalated; awaiting a manager.")],
        }


class InterruptingHumanReviewNode(BaseSupportNode):
    """Human review implemented with LangGraph's **dynamic** ``interrupt()``.

    The spec names ``graph.interrupt()`` explicitly, so this node exists
    alongside :class:`HumanReviewNode` to cover that wording literally. The two
    differ only in *where* the pause is declared:

    ===================  ==========================================  =======================
    mechanism            how it pauses                               how it resumes
    ===================  ==========================================  =======================
    ``HumanReviewNode``  ``interrupt_before=["human_review"]`` set    ``update_state(...)``
                         when the graph is compiled                   then ``invoke(None)``
    this class           ``interrupt(payload)`` called inside the     ``invoke(Command(
                         node, at the moment the pause is needed      resume=...))``
    ===================  ==========================================  =======================

    The dynamic form is the more expressive one: the ``payload`` handed to
    ``interrupt()`` is exactly what the support manager is shown -- the spec's
    "این تابع باید بتواند وضعیت را بررسی کرده و پاسخ ایجنت را تایید یا رد کند" --
    and whatever the manager sends back becomes the return value of that call.

    Accepted resume values:

    * ``str``   -- the manager's replacement reply;
    * ``dict``  -- ``{"approved": bool, "reply": str, "note": str}``;
    * ``True``  -- approve the agent's draft unchanged.
    """

    node_name = "human_review"

    def __init__(self, interrupt_fn=None) -> None:
        # Injectable so the node can be unit-tested without a running graph.
        self._interrupt_fn = interrupt_fn

    def _interrupt(self, payload: dict):
        if self._interrupt_fn is not None:
            return self._interrupt_fn(payload)
        # Imported lazily: the module must stay importable without LangGraph.
        from langgraph.types import interrupt

        return interrupt(payload)

    def handle(self, state) -> Mapping[str, Any]:
        draft = state.get("draft_response", "")

        # Everything the manager needs in order to decide, in one payload.
        decision = self._interrupt(
            {
                "reason": "negative sentiment -- human approval required",
                "user_id": state.get("user_id", ""),
                "department": state.get("department", ""),
                "sentiment": state.get("sentiment", ""),
                "customer_message": self.current_message(state),
                "agent_draft": draft,
                "instructions": (
                    "Reply with the text to send, or {'approved': true} to send the "
                    "draft unchanged."
                ),
            }
        )

        approved, reply, note = self._parse(decision, draft)
        logger.info("Human decision received: approved=%s", approved)

        return {
            "escalated": True,
            "final_response": reply,
            "next_step": NextStep.FINISH.value,
            "messages": [
                f"manager: {reply}",
                self.say(
                    f"Human decision: {'approved' if approved else 'overridden'}. {note}".strip()
                ),
            ],
        }

    @staticmethod
    def _parse(decision: Any, draft: str) -> tuple[bool, str, str]:
        """Normalise whatever the manager sent back into (approved, reply, note)."""
        if isinstance(decision, dict):
            approved = bool(decision.get("approved", False))
            reply = str(decision.get("reply", "")).strip()
            note = str(decision.get("note", ""))
            return approved, (draft if approved and not reply else reply or draft), note
        if decision is True:
            return True, draft, "Draft approved unchanged."
        text = str(decision).strip()
        if not text:
            # An empty resume value must not send an angry customer an empty reply.
            return False, draft, "Empty decision; fell back to the draft."
        return False, text, "Replaced by the manager."
