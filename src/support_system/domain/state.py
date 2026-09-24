"""
Graph state -- Step 1 of the assignment ("تعریف وضعیت").

The spec asks for this exact shape::

    class SupportState(TypedDict):
        messages: Annotated[List[str], operator.add]   # Chat History
        user_id: str
        sentiment: str        # "Positive", "Neutral", "Negative"
        department: str       # "Billing", "Technical", "Triage"
        next_step: str        # Control flow variable

We keep every one of those five keys with the same names and the same types, and
add a small number of *optional* bookkeeping keys that the later steps need
(draft answer, audit trail of tool calls, escalation flag...).  The optional keys
live in a second ``TypedDict`` with ``total=False`` so that:

* the five required keys stay required (type checkers enforce them), and
* a node may return a partial dict -- which is exactly how LangGraph updates
  state: a node returns only the keys it changed.

Reducer note
------------
``messages`` is annotated with ``operator.add``.  LangGraph reads that annotation
and *appends* whatever a node returns instead of overwriting, so every node can
emit ``{"messages": ["..."]}`` without knowing about the rest of the history.
Un-annotated keys (``sentiment``, ``department``, ...) use the default
"last write wins" reducer, which is what we want for control-flow values.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, TypedDict

from .enums import Awaiting, Department, NextStep, Sentiment


class _SupportStateRequired(TypedDict):
    """The five keys mandated by the assignment specification."""

    # Chat history.  ``operator.add`` makes this an append-only channel.
    messages: Annotated[List[str], operator.add]
    # Identifier used by the billing tools (e.g. "12345").
    user_id: str
    # Value of a :class:`Sentiment` member -- "Positive" | "Neutral" | "Negative".
    sentiment: str
    # Value of a :class:`Department` member -- "Billing" | "Technical" | ...
    department: str
    # Value of a :class:`NextStep` member -- drives the conditional edges.
    next_step: str


class SupportState(_SupportStateRequired, total=False):
    """Full state object flowing through the graph.

    Everything below is optional: nodes add these keys as the conversation
    progresses, and the notebook can inspect them to explain what happened.
    """

    # The raw text of the request currently being handled.  Kept separate from
    # ``messages`` so agents never have to re-parse the transcript.
    user_query: str
    # Transaction id for refunds, supplied by the user or extracted by the agent.
    transaction_id: str
    # Answer produced by a specialist *before* the guardrail approves it.
    draft_response: str
    # Answer actually shown to the user (guardrail-approved or human-written).
    final_response: str
    # True once the sentiment guardrail has handed control to a human.
    escalated: bool
    # Append-only audit trail: one line per tool invocation. Useful for grading.
    tool_calls: Annotated[List[str], operator.add]
    # How many times the request has bounced back to triage.  Guards against the
    # "specialist rejects -> triage re-routes -> specialist rejects" infinite loop.
    triage_attempts: int

    # Set by triage when the conversation changes department, e.g.
    # "Connecting you to the Billing team." Shown to the customer ahead of the
    # specialist's reply, so the hand-off between agents is visible rather than
    # happening silently inside the graph.
    routing_notice: str

    # --- multi-turn follow-up ------------------------------------------ #
    # Value of an :class:`Awaiting` member.  Non-empty means the last reply was
    # a question and the customer's next message is the answer to it.
    awaiting: str
    # Which department asked, so the follow-up goes straight back to it instead
    # of being re-classified by triage.
    pending_department: str
    # The action to resume once the missing value arrives (a BillingAction).
    pending_action: str
    # The question that triggered the request, kept so the specialist still has
    # the original context after the customer replies with a bare id.
    pending_query: str
    # How many times we have asked for the same value.  Two strikes, then we
    # stop pestering the customer and move on.
    ask_attempts: int


def initial_state(
    user_query: str,
    user_id: str = "",
    *,
    transaction_id: str = "",
) -> SupportState:
    """Build a well-formed starting state.

    Factory function rather than a constructor call scattered across the
    notebook: one place decides what a fresh conversation looks like, so the
    three delivery scenarios cannot drift apart.
    """
    return SupportState(
        messages=[f"user: {user_query}"],
        user_id=user_id,
        sentiment=Sentiment.NEUTRAL.value,   # assume good faith until judged
        department=Department.TRIAGE.value,  # everyone starts at reception
        next_step=NextStep.TRIAGE.value,
        user_query=user_query,
        transaction_id=transaction_id,
        draft_response="",
        final_response="",
        escalated=False,
        tool_calls=[],
        triage_attempts=0,
        routing_notice="",
        awaiting=Awaiting.NOTHING.value,
        pending_department="",
        pending_action="",
        pending_query="",
        ask_attempts=0,
    )


def turn_update(message: str, *, user_id: str = "") -> dict:
    """Per-turn state for a *continuing* conversation.

    Deliberately partial. A follow-up turn must reset the control-flow keys so
    the message is handled afresh, while leaving the append-only transcript and
    the follow-up bookkeeping (``awaiting``, ``pending_*``) untouched -- those
    are what let the specialist pick up where it left off.
    """
    update: dict = {
        "messages": [f"user: {message}"],
        "user_query": message,
        "next_step": NextStep.TRIAGE.value,
        "triage_attempts": 0,
        "escalated": False,
        "draft_response": "",
        "final_response": "",
        "routing_notice": "",
    }
    if user_id:
        update["user_id"] = user_id
    return update


def transcript(state: SupportState) -> str:
    """Render the chat history as a single string for prompting/printing."""
    return "\n".join(state.get("messages", []))
