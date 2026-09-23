"""
Graph assembly -- wires the four nodes into a LangGraph state machine.

    START -> triage ──┬─> billing   ─┐
                      ├─> technical ─┼─> guardrail ─┬─> END          (calm)
                      └─> general   ─┘              └─> human_review -> END
                          ▲                                            (angry)
                          └── billing may send a non-billing request back

Two design points worth calling out:

**Routing lives in the state, not in the nodes.** Every node writes
``next_step``; a single ``_route`` function maps that value to a node name.
Nodes therefore never name each other, so adding a department later touches the
registry and the enum -- not the existing agents (Open/Closed).

**The interrupt is declared on the graph, not called inside a node.** Compiling
with ``interrupt_before=["human_review"]`` is what implements the spec's
"execution must stop and wait for a human". Because the halt is a property of
the graph, the guardrail node stays a pure function and remains testable
without a checkpointer.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..domain.enums import NextStep
from ..domain.state import SupportState
from ..interfaces.nodes import SupportNode

logger = logging.getLogger(__name__)

#: ``next_step`` value -> node name. The only place the two vocabularies meet.
ROUTES: Mapping[str, str] = {
    NextStep.TRIAGE.value: "triage",
    NextStep.BILLING.value: "billing",
    NextStep.TECHNICAL.value: "technical",
    NextStep.GENERAL.value: "general",
    NextStep.GUARDRAIL.value: "guardrail",
    NextStep.HUMAN_REVIEW.value: "human_review",
    NextStep.FINISH.value: END,
}


def _route(state: SupportState) -> str:
    """Read ``next_step`` and name the node that should run next."""
    step = str(state.get("next_step", NextStep.FINISH.value))
    destination = ROUTES.get(step)
    if destination is None:
        # An unknown control value must end the run rather than loop forever.
        logger.error("Unknown next_step %r; ending the conversation.", step)
        return END
    return destination


def build_support_graph(
    *,
    triage: SupportNode,
    billing: SupportNode,
    technical: SupportNode,
    general: SupportNode,
    guardrail: SupportNode,
    human_review: SupportNode,
    checkpointer: Any | None = None,
    interrupt_before_human: bool = True,
) -> Any:
    """Compile the support graph.

    Args:
        triage ... human_review: The six nodes, injected. The builder depends on
            the :class:`SupportNode` Protocol only, so any of them can be
            replaced by a stub in a test.
        checkpointer: LangGraph checkpointer. **Required** for the
            human-in-the-loop flow: without one there is no state to pause and
            resume, and ``update_state`` has nothing to update. Defaults to an
            in-memory saver.
        interrupt_before_human: Set False to run the whole graph unattended
            (the escalation node then answers with a holding message).

    Returns:
        The compiled graph, ready for ``.invoke()`` / ``.stream()``.
    """
    builder = StateGraph(SupportState)

    for node in (triage, billing, technical, general, guardrail, human_review):
        builder.add_node(node.name, node)

    builder.add_edge(START, "triage")

    # Triage fans out to the three specialists.
    builder.add_conditional_edges(
        "triage", _route,
        {
            "billing": "billing",
            "technical": "technical",
            "general": "general",
            END: END,
        },
    )

    # A specialist either produces a draft (-> guardrail) or, in billing's case,
    # returns the request to triage.
    for specialist in ("billing", "technical", "general"):
        builder.add_conditional_edges(
            specialist, _route,
            {"guardrail": "guardrail", "triage": "triage", END: END},
        )

    # The guardrail either releases the answer or escalates.
    builder.add_conditional_edges(
        "guardrail", _route,
        {"human_review": "human_review", END: END},
    )
    builder.add_edge("human_review", END)

    compile_kwargs: dict[str, Any] = {"checkpointer": checkpointer or MemorySaver()}
    if interrupt_before_human:
        compile_kwargs["interrupt_before"] = ["human_review"]

    graph = builder.compile(**compile_kwargs)
    logger.info("Support graph compiled (interrupt=%s).", interrupt_before_human)
    return graph


def node_names() -> Sequence[str]:
    """Names registered on the graph, in execution order. Used by the tests."""
    return ("triage", "billing", "technical", "general", "guardrail", "human_review")
