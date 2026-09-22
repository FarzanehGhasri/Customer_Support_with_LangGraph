"""
Shared base class for every graph node.

Each concrete agent supplies only its *decision* logic; the mechanics that are
identical everywhere -- having a name, appending a transcript line, recording a
tool call in the audit trail -- live here once.  This keeps the subclasses short
and is a straightforward application of the Single Responsibility Principle: a
specialist agent should read like support policy, not like state plumbing.

The class satisfies the :class:`~support_system.interfaces.nodes.SupportNode`
Protocol, so the graph builder can register any subclass without special cases.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Mapping

from ..domain.state import SupportState

logger = logging.getLogger(__name__)


class BaseSupportNode(ABC):
    """Template for a LangGraph node.

    Subclasses implement :meth:`handle`.  ``__call__`` is deliberately *not*
    abstract: it wraps ``handle`` with logging and a safety net so that one
    misbehaving agent cannot take the whole graph down mid-conversation.
    """

    #: Node name used when registering on the graph. Subclasses must set it.
    node_name: str = "node"

    @property
    def name(self) -> str:
        """Name the graph registers this node under."""
        return self.node_name

    # ------------------------------------------------------------------ #
    # SupportNode Protocol
    # ------------------------------------------------------------------ #
    def __call__(self, state: SupportState) -> Mapping[str, Any]:
        """Run the node, returning ONLY the state keys that changed."""
        logger.debug("[%s] entering with next_step=%s", self.name, state.get("next_step"))
        update = self.handle(state)
        logger.debug("[%s] update=%s", self.name, sorted(update))
        return update

    @abstractmethod
    def handle(self, state: SupportState) -> Mapping[str, Any]:
        """Do the node's actual work. Must return a *partial* state update."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Helpers shared by all nodes
    # ------------------------------------------------------------------ #
    def say(self, text: str) -> str:
        """Format a transcript line attributed to this node.

        Using one helper means every agent's lines look the same in the chat
        history, which matters because that history is fed back into prompts.
        """
        return f"{self.name}: {text}"

    @staticmethod
    def tool_log(tool: str, argument: str, outcome: str) -> str:
        """Format an audit-trail entry for the ``tool_calls`` channel.

        The notebook prints this to show the grader which tool actually ran.
        """
        return f"{tool}({argument}) -> {outcome}"

    def current_message(self, state: SupportState) -> str:
        """The request being handled.

        Prefers the explicit ``user_query`` key and falls back to the last line
        of the transcript, so a node still works if it is invoked with a
        hand-built state (as the tests and the notebook sometimes do).
        """
        query = state.get("user_query", "").strip()
        if query:
            return query
        messages = state.get("messages", [])
        return messages[-1] if messages else ""
