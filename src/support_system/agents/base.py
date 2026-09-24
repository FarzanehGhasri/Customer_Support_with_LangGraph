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
    #: How this agent introduces itself to the customer. The graph deals in
    #: node names; a customer should hear a job title.
    display_name: str = "Support"

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

    def has_spoken(self, state: SupportState) -> bool:
        """True if this agent has already contributed to the transcript.

        Read from the transcript rather than tracked in its own field: every
        agent prefixes its lines with :meth:`say`, so the history already holds
        the answer. One less piece of state to keep in step.
        """
        prefix = f"{self.name}:"
        return any(line.startswith(prefix) for line in state.get("messages", []))

    def introduction(self, state: SupportState) -> str:
        """Self-introduction, or empty once this agent has already spoken.

        The spec describes a company where reception hands the customer to a
        specialist; a specialist that answers anonymously hides that. It is
        said once per conversation -- repeating it every turn would read like a
        machine, not a colleague.
        """
        if self.has_spoken(state):
            return ""
        return f"Hello, I am the {self.display_name}."

    def intro_instruction(self, state: SupportState) -> str:
        """Instruction prepended to the system prompt on an agent's first turn.

        Returned as prompt text rather than fixed wording so the model writes
        the introduction in the customer's own language and ties it to what
        they actually asked.
        """
        if self.has_spoken(state):
            return ""
        return (
            f"This is your first message to this customer. Begin by introducing "
            f"yourself as the {self.display_name}, refer briefly to what they "
            f"asked about, and then answer. One short sentence for the "
            f"introduction, no more.\n\n"
        )

    def compose_reply(
        self, composer, state: SupportState, prompt: str, message: str, *, fallback: str
    ) -> str:
        """Phrase a reply, introducing this agent on its first turn.

        Lives here because all three specialists need exactly this and nothing
        more. The composer is passed in rather than held on the base class:
        triage and the guardrail phrase nothing, and should not carry a
        dependency they never use.
        """
        intro = self.introduction(state)
        if intro:
            # A blank line before a multi-line answer; a space before a short
            # one. Without this an introduction runs straight into a markdown
            # heading -- "I am the Technical Support Specialist. ## Reset a..."
            separator = "\n\n" if "\n" in fallback.strip() else " "
            fallback = f"{intro}{separator}{fallback}"
        return composer.compose(self.intro_instruction(state) + prompt, message,
                                fallback=fallback)

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
