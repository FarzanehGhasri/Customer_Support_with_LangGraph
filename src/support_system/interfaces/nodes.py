"""
Graph-node abstraction.

Every LangGraph node in this project is a *callable object* that takes the
current :class:`SupportState` and returns a partial state update.  Modelling
that as a Protocol lets the graph builder treat Triage, the specialists and the
guardrail uniformly, and lets tests substitute a stub node anywhere.
"""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

from ..domain.state import SupportState


@runtime_checkable
class SupportNode(Protocol):
    """A unit of work in the support graph."""

    @property
    def name(self) -> str:
        """Node name used when registering it on the graph."""
        ...

    def __call__(self, state: SupportState) -> Mapping[str, object]:
        """Process ``state`` and return ONLY the keys that changed.

        Returning a partial mapping (not the whole state) is what lets
        LangGraph's reducers merge updates correctly -- in particular the
        append-only ``messages`` channel.
        """
        ...
