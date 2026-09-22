"""
Human-in-the-loop abstraction.

Step 3 of the assignment asks for "a simple function playing the role of the
support manager" that can approve, reject or replace the agent's answer.  We
express that as a Protocol so the notebook can supply an *interactive* reviewer
(``input()``), the tests can supply a *scripted* one, and neither the graph nor
the guardrail has to care which is in use.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.schemas import HumanDecision
from ..domain.state import SupportState


@runtime_checkable
class HumanReviewer(Protocol):
    """The escalation target for angry customers."""

    def review(self, state: SupportState) -> HumanDecision:
        """Inspect the interrupted conversation and decide how to answer."""
        ...
