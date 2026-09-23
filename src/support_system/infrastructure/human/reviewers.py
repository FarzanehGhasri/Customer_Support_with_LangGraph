"""
Concrete :class:`HumanReviewer` implementations -- the "support manager".

Three flavours, all interchangeable:

* :class:`ScriptedReviewer`  -- fixed decision; used by the tests and by the
  notebook's scenario 3, where the manager's reply is known in advance.
* :class:`ConsoleReviewer`   -- asks a real person at the terminal.
* :class:`AutoApproveReviewer` -- approves the draft unchanged; a control case.
"""

from __future__ import annotations

import logging

from ...domain.schemas import HumanDecision
from ...domain.state import SupportState, transcript

logger = logging.getLogger(__name__)


class ScriptedReviewer:
    """Always returns the same decision."""

    def __init__(self, decision: HumanDecision) -> None:
        self._decision = decision

    def review(self, state: SupportState) -> HumanDecision:
        return self._decision


class AutoApproveReviewer:
    """Approves whatever the agent drafted."""

    def review(self, state: SupportState) -> HumanDecision:
        return HumanDecision(approved=True, note="Auto-approved (no human available).")


class ConsoleReviewer:
    """Prompts a person at the terminal.

    Shows the manager exactly what the spec says they should see -- the state of
    the conversation and the agent's draft -- then lets them approve it or
    replace it.
    """

    def review(self, state: SupportState) -> HumanDecision:
        print("\n" + "=" * 68)
        print("HUMAN REVIEW REQUIRED -- the customer appears angry")
        print("=" * 68)
        print(f"user_id   : {state.get('user_id', '-')}")
        print(f"department: {state.get('department', '-')}")
        print(f"sentiment : {state.get('sentiment', '-')}")
        print("-" * 68)
        print("conversation:")
        print(transcript(state))
        print("-" * 68)
        print("agent's draft reply:")
        print(state.get("draft_response", "(none)"))
        print("=" * 68)

        answer = input("Send this reply? [y/N] ").strip().lower()
        if answer in {"y", "yes"}:
            return HumanDecision(approved=True, note="Approved by the manager at the console.")

        replacement = input("Type the reply to send instead: ").strip()
        return HumanDecision(
            approved=False,
            replacement_response=replacement,
            note="Replaced by the manager at the console.",
        )
