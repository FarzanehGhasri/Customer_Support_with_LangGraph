"""
Node 1 -- The Triage Agent ("ایجنت دسته‌بندی و توزیع").

Role: the reception desk.  It reads the incoming message, decides which team
owns it, and writes that decision into the state as the control-flow variable
``next_step``.  It never answers the customer.

Separation of concerns
----------------------
Two objects, not one:

* :class:`~support_system.interfaces.classification.IntentClassifier` decides
  *what kind* of request this is (LLM-backed in production, keyword-based in
  tests).
* :class:`TriageNode` decides *what that means for the graph* -- which state
  keys change, how loops are prevented.

The node depends on the Protocol, so swapping the classifier is a constructor
argument, not a code change (Dependency Inversion).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from ..domain.enums import Department, NextStep
from ..domain.state import SupportState, transcript
from ..interfaces.classification import IntentClassifier
from .base import BaseSupportNode

logger = logging.getLogger(__name__)


class TriageNode(BaseSupportNode):
    """Routes a request to the department that owns it.

    Args:
        classifier: Any :class:`IntentClassifier`.
        max_attempts: How many times one conversation may pass through triage.
            A specialist is allowed to bounce an off-topic request back here
            (the spec requires the Billing agent to do so); without a ceiling
            that could ping-pong forever, so after ``max_attempts`` the request
            is forced to the General queue and answered there.
        use_history: Whether to show the classifier the prior transcript. On a
            bounce-back this is what lets it avoid repeating its first choice.
    """

    node_name = "triage"

    def __init__(
        self,
        classifier: IntentClassifier,
        *,
        max_attempts: int = 2,
        use_history: bool = True,
    ) -> None:
        self._classifier = classifier
        self._max_attempts = max(1, max_attempts)
        self._use_history = use_history

    # ------------------------------------------------------------------ #
    def handle(self, state: SupportState) -> Mapping[str, Any]:
        """Classify the request and set ``department`` / ``next_step``."""
        message = self.current_message(state)
        attempt = int(state.get("triage_attempts", 0)) + 1

        # Loop guard: we have already routed this request and a specialist sent
        # it back. Stop bouncing and let the General agent handle it.
        if attempt > self._max_attempts:
            logger.info("Triage attempt %s exceeds limit; forcing General.", attempt)
            return {
                "department": Department.GENERAL.value,
                "next_step": NextStep.GENERAL.value,
                "triage_attempts": attempt,
                "messages": [
                    self.say(
                        "Routed to the general queue after "
                        f"{self._max_attempts} unsuccessful routing attempts."
                    )
                ],
            }

        history = transcript(state) if (self._use_history and attempt > 1) else ""
        decision = self._classifier.classify(message, history=history)

        department = decision.department
        next_step = NextStep.for_department(department)

        logger.info(
            "Triage -> %s (confidence=%.2f): %s",
            department.value, decision.confidence, decision.reasoning,
        )

        return {
            "department": department.value,
            "next_step": next_step.value,
            "triage_attempts": attempt,
            "messages": [
                self.say(
                    f"Classified as {department.value} "
                    f"(confidence {decision.confidence:.2f}). {decision.reasoning}".strip()
                )
            ],
            # Keep the raw query handy for the specialist that runs next.
            "user_query": message,
        }
