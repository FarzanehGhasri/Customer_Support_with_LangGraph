"""
LLM-backed intent classifier -- assignment Step 2.

This is the component that fulfils the requirement:

    "با استفاده از قابلیت with_structured_output در لنگچین، ایجنت تریاژ را
     طوری طراحی کنید که خروجی آن حتماً یک JSON باشد"

    (design the triage agent with LangChain's `with_structured_output` so its
     output is always JSON)

The class implements :class:`~support_system.interfaces.classification.IntentClassifier`
and depends only on the :class:`ChatModelProvider` Protocol, never on a concrete
vendor SDK.
"""

from __future__ import annotations

import logging

from ...domain.enums import Department
from ...domain.schemas import TriageDecision
from ...interfaces.llm import ChatModelProvider
from ...prompts.triage import (
    TRIAGE_SYSTEM_PROMPT,
    TRIAGE_USER_PROMPT,
    build_history_block,
)

logger = logging.getLogger(__name__)


class LLMIntentClassifier:
    """Classifies a customer message into a department using an LLM.

    Args:
        provider: Supplies the chat model. Injected, so the same class works
            against OpenAI, a gateway like GapGPT, Anthropic or a test double.
        temperature: Forced to 0.0 by default -- classification must be
            reproducible, otherwise the same message could be routed to two
            different teams on two runs.
        fallback_department: Used when the model or the network fails. Routing
            to General is the safe choice: a human still sees the message,
            whereas crashing the graph loses it entirely.
    """

    def __init__(
        self,
        provider: ChatModelProvider,
        *,
        temperature: float = 0.0,
        fallback_department: Department = Department.GENERAL,
    ) -> None:
        self._provider = provider
        self._temperature = temperature
        self._fallback_department = fallback_department
        # Built lazily on first use: constructing the classifier must not
        # require credentials or a network round-trip.
        self._runnable = None

    # ------------------------------------------------------------------ #
    # IntentClassifier Protocol
    # ------------------------------------------------------------------ #
    def classify(self, user_message: str, *, history: str = "") -> TriageDecision:
        """Return the department that should handle ``user_message``."""
        if not user_message.strip():
            # Nothing to classify; don't waste a paid API call on empty input.
            return TriageDecision(
                department=self._fallback_department,
                reasoning="Empty message.",
                confidence=0.0,
            )

        messages = [
            ("system", TRIAGE_SYSTEM_PROMPT),
            (
                "human",
                TRIAGE_USER_PROMPT.format(
                    history_block=build_history_block(history),
                    user_message=user_message.strip(),
                ),
            ),
        ]

        try:
            decision = self._structured_model().invoke(messages)
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            # A classification failure must degrade, not abort: the graph keeps
            # running and the customer still gets an answer.
            logger.warning("Triage classification failed (%s); falling back.", exc)
            return TriageDecision(
                department=self._fallback_department,
                reasoning=f"Classifier unavailable ({type(exc).__name__}); routed to fallback.",
                confidence=0.0,
            )

        # `with_structured_output` normally returns the Pydantic object already,
        # but some providers hand back a dict. Normalise so callers see one type.
        if isinstance(decision, dict):
            decision = TriageDecision.model_validate(decision)

        # The schema allows Department.TRIAGE (specialists use it to bounce a
        # request back), but triage itself must never route to itself.
        if decision.department is Department.TRIAGE:
            logger.warning("Triage returned TRIAGE; coercing to %s.", self._fallback_department)
            decision = decision.model_copy(update={"department": self._fallback_department})

        return decision

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _structured_model(self):
        """Build the schema-bound runnable once and reuse it."""
        if self._runnable is None:
            self._runnable = self._provider.get_structured_model(
                TriageDecision, temperature=self._temperature
            )
        return self._runnable
