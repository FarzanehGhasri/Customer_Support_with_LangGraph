"""
Concrete :class:`ResponseComposer` implementations.

* :class:`LLMResponseComposer` -- the real one: sends the prompt to the model.
* :class:`TemplateResponseComposer` -- returns the caller's ``fallback``
  verbatim. Not a mock: the fallback is the verified tool output or the
  retrieved documentation, so the answer stays *true*, just unpolished. This is
  what lets the entire graph run with no credentials.
"""

from __future__ import annotations

import logging

from ...interfaces.llm import ChatModelProvider

logger = logging.getLogger(__name__)


class LLMResponseComposer:
    """Phrases answers with a chat model."""

    def __init__(self, provider: ChatModelProvider, *, temperature: float = 0.0) -> None:
        self._provider = provider
        self._temperature = temperature

    def compose(self, system_prompt: str, user_message: str, *, fallback: str) -> str:
        try:
            response = self._provider.get_chat_model(temperature=self._temperature).invoke(
                [("system", system_prompt), ("human", user_message)]
            )
        except Exception as exc:  # noqa: BLE001 - never let wording break the graph
            logger.warning("Composition failed (%s); using the verified fallback.", exc)
            return fallback

        text = getattr(response, "content", str(response)).strip()
        # An empty completion is a failure, not an answer.
        return text or fallback


class TemplateResponseComposer:
    """Returns the verified fallback text unchanged."""

    def compose(self, system_prompt: str, user_message: str, *, fallback: str) -> str:
        return fallback
