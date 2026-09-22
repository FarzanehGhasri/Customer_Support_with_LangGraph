"""
LLM abstraction (Dependency Inversion).

Nothing in ``agents/`` or ``graph/`` may import ``langchain_openai`` or
``langchain_anthropic`` directly.  They depend on :class:`ChatModelProvider`
instead, and the concrete provider is injected at construction time.  Benefits:

* the agents are unit-testable with a fake provider and no API key;
* switching vendor (or pointing at an OpenAI-compatible proxy) is a config
  change, not a code change.

``Protocol`` is used rather than an ABC so that any object with the right
methods qualifies -- no inheritance required from third-party classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Type, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    # Imported lazily so this module can be loaded without LangChain installed.
    from langchain_core.language_models import BaseChatModel
    from pydantic import BaseModel


@runtime_checkable
class ChatModelProvider(Protocol):
    """Supplies configured chat models to the agents."""

    def get_chat_model(self, *, temperature: float | None = None) -> "BaseChatModel":
        """Return a ready-to-use chat model.

        Args:
            temperature: Optional per-call override.  Classification nodes want
                0.0 for determinism; a chatty General agent may want more.
        """
        ...

    def get_structured_model(
        self,
        schema: "Type[BaseModel]",
        *,
        temperature: float | None = None,
    ) -> Any:
        """Return a model bound to ``schema`` via ``with_structured_output``.

        The return value is a Runnable whose ``.invoke()`` yields an instance of
        ``schema`` rather than free text.  This is the mechanism Step 2 of the
        assignment explicitly asks for.
        """
        ...
