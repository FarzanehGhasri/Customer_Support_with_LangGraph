"""
Concrete :class:`~support_system.interfaces.llm.ChatModelProvider` implementations.

Design notes
------------
*Open/Closed*: vendors are registered in a dictionary through the
``@register_provider`` decorator.  Supporting a new vendor means adding a new
builder function below (or in your own module, even outside this package) --
the factory itself never changes.

*Lazy imports*: each builder imports its LangChain integration package **inside**
the function.  That way you only need to install the integration you actually
use, and merely importing this module never fails because
``langchain_anthropic`` happens to be missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Type

from ...config.settings import Settings

#: provider name -> builder(settings, temperature) -> BaseChatModel
_PROVIDER_REGISTRY: Dict[str, Callable[[Settings, float], Any]] = {}


def register_provider(name: str) -> Callable[[Callable[[Settings, float], Any]], Callable[[Settings, float], Any]]:
    """Decorator that adds a chat-model builder to the registry."""

    def decorator(builder: Callable[[Settings, float], Any]) -> Callable[[Settings, float], Any]:
        _PROVIDER_REGISTRY[name.lower()] = builder
        return builder

    return decorator


def available_providers() -> tuple[str, ...]:
    """Names accepted by :class:`LangChainModelProvider`."""
    return tuple(sorted(_PROVIDER_REGISTRY))


# --------------------------------------------------------------------------- #
# Builders -- one per vendor.  Each one is small and knows a single SDK.
# --------------------------------------------------------------------------- #


@register_provider("openai")
def _build_openai(settings: Settings, temperature: float) -> Any:
    """OpenAI, and any OpenAI-compatible gateway via ``base_url``."""
    from langchain_openai import ChatOpenAI  # imported lazily on purpose

    kwargs: dict[str, Any] = {
        "model": settings.model,
        "temperature": temperature,
        "api_key": settings.api_key,
        "timeout": settings.request_timeout,
        "max_retries": settings.max_retries,
    }
    if settings.base_url:
        # Lets the same code talk to a proxy / self-hosted compatible endpoint.
        kwargs["base_url"] = settings.base_url
    return ChatOpenAI(**kwargs)


@register_provider("anthropic")
def _build_anthropic(settings: Settings, temperature: float) -> Any:
    """Anthropic Claude models."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.model,
        temperature=temperature,
        api_key=settings.api_key,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )


@register_provider("google")
def _build_google(settings: Settings, temperature: float) -> Any:
    """Google Gemini models."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.model,
        temperature=temperature,
        google_api_key=settings.api_key,
    )


# --------------------------------------------------------------------------- #
# The provider object the rest of the system depends on
# --------------------------------------------------------------------------- #


class LangChainModelProvider:
    """Adapter turning :class:`Settings` into LangChain chat models.

    Implements the :class:`ChatModelProvider` Protocol.  Instances are cheap and
    cache the built model, so all agents share one client (and one connection
    pool) per temperature.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: Dict[float, Any] = {}

    @property
    def settings(self) -> Settings:
        return self._settings

    def get_chat_model(self, *, temperature: float | None = None) -> Any:
        """Build (or reuse) a chat model for the configured provider."""
        temp = self._settings.temperature if temperature is None else temperature
        if temp in self._cache:
            return self._cache[temp]

        # Fail fast with a readable message instead of a vendor auth error.
        self._settings.require_credentials()

        provider = self._settings.provider.lower()
        try:
            builder = _PROVIDER_REGISTRY[provider]
        except KeyError as exc:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Available: {', '.join(available_providers())}. "
                "Add a new one with @register_provider without touching this class."
            ) from exc

        model = builder(self._settings, temp)
        self._cache[temp] = model
        return model

    def get_structured_model(
        self,
        schema: Type[Any],
        *,
        temperature: float | None = None,
    ) -> Any:
        """Bind a Pydantic schema so the model must answer with valid JSON.

        This is the ``with_structured_output`` requirement from Step 2 of the
        assignment, kept behind the provider interface so every agent gets it
        the same way.
        """
        return self.get_chat_model(temperature=temperature).with_structured_output(schema)


# --------------------------------------------------------------------------- #
# Embeddings -- a separate registry and a separate class, because embeddings
# are a separate interface (see interfaces/embeddings.py).
# --------------------------------------------------------------------------- #

_EMBEDDING_REGISTRY: Dict[str, Callable[[Settings], Any]] = {}


def register_embedding_provider(name: str) -> Callable[[Callable[[Settings], Any]], Callable[[Settings], Any]]:
    """Decorator that adds an embedding builder to the registry."""

    def decorator(builder: Callable[[Settings], Any]) -> Callable[[Settings], Any]:
        _EMBEDDING_REGISTRY[name.lower()] = builder
        return builder

    return decorator


@register_embedding_provider("openai")
def _build_openai_embeddings(settings: Settings) -> Any:
    from langchain_openai import OpenAIEmbeddings

    kwargs: dict[str, Any] = {
        "model": settings.embedding_model,
        "api_key": settings.api_key,
        "timeout": settings.request_timeout,
    }
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    return OpenAIEmbeddings(**kwargs)


@register_embedding_provider("google")
def _build_google_embeddings(settings: Settings) -> Any:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model, google_api_key=settings.api_key
    )


class LangChainEmbeddingProvider:
    """Adapter implementing the :class:`EmbeddingProvider` Protocol."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None

    @property
    def is_configured(self) -> bool:
        """True when this provider can actually embed anything.

        The graph builder checks this to decide between the vector retriever
        and the keyword one, rather than discovering the problem mid-run.
        """
        return bool(
            self._settings.embedding_model
            and self._settings.has_credentials
            and self._settings.provider.lower() in _EMBEDDING_REGISTRY
        )

    def _get(self) -> Any:
        if self._model is None:
            self._settings.require_credentials()
            provider = self._settings.provider.lower()
            if provider not in _EMBEDDING_REGISTRY:
                raise ValueError(
                    f"Provider '{provider}' has no embedding support registered. "
                    f"Available: {', '.join(sorted(_EMBEDDING_REGISTRY))}."
                )
            self._model = _EMBEDDING_REGISTRY[provider](self._settings)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._get().embed_query(text)


# --------------------------------------------------------------------------- #
# Reachability probe
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProviderCapabilities:
    """What the configured endpoint can actually do.

    Two capabilities, probed separately, because they fail independently and
    the system needs them for different jobs:

    * ``chat`` -- ordinary completions. The composers need this to phrase
      answers.
    * ``structured_output`` -- ``with_structured_output``, i.e. JSON-schema or
      function calling. **Triage and the sentiment guardrail depend on it
      entirely.**

    OpenAI-compatible gateways very often support the first and not the second.
    Treating them as one capability is what produced the bug this class exists
    to prevent: chat worked, the probe passed, the system ran "live", and every
    classification silently fell back to General -- routing nothing, calling no
    tools, and never escalating an angry customer.
    """

    chat: bool = False
    structured_output: bool = False
    #: Why chat is unavailable, if it is.
    chat_error: str = ""
    #: Why structured output is unavailable, if it is.
    structured_error: str = ""

    @property
    def fully_usable(self) -> bool:
        """True when every component can run in its model-backed form."""
        return self.chat and self.structured_output

    @property
    def summary(self) -> str:
        """One line suitable for printing to a user."""
        if self.fully_usable:
            return "chat + structured output"
        if self.chat:
            return f"chat only -- structured output unavailable ({self.structured_error})"
        return f"unusable ({self.chat_error})"


def probe_capabilities(settings: Settings, *, timeout: int = 20) -> ProviderCapabilities:
    """Find out what the endpoint supports, with two small calls.

    Every component in this project degrades gracefully when a model call
    fails. That is right in production but it makes a *misconfigured* system
    look like a working one: the graph completes, every answer is wrong, and
    nothing raises. So the capabilities are established up front, deliberately,
    rather than discovered in the output.
    """
    if not settings.has_credentials:
        return ProviderCapabilities(
            chat_error=f"no API key ({settings.api_key_env_var} is not set)"
        )

    provider = LangChainModelProvider(settings)

    # --- 1. plain chat -------------------------------------------------- #
    try:
        model = provider.get_chat_model()
        response = model.invoke(
            [("human", "Reply with the single word: ok")],
            config={"max_tokens": 5, "timeout": timeout},
        )
    except Exception as exc:  # noqa: BLE001
        detail = str(exc).strip().splitlines()[0][:160]
        return ProviderCapabilities(chat_error=f"{type(exc).__name__}: {detail}")

    if not str(getattr(response, "content", "")).strip():
        return ProviderCapabilities(chat_error="the endpoint answered with empty content")

    # --- 2. structured output ------------------------------------------- #
    # Probed with the real schema the triage agent uses, on a message whose
    # correct answer is unambiguous -- so a gateway that accepts the request but
    # returns nonsense is caught here too, not in front of a customer.
    from ...domain.enums import Department
    from ...domain.schemas import TriageDecision

    try:
        decision = provider.get_structured_model(TriageDecision, temperature=0.0).invoke(
            [
                ("system", "Classify the customer's message into a department."),
                ("human", "I want a refund for my subscription payment."),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        detail = str(exc).strip().splitlines()[0][:160]
        return ProviderCapabilities(chat=True, structured_error=f"{type(exc).__name__}: {detail}")

    if isinstance(decision, dict):
        try:
            decision = TriageDecision.model_validate(decision)
        except Exception as exc:  # noqa: BLE001
            return ProviderCapabilities(
                chat=True, structured_error=f"returned an unusable object: {exc}"
            )
    if not isinstance(decision, TriageDecision):
        return ProviderCapabilities(
            chat=True,
            structured_error=f"returned {type(decision).__name__}, not a TriageDecision",
        )
    if decision.department is not Department.BILLING:
        # It produced valid JSON but classified a plain refund request wrongly.
        return ProviderCapabilities(
            chat=True,
            structured_error=(
                f"schema honoured but the answer was wrong "
                f"(a refund request classified as {decision.department.value})"
            ),
        )

    return ProviderCapabilities(chat=True, structured_output=True)


def probe_provider(settings: Settings, *, timeout: int = 15) -> tuple[bool, str]:
    """Back-compatible wrapper: is the endpoint usable for chat at all?"""
    capabilities = probe_capabilities(settings, timeout=timeout)
    return capabilities.chat, capabilities.chat_error
