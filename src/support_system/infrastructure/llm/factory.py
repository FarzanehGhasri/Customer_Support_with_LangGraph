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
