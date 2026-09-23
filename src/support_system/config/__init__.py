"""Configuration layer -- the only place that reads the environment."""

from .dotenv import load_dotenv
from .settings import (
    API_KEY_ENV_VARS,
    DEFAULT_EMBEDDING_MODELS,
    DEFAULT_MODELS,
    PROJECT_ROOT,
    Settings,
)

__all__ = [
    "Settings",
    "PROJECT_ROOT",
    "API_KEY_ENV_VARS",
    "DEFAULT_MODELS",
    "DEFAULT_EMBEDDING_MODELS",
    "load_dotenv",
]
