"""Abstract contracts (Protocols) that every concrete component implements.

Importing from this package -- never from a concrete implementation -- is how
the higher-level policy (agents, graph) stays independent of the low-level
details (which LLM vendor, which vector store, which database).
"""

from .billing import RefundGateway, SubscriptionRepository
from .classification import IntentClassifier, SentimentAnalyzer
from .embeddings import EmbeddingProvider
from .human import HumanReviewer
from .llm import ChatModelProvider
from .nodes import SupportNode
from .retrieval import KnowledgeRetriever

__all__ = [
    "ChatModelProvider",
    "EmbeddingProvider",
    "IntentClassifier",
    "SentimentAnalyzer",
    "KnowledgeRetriever",
    "SubscriptionRepository",
    "RefundGateway",
    "SupportNode",
    "HumanReviewer",
]
