"""
Embedding abstraction, kept separate from :class:`ChatModelProvider`.

Interface Segregation again: the triage agent and the specialists need a *chat*
model and nothing else, so they must not depend on an interface that also
promises embeddings. Only the vector retriever asks for this one.
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors for similarity search."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents (indexing)."""
        ...

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query (searching).

        Kept distinct from :meth:`embed_documents` because some providers use
        different instructions for queries and documents.
        """
        ...
