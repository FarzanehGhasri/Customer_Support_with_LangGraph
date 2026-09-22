"""
Retrieval abstraction for the RAG-backed ``search_knowledge_base`` tool.

The Technical Support agent depends on this Protocol only.  Step 3 can therefore
start with a dependency-free keyword retriever and later drop in a FAISS/Chroma
vector store without touching the agent -- the essence of the Liskov
Substitution Principle: any retriever honouring this contract is a valid
stand-in for any other.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.schemas import RetrievalResult


@runtime_checkable
class KnowledgeRetriever(Protocol):
    """Searches the product documentation."""

    def search(self, query: str, *, top_k: int = 3) -> RetrievalResult:
        """Return the passages most relevant to ``query``.

        Implementations MUST set ``RetrievalResult.has_grounding`` to False when
        nothing passes their relevance threshold, so the calling agent can admit
        it does not know instead of inventing an answer.
        """
        ...
