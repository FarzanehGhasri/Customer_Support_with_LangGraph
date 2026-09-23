"""
Vector-based knowledge retriever -- the "RAG" the spec asks for.

Chunks are embedded once and cached; a query is embedded and compared by cosine
similarity. Same :class:`KnowledgeRetriever` Protocol as the keyword retriever,
so the Technical agent cannot tell them apart (Liskov).

Why a hand-rolled vector store instead of FAISS or Chroma
---------------------------------------------------------
The knowledge base is six articles / ~26 chunks. An exact cosine scan over 26
vectors is instant and exact, whereas FAISS/Chroma add a large binary
dependency for an approximate index we do not need at this size. If the corpus
ever grows, this class is the only thing that has to change.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Sequence

from ...domain.schemas import KnowledgeSnippet, RetrievalResult
from ...interfaces.embeddings import EmbeddingProvider
from .documents import Chunk, load_chunks

logger = logging.getLogger(__name__)


class EmbeddingKnowledgeRetriever:
    """Semantic search over the knowledge base.

    Args:
        embeddings: Any :class:`EmbeddingProvider`.
        knowledge_base_dir: Directory of markdown articles.
        min_score: Cosine similarity below which the result is ungrounded.
            Embedding similarity has a much narrower dynamic range than TF-IDF
            (unrelated text still scores ~0.1-0.3 with most models), so this
            default is higher than the keyword retriever's.
        fallback: Optional retriever used when embedding fails -- a network
            outage then degrades search quality instead of breaking the agent.
    """

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        knowledge_base_dir: Path | str | None = None,
        *,
        min_score: float = 0.35,
        chunks: Sequence[Chunk] | None = None,
        fallback: object | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._dir = Path(knowledge_base_dir) if knowledge_base_dir else None
        self._min_score = min_score
        self._chunks: List[Chunk] | None = list(chunks) if chunks is not None else None
        self._vectors: List[List[float]] = []
        self._fallback = fallback
        self._built = False

    # ------------------------------------------------------------------ #
    # KnowledgeRetriever Protocol
    # ------------------------------------------------------------------ #
    def search(self, query: str, *, top_k: int = 3) -> RetrievalResult:
        """Return the passages semantically closest to ``query``."""
        if not query.strip():
            return RetrievalResult(query=query, snippets=[], has_grounding=False)

        try:
            self._build()
            query_vector = self._normalise(self._embeddings.embed_query(query))
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the graph
            logger.warning("Embedding search failed (%s).", exc)
            if self._fallback is not None:
                logger.info("Falling back to the keyword retriever.")
                return self._fallback.search(query, top_k=top_k)  # type: ignore[attr-defined]
            return RetrievalResult(query=query, snippets=[], has_grounding=False)

        if not self._chunks:
            return RetrievalResult(query=query, snippets=[], has_grounding=False)

        scored = sorted(
            (
                (self._dot(query_vector, vector), chunk)
                for vector, chunk in zip(self._vectors, self._chunks)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        snippets = [
            KnowledgeSnippet(content=chunk.content, source=chunk.source, score=round(score, 4))
            for score, chunk in scored[:top_k]
        ]
        has_grounding = bool(snippets) and snippets[0].score >= self._min_score
        if not has_grounding:
            logger.info("No grounded answer for %r (best=%.3f)", query,
                        snippets[0].score if snippets else 0.0)
        return RetrievalResult(query=query, snippets=snippets, has_grounding=has_grounding)

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        """Embed the corpus once, on first search."""
        if self._built:
            return
        if self._chunks is None:
            self._chunks = load_chunks(self._dir) if self._dir else []
        if self._chunks:
            raw = self._embeddings.embed_documents([c.searchable_text for c in self._chunks])
            self._vectors = [self._normalise(v) for v in raw]
            logger.info("Embedded %d chunks.", len(self._vectors))
        self._built = True

    @staticmethod
    def _normalise(vector: Sequence[float]) -> List[float]:
        """L2-normalise so cosine similarity is a plain dot product."""
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return list(vector)
        return [value / norm for value in vector]

    @staticmethod
    def _dot(left: Sequence[float], right: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(left, right))
