"""
Dependency-free knowledge retriever (TF-IDF + cosine similarity).

Implements :class:`KnowledgeRetriever` with no API calls, so:

* the test suite exercises the real retrieval path offline and for free;
* the Technical agent still works if the embedding endpoint is unavailable.

TF-IDF is used rather than raw keyword counting because *idf* is what stops
common words ("the app", "settings") from dominating: a rare term like
"E-204" should outweigh a term appearing in every article.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

from ...domain.schemas import KnowledgeSnippet, RetrievalResult
from .documents import Chunk, load_chunks, tokenize

logger = logging.getLogger(__name__)

#: Words carrying no topical information. Without this list a question like
#: "What is the capital of France?" scores well against any article simply
#: because "what/is/the/of" occur everywhere -- which would let the Technical
#: agent believe it had found an answer when it had not.
STOPWORDS = frozenset("""
a an and are as at be been but by can cannot could did do does doing for from
get go had has have how i if in into is it its me my no not of on or our out
so some that the their them then there these they this to too us was we were
what when where which who why will with would you your
""".split())


def content_terms(text: str) -> list[str]:
    """Topical tokens of ``text`` -- everything that is not a stopword."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


class KeywordKnowledgeRetriever:
    """Ranks chunks by TF-IDF cosine similarity to the query.

    Args:
        knowledge_base_dir: Directory of markdown articles.
        min_score: Cosine similarity below which a result is ungrounded.
        min_coverage: Fraction of the query's *distinctive* terms (weighted by
            idf) that the best chunk must actually contain. Similarity alone is
            not enough: a short off-topic question can still score against a
            long article. Requiring the question's own rare words to appear is
            what makes "I don't know" reliable, as the spec demands.
        chunks: Pre-built chunks, mainly for tests.
    """

    def __init__(
        self,
        knowledge_base_dir: Path | str | None = None,
        *,
        min_score: float = 0.10,
        min_coverage: float = 0.34,
        chunks: Sequence[Chunk] | None = None,
    ) -> None:
        self._dir = Path(knowledge_base_dir) if knowledge_base_dir else None
        self._min_score = min_score
        self._min_coverage = min_coverage
        self._chunks: List[Chunk] | None = list(chunks) if chunks is not None else None
        self._vectors: List[Dict[str, float]] = []
        self._idf: Dict[str, float] = {}
        self._built = False

    # ------------------------------------------------------------------ #
    # KnowledgeRetriever Protocol
    # ------------------------------------------------------------------ #
    def search(self, query: str, *, top_k: int = 3) -> RetrievalResult:
        """Return the passages most similar to ``query``."""
        self._build()
        if not self._chunks or not query.strip():
            return RetrievalResult(query=query, snippets=[], has_grounding=False)

        query_terms = content_terms(query)
        query_vector = self._vectorize(query_terms)
        scored = [
            (self._cosine(query_vector, doc_vector), chunk)
            for doc_vector, chunk in zip(self._vectors, self._chunks)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        snippets = [
            KnowledgeSnippet(content=chunk.content, source=chunk.source, score=round(score, 4))
            for score, chunk in scored[:top_k]
            if score > 0
        ]
        # Grounding is decided by the *best* hit only: three weak matches are
        # not evidence, one strong match is. Both tests must pass.
        has_grounding = False
        coverage = 0.0
        if snippets:
            best_chunk = scored[0][1]
            coverage = self._coverage(query_terms, best_chunk)
            has_grounding = snippets[0].score >= self._min_score and coverage >= self._min_coverage
        if not has_grounding:
            logger.info(
                "No grounded answer for %r (best score=%.3f, coverage=%.2f)",
                query, snippets[0].score if snippets else 0.0, coverage,
            )
        return RetrievalResult(query=query, snippets=snippets, has_grounding=has_grounding)

    def _coverage(self, query_terms: Sequence[str], chunk: Chunk) -> float:
        """Share of the query's idf mass that ``chunk`` actually contains.

        A term the corpus has never seen counts as maximally distinctive and
        therefore unmatched, which is what makes a question about an unrelated
        topic fall through.
        """
        if not query_terms:
            return 0.0
        unseen_weight = max(self._idf.values(), default=1.0)
        chunk_terms = set(tokenize(chunk.searchable_text))
        matched = total = 0.0
        for term in set(query_terms):
            weight = self._idf.get(term, unseen_weight)
            total += weight
            if term in chunk_terms:
                matched += weight
        return matched / total if total else 0.0

    # ------------------------------------------------------------------ #
    # Index construction
    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        """Build the TF-IDF index once, on first search."""
        if self._built:
            return
        if self._chunks is None:
            self._chunks = load_chunks(self._dir) if self._dir else []

        documents = [content_terms(c.searchable_text) for c in self._chunks]
        total = len(documents)

        document_frequency: Counter[str] = Counter()
        for tokens in documents:
            document_frequency.update(set(tokens))

        # Smoothed idf; +1 keeps the weight positive for a term in every doc.
        self._idf = {
            term: math.log((total + 1) / (count + 1)) + 1.0
            for term, count in document_frequency.items()
        }
        self._vectors = [self._vectorize(tokens) for tokens in documents]
        self._built = True

    def _vectorize(self, tokens: Sequence[str]) -> Dict[str, float]:
        """Turn tokens into an L2-normalised tf-idf vector."""
        if not tokens:
            return {}
        counts = Counter(tokens)
        vector = {
            term: (count / len(tokens)) * self._idf.get(term, 1.0)
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return {}
        return {term: value / norm for term, value in vector.items()}

    @staticmethod
    def _cosine(left: Dict[str, float], right: Dict[str, float]) -> float:
        """Both vectors are already normalised, so this is just a dot product."""
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(term, 0.0) for term, value in left.items())
