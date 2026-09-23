"""
Loading and chunking the knowledge base.

Split out from the retrievers because *how documents are cut up* is a separate
concern from *how they are searched*: the keyword retriever and the embedding
retriever consume the identical chunks, which is also what makes comparing them
meaningful.

Chunking strategy: split each markdown file on its ``##`` headings. Support
articles are already organised by question ("Reset a forgotten password",
"Reset link does not work"), so the author's own structure is a better boundary
than a fixed character count -- each chunk is one self-contained answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

logger = logging.getLogger(__name__)

#: A line starting a level-2 section.
_SECTION_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage."""

    content: str
    #: File the passage came from, e.g. "password_reset.md".
    source: str
    #: The ``##`` heading, or the document title for the preamble.
    heading: str

    @property
    def searchable_text(self) -> str:
        """Text used for matching.

        The heading is included so a query like "reset password" scores against
        the section title as well as the body.
        """
        return f"{self.heading}\n{self.content}"


def load_chunks(directory: Path | str, *, pattern: str = "*.md") -> List[Chunk]:
    """Read every markdown file in ``directory`` and split it into chunks."""
    directory = Path(directory)
    if not directory.is_dir():
        logger.warning("Knowledge base directory not found: %s", directory)
        return []

    chunks: List[Chunk] = []
    for path in sorted(directory.glob(pattern)):
        chunks.extend(split_document(path.read_text(encoding="utf-8"), source=path.name))
    logger.info("Loaded %d chunks from %s", len(chunks), directory)
    return chunks


def split_document(text: str, *, source: str) -> List[Chunk]:
    """Split one markdown document on its ``##`` headings."""
    title = _document_title(text) or source
    matches = list(_SECTION_RE.finditer(text))

    # No ## headings: keep the document whole rather than inventing boundaries.
    if not matches:
        body = text.strip()
        return [Chunk(content=body, source=source, heading=title)] if body else []

    chunks: List[Chunk] = []

    # Text before the first ## (title, keywords line, "applies to") is kept as
    # its own chunk: it carries the keyword hints that help retrieval.
    preamble = text[: matches[0].start()].strip()
    if preamble:
        chunks.append(Chunk(content=preamble, source=source, heading=title))

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.start():end].strip()
        if body:
            # Prefix the document title so a chunk read in isolation still says
            # which article it belongs to.
            chunks.append(
                Chunk(content=body, source=source, heading=f"{title} — {match.group(1).strip()}")
            )
    return chunks


def _document_title(text: str) -> str:
    """First level-1 heading of the document, if any."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens, used by the keyword retriever and the tests."""
    return re.findall(r"[a-z0-9]+", text.lower())


def unique(items: Iterable[str]) -> List[str]:
    """Order-preserving de-duplication."""
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
