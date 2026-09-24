"""
Node 3 -- Technical Support ("متخصص پشتیبانی فنی").

Owns one tool: ``search_knowledge_base`` (RAG). The spec's hard rule is that it
must not hallucinate -- "اگر پاسخ را در داکیومنت‌ها پیدا نکرد، نباید توهم داشته
باشد و باید اعلام کند که پاسخ را نمی‌داند".

That rule is enforced in two places, deliberately:

1. **Structurally** -- when the retriever reports no grounding the node takes a
   different branch entirely, with a prompt that contains no documentation to
   quote. The model is never given the *opportunity* to invent.
2. **In the prompt** -- the grounded branch is told to use only the passages.

Relying on the prompt alone would be a single point of failure.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from ..domain.enums import NextStep
from ..interfaces.composition import ResponseComposer
from ..interfaces.retrieval import KnowledgeRetriever
from ..prompts.specialists import TECHNICAL_ANSWER_PROMPT, TECHNICAL_NO_ANSWER_PROMPT
from .base import BaseSupportNode

logger = logging.getLogger(__name__)


class TechnicalAgent(BaseSupportNode):
    """Answers product questions strictly from the knowledge base.

    Args:
        retriever: Any :class:`KnowledgeRetriever` -- keyword or vector.
        composer: Phrases the answer from the retrieved passages.
        top_k: How many passages to put in front of the model.
    """

    node_name = "technical"
    display_name = "Technical Support Specialist"

    def __init__(
        self,
        retriever: KnowledgeRetriever,
        composer: ResponseComposer,
        *,
        top_k: int = 3,
    ) -> None:
        self._retriever = retriever
        self._composer = composer
        self._top_k = top_k

    # ------------------------------------------------------------------ #
    def handle(self, state) -> Mapping[str, Any]:
        message = self.current_message(state)
        result = self._retriever.search(message, top_k=self._top_k)

        sources = ", ".join(sorted({s.source for s in result.snippets})) or "none"
        best = result.snippets[0].score if result.snippets else 0.0
        tool_log = self.tool_log(
            "search_knowledge_base", message[:40],
            f"{'grounded' if result.has_grounding else 'no grounding'} "
            f"(best={best:.2f}, sources={sources})",
        )

        if result.has_grounding:
            draft = self._answer_from_documents(state, message, result)
        else:
            logger.info("No grounding for %r; admitting ignorance.", message)
            draft = self._admit_ignorance(state, message)

        return {
            "draft_response": draft,
            "next_step": NextStep.GUARDRAIL.value,
            "tool_calls": [tool_log],
            "messages": [self.say(draft)],
        }

    # ------------------------------------------------------------------ #
    def _answer_from_documents(self, state, message: str, result) -> str:
        """Answer from the retrieved passages.

        The fallback is the best passage itself rather than the whole context
        block: if the wording step fails, the customer still gets the correct
        documented steps instead of a dump of every candidate.
        """
        prompt = TECHNICAL_ANSWER_PROMPT.format(
            context=result.as_context(), user_message=message
        )
        return self._compose(state, prompt, message, fallback=result.snippets[0].content)

    def _admit_ignorance(self, state, message: str) -> str:
        """Branch taken when nothing was retrieved.

        Note the fallback text: if even this call fails we still say "I don't
        know" rather than going silent, because silence would be read as a
        successful answer by the guardrail downstream.
        """
        prompt = TECHNICAL_NO_ANSWER_PROMPT.format(user_message=message)
        return self._compose(
            state,
            prompt,
            message,
            fallback=(
                "I could not find an answer to this in our documentation, so I "
                "will pass your question to a human colleague."
            ),
        )

    def _compose(self, state, prompt: str, message: str, *, fallback: str) -> str:
        return self.compose_reply(self._composer, state, prompt, message, fallback=fallback)
