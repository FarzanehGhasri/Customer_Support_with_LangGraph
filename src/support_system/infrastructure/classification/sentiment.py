"""
Sentiment analysers for the guardrail (Node 4).

Two implementations of the same :class:`SentimentAnalyzer` Protocol:

* :class:`LLMSentimentAnalyzer` -- the real one.
* :class:`KeywordSentimentAnalyzer` -- offline, deterministic, and the safety
  net when the model is unavailable.

The fallback direction matters. If the analyser fails we return **Neutral**,
not Negative: escalating every conversation to a human on an API outage would
be worse than occasionally missing an angry customer, and the human reviewer
can still be invoked manually.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from ...domain.enums import Sentiment
from ...domain.schemas import SentimentAssessment
from ...interfaces.llm import ChatModelProvider
from ...prompts.guardrail import SENTIMENT_SYSTEM_PROMPT, SENTIMENT_USER_PROMPT
from ...prompts.triage import build_history_block

logger = logging.getLogger(__name__)


class LLMSentimentAnalyzer:
    """Judges tone with a structured-output call."""

    def __init__(self, provider: ChatModelProvider, *, temperature: float = 0.0) -> None:
        self._provider = provider
        self._temperature = temperature
        self._runnable = None

    def analyze(self, user_message: str, *, history: str = "") -> SentimentAssessment:
        if not user_message.strip():
            return SentimentAssessment(sentiment=Sentiment.NEUTRAL, reasoning="Empty message.")

        if self._runnable is None:
            self._runnable = self._provider.get_structured_model(
                SentimentAssessment, temperature=self._temperature
            )
        try:
            assessment = self._runnable.invoke([
                ("system", SENTIMENT_SYSTEM_PROMPT),
                ("human", SENTIMENT_USER_PROMPT.format(
                    history_block=build_history_block(history),
                    user_message=user_message.strip(),
                )),
            ])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sentiment analysis failed (%s); assuming Neutral.", exc)
            return SentimentAssessment(
                sentiment=Sentiment.NEUTRAL,
                reasoning=f"Analyzer unavailable ({type(exc).__name__}).",
            )

        if isinstance(assessment, dict):
            assessment = SentimentAssessment.model_validate(assessment)
        return assessment


#: Words and phrases that mark real anger. Kept explicit and auditable rather
#: than hidden in a model, because this list decides when a human is called.
ANGRY_MARKERS: Sequence[str] = (
    "stole", "stolen", "steal", "thief", "scam", "fraud", "cheated", "rip off",
    "ripped me off", "useless", "terrible", "awful", "horrible", "disgusting",
    "pathetic", "ridiculous", "unacceptable", "furious", "angry", "outrageous",
    "worst", "garbage", "rubbish", "sue", "lawyer", "legal action",
    "speak to a manager", "talk to a manager", "speak to your manager",
    "want a human", "real person", "cancel everything", "never again",
    "fed up", "sick of", "disgrace", "shameful", "incompetent",
)

HAPPY_MARKERS: Sequence[str] = (
    "thank you", "thanks", "great", "excellent", "perfect", "appreciate",
    "wonderful", "brilliant", "helpful", "love it", "amazing",
)


class KeywordSentimentAnalyzer:
    """Rule-based stand-in for :class:`LLMSentimentAnalyzer`."""

    def __init__(
        self,
        angry: Sequence[str] = ANGRY_MARKERS,
        happy: Sequence[str] = HAPPY_MARKERS,
        *,
        shouting_ratio: float = 0.6,
        min_shouting_length: int = 12,
    ) -> None:
        self._angry = tuple(angry)
        self._happy = tuple(happy)
        self._shouting_ratio = shouting_ratio
        self._min_shouting_length = min_shouting_length

    def analyze(self, user_message: str, *, history: str = "") -> SentimentAssessment:
        text = user_message.lower()
        if not text.strip():
            return SentimentAssessment(sentiment=Sentiment.NEUTRAL, reasoning="Empty message.")

        hits = [marker for marker in self._angry if marker in text]
        if hits:
            return SentimentAssessment(
                sentiment=Sentiment.NEGATIVE,
                reasoning=f"Angry wording: {', '.join(hits[:3])}.",
            )

        if self._is_shouting(user_message):
            return SentimentAssessment(
                sentiment=Sentiment.NEGATIVE, reasoning="Message is written in capitals."
            )

        # Multiple exclamation marks read as shouting too.
        if user_message.count("!") >= 3:
            return SentimentAssessment(
                sentiment=Sentiment.NEGATIVE, reasoning="Repeated exclamation marks."
            )

        happy = [marker for marker in self._happy if marker in text]
        if happy:
            return SentimentAssessment(
                sentiment=Sentiment.POSITIVE, reasoning=f"Positive wording: {happy[0]}."
            )

        return SentimentAssessment(
            sentiment=Sentiment.NEUTRAL, reasoning="No emotional markers found."
        )

    def _is_shouting(self, message: str) -> bool:
        """ALL CAPS, but only for messages long enough for it to be meaningful."""
        letters = re.findall(r"[A-Za-z]", message)
        if len(letters) < self._min_shouting_length:
            return False
        upper = sum(1 for c in letters if c.isupper())
        return upper / len(letters) >= self._shouting_ratio
