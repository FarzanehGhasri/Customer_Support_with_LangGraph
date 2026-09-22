"""
Classification abstractions.

Two *separate* small interfaces instead of one ``LLMAnalyzer`` with both
methods: the Triage node needs only :class:`IntentClassifier`, the guardrail
needs only :class:`SentimentAnalyzer`.  Neither is forced to depend on a method
it never calls (Interface Segregation Principle).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.schemas import SentimentAssessment, TriageDecision


@runtime_checkable
class IntentClassifier(Protocol):
    """Decides which department a request belongs to."""

    def classify(self, user_message: str, *, history: str = "") -> TriageDecision:
        """Classify ``user_message`` into a :class:`~..domain.enums.Department`."""
        ...


@runtime_checkable
class SentimentAnalyzer(Protocol):
    """Judges the tone of the user's message."""

    def analyze(self, user_message: str, *, history: str = "") -> SentimentAssessment:
        """Return the tone of ``user_message``."""
        ...
