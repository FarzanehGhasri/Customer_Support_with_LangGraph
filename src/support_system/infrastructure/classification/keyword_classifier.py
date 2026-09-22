"""
Dependency-free intent classifier.

Purpose is twofold:

1. **Testing.** The triage node can be exercised end to end with no API key,
   no network and no cost -- which is how the Step 2 tests run.
2. **Liskov demonstration.** It implements exactly the same
   :class:`IntentClassifier` Protocol as :class:`LLMIntentClassifier`, so it can
   be substituted anywhere the LLM version is used without the caller noticing.

It is a keyword scorer, not a pretend LLM: it is intentionally simple and its
confidence reflects how weak the evidence is.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from ...domain.enums import Department
from ...domain.schemas import TriageDecision

#: Keyword -> weight, per department. Multi-word phrases are matched as phrases.
DEFAULT_KEYWORDS: Mapping[Department, Sequence[str]] = {
    Department.BILLING: (
        "refund", "refunded", "invoice", "charge", "charged", "payment", "pay",
        "billing", "bill", "subscription", "subscribe", "renew", "renewal",
        "cancel my plan", "plan", "price", "pricing", "money", "credit card",
        "double charged", "overcharged", "receipt", "transaction",
    ),
    Department.TECHNICAL: (
        "error", "bug", "crash", "crashes", "crashing", "freeze", "frozen",
        "not working", "broken", "install", "installation", "update", "upgrade",
        "sync", "syncing", "password", "reset my password", "forgot my password",
        "log in", "login", "sign in", "2fa", "two-factor", "app", "how do i",
        "how can i", "white screen", "slow",
    ),
    Department.GENERAL: (
        "hello", "hi ", "hey", "good morning", "good evening", "thanks",
        "thank you", "who are you", "what do you do", "opening hours",
        "contact", "bye",
    ),
}

#: A Billing keyword outweighs a Technical one, because the spec's scenario 2
#: ("my subscription is not working") contains both and must land in Billing.
DEPARTMENT_WEIGHTS: Mapping[Department, float] = {
    Department.BILLING: 1.5,
    Department.TECHNICAL: 1.0,
    Department.GENERAL: 0.5,
}


class KeywordIntentClassifier:
    """Rule-based stand-in for :class:`LLMIntentClassifier`."""

    def __init__(
        self,
        keywords: Mapping[Department, Sequence[str]] | None = None,
        *,
        default_department: Department = Department.GENERAL,
    ) -> None:
        self._keywords = dict(keywords or DEFAULT_KEYWORDS)
        self._default = default_department

    def classify(self, user_message: str, *, history: str = "") -> TriageDecision:
        """Score the message against each department's keyword list."""
        text = f" {user_message.lower().strip()} "
        if not text.strip():
            return TriageDecision(
                department=self._default, reasoning="Empty message.", confidence=0.0
            )

        scores: dict[Department, float] = {}
        hits: dict[Department, list[str]] = {}
        for department, words in self._keywords.items():
            weight = DEPARTMENT_WEIGHTS.get(department, 1.0)
            matched = [w for w in words if self._contains(text, w)]
            if matched:
                scores[department] = len(matched) * weight
                hits[department] = matched

        if not scores:
            return TriageDecision(
                department=self._default,
                reasoning="No keyword matched; defaulting.",
                confidence=0.2,
            )

        best = max(scores, key=lambda d: scores[d])
        total = sum(scores.values())
        # Confidence = share of the winning department in the total score, so a
        # message matching two departments equally reports low confidence.
        confidence = round(min(scores[best] / total, 1.0), 2)
        return TriageDecision(
            department=best,
            reasoning=f"Matched {best.value} keywords: {', '.join(hits[best][:4])}.",
            confidence=confidence,
        )

    @staticmethod
    def _contains(haystack: str, needle: str) -> bool:
        """Whole-word / phrase match so 'app' does not match 'happy'."""
        return re.search(rf"(?<![a-z]){re.escape(needle.strip())}(?![a-z])", haystack) is not None
