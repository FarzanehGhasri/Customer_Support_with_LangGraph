"""Pure domain layer: enums, state shape and data contracts.

This package has no dependency on LangChain, LangGraph or any vendor SDK --
only on Pydantic.  Keeping it dependency-free means the business rules can be
tested (and understood) without an API key.
"""

from .enums import Department, NextStep, Sentiment
from .schemas import (
    HumanDecision,
    KnowledgeSnippet,
    RefundReceipt,
    RetrievalResult,
    SentimentAssessment,
    SubscriptionStatus,
    TriageDecision,
)
from .state import SupportState, initial_state, transcript

__all__ = [
    "Department",
    "NextStep",
    "Sentiment",
    "TriageDecision",
    "SentimentAssessment",
    "SubscriptionStatus",
    "RefundReceipt",
    "KnowledgeSnippet",
    "RetrievalResult",
    "HumanDecision",
    "SupportState",
    "initial_state",
    "transcript",
]
