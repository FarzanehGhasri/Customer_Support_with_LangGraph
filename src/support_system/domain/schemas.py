"""
Structured data contracts.

These Pydantic models are the *typed boundary* between the LLM and our code.
Two distinct jobs are covered here, and they are deliberately kept in separate
models rather than one fat "result" object (Interface Segregation):

1. **LLM output schemas** -- ``TriageDecision`` and ``SentimentAssessment`` are
   handed to ``llm.with_structured_output(...)`` (Step 2 of the assignment), so
   the model is forced to answer with valid JSON instead of prose we would have
   to parse with regexes.
2. **Tool result schemas** -- ``SubscriptionStatus``, ``RefundReceipt`` and
   ``KnowledgeSnippet`` describe what the billing/RAG tools return.  Agents
   depend on these models, never on the concrete data source, which is what lets
   us swap the mock JSON store for a real API later (Dependency Inversion).
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from .enums import Department, Sentiment

# --------------------------------------------------------------------------- #
# 1. LLM output schemas (used with ``with_structured_output``)
# --------------------------------------------------------------------------- #


class TriageDecision(BaseModel):
    """Structured verdict of the Triage ("reception desk") agent.

    The field descriptions are not decoration: LangChain turns this model into a
    JSON schema and sends the descriptions to the model, so they are effectively
    part of the prompt.  Keep them short and unambiguous.
    """

    department: Department = Field(
        description=(
            "Which team must handle this request. "
            "'Billing' for payments, subscriptions, invoices and refunds. "
            "'Technical' for bugs, errors, how-to and product questions. "
            "'General' for greetings, small talk and anything else."
        )
    )
    reasoning: str = Field(
        description="One short sentence justifying the chosen department.",
        default="",
    )
    confidence: float = Field(
        description="How certain the classification is, between 0.0 and 1.0.",
        ge=0.0,
        le=1.0,
        default=1.0,
    )


class SentimentAssessment(BaseModel):
    """Structured verdict of the Sentiment Guardrail agent."""

    sentiment: Sentiment = Field(
        description=(
            "Tone of the user's own message. "
            "'Negative' for anger, insults, accusations, threats to leave or "
            "demands to speak to a manager. "
            "'Positive' for thanks and praise. 'Neutral' for everything else."
        )
    )
    reasoning: str = Field(
        description="One short sentence justifying the judgement.",
        default="",
    )

    @property
    def requires_human(self) -> bool:
        """Single place that decides whether to interrupt the graph."""
        return self.sentiment.needs_human


# --------------------------------------------------------------------------- #
# 2. Tool result schemas
# --------------------------------------------------------------------------- #


class SubscriptionStatus(BaseModel):
    """Result of ``check_subscription_status(user_id)``."""

    user_id: str
    found: bool = Field(description="False when no such customer exists.")
    plan: str = ""
    status: str = Field(
        default="unknown",
        description="'active' | 'expired' | 'cancelled' | 'trial' | 'unknown'.",
    )
    expires_on: Optional[date] = None
    auto_renew: bool = False

    @property
    def is_active(self) -> bool:
        return self.status in {"active", "trial"}

    def summary(self) -> str:
        """Human-readable one-liner the agent can quote verbatim."""
        if not self.found:
            return f"No subscription record found for user id '{self.user_id}'."
        expiry = f", expires on {self.expires_on.isoformat()}" if self.expires_on else ""
        return (
            f"User '{self.user_id}' is on the '{self.plan}' plan; "
            f"status: {self.status}{expiry}; auto-renew: {self.auto_renew}."
        )


class RefundReceipt(BaseModel):
    """Result of the mocked ``process_refund(transaction_id)``."""

    transaction_id: str
    approved: bool
    amount: float = 0.0
    currency: str = "USD"
    reference: str = Field(default="", description="Mock refund confirmation code.")
    reason: str = Field(default="", description="Why a refund was refused, if it was.")

    def summary(self) -> str:
        if self.approved:
            return (
                f"Refund of {self.amount:.2f} {self.currency} for transaction "
                f"'{self.transaction_id}' approved (reference {self.reference})."
            )
        return f"Refund for transaction '{self.transaction_id}' refused: {self.reason}"


class KnowledgeSnippet(BaseModel):
    """One retrieved passage from ``search_knowledge_base(query)``."""

    content: str
    source: str = Field(description="File or document the passage came from.")
    score: float = Field(
        default=0.0,
        description="Relevance score; higher is better. Scale depends on retriever.",
    )


class RetrievalResult(BaseModel):
    """All snippets for a query, plus whether the retriever considers them usable.

    ``has_grounding`` is what stops the Technical agent from hallucinating: when
    it is False the agent must answer "I don't know", as the spec requires.
    """

    query: str
    snippets: List[KnowledgeSnippet] = Field(default_factory=list)
    has_grounding: bool = False

    @field_validator("snippets")
    @classmethod
    def _sort_by_score(cls, value: List[KnowledgeSnippet]) -> List[KnowledgeSnippet]:
        """Always hand the agent the best passage first."""
        return sorted(value, key=lambda s: s.score, reverse=True)

    def as_context(self) -> str:
        """Format the snippets as a context block for the prompt."""
        if not self.snippets:
            return "(no relevant documentation found)"
        return "\n\n".join(
            f"[source: {s.source}]\n{s.content}" for s in self.snippets
        )


# --------------------------------------------------------------------------- #
# 3. Human-in-the-loop contract
# --------------------------------------------------------------------------- #


class HumanDecision(BaseModel):
    """What the "support manager" did after the graph was interrupted.

    Step 3 of the assignment asks for a function that can *approve*, *reject* or
    *replace* the agent's draft answer; this model is that function's return type.
    """

    approved: bool = Field(
        description="True to send the agent's draft, False to override or drop it."
    )
    replacement_response: str = Field(
        default="",
        description="Manager-written reply that replaces the draft when approved is False.",
    )
    note: str = Field(default="", description="Internal note for the audit trail.")

    def resolve(self, draft: str) -> str:
        """Pick the text that will actually be sent to the user."""
        if self.approved:
            return draft
        return self.replacement_response or draft
