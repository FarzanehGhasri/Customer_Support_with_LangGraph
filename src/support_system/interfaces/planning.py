"""
Planning abstraction for the billing specialist.

"Which tool should run, with which argument" is a decision that can be made by
an LLM with structured output or by deterministic rules. The billing agent
should not care which, so it depends on this Protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.schemas import BillingPlan


@runtime_checkable
class BillingPlanner(Protocol):
    """Chooses the billing action for a customer message."""

    def plan(self, user_message: str) -> BillingPlan:
        """Decide what the billing specialist should do."""
        ...
