"""
Billing abstractions -- one Protocol per tool the spec requires.

``check_subscription_status`` is a *read* against a customer record, while
``process_refund`` is a *write* against a payment system.  They are different
responsibilities with different failure modes and different blast radius, so
they get two interfaces rather than one ``BillingService`` god-object
(Single Responsibility + Interface Segregation).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.schemas import RefundReceipt, SubscriptionStatus


@runtime_checkable
class SubscriptionRepository(Protocol):
    """Read-only access to customer subscription records."""

    def get_status(self, user_id: str) -> SubscriptionStatus:
        """Look up ``user_id``.

        Implementations MUST return a :class:`SubscriptionStatus` with
        ``found=False`` for unknown ids rather than raising, so the agent can
        respond politely instead of crashing the graph.
        """
        ...


@runtime_checkable
class RefundGateway(Protocol):
    """Executes refunds (mocked for this assignment)."""

    def process_refund(self, transaction_id: str) -> RefundReceipt:
        """Attempt to refund ``transaction_id``."""
        ...
