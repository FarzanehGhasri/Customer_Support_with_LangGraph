"""
Mocked refund gateway -- the spec asks for ``process_refund`` "به صورت Mock".

Even mocked, it enforces real refund policy, because the point of the exercise
is that the *agent* must not be the thing deciding whether money moves:

* an unknown transaction is refused, not invented;
* a transaction already refunded is refused (no double refunds);
* a transaction the payment provider marked non-refundable is refused.

That keeps the dangerous decision in deterministic code and leaves the LLM to
explain the outcome.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

from ...domain.schemas import RefundReceipt

logger = logging.getLogger(__name__)


class MockRefundGateway:
    """Approves refunds against a JSON list of transactions."""

    def __init__(self, path: Path | str, *, reference_prefix: str = "RF") -> None:
        self._path = Path(path)
        self._prefix = reference_prefix
        self._records: dict[str, Mapping[str, Any]] | None = None
        #: Transactions refunded during this session, so a second attempt in the
        #: same conversation is refused just like a previously-refunded one.
        self._refunded_now: set[str] = set()

    # ------------------------------------------------------------------ #
    # RefundGateway Protocol
    # ------------------------------------------------------------------ #
    def process_refund(self, transaction_id: str) -> RefundReceipt:
        """Attempt to refund ``transaction_id``."""
        transaction_id = (transaction_id or "").strip()
        if not transaction_id:
            return RefundReceipt(
                transaction_id="", approved=False,
                reason="No transaction id was provided.",
            )

        record = self._load().get(transaction_id)
        if record is None:
            logger.info("Refund refused: unknown transaction %r", transaction_id)
            return RefundReceipt(
                transaction_id=transaction_id, approved=False,
                reason="No such transaction could be found.",
            )

        if transaction_id in self._refunded_now or record.get("already_refunded"):
            return RefundReceipt(
                transaction_id=transaction_id, approved=False,
                amount=float(record.get("amount", 0.0)),
                currency=str(record.get("currency", "USD")),
                reason=str(record.get("reason") or "This transaction was already refunded."),
            )

        if not record.get("refundable", False):
            return RefundReceipt(
                transaction_id=transaction_id, approved=False,
                amount=float(record.get("amount", 0.0)),
                currency=str(record.get("currency", "USD")),
                reason=str(record.get("reason") or "This transaction is not refundable."),
            )

        self._refunded_now.add(transaction_id)
        receipt = RefundReceipt(
            transaction_id=transaction_id,
            approved=True,
            amount=float(record.get("amount", 0.0)),
            currency=str(record.get("currency", "USD")),
            reference=f"{self._prefix}-{transaction_id}",
        )
        logger.info("Refund approved: %s", receipt.summary())
        return receipt

    # ------------------------------------------------------------------ #
    def _load(self) -> dict[str, Mapping[str, Any]]:
        if self._records is None:
            if not self._path.is_file():
                logger.warning("Transaction file missing: %s", self._path)
                self._records = {}
            else:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
                self._records = {
                    str(t["transaction_id"]): t for t in payload.get("transactions", [])
                }
        return self._records
