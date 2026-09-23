"""
JSON-file implementation of :class:`SubscriptionRepository`.

Backs the ``check_subscription_status(user_id)`` tool the spec requires.
It reads ``data/mock/subscriptions.json``; swapping in a real billing API later
means writing another class against the same Protocol and changing one
constructor argument -- no agent code moves.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from ...domain.schemas import SubscriptionStatus

logger = logging.getLogger(__name__)


class JsonSubscriptionRepository:
    """Looks customers up in a JSON file."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        # Loaded lazily and cached: constructing a repository must not touch the
        # disk, so tests and notebooks can build one cheaply.
        self._records: dict[str, Mapping[str, Any]] | None = None

    # ------------------------------------------------------------------ #
    # SubscriptionRepository Protocol
    # ------------------------------------------------------------------ #
    def get_status(self, user_id: str) -> SubscriptionStatus:
        """Return the subscription for ``user_id``.

        An unknown id yields ``found=False`` rather than an exception: the
        billing agent must be able to answer "I can't find that account"
        politely instead of crashing the graph.
        """
        user_id = (user_id or "").strip()
        record = self._load().get(user_id)

        if record is None:
            logger.info("No subscription record for user_id=%r", user_id)
            return SubscriptionStatus(user_id=user_id, found=False)

        return SubscriptionStatus(
            user_id=user_id,
            found=True,
            plan=str(record.get("plan", "")),
            status=str(record.get("status", "unknown")),
            expires_on=self._parse_date(record.get("expires_on")),
            auto_renew=bool(record.get("auto_renew", False)),
        )

    # ------------------------------------------------------------------ #
    def _load(self) -> dict[str, Mapping[str, Any]]:
        if self._records is None:
            if not self._path.is_file():
                logger.warning("Subscription file missing: %s", self._path)
                self._records = {}
            else:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
                self._records = {
                    str(c["user_id"]): c for c in payload.get("customers", [])
                }
        return self._records

    @staticmethod
    def _parse_date(value: Any) -> date | None:
        if not value:
            return None
        try:
            return date.fromisoformat(str(value))
        except ValueError:
            logger.warning("Unparseable date in subscription record: %r", value)
            return None
