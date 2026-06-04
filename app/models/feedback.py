"""Feedback ORM model — persists user feedback so it can feed future tuning.

Previously ``/feedback`` only logged and discarded the rating; for a service
whose thesis is self-optimization, that signal is worth keeping. Rows carry the
``request_id`` correlation id (see ``app/observability.py``) so a feedback entry
can be tied back to its log trail and, eventually, the query it rates.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _now() -> datetime:
    return datetime.now(UTC)


class Feedback(Base):
    """A single feedback submission against a prior query."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[str] = mapped_column(String(64), index=True)
    rating: Mapped[int] = mapped_column(Integer)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrections: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
