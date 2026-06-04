"""Tests that /feedback persists to the DB (no LM, in-memory sqlite)."""

from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.endpoints.core import submit_feedback
from app.database import Base
from app.models.feedback import Feedback
from app.schemas.request import FeedbackRequest


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_feedback_model_roundtrip() -> None:
    s = _session()
    s.add(
        Feedback(
            query_id="qry_1", rating=5, feedback="great", corrections={"fix": 1}, request_id="r1"
        )
    )
    s.commit()
    rows = s.execute(select(Feedback)).scalars().all()
    assert len(rows) == 1
    assert rows[0].rating == 5
    assert rows[0].corrections == {"fix": 1}
    assert rows[0].created_at is not None


async def test_submit_feedback_persists() -> None:
    s = _session()
    resp = await submit_feedback(
        FeedbackRequest(query_id="qry_2", rating=4, feedback=None, corrections=None),
        db=s,
    )
    assert resp.success
    rows = s.execute(select(Feedback)).scalars().all()
    assert len(rows) == 1
    assert rows[0].query_id == "qry_2"
    assert rows[0].rating == 4
