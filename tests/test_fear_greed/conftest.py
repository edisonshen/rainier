"""Shared fixtures for the Fear & Greed ingest suite.

Everything here is deterministic and offline: the CNN payload is a recorded
JSON fixture and the DB is an in-memory SQLite created straight from the ORM
model (the `raw` JSONB column carries a SQLite JSON variant so `create_all`
works without Postgres). No live network, no timing sleeps.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from rainier.core.models import FearGreedIndex

FIXTURE = Path(__file__).parent / "fixtures" / "cnn_graphdata.json"


@pytest.fixture
def payload() -> dict:
    """The recorded CNN graphdata response (3 trading days, 9 components)."""
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def session_factory():
    """A context-manager session factory backed by a fresh in-memory SQLite.

    Mirrors the signature of ``rainier.core.database.get_session`` so the ingest
    orchestrators can be driven against SQLite in tests.
    """
    engine = create_engine("sqlite://")
    FearGreedIndex.__table__.create(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def _factory() -> Iterator[Session]:
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return _factory
