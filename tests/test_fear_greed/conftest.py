"""Shared fixtures for the Fear & Greed ingest suite.

Everything here is deterministic and offline: the CNN payload is a recorded
JSON fixture and the DB is an in-memory SQLite created straight from the ORM
model (the `raw` JSONB column carries a SQLite JSON variant so `create_all`
works without Postgres). No live network, no timing sleeps.
"""

from __future__ import annotations

import copy
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
def payload_dup_current_day() -> dict:
    """Payload whose current (last) trading day appears TWICE with different
    scores — the live CNN shape (verified 2026-07-29: today served as both
    33.4531 and 33.4694). The SECOND/later point (score 63.0) is CNN's latest
    reading; the earlier (61.0) is the fixture's original last day. Every
    component series carries the same intra-day duplicate.
    """
    data = copy.deepcopy(json.loads(FIXTURE.read_text()))
    hist = data["fear_and_greed_historical"]["data"]
    dup = copy.deepcopy(hist[-1])
    dup["y"] = 63.0  # CNN's latest reading for the current, unsettled day
    dup["rating"] = "greed"
    hist.append(dup)
    for key in (
        "market_momentum_sp500",
        "market_momentum_sp125",
        "stock_price_strength",
        "stock_price_breadth",
        "put_call_options",
        "market_volatility_vix",
        "market_volatility_vix_50",
        "junk_bond_demand",
        "safe_haven_demand",
    ):
        series = data[key]["data"]
        cdup = copy.deepcopy(series[-1])
        cdup["y"] = (cdup.get("y") or 0.0) + 1.0
        series.append(cdup)
    return data


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
