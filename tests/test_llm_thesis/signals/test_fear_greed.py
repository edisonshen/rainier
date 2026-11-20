"""Spec: the fear_greed thesis signal reads CNN F&G as of the PRIOR close only."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from rainier.core.models import FearGreedIndex
from rainier.core.types import StockCandidate
from rainier.llm_thesis.signals import REGISTRY
from rainier.llm_thesis.signals.base import SignalContext
from rainier.llm_thesis.signals.fear_greed import FearGreedSignal

SCAN = date(2026, 10, 16)


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    FearGreedIndex.__table__.create(engine)

    @contextmanager
    def _session():
        with Session(engine) as s:
            yield s
            s.commit()

    with patch("rainier.llm_thesis.signals.fear_greed.get_session", _session):
        yield _session


def _add(db, day, score, rating, *, observed=None, components=None):
    with db() as s:
        s.add(FearGreedIndex(
            date=day, observed_at=observed or datetime(2026, 10, 16, tzinfo=timezone.utc),
            score=score, rating=rating, raw={"components": components or {}},
            source_version="backfill",
        ))


def _ctx():
    cand = StockCandidate(symbol="CRWV", rank=0, rank_change=0, long_short="Not in QU100",
                          capital_flow_direction="N", sector="Unknown", signal_strength=0.0)
    return SignalContext(symbol="CRWV", scan_date=SCAN, session_name="afternoon",
                         candidate=cand, params={"lookback_days": 45})


def test_reads_prior_close_with_week_and_month_change(db):
    for back in range(1, 40):
        _add(db, SCAN - timedelta(days=back), 50.0 - back, "fear")
    _add(db, SCAN, 5.0, "extreme fear")  # same-day row must be ignored (lookahead)
    _add(db, SCAN - timedelta(days=1), 30.3, "fear",
         observed=datetime(2026, 10, 17, tzinfo=timezone.utc),
         components={"market_volatility_vix": {"rating": "extreme fear"},
                     "junk_bond_demand": {"rating": "fear"}})

    value = FearGreedSignal().compute(_ctx())

    assert value["as_of"] == "2026-10-15"
    assert value["score"] == 30.3  # newest observation for that date wins
    assert value["week_ago"] == 42.0 and value["month_ago"] == 19.0
    text = FearGreedSignal().render_for_prompt(value)
    assert text.startswith("Market Fear & Greed (CNN, close 2026-10-15): 30.3 fear;")
    assert text.endswith("1w ago 42.0, 1m ago 19.0.")


def test_no_history_before_scan_skips_the_signal(db):
    _add(db, SCAN, 17.9, "extreme fear")
    assert FearGreedSignal().compute(_ctx()) is None


def test_registered_and_enabled_by_default():
    from rainier.core.config import LLMThesisConfig

    assert REGISTRY["fear_greed"] is FearGreedSignal
    assert LLMThesisConfig().signals["fear_greed"].enabled
