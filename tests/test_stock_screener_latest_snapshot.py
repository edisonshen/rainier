"""Regression: _screen_money_flow must scope the latest snapshot by trading day.

P0 (2026-06-04): the QU100 backfill stamped all 1,373 trading days with one
shared `captured_at`. The old `captured_at == max(captured_at)` selected every
backfilled day at once (45,921 "Long in" rows, duplicate symbols across days),
which then crashed the screened_stocks upsert with a CardinalityViolation.

The fix scopes "latest snapshot" by `max(data_date)`, then the latest
`captured_at` WITHIN that date — isolating one clean trading-day snapshot
regardless of insert-time collisions. Staleness is judged on that date's
captured_at. Signals are deduped by symbol.

These tests build a raw-DDL SQLite table that mirrors the
`money_flow_snapshots` / `stock_capital_flow` columns (the ORM models can't
create on SQLite — composite PK + JSONB). The ORM-class queries inside
`_screen_money_flow` run unmodified against those tables.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from rainier.analysis.stock_screener import _screen_money_flow

_DDL_SNAPSHOTS = """
CREATE TABLE money_flow_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TIMESTAMP NOT NULL,
    capture_session TEXT,
    data_date DATE NOT NULL,
    view_type TEXT,
    ranking_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    rank INTEGER NOT NULL,
    daily_change INTEGER,
    sector TEXT,
    industry TEXT,
    long_short TEXT,
    raw_data TEXT
)
"""

_DDL_CAPITAL_FLOW = """
CREATE TABLE stock_capital_flow (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    captured_at TIMESTAMP NOT NULL,
    flow_date DATE NOT NULL,
    period_type TEXT NOT NULL,
    week_start DATE,
    week_end DATE,
    capital_flow_direction TEXT,
    long_short TEXT,
    rank INTEGER,
    rank_total INTEGER,
    raw_data TEXT
)
"""


@pytest.fixture
def db_session():
    eng = create_engine("sqlite://")
    with eng.begin() as conn:
        conn.execute(text(_DDL_SNAPSHOTS))
        conn.execute(text(_DDL_CAPITAL_FLOW))
    sess = Session(eng)
    try:
        yield sess
    finally:
        sess.close()
        eng.dispose()


def _insert_snapshot(
    session: Session,
    *,
    symbol: str,
    rank: int,
    data_date: str,
    captured_at: datetime,
    ranking_type: str = "top100",
    long_short: str = "Long in",
    sector: str = "Technology",
):
    session.execute(
        text(
            "INSERT INTO money_flow_snapshots "
            "(captured_at, capture_session, data_date, view_type, ranking_type, "
            " symbol, rank, daily_change, sector, industry, long_short) "
            "VALUES (:captured_at, 'afternoon', :data_date, 'daily', :ranking_type, "
            " :symbol, :rank, 0, :sector, 'X', :long_short)"
        ),
        {
            "captured_at": captured_at,
            "data_date": data_date,
            "ranking_type": ranking_type,
            "symbol": symbol,
            "rank": rank,
            "sector": sector,
            "long_short": long_short,
        },
    )
    session.commit()


def _recent_ts() -> datetime:
    """A captured_at well within the 24h staleness window.

    Stored tz-naive UTC: SQLite's TIMESTAMP affinity drops tz offsets on the
    equality-bind path, so a tz-aware value would round-trip to a string that
    never equality-matches the stored row (a SQLite harness artifact — real
    Postgres `timestamptz` matches fine). The screener's staleness check
    re-stamps naive values as UTC, so this faithfully mirrors production.
    """
    return (datetime.now(timezone.utc) - timedelta(hours=1)).replace(tzinfo=None)


def test_backfill_shape_returns_only_latest_date(db_session):
    """Many data_dates sharing ONE captured_at -> only the latest date's rows."""
    shared_ts = _recent_ts()
    # Backfill: 5 trading days, each with its own 2-symbol "Long in" snapshot,
    # ALL stamped with the same captured_at (the backfill collision).
    for i, d in enumerate(
        ["2026-05-30", "2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"]
    ):
        _insert_snapshot(db_session, symbol="NVDA", rank=1, data_date=d, captured_at=shared_ts)
        _insert_snapshot(db_session, symbol="TSLA", rank=2, data_date=d, captured_at=shared_ts)

    signals = _screen_money_flow(db_session)

    # Only the latest data_date (2026-06-04) -> 2 unique symbols, not 10.
    assert len(signals) == 2
    symbols = sorted(s.symbol for s in signals)
    assert symbols == ["NVDA", "TSLA"]


def test_signals_are_unique_by_symbol(db_session):
    """Even if the same symbol appears across days at one captured_at, dedup."""
    shared_ts = _recent_ts()
    for d in ["2026-06-02", "2026-06-03", "2026-06-04"]:
        _insert_snapshot(db_session, symbol="NVDA", rank=1, data_date=d, captured_at=shared_ts)

    signals = _screen_money_flow(db_session)
    assert [s.symbol for s in signals] == ["NVDA"]


def test_latest_date_picks_latest_captured_at(db_session):
    """Latest date has two captured_at (morning scrape + backfill) -> pick latest."""
    latest_date = "2026-06-04"
    early_ts = _recent_ts() - timedelta(hours=2)
    late_ts = _recent_ts()
    # Early snapshot: NVDA only. Late snapshot: NVDA + AMD (the real refresh).
    _insert_snapshot(db_session, symbol="NVDA", rank=9, data_date=latest_date, captured_at=early_ts)
    _insert_snapshot(db_session, symbol="NVDA", rank=1, data_date=latest_date, captured_at=late_ts)
    _insert_snapshot(db_session, symbol="AMD", rank=2, data_date=latest_date, captured_at=late_ts)

    signals = _screen_money_flow(db_session)
    by_symbol = {s.symbol: s for s in signals}
    assert set(by_symbol) == {"NVDA", "AMD"}
    # NVDA must reflect the LATE snapshot (rank 1), not the early one (rank 9).
    assert by_symbol["NVDA"].rank == 1


def test_stale_latest_date_skipped(db_session):
    """Latest data_date's captured_at older than 24h -> skip (empty)."""
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=30)).replace(tzinfo=None)
    _insert_snapshot(db_session, symbol="NVDA", rank=1, data_date="2026-06-04", captured_at=stale_ts)

    signals = _screen_money_flow(db_session)
    assert signals == []


def test_normal_single_day_unchanged(db_session):
    """Normal single-day snapshot -> all its Long-in rows, unique symbols."""
    ts = _recent_ts()
    for i, sym in enumerate(["NVDA", "TSLA", "AMD"], start=1):
        _insert_snapshot(db_session, symbol=sym, rank=i, data_date="2026-06-04", captured_at=ts)
    # A short-in row on the same day must be excluded.
    _insert_snapshot(
        db_session, symbol="META", rank=4, data_date="2026-06-04",
        captured_at=ts, long_short="Short in",
    )

    signals = _screen_money_flow(db_session)
    assert sorted(s.symbol for s in signals) == ["AMD", "NVDA", "TSLA"]


def test_empty_db_returns_empty(db_session):
    assert _screen_money_flow(db_session) == []
