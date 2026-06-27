"""WS B — `analyze_sectors_at` keyed on an as-of DATE, per-ranking_type.

`analyze_sectors_at` used to take a single global `captured_at` timestamp
(`sector_momentum` passed `max(captured_at)`). Under the rebuild-the-day fix a
day holds one `captured_at` per `(data_date, ranking_type)`, so a single global
timestamp can only match ONE ranking type → half-book. The new contract takes an
as-of `date` and resolves, independently per ranking type, the latest
`(data_date <= as_of, captured_at)` generation, then unions.

Logic-lane SQLite harness (rowid-alias `id`).
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from rainier.analysis.sector_analyzer import analyze_sectors_at
from rainier.core.models import MoneyFlowSnapshot, Stock

_DDL = """
CREATE TABLE stocks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol VARCHAR(10) NOT NULL UNIQUE,
  name VARCHAR(255), sector VARCHAR(100), industry VARCHAR(200),
  is_active BOOLEAN DEFAULT 1,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE money_flow_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  captured_at TIMESTAMP NOT NULL,
  capture_session VARCHAR(20) NOT NULL,
  data_date DATE NOT NULL,
  view_type VARCHAR(10) NOT NULL DEFAULT 'daily',
  ranking_type VARCHAR(10) NOT NULL,
  symbol VARCHAR(10) NOT NULL REFERENCES stocks(symbol),
  rank INTEGER NOT NULL,
  daily_change INTEGER, sector VARCHAR(100), industry VARCHAR(200),
  long_short VARCHAR(50), raw_data JSON
);
"""

DAY1 = date(2026, 6, 1)
DAY2 = date(2026, 6, 2)
T_EARLY = datetime(2026, 6, 1, 15, 45, tzinfo=timezone.utc)
T_LATE = datetime(2026, 6, 1, 22, 0, tzinfo=timezone.utc)
T_DAY2 = datetime(2026, 6, 2, 22, 0, tzinfo=timezone.utc)


@pytest.fixture
def session():
    engine = create_engine("sqlite://", future=True)
    with engine.begin() as conn:
        for stmt in _DDL.strip().split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    s = factory()
    try:
        yield s
    finally:
        s.close()
        engine.dispose()


def _snap(s, symbol, sector, long_short, rank, ranking_type, data_date, captured_at):
    if not s.query(Stock).filter_by(symbol=symbol).first():
        s.add(Stock(symbol=symbol, sector=sector))
        s.flush()
    s.add(
        MoneyFlowSnapshot(
            captured_at=captured_at, capture_session="x", data_date=data_date,
            ranking_type=ranking_type, symbol=symbol, rank=rank,
            sector=sector, long_short=long_short, raw_data={},
        )
    )


def test_as_of_date_unions_both_ranking_types(session):
    """top100 @T_LATE + bottom100 @T_EARLY on the same day: as-of DAY1 reads BOTH."""
    _snap(session, "AAA", "Technology", "Long in", 1, "top100", DAY1, T_LATE)
    _snap(session, "EEE", "Energy", "Short in", 1, "bottom100", DAY1, T_EARLY)
    session.commit()

    trends = analyze_sectors_at(DAY1, session=session)
    sectors = {t.sector for t in trends}
    assert sectors == {"Technology", "Energy"}


def test_as_of_date_picks_latest_generation_on_or_before(session):
    """Two generations on DAY1 for top100 (T_EARLY then T_LATE): the LATER wins;
    DAY2 rows are excluded by an as-of of DAY1."""
    _snap(session, "AAA", "Technology", "Short in", 1, "top100", DAY1, T_EARLY)
    _snap(session, "AAA", "Technology", "Long in", 1, "top100", DAY1, T_LATE)
    _snap(session, "AAA", "Technology", "Short in", 1, "top100", DAY2, T_DAY2)
    session.commit()

    trends = analyze_sectors_at(DAY1, session=session)
    tech = next(t for t in trends if t.sector == "Technology")
    # latest generation on/before DAY1 is T_LATE (Long in) — DAY2 excluded.
    assert tech.long_in_count == 1
    assert tech.short_in_count == 0


def test_as_of_date_no_data_returns_empty(session):
    assert analyze_sectors_at(date(2020, 1, 1), session=session) == []
