"""WS B — sector reads resolve latest snapshot PER ranking_type, not globally.

Under the rebuild-the-day fix a day holds ONE `captured_at` per
`(data_date, ranking_type)`. On a partial-failure scrape one ranking type can
advance (e.g. top100 rebuilt at T2) while the other stays at an earlier
`captured_at` (bottom100 still at T1, or a different `data_date` entirely if it
was never scraped today).

`_analyze_sectors` used a GLOBAL `max(captured_at)` with no `ranking_type`
scope. After the fix that reads only ONE ranking type's rows (the most recent
one) and silently drops the other half-book. The fix resolves the latest
`(data_date, captured_at)` INDEPENDENTLY per ranking type and unions the
batches, so the analysis always sees the current full book.

Logic-lane SQLite harness (rowid-alias `id`; composite PK is a hypertable
artifact). Never touches a provided URL / `public.*`.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from rainier.analysis.sector_analyzer import _analyze_sectors
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
T1 = datetime(2026, 6, 1, 15, 45, tzinfo=timezone.utc)
T2 = datetime(2026, 6, 1, 22, 0, tzinfo=timezone.utc)


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


def _stock(s, symbol, sector):
    if not s.query(Stock).filter_by(symbol=symbol).first():
        s.add(Stock(symbol=symbol, sector=sector))
        s.flush()


def _snap(s, symbol, sector, long_short, rank, ranking_type, data_date, captured_at):
    _stock(s, symbol, sector)
    s.add(
        MoneyFlowSnapshot(
            captured_at=captured_at, capture_session="x", data_date=data_date,
            ranking_type=ranking_type, symbol=symbol, rank=rank,
            sector=sector, long_short=long_short, raw_data={},
        )
    )


def test_partial_fail_reads_full_book_per_ranking_type(session):
    """top100 rebuilt @T2, bottom100 still @T1 (same day). A global max(captured_at)
    would read only top100; per-ranking_type reads BOTH books."""
    # top100 @T2 — Technology bullish
    _snap(session, "AAA", "Technology", "Long in", 1, "top100", DAY1, T2)
    _snap(session, "BBB", "Technology", "Long in", 2, "top100", DAY1, T2)
    # bottom100 @T1 — Energy bearish (earlier captured_at, same day)
    _snap(session, "EEE", "Energy", "Short in", 1, "bottom100", DAY1, T1)
    _snap(session, "FFF", "Energy", "Short in", 2, "bottom100", DAY1, T1)
    session.commit()

    trends = _analyze_sectors(session)
    sectors = {t.sector for t in trends}
    # Both ranking types' books are present — NOT just the latest captured_at half.
    assert "Technology" in sectors
    assert "Energy" in sectors, "bottom100 (@T1) must not be dropped by a global max"
    energy = next(t for t in trends if t.sector == "Energy")
    assert energy.short_in_count == 2


def test_ranking_type_on_different_data_date_not_dropped(session):
    """bottom100 was never scraped DAY2 (stuck on DAY1); top100 advanced to DAY2.
    A globally-resolved latest data_date=DAY2 would drop bottom100 entirely."""
    # top100 latest on DAY2
    _snap(session, "AAA", "Technology", "Long in", 1, "top100", DAY2, T1)
    # bottom100's last real rows are DAY1 only
    _snap(session, "EEE", "Energy", "Short in", 1, "bottom100", DAY1, T1)
    session.commit()

    trends = _analyze_sectors(session)
    sectors = {t.sector for t in trends}
    assert "Technology" in sectors
    assert "Energy" in sectors, "a ranking type stuck on an earlier data_date must survive"


def test_non_qu100_ranking_type_excluded_from_sentiment(session):
    """Codex P1 regression — only the QU100 books (top100 + bottom100) feed sector
    sentiment. A stray auxiliary ranking type (e.g. 'concept') in the table must
    NOT be unioned into the analysis, or it would silently skew the screener."""
    # QU100 books: Technology bullish, Energy bearish.
    _snap(session, "AAA", "Technology", "Long in", 1, "top100", DAY1, T2)
    _snap(session, "EEE", "Energy", "Short in", 1, "bottom100", DAY1, T2)
    # Auxiliary non-QU100 board for a sector that must NOT appear.
    _snap(session, "CCC", "Materials", "Long in", 1, "concept", DAY1, T2)
    session.commit()

    trends = _analyze_sectors(session)
    sectors = {t.sector for t in trends}
    assert "Technology" in sectors
    assert "Energy" in sectors
    assert "Materials" not in sectors, "non-QU100 ranking_type must be excluded"


def test_no_snapshots_returns_empty(session):
    assert _analyze_sectors(session) == []
