"""WS A — `_persist_qu100` rebuilds the day's snapshot (override + carry-forward).

Of the 4 daily QU100 scrapes only the first persisted: a skip-guard keyed on
`(data_date, ranking_type)` (without `captured_at`) no-op'd every later scrape,
so the screener re-posted frozen morning dominance all day. The fix makes every
non-empty later scrape REBUILD that day's `(data_date, ranking_type)` snapshot:

  * symbols the scrape returns are OVERRIDDEN with fresh values,
  * symbols missing from the scrape are CARRIED FORWARD (last data, original
    `capture_session`),
  * the whole day's rows are re-stamped with the latest scrape's `captured_at`
    (ONE generation per `(data_date, ranking_type)`),
  * an empty scrape (0 rows) is a no-op (never wipes a good snapshot).

Logic-lane harness: SQLite bound to the legacy `core.database` singleton. The
real prod table is a TimescaleDB hypertable with a composite PK `(id,
captured_at)`; here `id` is a single-column autoincrement PK (the composite is a
hypertable artifact, irrelevant to the rebuild logic) so the ORM autoincrement
that `_persist_qu100` relies on works on SQLite.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from rainier.core.models import MoneyFlowSnapshot
from rainier.scrapers.qu.parsers import QU100Row
from rainier.scrapers.qu.scraper import QUScraper

# Hand-DDL so SQLite gets a rowid-alias autoincrement `id` (composite-PK
# autoincrement is unsupported on SQLite and is a hypertable-only artifact).
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

DAY = date(2026, 6, 1)
T1 = datetime(2026, 6, 1, 15, 45, tzinfo=timezone.utc)  # morning
T2 = datetime(2026, 6, 1, 17, 45, tzinfo=timezone.utc)  # midday
T3 = datetime(2026, 6, 1, 22, 0, tzinfo=timezone.utc)  # close


@pytest.fixture
def sqlite_session_factory():
    """SQLite engine + factory bound onto the legacy `core.database` singleton.

    Never touches a provided URL / `public.*` (shared-DB-clobber learning): a
    fresh in-memory DB per test.
    """
    from rainier.core import config, database

    engine = create_engine("sqlite://", future=True)
    with engine.begin() as conn:
        for stmt in _DDL.strip().split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    prev_engine = database._engine
    prev_factory = database._session_factory
    prev_settings = config._settings
    database._engine = engine
    database._session_factory = factory
    try:
        yield factory
    finally:
        database._engine = prev_engine
        database._session_factory = prev_factory
        config._settings = prev_settings
        engine.dispose()


def _row(symbol: str, rank: int, long_short: str, sector: str = "Technology") -> QU100Row:
    return QU100Row(
        rank=rank,
        symbol=symbol,
        daily_change=rank,
        sector=sector,
        industry="Software",
        long_short=long_short,
        raw={"ticker": symbol, "rank": rank, "long_short": long_short},
    )


def _scraper() -> QUScraper:
    """A QUScraper with just the logging attr `_persist_qu100` touches."""
    import structlog

    scraper = QUScraper.__new__(QUScraper)
    scraper.log = structlog.get_logger().bind(scraper="qu")
    return scraper


def _snapshots(factory, ranking_type: str = "top100") -> list[MoneyFlowSnapshot]:
    with factory() as s:
        rows = (
            s.query(MoneyFlowSnapshot)
            .filter(MoneyFlowSnapshot.ranking_type == ranking_type)
            .order_by(MoneyFlowSnapshot.symbol)
            .all()
        )
        # detach so attribute access survives the closed session
        for r in rows:
            s.expunge(r)
        return rows


def _naive(dt: datetime) -> datetime:
    """SQLite stores datetimes naive; compare on UTC wall-clock without tzinfo."""
    return dt.replace(tzinfo=None)


def test_override_full_flip(sqlite_session_factory):
    """A→B full scrape where LLY flips Long-in→No-dominance keeps ONE LLY row."""
    scraper = _scraper()
    scraper._persist_qu100(
        [_row("LLY", 1, "Long in"), _row("NVDA", 2, "Long in")],
        "top100", "morning", T1, DAY,
    )
    scraper._persist_qu100(
        [_row("LLY", 5, "No dominance"), _row("NVDA", 2, "Long in")],
        "top100", "midday", T2, DAY,
    )

    rows = _snapshots(sqlite_session_factory)
    by_symbol = {r.symbol: r for r in rows}
    assert set(by_symbol) == {"LLY", "NVDA"}
    # exactly one LLY row, holding the SECOND value
    assert sum(1 for r in rows if r.symbol == "LLY") == 1
    assert by_symbol["LLY"].long_short == "No dominance"
    assert by_symbol["LLY"].rank == 5
    # whole day re-stamped to the latest scrape's captured_at
    assert all(_naive(r.captured_at) == _naive(T2) for r in rows)
    assert by_symbol["LLY"].capture_session == "midday"


def test_carry_forward_partial(sqlite_session_factory):
    """A{X,Y}@T1 → B{X}@T2: X fresh@T2; Y carried (old data, captured_at=T2,
    capture_session still Y's original); neither dropped."""
    scraper = _scraper()
    scraper._persist_qu100(
        [_row("X", 1, "Long in"), _row("Y", 2, "Short in")],
        "top100", "morning", T1, DAY,
    )
    scraper._persist_qu100(
        [_row("X", 3, "No dominance")],
        "top100", "midday", T2, DAY,
    )

    rows = _snapshots(sqlite_session_factory)
    by_symbol = {r.symbol: r for r in rows}
    assert set(by_symbol) == {"X", "Y"}, "carry-forward must not drop Y"

    # X overridden with fresh data + this session
    assert by_symbol["X"].long_short == "No dominance"
    assert by_symbol["X"].rank == 3
    assert by_symbol["X"].capture_session == "midday"

    # Y carried: OLD data preserved, but re-stamped captured_at=T2 and KEEPS its
    # original capture_session (truthful per row).
    assert by_symbol["Y"].long_short == "Short in"
    assert by_symbol["Y"].rank == 2
    assert by_symbol["Y"].capture_session == "morning"

    # one generation: the whole day shares the latest captured_at.
    assert all(_naive(r.captured_at) == _naive(T2) for r in rows)


def test_empty_scrape_is_noop(sqlite_session_factory):
    """A(2 rows) → C(0 rows): C is a no-op; A snapshot intact."""
    scraper = _scraper()
    scraper._persist_qu100(
        [_row("LLY", 1, "Long in"), _row("NVDA", 2, "Long in")],
        "top100", "morning", T1, DAY,
    )
    returned = scraper._persist_qu100([], "top100", "close", T3, DAY)

    assert returned == 0
    rows = _snapshots(sqlite_session_factory)
    by_symbol = {r.symbol: r for r in rows}
    assert set(by_symbol) == {"LLY", "NVDA"}
    # snapshot UNTOUCHED — still morning's captured_at, never re-stamped to T3.
    assert all(_naive(r.captured_at) == _naive(T1) for r in rows)
    assert by_symbol["LLY"].capture_session == "morning"


def test_one_row_per_symbol_after_rebuild(sqlite_session_factory):
    """Three consecutive scrapes never accumulate duplicate symbol rows."""
    scraper = _scraper()
    for ts, sess in ((T1, "morning"), (T2, "midday"), (T3, "close")):
        scraper._persist_qu100(
            [_row("LLY", 1, "Long in"), _row("NVDA", 2, "Long in")],
            "top100", sess, ts, DAY,
        )
    rows = _snapshots(sqlite_session_factory)
    assert len(rows) == 2  # one row per symbol, not 6
    assert all(_naive(r.captured_at) == _naive(T3) for r in rows)


def test_ranking_types_rebuilt_independently(sqlite_session_factory):
    """top100 and bottom100 are merged independently; rebuilding one leaves the
    other's snapshot at its own captured_at."""
    scraper = _scraper()
    scraper._persist_qu100([_row("AAA", 1, "Long in")], "top100", "morning", T1, DAY)
    scraper._persist_qu100([_row("ZZZ", 1, "Short in")], "bottom100", "morning", T1, DAY)
    # only top100 advances at T2
    scraper._persist_qu100([_row("AAA", 2, "No dominance")], "top100", "midday", T2, DAY)

    top = _snapshots(sqlite_session_factory, "top100")
    bottom = _snapshots(sqlite_session_factory, "bottom100")
    assert [_naive(r.captured_at) for r in top] == [_naive(T2)]
    assert top[0].long_short == "No dominance"
    assert [_naive(r.captured_at) for r in bottom] == [_naive(T1)]  # untouched
    assert bottom[0].symbol == "ZZZ"
