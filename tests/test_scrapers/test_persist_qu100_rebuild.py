"""WS A — `_persist_qu100` rebuilds the day's snapshot by RANK slot.

Of the 4 daily QU100 scrapes only the first persisted: a skip-guard keyed on
`(data_date, ranking_type)` (without `captured_at`) no-op'd every later scrape,
so the screener re-posted frozen morning dominance all day. The fix makes every
non-empty later scrape REBUILD that day's `(data_date, ranking_type)` snapshot
keyed on the RANK SLOT (1..100), NOT the symbol:

  * the batch is first CANONICALIZED — out-of-range ranks dropped, a duplicate
    symbol or duplicate rank collapsed to one row (the cohort reader has no dedup
    of its own, so this is the only guard against a glitched API response),
  * ranks the scrape returns are OVERRIDDEN with fresh values,
  * ranks MISSING from the scrape carry forward the prior occupant (last data,
    original `capture_session`, the new `captured_at`),
  * a carried slot whose symbol was freshly scraped at ANOTHER rank is DROPPED —
    fresh wins; that rank is left unfilled (dedup by symbol),
  * the whole day's rows are re-stamped with the latest scrape's `captured_at`
    (ONE generation per `(data_date, ranking_type)`),
  * an empty scrape (0 rows) is a no-op (never wipes a good snapshot).

Because carry-forward is rank-keyed, a FULL scrape overrides every slot, so a
stock that fell out of the top 100 holds no rank and disappears — the cohort
stays <=100 with no fallen-out members. The reader contract is therefore
"<=100, rank-ordered, current as of the latest full scrape", NOT an
unconditional exactly-100 (a legal partial/dedup-gap generation can be <100).

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


def _row(symbol: str, rank: int, long_short: str = "Long in", sector: str = "Technology") -> QU100Row:
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
            .order_by(MoneyFlowSnapshot.rank)
            .all()
        )
        # detach so attribute access survives the closed session
        for r in rows:
            s.expunge(r)
        return rows


def _naive(dt: datetime) -> datetime:
    """SQLite stores datetimes naive; compare on UTC wall-clock without tzinfo."""
    return dt.replace(tzinfo=None)


# --- Acceptance 1: flip override (full) ------------------------------------


def test_override_full_flip(sqlite_session_factory):
    """A→B full scrape where LLY flips Long-in→No-dominance keeps ONE LLY row."""
    scraper = _scraper()
    scraper._persist_qu100(
        [_row("LLY", 1, "Long in"), _row("NVDA", 2, "Long in")],
        "top100", "morning", T1, DAY,
    )
    scraper._persist_qu100(
        [_row("LLY", 1, "No dominance"), _row("NVDA", 2, "Long in")],
        "top100", "midday", T2, DAY,
    )

    rows = _snapshots(sqlite_session_factory)
    by_symbol = {r.symbol: r for r in rows}
    assert set(by_symbol) == {"LLY", "NVDA"}
    assert sum(1 for r in rows if r.symbol == "LLY") == 1
    assert by_symbol["LLY"].long_short == "No dominance"
    assert by_symbol["LLY"].rank == 1
    # whole day re-stamped to the latest scrape's captured_at
    assert all(_naive(r.captured_at) == _naive(T2) for r in rows)
    assert by_symbol["LLY"].capture_session == "midday"


# --- Acceptance 2a: dropout on a FULL scrape + cohort end-to-end ------------


def test_dropout_on_full_scrape_cohort(sqlite_session_factory):
    """A fallen-out stock holds no rank on a full scrape and disappears.

    Load-bearing: `get_current_qu100_cohort` has NO dedup/LIMIT of its own — the
    rank-keyed rebuild is the sole guarantor of <=100 / no fallen-out members.
    """
    from rainier.paper.ingest import get_current_qu100_cohort

    scraper = _scraper()
    # A: ranks 1..3, Z occupies rank 3.
    scraper._persist_qu100(
        [_row("P", 1), _row("Q", 2), _row("Z", 3)],
        "top100", "morning", T1, DAY,
    )
    # B full: rank 3 now W; Z fell out and holds no rank.
    scraper._persist_qu100(
        [_row("P", 1), _row("Q", 2), _row("W", 3)],
        "top100", "midday", T2, DAY,
    )

    rows = _snapshots(sqlite_session_factory)
    symbols = {r.symbol for r in rows}
    assert symbols == {"P", "Q", "W"}, "Z must NOT linger after a full scrape"
    assert "Z" not in symbols
    # one slot per rank, no fallen-out member; cohort = exactly the rebuilt set.
    assert len(rows) == 3
    assert all(_naive(r.captured_at) == _naive(T2) for r in rows)

    cohort = get_current_qu100_cohort(DAY)
    assert [c["symbol"] for c in cohort] == ["P", "Q", "W"]
    assert len(cohort) == 3
    assert "Z" not in {c["symbol"] for c in cohort}


# --- Acceptance 2b: carry-forward across a partial gap ----------------------


def test_carry_forward_partial_gap(sqlite_session_factory):
    """B omits rank 2 (glitch) and changes rank 3: slot 2 carries A's stock
    (old data, captured_at=T2, original capture_session); slots 1,3 override."""
    scraper = _scraper()
    scraper._persist_qu100(
        [_row("X", 1, "Long in"), _row("Y", 2, "Short in"), _row("V", 3, "No dominance")],
        "top100", "morning", T1, DAY,
    )
    # B: rank 1 overridden, rank 2 MISSING, rank 3 changed.
    scraper._persist_qu100(
        [_row("X", 1, "No dominance"), _row("V", 3, "Long in")],
        "top100", "midday", T2, DAY,
    )

    rows = _snapshots(sqlite_session_factory)
    by_rank = {r.rank: r for r in rows}
    assert set(by_rank) == {1, 2, 3}

    # slot 1 overridden with fresh data + this session
    assert by_rank[1].symbol == "X"
    assert by_rank[1].long_short == "No dominance"
    assert by_rank[1].capture_session == "midday"

    # slot 2 carried: OLD data preserved, re-stamped captured_at=T2, KEEPS its
    # original capture_session (truthful per row).
    assert by_rank[2].symbol == "Y"
    assert by_rank[2].long_short == "Short in"
    assert by_rank[2].capture_session == "morning"

    # slot 3 overridden
    assert by_rank[3].symbol == "V"
    assert by_rank[3].long_short == "Long in"

    # one generation: the whole day shares the latest captured_at.
    assert all(_naive(r.captured_at) == _naive(T2) for r in rows)


# --- Acceptance 2c: symbol-moved dedup pins the "<100 acceptable" invariant --


def test_symbol_moved_dedup_leaves_rank_unfilled(sqlite_session_factory):
    """M moves rank 50→51 while rank 50 is omitted: M appears once (fresh at 51),
    slot 50 is NOT a carried duplicate M, and the generation holds 99 rows."""
    scraper = _scraper()
    # A: full board ranks 1..100, M at rank 50.
    a = [_row("M" if r == 50 else f"S{r}", r) for r in range(1, 101)]
    scraper._persist_qu100(a, "top100", "morning", T1, DAY)

    # B: ranks 1..49 + 51..100 (rank 50 omitted); M now at rank 51.
    b = []
    for r in range(1, 101):
        if r == 50:
            continue  # omitted gap
        if r == 51:
            b.append(_row("M", 51))  # M moved here (displaces original S51)
        else:
            b.append(_row(f"S{r}", r))
    scraper._persist_qu100(b, "top100", "midday", T2, DAY)

    rows = _snapshots(sqlite_session_factory)
    by_rank = {r.rank: r for r in rows}
    # 99 slots: rank 50 left unfilled (carried M dropped because fresh M wins @51).
    assert len(rows) == 99
    assert 50 not in by_rank
    # M appears exactly once, at rank 51.
    m_rows = [r for r in rows if r.symbol == "M"]
    assert len(m_rows) == 1
    assert m_rows[0].rank == 51
    assert all(_naive(r.captured_at) == _naive(T2) for r in rows)


# --- Acceptance 2d: first-ever scrape (no carry path) -----------------------


def test_first_ever_scrape_plain_insert(sqlite_session_factory):
    """No prior (data_date, ranking_type) rows -> plain insert, all fresh."""
    scraper = _scraper()
    n = scraper._persist_qu100(
        [_row("A", 1), _row("B", 2), _row("C", 3)],
        "top100", "morning", T1, DAY,
    )
    assert n == 3
    rows = _snapshots(sqlite_session_factory)
    assert {r.symbol for r in rows} == {"A", "B", "C"}
    assert all(r.capture_session == "morning" for r in rows)
    assert all(_naive(r.captured_at) == _naive(T1) for r in rows)


# --- Acceptance 2e: degenerate prior (<100 ranks) carries without error -----


def test_degenerate_prior_lt_100_carries(sqlite_session_factory):
    """A prior generation with <100 ranks; a later partial scrape carries the
    ranks that existed, with no error reading the short base."""
    scraper = _scraper()
    scraper._persist_qu100(
        [_row("P", 1, "Long in"), _row("Q", 2, "Short in")],
        "top100", "morning", T1, DAY,
    )
    # later partial scrape: rank 1 changed, rank 2 omitted.
    scraper._persist_qu100(
        [_row("P", 1, "No dominance")],
        "top100", "midday", T2, DAY,
    )
    rows = _snapshots(sqlite_session_factory)
    by_rank = {r.rank: r for r in rows}
    assert set(by_rank) == {1, 2}
    assert by_rank[1].long_short == "No dominance"
    assert by_rank[2].symbol == "Q"  # carried
    assert by_rank[2].capture_session == "morning"
    assert all(_naive(r.captured_at) == _naive(T2) for r in rows)


# --- Acceptance 2f: glitched batch is canonicalized -------------------------


def test_glitched_batch_canonicalized(sqlite_session_factory):
    """A batch with a duplicate symbol, a duplicate rank, and out-of-range ranks
    is canonicalized so each symbol and each rank appears once and out-of-range
    rows are dropped — the cohort cannot double-count."""
    scraper = _scraper()
    glitched = [
        _row("DUP", 1),       # kept
        _row("DUP", 2),       # dropped: symbol DUP already seen
        _row("A3", 3),        # kept
        _row("B3", 3),        # dropped: rank 3 already seen
        _row("OOR0", 0),      # dropped: rank out of range (<1)
        _row("OOR101", 101),  # dropped: rank out of range (>100)
        _row("GOOD", 4),      # kept
    ]
    n = scraper._persist_qu100(glitched, "top100", "morning", T1, DAY)

    rows = _snapshots(sqlite_session_factory)
    ranks = sorted(r.rank for r in rows)
    symbols = sorted(r.symbol for r in rows)
    assert ranks == [1, 3, 4]  # no rank twice, no out-of-range rank
    assert symbols == ["A3", "DUP", "GOOD"]  # DUP once, B3 dropped
    assert len(rows) == 3
    assert n == 3


# --- Acceptance 3: empty scrape is a no-op ----------------------------------


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


def test_all_invalid_batch_is_noop(sqlite_session_factory):
    """A batch whose every row is out-of-range canonicalizes to empty -> no-op
    (never wipes a good snapshot with a fully-glitched pull)."""
    scraper = _scraper()
    scraper._persist_qu100([_row("LLY", 1, "Long in")], "top100", "morning", T1, DAY)
    returned = scraper._persist_qu100(
        [_row("BAD", 0), _row("WORSE", 200)], "top100", "close", T3, DAY
    )
    assert returned == 0
    rows = _snapshots(sqlite_session_factory)
    assert {r.symbol for r in rows} == {"LLY"}
    assert all(_naive(r.captured_at) == _naive(T1) for r in rows)


# --- Idempotence / independence --------------------------------------------


def test_one_row_per_rank_after_rebuild(sqlite_session_factory):
    """Three consecutive identical scrapes never accumulate duplicate rows."""
    scraper = _scraper()
    for ts, sess in ((T1, "morning"), (T2, "midday"), (T3, "close")):
        scraper._persist_qu100(
            [_row("LLY", 1, "Long in"), _row("NVDA", 2, "Long in")],
            "top100", sess, ts, DAY,
        )
    rows = _snapshots(sqlite_session_factory)
    assert len(rows) == 2  # one row per rank, not 6
    assert all(_naive(r.captured_at) == _naive(T3) for r in rows)


def test_ranking_types_rebuilt_independently(sqlite_session_factory):
    """top100 and bottom100 are merged independently; rebuilding one leaves the
    other's snapshot at its own captured_at."""
    scraper = _scraper()
    scraper._persist_qu100([_row("AAA", 1, "Long in")], "top100", "morning", T1, DAY)
    scraper._persist_qu100([_row("ZZZ", 1, "Short in")], "bottom100", "morning", T1, DAY)
    # only top100 advances at T2 (AAA stays at rank 1, value flips)
    scraper._persist_qu100([_row("AAA", 1, "No dominance")], "top100", "midday", T2, DAY)

    top = _snapshots(sqlite_session_factory, "top100")
    bottom = _snapshots(sqlite_session_factory, "bottom100")
    assert [_naive(r.captured_at) for r in top] == [_naive(T2)]
    assert top[0].long_short == "No dominance"
    assert [_naive(r.captured_at) for r in bottom] == [_naive(T1)]  # untouched
    assert bottom[0].symbol == "ZZZ"
