"""As-of money-flow selectors for A/B replay (task ab-asof-selectors-8f6f).

The replay must answer "which stocks would the screener have picked on day t?"
without leaking future data. This covers:

* `_screen_money_flow_as_of(session, t, *, normalize_casing=False)` — mirrors
  the #148 per-(data_date, ranking_type) generation convention (max data_date
  <= t for top100, then max captured_at within that generation) plus the live
  per-symbol dedup. NO 24h staleness guard (replay reads history).
* `capital_flow_as_of(session, symbol, t, limit=5, *, normalize_casing=False)`
  — closes the look-ahead in the capital-flow enrichment (live grabs the
  latest 5 rows with no date filter; as-of filters flow_date <= t).
* `normalize_casing` — the live/replay split for the `long_short` casing trap
  ('Long in' 1,011 days vs 'Long In' 406 days). `True` admits both spellings
  and canonicalizes so scorer + sector leg agree; `False` (default, live)
  keeps exact-match byte-identical.
* `analyze_sectors_at(..., normalize_casing=...)` — the sector leg queries
  MoneyFlowSnapshot rows itself, so it needs the flag directly.

Raw-DDL SQLite harness mirroring `money_flow_snapshots` / `stock_capital_flow`
(the ORM models can't create on SQLite — composite PK + JSONB); the ORM-class
queries run unmodified against these tables. Timestamps are stored tz-naive
UTC (SQLite TIMESTAMP-affinity artifact, see
tests/test_stock_screener_latest_snapshot.py).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from rainier.analysis.sector_analyzer import analyze_sectors_at, get_sector_boost
from rainier.analysis.stock_screener import (
    _screen_money_flow,
    _screen_money_flow_as_of,
    capital_flow_as_of,
)

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

# Fixed historical day t and generation timestamps — all far outside the 24h
# staleness window on purpose: the as-of path must not care.
T = date(2026, 6, 4)
TS_EARLY = datetime(2026, 6, 4, 15, 45)
TS_LATE = datetime(2026, 6, 4, 22, 0)


def _ts_param(ts: datetime) -> str:
    """Bind-format a captured_at for raw INSERT (SQLite harness artifact).

    sqlite3's default datetime adapter omits a zero microsecond field
    ('... 22:00:00') while SQLAlchemy's DateTime equality bind always emits
    it ('... 22:00:00.000000') — a raw-inserted whole-second timestamp would
    never equality-match. Store the SQLAlchemy spelling.
    """
    return ts.isoformat(sep=" ", timespec="microseconds")


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
    daily_change: int = 0,
):
    session.execute(
        text(
            "INSERT INTO money_flow_snapshots "
            "(captured_at, capture_session, data_date, view_type, ranking_type, "
            " symbol, rank, daily_change, sector, industry, long_short) "
            "VALUES (:captured_at, 'afternoon', :data_date, 'daily', :ranking_type, "
            " :symbol, :rank, :daily_change, :sector, 'X', :long_short)"
        ),
        {
            "captured_at": _ts_param(captured_at),
            "data_date": data_date,
            "ranking_type": ranking_type,
            "symbol": symbol,
            "rank": rank,
            "daily_change": daily_change,
            "sector": sector,
            "long_short": long_short,
        },
    )
    session.commit()


def _insert_capital_flow(
    session: Session,
    *,
    symbol: str,
    flow_date: str,
    direction: str = "+",
    long_short: str = "Long in",
    rank: int = 1,
    period_type: str = "daily",
):
    session.execute(
        text(
            "INSERT INTO stock_capital_flow "
            "(symbol, captured_at, flow_date, period_type, capital_flow_direction, "
            " long_short, rank, rank_total) "
            "VALUES (:symbol, :captured_at, :flow_date, :period_type, :direction, "
            " :long_short, :rank, 100)"
        ),
        {
            "symbol": symbol,
            "captured_at": _ts_param(datetime(2026, 6, 4, 22, 0)),
            "flow_date": flow_date,
            "period_type": period_type,
            "direction": direction,
            "long_short": long_short,
            "rank": rank,
        },
    )
    session.commit()


def _recent_ts() -> datetime:
    """A captured_at within the live 24h staleness window (tz-naive UTC)."""
    return (datetime.now(timezone.utc) - timedelta(hours=1)).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# As-of resolution (#148 generation convention)
# ---------------------------------------------------------------------------


def test_as_of_trading_day_selects_that_day(db_session):
    """t is a trading day with data -> exactly that day's generation."""
    _insert_snapshot(db_session, symbol="NVDA", rank=1, data_date="2026-06-04", captured_at=TS_LATE)
    _insert_snapshot(db_session, symbol="TSLA", rank=2, data_date="2026-06-04", captured_at=TS_LATE)
    # Future day must never leak into an as-of t read.
    _insert_snapshot(
        db_session, symbol="AMD", rank=1, data_date="2026-06-05",
        captured_at=datetime(2026, 6, 5, 22, 0),
    )

    signals = _screen_money_flow_as_of(db_session, T)
    assert sorted(s.symbol for s in signals) == ["NVDA", "TSLA"]


def test_as_of_non_trading_day_falls_back_to_nearest(db_session):
    """t is a non-trading day (weekend) -> nearest data_date <= t."""
    _insert_snapshot(db_session, symbol="NVDA", rank=1, data_date="2026-06-04", captured_at=TS_LATE)
    _insert_snapshot(db_session, symbol="TSLA", rank=2, data_date="2026-06-04", captured_at=TS_LATE)

    signals = _screen_money_flow_as_of(db_session, date(2026, 6, 7))  # Sunday
    assert sorted(s.symbol for s in signals) == ["NVDA", "TSLA"]


def test_as_of_picks_latest_generation_within_day(db_session):
    """Two captured_at batches for one (data_date, top100) -> latest generation only."""
    _insert_snapshot(db_session, symbol="NVDA", rank=9, data_date="2026-06-04", captured_at=TS_EARLY)
    _insert_snapshot(db_session, symbol="NVDA", rank=1, data_date="2026-06-04", captured_at=TS_LATE)
    _insert_snapshot(db_session, symbol="AMD", rank=2, data_date="2026-06-04", captured_at=TS_LATE)

    signals = _screen_money_flow_as_of(db_session, T)
    by_symbol = {s.symbol: s for s in signals}
    assert set(by_symbol) == {"NVDA", "AMD"}
    assert by_symbol["NVDA"].rank == 1  # late generation wins


def test_as_of_generation_scoped_to_top100(db_session):
    """A newer data_date holding only bottom100 must not blank the as-of read
    (same scope rule as the live selector)."""
    _insert_snapshot(db_session, symbol="NVDA", rank=1, data_date="2026-06-03", captured_at=TS_EARLY)
    _insert_snapshot(
        db_session, symbol="META", rank=1, data_date="2026-06-04",
        captured_at=TS_LATE, ranking_type="bottom100",
    )

    signals = _screen_money_flow_as_of(db_session, T)
    assert [s.symbol for s in signals] == ["NVDA"]


def test_as_of_dedup_per_symbol_mirrors_live(db_session):
    """Duplicate symbol rows within one generation -> one signal, best rank kept."""
    _insert_snapshot(db_session, symbol="NVDA", rank=1, data_date="2026-06-04", captured_at=TS_LATE)
    _insert_snapshot(db_session, symbol="NVDA", rank=5, data_date="2026-06-04", captured_at=TS_LATE)

    signals = _screen_money_flow_as_of(db_session, T)
    assert [s.symbol for s in signals] == ["NVDA"]
    assert signals[0].rank == 1


def test_as_of_has_no_staleness_guard(db_session):
    """Replay reads history: a generation weeks/months old must still return.
    (TS_LATE is 2026-06-04 — far beyond the live 24h window.)"""
    _insert_snapshot(db_session, symbol="NVDA", rank=1, data_date="2026-06-04", captured_at=TS_LATE)

    signals = _screen_money_flow_as_of(db_session, T)
    assert [s.symbol for s in signals] == ["NVDA"]


def test_as_of_empty_db_returns_empty(db_session):
    assert _screen_money_flow_as_of(db_session, T) == []


# ---------------------------------------------------------------------------
# Casing normalization (the live/replay split)
# ---------------------------------------------------------------------------


def test_normalize_true_admits_and_scores_title_cased_day(db_session):
    """'Long In' (406 historical days) admitted AND scored under the flag.

    Canonicalized long_short means _compute_money_flow_score's exact
    'Long in' match awards the 0.5 base — an admitted-but-mis-scored row
    (0.15 only) is exactly the silent-corruption trap this flag closes.
    """
    _insert_snapshot(
        db_session, symbol="NVDA", rank=1, data_date="2026-06-04",
        captured_at=TS_LATE, long_short="Long In",
    )

    signals = _screen_money_flow_as_of(db_session, T, normalize_casing=True)
    assert [s.symbol for s in signals] == ["NVDA"]
    assert signals[0].long_short == "Long in"  # canonical casing out
    # 0.5 (Long in base) + 0.15 (rank <= 30); no capital flow rows.
    assert signals[0].signal_strength == pytest.approx(0.65)


def test_normalize_false_keeps_exact_match(db_session):
    """Default (live) path: 'Long In' rows are NOT admitted — regression on
    byte-identical live behavior."""
    _insert_snapshot(
        db_session, symbol="NVDA", rank=1, data_date="2026-06-04",
        captured_at=TS_LATE, long_short="Long In",
    )

    assert _screen_money_flow_as_of(db_session, T) == []
    assert _screen_money_flow_as_of(db_session, T, normalize_casing=False) == []


def test_sector_leg_normalize_true_counts_title_cased(db_session):
    """analyze_sectors_at(normalize_casing=True): a 'Long In' day is counted
    long, turns the sector bullish, and yields the 0.1 boost."""
    _insert_snapshot(
        db_session, symbol="NVDA", rank=1, data_date="2026-06-04",
        captured_at=TS_LATE, long_short="Long In", sector="Technology",
    )

    trends = analyze_sectors_at(T, session=db_session, normalize_casing=True)
    tech = next(t for t in trends if t.sector == "Technology")
    assert tech.long_in_count == 1
    assert tech.short_in_count == 0
    assert tech.trend_direction == "bullish"
    assert tech.top_stocks == ["NVDA"]
    assert get_sector_boost("Technology", trends) == pytest.approx(0.1)


def test_sector_leg_normalize_true_counts_title_cased_short(db_session):
    """'Short In' normalizes into the short count too."""
    _insert_snapshot(
        db_session, symbol="XOM", rank=1, data_date="2026-06-04",
        captured_at=TS_LATE, long_short="Short In", sector="Energy",
        ranking_type="bottom100",
    )

    trends = analyze_sectors_at(T, session=db_session, normalize_casing=True)
    energy = next(t for t in trends if t.sector == "Energy")
    assert energy.short_in_count == 1
    assert energy.trend_direction == "bearish"


def test_recorded_decision_sector_momentum_live_exposure(db_session):
    """Recorded decision (design §9.5, accepted): the live default path of
    analyze_sectors_at — as consumed by llm_thesis sector_momentum over PRIOR
    data_dates — keeps exact-match casing. Historical 'Long In' days stay
    uncounted live; that exposure is resolved later through the substrate's
    designated first experiment, NOT by this task."""
    _insert_snapshot(
        db_session, symbol="NVDA", rank=1, data_date="2026-06-04",
        captured_at=TS_LATE, long_short="Long In", sector="Technology",
    )

    trends = analyze_sectors_at(T, session=db_session)  # default: no normalization
    tech = next(t for t in trends if t.sector == "Technology")
    assert tech.long_in_count == 0
    assert tech.trend_direction == "neutral"


# ---------------------------------------------------------------------------
# As-of capital flow (look-ahead close)
# ---------------------------------------------------------------------------


def test_capital_flow_as_of_excludes_future_rows(db_session):
    """Rows with flow_date > t never returned; results newest-first."""
    for d in ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"]:
        _insert_capital_flow(db_session, symbol="NVDA", flow_date=d)

    rows = capital_flow_as_of(db_session, "NVDA", date(2026, 6, 3))
    assert [r.flow_date for r in rows] == [
        date(2026, 6, 3), date(2026, 6, 2), date(2026, 6, 1),
    ]


def test_capital_flow_as_of_limit_and_period_type(db_session):
    """limit caps to the newest N on-or-before t; non-daily rows excluded."""
    for i in range(1, 8):
        _insert_capital_flow(db_session, symbol="NVDA", flow_date=f"2026-06-0{i}")
    _insert_capital_flow(
        db_session, symbol="NVDA", flow_date="2026-06-07", period_type="weekly",
    )

    rows = capital_flow_as_of(db_session, "NVDA", date(2026, 6, 7), limit=5)
    assert len(rows) == 5
    assert rows[0].flow_date == date(2026, 6, 7)
    assert rows[-1].flow_date == date(2026, 6, 3)


def test_capital_flow_as_of_normalize_casing(db_session):
    """normalize_casing=True canonicalizes long_short; False returns raw."""
    _insert_capital_flow(
        db_session, symbol="NVDA", flow_date="2026-06-04", long_short="Long In",
    )

    raw = capital_flow_as_of(db_session, "NVDA", T)
    assert raw[0].long_short == "Long In"

    normalized = capital_flow_as_of(db_session, "NVDA", T, normalize_casing=True)
    assert normalized[0].long_short == "Long in"


# ---------------------------------------------------------------------------
# No look-ahead through the composed selector
# ---------------------------------------------------------------------------


def test_no_lookahead_capital_flow_in_selector(db_session):
    """Future capital-flow rows must not affect direction / streak / score.

    A '+' row dated after t would flip direction to '+' (+0.2) and extend the
    streak if the enrichment leg leaked — the exact live bug (no date filter).
    """
    _insert_snapshot(db_session, symbol="NVDA", rank=1, data_date="2026-06-04", captured_at=TS_LATE)
    _insert_capital_flow(db_session, symbol="NVDA", flow_date="2026-06-05", direction="+")
    _insert_capital_flow(db_session, symbol="NVDA", flow_date="2026-06-04", direction="-")
    _insert_capital_flow(db_session, symbol="NVDA", flow_date="2026-06-03", direction="+")

    signals = _screen_money_flow_as_of(db_session, T)
    assert [s.symbol for s in signals] == ["NVDA"]
    sig = signals[0]
    assert sig.capital_flow_direction == "-"  # 2026-06-04 row, not the future '+'
    assert sig.days_in_top100 == 0
    # 0.5 base + 0.15 rank; no '+' direction bonus, no streak bonus.
    assert sig.signal_strength == pytest.approx(0.65)


def test_no_lookahead_money_flow_snapshots(db_session):
    """Snapshot days after t never affect the as-of basket."""
    _insert_snapshot(db_session, symbol="NVDA", rank=1, data_date="2026-06-04", captured_at=TS_LATE)
    _insert_snapshot(
        db_session, symbol="TSLA", rank=1, data_date="2026-06-08",
        captured_at=datetime(2026, 6, 8, 22, 0),
    )

    signals = _screen_money_flow_as_of(db_session, T)
    assert [s.symbol for s in signals] == ["NVDA"]


# ---------------------------------------------------------------------------
# Live parity (byte-identical delegation)
# ---------------------------------------------------------------------------


def test_live_delegates_to_as_of_identically(db_session):
    """_screen_money_flow == _screen_money_flow_as_of(today, normalize=False)
    on a fresh fixture — the extraction changes nothing live."""
    ts = _recent_ts()
    today = datetime.now(timezone.utc).date().isoformat()
    for i, sym in enumerate(["NVDA", "TSLA", "AMD"], start=1):
        _insert_snapshot(
            db_session, symbol=sym, rank=i, data_date=today,
            captured_at=ts, daily_change=1,
        )
    _insert_snapshot(
        db_session, symbol="META", rank=4, data_date=today,
        captured_at=ts, long_short="Short in",
    )
    # 'Long In' rows stay excluded on BOTH live paths (exact match).
    _insert_snapshot(
        db_session, symbol="ORCL", rank=5, data_date=today,
        captured_at=ts, long_short="Long In",
    )
    for d_off in range(4):
        d = (datetime.now(timezone.utc).date() - timedelta(days=d_off)).isoformat()
        _insert_capital_flow(db_session, symbol="NVDA", flow_date=d, direction="+")

    live = _screen_money_flow(db_session)
    as_of = _screen_money_flow_as_of(
        db_session, datetime.now(timezone.utc).date(), normalize_casing=False
    )
    assert live  # non-vacuous
    assert live == as_of
    assert sorted(s.symbol for s in live) == ["AMD", "NVDA", "TSLA"]


def test_live_staleness_guard_retained(db_session):
    """The 24h guard still protects the LIVE path only."""
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=30)).replace(tzinfo=None)
    d = (datetime.now(timezone.utc) - timedelta(hours=30)).date()
    _insert_snapshot(
        db_session, symbol="NVDA", rank=1, data_date=d.isoformat(), captured_at=stale_ts,
    )

    assert _screen_money_flow(db_session) == []
    # Same data via the as-of path (replay) still returns.
    assert [s.symbol for s in _screen_money_flow_as_of(db_session, date.today())] == ["NVDA"]
