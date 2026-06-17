"""Price-coverage backfill (TASK-PLAN p1-cleanup-price-coverag-18ff).

`db backfill-prices` historically decided what to fetch with a PRESENCE check
(`StockPrice.date >= start`). A separate incremental ingest seeded thin RECENT
slivers for large-caps (e.g. AMZN: 40 rows from 2026-03-03), so they passed the
presence check and were skipped — their 2022-2025 history (most of the sweep
window) was never fetched.

The fix is COVERAGE-based: a symbol is "covered" only if its rows span the
required window at BOTH ends (a bar near the window START and a recent bar near
the window END). Both failure modes (recent-sliver, stale-tail) must re-fetch.
Per-symbol gaps are surfaced loudly (`yf_batch_dropped_symbols`) and a post-run
assertion checks 100% current-cohort coverage over the sweep window.

DB-backed cases run under `requires_postgres` (pg_legacy_session binds the
legacy `core.database` singleton the backfill uses).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest
from sqlalchemy import select, text

from rainier.backtest.qu100_portfolio import (
    COVERAGE_BOUNDARY_TOLERANCE_DAYS,
    COVERAGE_MAX_INTERIOR_GAP_SESSIONS,
    _is_covered,
    _yf_to_long,
    assert_cohort_coverage,
    select_symbols_needing_backfill,
    sweep_window_start,
)
from rainier.core.models import StockPrice
from rainier.paper.calendar import DEFAULT_CALENDAR

# Most cases need a live Postgres (pg_legacy_session binds the legacy
# core.database singleton the backfill uses). The pure `_is_covered` /
# `_yf_to_long` cases below need no DB but inherit the module mark harmlessly
# (the mark only labels for `-m` selection; the DB skip comes from the fixture).
pytestmark = pytest.mark.requires_postgres

WINDOW_START = date(2022, 5, 25)
WINDOW_END = date(2026, 6, 12)


# ---------------------------------------------------------------------------
# Pure coverage-decision logic (_is_covered) — no DB. The span check is the
# crux of the fix; pin every branch independent of Postgres availability.
# ---------------------------------------------------------------------------

TOL = COVERAGE_BOUNDARY_TOLERANCE_DAYS


def _all_sessions(start: date, end: date) -> set[date]:
    """Every trading session in [start, end] — a dense, fully-covered set."""
    return set(DEFAULT_CALENDAR.sessions_between(start, end))


def test_is_covered_full_span_true():
    # Dense bars across the whole window → covered.
    assert _is_covered(_all_sessions(WINDOW_START, WINDOW_END), WINDOW_START, WINDOW_END)


def test_is_covered_recent_sliver_false():
    # Left edge missing (AMZN case): bars only near the end.
    present = _all_sessions(date(2026, 3, 3), WINDOW_END)
    assert not _is_covered(present, WINDOW_START, WINDOW_END)


def test_is_covered_stale_tail_false():
    # Right edge missing: bars only in the early window.
    present = _all_sessions(WINDOW_START, date(2024, 6, 6))
    assert not _is_covered(present, WINDOW_START, WINDOW_END)


def test_is_covered_absent_false():
    assert not _is_covered(set(), WINDOW_START, WINDOW_END)


def test_is_covered_split_history_false():
    # Both edges present but a multi-year interior hole → NOT covered. A min/max
    # span check would wrongly pass this (codex P1). The interior-gap scan fails.
    present = _all_sessions(WINDOW_START, date(2022, 8, 1)) | _all_sessions(
        date(2026, 1, 1), WINDOW_END
    )
    assert not _is_covered(present, WINDOW_START, WINDOW_END)


def test_is_covered_small_interior_gap_true():
    # A short gap (<= max interior tolerance) is fine — holidays/data hiccups.
    present = _all_sessions(WINDOW_START, WINDOW_END)
    # Drop a handful of consecutive sessions in the middle (within tolerance).
    mid = DEFAULT_CALENDAR.sessions_between(date(2024, 1, 8), date(2024, 1, 12))
    present -= set(mid)  # 5 sessions ≤ COVERAGE_MAX_INTERIOR_GAP_SESSIONS (10)
    assert COVERAGE_MAX_INTERIOR_GAP_SESSIONS >= 5
    assert _is_covered(present, WINDOW_START, WINDOW_END)


def test_is_covered_within_tolerance_true():
    # Edges a few days inside the window but within boundary tolerance → covered.
    present = _all_sessions(
        WINDOW_START + timedelta(days=TOL), WINDOW_END - timedelta(days=TOL)
    )
    assert _is_covered(present, WINDOW_START, WINDOW_END)


def test_is_covered_just_outside_tolerance_false():
    # Left edge starts just past tolerance → not left-covered.
    present = _all_sessions(WINDOW_START + timedelta(days=TOL + 5), WINDOW_END)
    assert not _is_covered(present, WINDOW_START, WINDOW_END)


def _instant(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _seed_stock(session, symbol: str) -> None:
    session.execute(
        text("INSERT INTO stocks (symbol) VALUES (:s) ON CONFLICT DO NOTHING"),
        {"s": symbol},
    )


def _seed_prices(session, symbol: str, days: list[date]) -> None:
    _seed_stock(session, symbol)
    for d in days:
        session.add(
            StockPrice(
                symbol=symbol, date=_instant(d),
                open=10.0, high=11.0, low=9.0, close=10.5, volume=1000,
            )
        )
    session.commit()


def _seed_dense(session, symbol: str, start: date, end: date) -> None:
    """Seed a usable bar for EVERY trading session in [start, end] — a fully
    covered history (no interior holes)."""
    _seed_prices(session, symbol, DEFAULT_CALENDAR.sessions_between(start, end))


def _seed_null_ohlc(session, symbol: str, days: list[date]) -> None:
    """Seed placeholder rows with NULL OHLC (a yfinance partial). These must be
    treated as GAPS, not coverage (codex P1)."""
    _seed_stock(session, symbol)
    for d in days:
        session.add(
            StockPrice(
                symbol=symbol, date=_instant(d),
                open=None, high=None, low=None, close=None, volume=None,
            )
        )
    session.commit()


# ---------------------------------------------------------------------------
# sweep_window_start — derived from the rankings, NOT hard-coded / --years.
# ---------------------------------------------------------------------------


def test_sweep_window_start_derives_from_rankings(pg_legacy_session):
    _seed_stock(pg_legacy_session, "AAA")
    for dd in [date(2022, 5, 25), date(2023, 1, 3), date(2026, 6, 12)]:
        pg_legacy_session.execute(
            text(
                "INSERT INTO money_flow_snapshots "
                "(captured_at, capture_session, data_date, ranking_type, symbol, rank) "
                "VALUES (:cap, 'close', :dd, 'top100', 'AAA', 1)"
            ),
            {"cap": _instant(dd), "dd": dd},
        )
    pg_legacy_session.commit()
    # The earliest ranking date is the sweep-consumption start (not --years).
    assert sweep_window_start() == date(2022, 5, 25)


# ---------------------------------------------------------------------------
# select_symbols_needing_backfill — span check at BOTH boundaries.
# ---------------------------------------------------------------------------


def test_recent_sliver_is_refetched(pg_legacy_session):
    # AMZN-shaped: only a recent sliver, no left-edge bar near the window start.
    _seed_prices(pg_legacy_session, "AMZN", [date(2026, 3, 3), date(2026, 3, 4)])
    pg_legacy_session.expire_all()
    need = select_symbols_needing_backfill(["AMZN"], WINDOW_START, WINDOW_END)
    assert "AMZN" in need


def test_stale_tail_is_refetched(pg_legacy_session):
    # Old history but no recent bar near today — the right edge is missing.
    days = [date(2022, 5, 25), date(2023, 1, 3), date(2024, 6, 6)]
    _seed_prices(pg_legacy_session, "STALE", days)
    pg_legacy_session.expire_all()
    need = select_symbols_needing_backfill(["STALE"], WINDOW_START, WINDOW_END)
    assert "STALE" in need


def test_fully_covered_symbol_is_skipped(pg_legacy_session):
    # Dense bars across the whole window (no interior hole) → covered, not
    # re-fetched.
    _seed_dense(pg_legacy_session, "FULL", WINDOW_START, WINDOW_END)
    pg_legacy_session.expire_all()
    need = select_symbols_needing_backfill(["FULL"], WINDOW_START, WINDOW_END)
    assert "FULL" not in need


def test_split_history_is_refetched(pg_legacy_session):
    # Both edges present but a multi-year interior hole → must re-fetch (codex
    # P1). A min/max span check would wrongly skip this.
    _seed_dense(pg_legacy_session, "SPLIT", WINDOW_START, date(2022, 8, 1))
    _seed_dense(pg_legacy_session, "SPLIT", date(2026, 1, 1), WINDOW_END)
    pg_legacy_session.expire_all()
    need = select_symbols_needing_backfill(["SPLIT"], WINDOW_START, WINDOW_END)
    assert "SPLIT" in need


def test_null_ohlc_rows_are_not_coverage(pg_legacy_session):
    # Placeholder rows with NULL OHLC at the boundaries must NOT count as
    # coverage (codex P1) — a covered span built only from NULL rows re-fetches.
    _seed_null_ohlc(
        pg_legacy_session, "NULLY", [WINDOW_START, date(2024, 1, 2), WINDOW_END]
    )
    pg_legacy_session.expire_all()
    need = select_symbols_needing_backfill(["NULLY"], WINDOW_START, WINDOW_END)
    assert "NULLY" in need


def test_absent_symbol_is_refetched(pg_legacy_session):
    # No rows at all → needs backfill.
    need = select_symbols_needing_backfill(["GHOST"], WINDOW_START, WINDOW_END)
    assert "GHOST" in need


def test_boundary_tolerates_weekends(pg_legacy_session):
    # Dense bars whose edges sit a couple of sessions inside a weekend-bounded
    # window — within boundary tolerance, so still covered.
    _seed_dense(pg_legacy_session, "NEARWE", date(2022, 5, 27), date(2026, 6, 10))
    # window start/end on weekend days; nearest bars a session or two in count.
    need = select_symbols_needing_backfill(
        ["NEARWE"], date(2022, 5, 23), date(2026, 6, 13)
    )
    assert "NEARWE" not in need


# ---------------------------------------------------------------------------
# _yf_to_long — surface dropped symbols (no silent omission).
# ---------------------------------------------------------------------------


def _capture_warnings(monkeypatch):
    """Capture qu100_portfolio's structlog `log.warning` calls reliably,
    independent of how structlog routes to stdlib (which `caplog` can miss)."""
    from rainier.backtest import qu100_portfolio as qp

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        qp.log, "warning", lambda event, **kw: events.append((event, kw))
    )
    return events


def test_yf_to_long_surfaces_batch_drop(monkeypatch):
    events = _capture_warnings(monkeypatch)
    idx = pd.to_datetime([date(2026, 6, 10), date(2026, 6, 11)])
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["AAA"]]
    )
    yf_df = pd.DataFrame(1.0, index=idx, columns=cols)
    # Request AAA + BBB; yfinance only returned AAA → BBB must be surfaced.
    out = _yf_to_long(yf_df, ["AAA", "BBB"])
    assert set(out["symbol"].unique()) == {"AAA"}
    drop_events = [kw for ev, kw in events if ev == "yf_batch_dropped_symbols"]
    assert drop_events, "expected a yf_batch_dropped_symbols warning"
    assert "BBB" in drop_events[0]["missing"]


def test_yf_to_long_no_drop_logs_nothing(monkeypatch):
    events = _capture_warnings(monkeypatch)
    idx = pd.to_datetime([date(2026, 6, 10)])
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["AAA"]]
    )
    yf_df = pd.DataFrame(1.0, index=idx, columns=cols)
    _yf_to_long(yf_df, ["AAA"])
    assert not any(ev == "yf_batch_dropped_symbols" for ev, _ in events)


# ---------------------------------------------------------------------------
# assert_cohort_coverage — post-run, reports missing cohort members loudly.
# ---------------------------------------------------------------------------


def _seed_cohort(session, symbols: list[str], data_date: date) -> None:
    for i, sym in enumerate(symbols, start=1):
        _seed_stock(session, sym)
        session.execute(
            text(
                "INSERT INTO money_flow_snapshots "
                "(captured_at, capture_session, data_date, ranking_type, symbol, rank) "
                "VALUES (:cap, 'close', :dd, 'top100', :sym, :rank)"
            ),
            {"cap": _instant(data_date), "dd": data_date, "sym": sym, "rank": i},
        )
    session.commit()


def test_cohort_coverage_reports_missing(pg_legacy_session):
    as_of = date(2026, 6, 12)
    _seed_cohort(pg_legacy_session, ["COVD", "GAPS"], as_of)
    # COVD spans the window densely; GAPS has only a recent sliver.
    _seed_dense(pg_legacy_session, "COVD", WINDOW_START, as_of)
    _seed_prices(pg_legacy_session, "GAPS", [as_of])
    pg_legacy_session.expire_all()
    missing = assert_cohort_coverage(WINDOW_START, as_of, as_of=as_of)
    assert missing == ["GAPS"]


def test_cohort_coverage_all_covered_empty(pg_legacy_session):
    as_of = date(2026, 6, 12)
    _seed_cohort(pg_legacy_session, ["COVD"], as_of)
    _seed_dense(pg_legacy_session, "COVD", WINDOW_START, as_of)
    pg_legacy_session.expire_all()
    assert assert_cohort_coverage(WINDOW_START, as_of, as_of=as_of) == []


# ---------------------------------------------------------------------------
# CLI `db backfill-prices` — fetch the full sweep window for incomplete
# symbols (NOT the --years window), and stay idempotent on re-run.
# ---------------------------------------------------------------------------


def _build_yf_df(symbols: list[str], days: list[date]) -> pd.DataFrame:
    idx = pd.to_datetime(days)
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], symbols]
    )
    return pd.DataFrame(1.0, index=idx, columns=cols)


def test_backfill_fetch_window_pinned_to_sweep_start(pg_legacy_session, monkeypatch):
    """A present-but-incomplete symbol must be fetched from the sweep-window
    start (the earliest ranking date), NOT the trailing `--years` window."""
    from click.testing import CliRunner

    from rainier import cli as cli_mod

    sweep_start = date(2022, 5, 25)
    today = date(2026, 6, 12)
    # Rankings reach back to 2022-05-25 → that's the sweep start.
    _seed_cohort(pg_legacy_session, ["AMZN"], today)
    pg_legacy_session.execute(
        text(
            "INSERT INTO money_flow_snapshots "
            "(captured_at, capture_session, data_date, ranking_type, symbol, rank) "
            "VALUES (:cap, 'close', :dd, 'top100', 'AMZN', 1)"
        ),
        {"cap": _instant(sweep_start), "dd": sweep_start},
    )
    # AMZN has only a recent sliver → present-but-incomplete.
    _seed_prices(pg_legacy_session, "AMZN", [date(2026, 3, 3)])
    pg_legacy_session.commit()

    captured = {}

    def _fake_download(tickers, start=None, end=None, **kwargs):
        captured["start"] = start
        syms = tickers.split()
        # Return a full-window frame so the fetch "repairs" history.
        return _build_yf_df(syms, [sweep_start, today])

    import yfinance as yf

    monkeypatch.setattr(yf, "download", _fake_download)

    runner = CliRunner()
    # --years 1 would, under the OLD presence logic, cap the fetch at 2025-06.
    result = runner.invoke(
        cli_mod.cli, ["db", "backfill-prices", "--years", "1", "--batch-size", "20"]
    )
    assert result.exit_code == 0, result.output
    # The download start must be the sweep-window start, not 1y ago.
    assert captured.get("start") == str(sweep_start), (
        f"fetch start was {captured.get('start')!r}; expected the sweep window "
        f"start {sweep_start}. Output:\n{result.output}"
    )


def test_backfill_idempotent_rerun(pg_legacy_session, monkeypatch):
    from click.testing import CliRunner

    from rainier import cli as cli_mod

    sweep_start = date(2022, 5, 25)
    today = date(2026, 6, 12)
    _seed_cohort(pg_legacy_session, ["NVDA"], today)
    pg_legacy_session.execute(
        text(
            "INSERT INTO money_flow_snapshots "
            "(captured_at, capture_session, data_date, ranking_type, symbol, rank) "
            "VALUES (:cap, 'close', :dd, 'top100', 'NVDA', 1)"
        ),
        {"cap": _instant(sweep_start), "dd": sweep_start},
    )
    pg_legacy_session.commit()

    bars = [sweep_start, date(2024, 1, 2), today]

    def _fake_download(tickers, start=None, end=None, **kwargs):
        return _build_yf_df(tickers.split(), bars)

    import yfinance as yf

    monkeypatch.setattr(yf, "download", _fake_download)

    runner = CliRunner()
    r1 = runner.invoke(cli_mod.cli, ["db", "backfill-prices", "--years", "1"])
    assert r1.exit_code == 0, r1.output
    pg_legacy_session.expire_all()
    n1 = len(_count := pg_legacy_session.execute(
        select(StockPrice).where(StockPrice.symbol == "NVDA")
    ).scalars().all())
    r2 = runner.invoke(cli_mod.cli, ["db", "backfill-prices", "--years", "1"])
    assert r2.exit_code == 0, r2.output
    pg_legacy_session.expire_all()
    n2 = len(pg_legacy_session.execute(
        select(StockPrice).where(StockPrice.symbol == "NVDA")
    ).scalars().all())
    assert n1 == n2 == 3  # ON CONFLICT DO NOTHING → no duplicates
