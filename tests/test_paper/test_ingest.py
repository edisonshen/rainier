"""Step 2 — price ingest (TEST-SPEC §B).

The upsert is `ON CONFLICT DO UPDATE` (Postgres dialect) so the DB-backed cases
run under `requires_postgres`. Pure tz-normalization is exercised separately.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import select, text

from rainier.core.models import PaperTrade, StockPrice
from rainier.paper.ingest import (
    active_symbols,
    canonical_instant,
    ingest_prices,
    normalize_to_trading_date,
    screened_symbols,
)

pytestmark = pytest.mark.requires_postgres


def _bar(d, o=10.0, h=11.0, low=9.0, c=10.5, v=1000):
    return {"date": d, "open": o, "high": h, "low": low, "close": c, "volume": v}


def _count_rows(session, symbol):
    return session.execute(
        select(StockPrice).where(StockPrice.symbol == symbol)
    ).scalars().all()


# ---------------------------------------------------------------------------
# Pure normalization (no DB) — also runs under requires_postgres but needs none.
# ---------------------------------------------------------------------------


def test_b4_normalization_pure():
    # Differing time-of-day → same trading date.
    morning = datetime(2026, 3, 10, 13, 30, tzinfo=timezone.utc)
    evening = datetime(2026, 3, 10, 20, 0, tzinfo=timezone.utc)
    assert normalize_to_trading_date(morning) == date(2026, 3, 10)
    assert normalize_to_trading_date(evening) == date(2026, 3, 10)
    # A US/Eastern midnight that lands on the adjacent UTC date still resolves
    # via UTC consistently (read & write use the same rule).
    assert canonical_instant(date(2026, 3, 10)) == datetime(
        2026, 3, 10, tzinfo=timezone.utc
    )


# ---------------------------------------------------------------------------
# DB-backed
# ---------------------------------------------------------------------------


def _ingest(symbols, as_of, bars_by_symbol, window_days=10):
    def fetch_fn(syms, start, end):
        return {s: bars_by_symbol.get(s, []) for s in syms}

    return ingest_prices(
        symbols, as_of=as_of, fetch_fn=fetch_fn, window_days=window_days
    )


def test_b1_bootstrap(pg_legacy_session):
    as_of = date(2026, 1, 9)  # Friday
    days = [date(2026, 1, 5 + i) for i in range(5)]  # Mon-Fri
    res = _ingest(["AAA"], as_of, {"AAA": [_bar(d) for d in days]})
    assert res["upserted"] == 5
    rows = _count_rows(pg_legacy_session, "AAA")
    assert len(rows) == 5


def test_b2_per_symbol_date_gap_fill(pg_legacy_session):
    as_of = date(2026, 1, 9)
    days = [date(2026, 1, 5 + i) for i in range(5)]
    # Present through day 3 only.
    _ingest(["AAA"], as_of, {"AAA": [_bar(d) for d in days[:3]]})
    pg_legacy_session.expire_all()
    assert len(_count_rows(pg_legacy_session, "AAA")) == 3
    # Re-ingest full window → days 4-5 added (stale-symbol DOES get new dates).
    _ingest(["AAA"], as_of, {"AAA": [_bar(d) for d in days]})
    pg_legacy_session.expire_all()
    assert len(_count_rows(pg_legacy_session, "AAA")) == 5


def test_b3_idempotent_rerun(pg_legacy_session):
    as_of = date(2026, 1, 9)
    days = [date(2026, 1, 5 + i) for i in range(5)]  # the 5 sessions 1/5..1/9
    # window_days MUST match the supplied data span: with a wider window, the
    # un-fetched earlier window sessions stay permanently gapped and re-trigger a
    # re-fetch every run (in production yfinance returns the whole window, so
    # gaps close after run 1 — the mock only supplies these 5). 5-day window =>
    # after run 1 there are NO gaps, so run 2 is a true no-op (idempotency).
    _ingest(["AAA"], as_of, {"AAA": [_bar(d) for d in days]}, window_days=5)
    res2 = _ingest(["AAA"], as_of, {"AAA": [_bar(d) for d in days]}, window_days=5)
    pg_legacy_session.expire_all()
    assert res2["upserted"] == 0  # no gaps → no re-fetch
    assert len(_count_rows(pg_legacy_session, "AAA")) == 5


def test_b4_db_one_row_per_trading_date(pg_legacy_session):
    as_of = date(2026, 3, 13)
    # Two bars for the SAME trading day with different intraday timestamps +
    # a DST-transition day (US DST 2026-03-08). Should collapse to one row each.
    days = [date(2026, 3, 9 + i) for i in range(5)]  # Mon 3/9 .. Fri 3/13
    bars = []
    for d in days:
        bars.append(_bar(datetime(d.year, d.month, d.day, 13, 30, tzinfo=timezone.utc)))
        bars.append(_bar(datetime(d.year, d.month, d.day, 20, 0, tzinfo=timezone.utc)))
    _ingest(["AAA"], as_of, {"AAA": bars})
    pg_legacy_session.expire_all()
    rows = _count_rows(pg_legacy_session, "AAA")
    assert len(rows) == 5  # one per trading day, not 10


def test_b5_upsert_overwrites_adjusted_together(pg_legacy_session):
    as_of = date(2026, 1, 9)
    d = date(2026, 1, 9)
    _ingest(["AAA"], as_of, {"AAA": [_bar(d, o=100, h=110, low=90, c=105, v=5000)]})
    # Re-fetch with split-adjusted values — all of o/h/l/c/v update together.
    _ingest(["AAA"], as_of, {"AAA": [_bar(d, o=50, h=55, low=45, c=52.5, v=10000)]})
    pg_legacy_session.expire_all()
    rows = _count_rows(pg_legacy_session, "AAA")
    assert len(rows) == 1
    r = rows[0]
    assert (r.open, r.high, r.low, r.close, r.volume) == (50, 55, 45, 52.5, 10000)


def test_b5_partial_nan_refetch_does_not_clobber(pg_legacy_session):
    as_of = date(2026, 1, 9)
    d = date(2026, 1, 9)
    _ingest(["AAA"], as_of, {"AAA": [_bar(d, o=100, h=110, low=90, c=105, v=5000)]})
    # A NaN/None re-fetch must NOT null-out the previously-good values.
    _ingest(
        ["AAA"], as_of,
        {"AAA": [{"date": d, "open": None, "high": None, "low": None,
                  "close": 106.0, "volume": None}]},
    )
    pg_legacy_session.expire_all()
    r = _count_rows(pg_legacy_session, "AAA")[0]
    assert r.open == 100 and r.high == 110 and r.low == 90 and r.volume == 5000
    assert r.close == 106.0  # the non-null field DID update


def test_b6_universe_active_and_screened(pg_legacy_session):
    # paper_trade fixtures: pending AAA, open BBB, closed CCC, expired DDD.
    pg_legacy_session.execute(
        text(
            "INSERT INTO analysis_results (id, llm_model, prompt_template) "
            "VALUES (1,'m','t'),(2,'m','t'),(3,'m','t'),(4,'m','t')"
        )
    )
    for tid, sym, status in [(1, "AAA", "pending"), (2, "BBB", "open"),
                             (3, "CCC", "closed"), (4, "DDD", "expired")]:
        pg_legacy_session.add(
            PaperTrade(
                thesis_id=tid, symbol=sym, scan_date=date(2026, 1, 5),
                session_name="close", status=status,
                planned_entry_price=10, stop_loss=9, target_price=12,
            )
        )
    # screened today
    pg_legacy_session.execute(
        text(
            "INSERT INTO screened_stocks (scan_date, session_name, symbol, "
            "rule_rank, composite_score) VALUES "
            "('2026-01-09','close','EEE',1,1.0),('2026-01-09','close','FFF',2,1.0)"
        )
    )
    pg_legacy_session.commit()
    assert active_symbols() == ["AAA", "BBB"]
    assert screened_symbols(date(2026, 1, 9)) == ["EEE", "FFF"]


def test_b7_missing_symbol_skipped_others_ingested(pg_legacy_session):
    as_of = date(2026, 1, 9)
    days = [date(2026, 1, 5 + i) for i in range(5)]
    # AAA returns bars, BBB returns nothing.
    _ingest(["AAA", "BBB"], as_of, {"AAA": [_bar(d) for d in days], "BBB": []})
    pg_legacy_session.expire_all()
    assert len(_count_rows(pg_legacy_session, "AAA")) == 5
    # Persisted absence (not log text): BBB has zero rows.
    assert len(_count_rows(pg_legacy_session, "BBB")) == 0
