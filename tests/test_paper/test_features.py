"""R-E daily JSONB feature snapshot (design Appendix C, feature_version 1).

Two lanes (same split as the rest of the paper suite):

* pure lane — `compute_features` on hand-built windows, no DB. Every pinned
  formula is asserted against hand-computed numbers.
* postgres lane (`requires_postgres`) — migration 0011, the
  `qu100_daily_features` upsert step, the cohort selector, and the
  changed-set recompute contract.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect, select, text

from rainier.core.models import QU100DailyFeatures
from rainier.paper.features import (
    FEATURE_VERSION,
    VRVP_BINS,
    compute_features,
    run_daily_feature_step,
)
from rainier.paper.ingest import canonical_instant, get_current_qu100_cohort

# ---------------------------------------------------------------------------
# Pure-lane helpers
# ---------------------------------------------------------------------------


def _bar(d, o=10.0, h=11.0, low=9.0, c=10.5, v=1000):
    return {"date": d, "open": o, "high": h, "low": low, "close": c, "volume": v}


def _days(n, start=date(2026, 1, 5)):
    """n consecutive weekday dates starting at `start` (a Monday)."""
    out, cur = [], start
    while len(out) < n:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def _window_n(n, close_fn=lambda i: 10.0 + i * 0.001):
    """n bars with gently varying closes (no accidental pivots/warm-up)."""
    return [
        _bar(d, o=close_fn(i), h=close_fn(i) + 1, low=close_fn(i) - 1,
             c=close_fn(i), v=1000)
        for i, d in enumerate(_days(n))
    ]


# ---------------------------------------------------------------------------
# Metadata / shape
# ---------------------------------------------------------------------------


def test_metadata_version_and_basis():
    w = _window_n(10)
    f = compute_features(w, w[-1]["date"])
    assert f["feature_version"] == FEATURE_VERSION == 1
    assert f["price_basis"] == "adjusted"
    assert "data_gap" not in f  # only present (true) on gap days
    assert VRVP_BINS == 200
    assert f["vrvp"]["bins"] == 200


def test_deterministic():
    w = _window_n(70)
    as_of = w[-1]["date"]
    assert compute_features(w, as_of) == compute_features(w, as_of)


# ---------------------------------------------------------------------------
# vwap / volume — day's typical price (H+L+C)/3 (pinned proxy)
# ---------------------------------------------------------------------------


def test_vwap_equals_typical_price_and_volume_is_days():
    w = _window_n(10)
    w[-1] = _bar(w[-1]["date"], o=12.0, h=13.0, low=11.0, c=13.0, v=4321)
    f = compute_features(w, w[-1]["date"])
    assert f["vwap"] == pytest.approx((13.0 + 11.0 + 13.0) / 3)
    assert f["volume"] == 4321


# ---------------------------------------------------------------------------
# SMAs — close-based, NULL under warm-up, never partial-window
# ---------------------------------------------------------------------------


def test_sma_warmup_under_5_all_null():
    w = _window_n(4)
    f = compute_features(w, w[-1]["date"])
    assert f["sma5"] is None and f["sma22"] is None
    assert f["sma44"] is None and f["sma60"] is None


def test_sma_warmup_59_bars_sma60_null_others_exact():
    w = [
        _bar(d, c=float(i + 1), o=float(i + 1), h=float(i + 2), low=float(i))
        for i, d in enumerate(_days(59))
    ]
    f = compute_features(w, w[-1]["date"])
    # closes are 1..59
    assert f["sma5"] == pytest.approx(57.0)    # mean(55..59)
    assert f["sma22"] == pytest.approx(48.5)   # mean(38..59)
    assert f["sma44"] == pytest.approx(37.5)   # mean(16..59)
    assert f["sma60"] is None                  # warm-up: NULL, never partial


def test_sma60_exact_at_60_bars():
    w = [
        _bar(d, c=float(i + 1), o=float(i + 1), h=float(i + 2), low=float(i))
        for i, d in enumerate(_days(60))
    ]
    f = compute_features(w, w[-1]["date"])
    assert f["sma60"] == pytest.approx(30.5)   # mean(1..60)


def test_sma_never_partial_window():
    w = _window_n(10)
    f = compute_features(w, w[-1]["date"])
    assert f["sma5"] is not None
    assert f["sma22"] is None  # 10 bars: NOT a mean over 10


# ---------------------------------------------------------------------------
# fractal — latest CONFIRMED pivot, carries the pivot's own date, never same-day
# ---------------------------------------------------------------------------


def _fractal_window(n=20, spike_high_at=None, dip_low_at=None, extra=()):
    """Flat highs/lows with a unique spike/dip so only those confirm as pivots."""
    bars = []
    for i, d in enumerate(_days(n)):
        h, low = 10.0, 5.0
        if i == spike_high_at:
            h = 15.0
        if i == dip_low_at:
            low = 2.0
        for j, hh, ll in extra:
            if i == j:
                h, low = hh, ll
        bars.append(_bar(d, o=7.0, h=h, low=low, c=7.0, v=100))
    return bars


def test_fractal_confirmed_pivots_carry_own_dates():
    w = _fractal_window(spike_high_at=8, dip_low_at=12)
    f = compute_features(w, w[-1]["date"])
    days = _days(20)
    assert f["fractal"]["last_pivot_high"] == {
        "date": days[8].isoformat(), "price": 15.0,
    }
    assert f["fractal"]["last_pivot_low"] == {
        "date": days[12].isoformat(), "price": 2.0,
    }
    assert f["fractal"]["latest"] == "low"  # day 12 is more recent than day 8


def test_fractal_never_same_day_signal():
    # A bigger spike on the LAST bar is NOT confirmed (centered lookback lags).
    w = _fractal_window(spike_high_at=8, extra=[(19, 16.0, 5.0)])
    f = compute_features(w, w[-1]["date"])
    assert f["fractal"]["last_pivot_high"]["date"] == _days(20)[8].isoformat()
    assert f["fractal"]["latest"] == "high"


def test_fractal_null_when_no_confirmed_pivot():
    # Strictly trending bars have no centered-window extreme → no pivots.
    w = [
        _bar(d, o=10 + i, h=11 + i, low=9 + i, c=10 + i)
        for i, d in enumerate(_days(20))
    ]
    f = compute_features(w, w[-1]["date"])
    assert f["fractal"] == {
        "last_pivot_high": None, "last_pivot_low": None, "latest": None,
    }


# ---------------------------------------------------------------------------
# vrvp — 200 uniform half-open bins, proportional-overlap allocation
# ---------------------------------------------------------------------------


def test_vrvp_hand_computed():
    """lo=0, hi=200 → bin width exactly 1.0; every number below is hand-derived.

    b1 spans all 200 bins (vol 200 → 1.0/bin); b2 is a high==low point day at
    10 (vol 500 → all to bin 10); b3 spans [11,13] (vol 100 → 50 to bin 11,
    50 to bin 12). Totals: bin10=501, bin11=bin12=51, others 1.0; total=800.
    """
    days = _days(3)
    w = [
        _bar(days[0], o=100.0, h=200.0, low=0.0, c=100.0, v=200),
        _bar(days[1], o=10.0, h=10.0, low=10.0, c=10.0, v=500),
        _bar(days[2], o=12.0, h=13.0, low=11.0, c=13.0, v=100),
    ]
    f = compute_features(w, days[2])
    v = f["vrvp"]
    assert v["bins"] == 200
    # POC = max bin (501 @ bin 10) midpoint.
    assert v["poc"] == pytest.approx(10.5)
    # VA: 501 → +bin11 (51 beats bin9's 1.0) → 552 → +bin12 → 603 ≥ 0.70*800.
    assert v["va_low"] == pytest.approx(10.5)
    assert v["va_high"] == pytest.approx(12.5)
    # Split at the day's close (13): bins 0..12 lie fully below.
    # below = 13*1.0 + 500 + 100 = 613; above = 800 - 613 = 187.
    assert v["vol_below"] == pytest.approx(613.0)
    assert v["vol_above"] == pytest.approx(187.0)
    assert v["vol_below"] + v["vol_above"] == pytest.approx(800.0)


def test_vrvp_poc_tie_takes_lower_price():
    days = _days(4)
    w = [
        _bar(days[0], o=1.0, h=200.0, low=0.0, c=1.0, v=0),     # bounds only
        _bar(days[1], o=10.5, h=10.5, low=10.5, c=10.5, v=100),  # bin 10
        _bar(days[2], o=50.5, h=50.5, low=50.5, c=50.5, v=100),  # bin 50 (tie)
        _bar(days[3], o=150.5, h=150.5, low=150.5, c=150.5, v=1),
    ]
    f = compute_features(w, days[3])
    assert f["vrvp"]["poc"] == pytest.approx(10.5)  # tie → lower price


def test_vrvp_value_area_tie_takes_lower_side():
    """POC bin10=100; bins 9 and 11 tie at 70 each; total 240 → 70% = 168.
    Lower-side tie pick: 100 + 70 = 170 ≥ 168 → VA stops at {9,10}."""
    days = _days(4)
    w = [
        _bar(days[0], o=1.0, h=200.0, low=0.0, c=1.0, v=0),      # bounds only
        _bar(days[1], o=10.5, h=10.5, low=10.5, c=10.5, v=100),  # bin 10 (POC)
        _bar(days[2], o=9.5, h=9.5, low=9.5, c=9.5, v=70),       # bin 9
        _bar(days[3], o=11.5, h=11.5, low=11.5, c=11.5, v=70),   # bin 11
    ]
    f = compute_features(w, days[3])
    assert f["vrvp"]["va_low"] == pytest.approx(9.5)
    assert f["vrvp"]["va_high"] == pytest.approx(10.5)  # bin 11 NOT included


def test_vrvp_degenerate_single_price_window():
    # Every bar at exactly one price: hi == lo → single-bin profile, no crash.
    w = [_bar(d, o=42.0, h=42.0, low=42.0, c=42.0, v=100) for d in _days(3)]
    f = compute_features(w, w[-1]["date"])
    v = f["vrvp"]
    assert v["poc"] == pytest.approx(42.0)
    assert v["va_low"] == pytest.approx(42.0)
    assert v["va_high"] == pytest.approx(42.0)
    assert v["vol_above"] == pytest.approx(0.0)
    assert v["vol_below"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# data-gap day — row written with NULLs + "data_gap": true
# ---------------------------------------------------------------------------


def test_data_gap_when_no_bar_for_as_of():
    w = _window_n(10)
    as_of = w[-1]["date"] + timedelta(days=1)  # no bar for as_of
    f = compute_features(w, as_of)
    assert f["data_gap"] is True
    for k in ("vwap", "volume", "sma5", "sma22", "sma44", "sma60",
              "fractal", "vrvp"):
        assert f[k] is None
    assert f["feature_version"] == 1
    assert f["price_basis"] == "adjusted"


def test_data_gap_empty_window():
    f = compute_features([], date(2026, 1, 9))
    assert f["data_gap"] is True and f["vwap"] is None


# ---------------------------------------------------------------------------
# ORM shape (no DB needed)
# ---------------------------------------------------------------------------


def test_qu100_daily_features_orm_shape():
    cols = {c.name for c in inspect(QU100DailyFeatures).columns}
    assert {"id", "symbol", "data_date", "ranking_type", "rank",
            "features", "computed_at"} <= cols
    t = QU100DailyFeatures.__table__
    assert t.c.rank.nullable
    assert not t.c.features.nullable
    assert "JSON" in str(t.c.features.type).upper()
    uniques = [
        c for c in t.constraints if c.__class__.__name__ == "UniqueConstraint"
    ]
    assert any(
        {col.name for col in c.columns} == {"symbol", "data_date", "ranking_type"}
        for c in uniques
    )
    from rainier.core.models import HYPERTABLES

    assert "qu100_daily_features" not in HYPERTABLES


# ---------------------------------------------------------------------------
# Postgres lane — migration, cohort selector, daily step, recompute
# ---------------------------------------------------------------------------

def _ensure_stock(session, symbol):
    # money_flow_snapshots.symbol / stock_prices.symbol both FK to stocks.
    session.execute(
        text("INSERT INTO stocks (symbol) VALUES (:s) ON CONFLICT DO NOTHING"),
        {"s": symbol},
    )


def _seed_snapshot(session, symbol, rank, data_date, captured_at,
                   ranking_type="top100"):
    from rainier.core.models import MoneyFlowSnapshot

    _ensure_stock(session, symbol)
    session.add(
        MoneyFlowSnapshot(
            captured_at=captured_at,
            capture_session="close",
            data_date=data_date,
            ranking_type=ranking_type,
            symbol=symbol,
            rank=rank,
        )
    )


def _seed_bars(session, symbol, bars):
    from rainier.core.models import StockPrice

    _ensure_stock(session, symbol)
    for b in bars:
        session.add(
            StockPrice(
                symbol=symbol,
                date=canonical_instant(b["date"]),
                open=b["open"],
                high=b["high"],
                low=b["low"],
                close=b["close"],
                volume=b["volume"],
            )
        )


def _feature_rows(session):
    return session.execute(
        select(QU100DailyFeatures).order_by(
            QU100DailyFeatures.symbol, QU100DailyFeatures.data_date
        )
    ).scalars().all()


@pytest.mark.requires_postgres
def test_0011_migration_creates_table_with_unique(pg_legacy_engine):
    insp = inspect(pg_legacy_engine)
    schema = pg_legacy_engine.rainier_schema
    assert "qu100_daily_features" in set(insp.get_table_names(schema=schema))
    uqs = insp.get_unique_constraints("qu100_daily_features", schema=schema)
    assert any(
        set(u["column_names"]) == {"symbol", "data_date", "ranking_type"}
        for u in uqs
    )
    feat_col = next(
        c for c in insp.get_columns("qu100_daily_features", schema=schema)
        if c["name"] == "features"
    )
    assert "JSON" in str(feat_col["type"]).upper()


@pytest.mark.requires_postgres
def test_0011_downgrade_drops_only_features_table(pg_legacy_engine):
    from pathlib import Path

    down = (
        Path(__file__).resolve().parents[2]
        / "migrations" / "0011_qu100_daily_features_downgrade.sql"
    )
    with pg_legacy_engine.begin() as conn:
        conn.execute(text(down.read_text()))
    insp = inspect(pg_legacy_engine)
    schema = pg_legacy_engine.rainier_schema
    tables = set(insp.get_table_names(schema=schema))
    assert "qu100_daily_features" not in tables
    # Neighbours untouched.
    assert {"paper_trade", "stock_prices", "money_flow_snapshots"} <= tables


@pytest.mark.requires_postgres
class TestCohortSelector:
    AS_OF = date(2026, 6, 9)
    T1 = datetime(2026, 6, 9, 18, 0, tzinfo=timezone.utc)
    T2 = datetime(2026, 6, 9, 21, 0, tzinfo=timezone.utc)

    def test_latest_capture_of_latest_date_wins(self, pg_legacy_session):
        # Earlier capture has BBB; the later capture (same data_date) replaces
        # it with CCC and re-ranks AAA — the later capture is authoritative.
        _seed_snapshot(pg_legacy_session, "AAA", 1, self.AS_OF, self.T1)
        _seed_snapshot(pg_legacy_session, "BBB", 2, self.AS_OF, self.T1)
        _seed_snapshot(pg_legacy_session, "AAA", 2, self.AS_OF, self.T2)
        _seed_snapshot(pg_legacy_session, "CCC", 1, self.AS_OF, self.T2)
        pg_legacy_session.commit()

        cohort = get_current_qu100_cohort(self.AS_OF)
        by_symbol = {m["symbol"]: m for m in cohort}
        assert set(by_symbol) == {"AAA", "CCC"}
        assert by_symbol["AAA"]["rank"] == 2  # latest capture's rank
        assert by_symbol["CCC"]["rank"] == 1
        assert by_symbol["AAA"]["data_date"] == self.AS_OF
        assert by_symbol["AAA"]["captured_at"] is not None

    def test_as_of_excludes_future_dates(self, pg_legacy_session):
        _seed_snapshot(pg_legacy_session, "AAA", 1, date(2026, 6, 8), self.T1)
        _seed_snapshot(pg_legacy_session, "BBB", 1, date(2026, 6, 10), self.T2)
        pg_legacy_session.commit()

        cohort = get_current_qu100_cohort(date(2026, 6, 9))
        assert [m["symbol"] for m in cohort] == ["AAA"]
        assert cohort[0]["data_date"] == date(2026, 6, 8)

    def test_non_top100_rows_never_win(self, pg_legacy_session):
        _seed_snapshot(pg_legacy_session, "AAA", 1, date(2026, 6, 8), self.T1)
        _seed_snapshot(
            pg_legacy_session, "BBB", 1, date(2026, 6, 9), self.T2,
            ranking_type="concept",
        )
        pg_legacy_session.commit()

        cohort = get_current_qu100_cohort(date(2026, 6, 9))
        assert [m["symbol"] for m in cohort] == ["AAA"]

    def test_backfill_shared_captured_at_date_precedence(self, pg_legacy_session):
        # Backfills stamp many data_dates with ONE captured_at — date wins first.
        t0 = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)
        _seed_snapshot(pg_legacy_session, "OLD", 1, date(2026, 6, 8), t0)
        _seed_snapshot(pg_legacy_session, "NEW", 1, date(2026, 6, 9), t0)
        pg_legacy_session.commit()

        cohort = get_current_qu100_cohort(date(2026, 6, 9))
        assert [m["symbol"] for m in cohort] == ["NEW"]

    def test_empty_db_returns_empty(self, pg_legacy_session):
        assert get_current_qu100_cohort(self.AS_OF) == []

    # NOTE: a "defensive symbol-dedup" case was dropped in the re-derive — the
    # spec (acceptance 6) reuses the shared get_current_qu100_cohort (PR #138),
    # which orders by rank without per-symbol dedup, rather than the worker's
    # own selector. The captured top100 ranking is unique per symbol per
    # capture, so the case is not a real-data scenario.


@pytest.mark.requires_postgres
class TestDailyFeatureStep:
    AS_OF = date(2026, 6, 9)
    T = datetime(2026, 6, 9, 21, 0, tzinfo=timezone.utc)

    def _seed_cohort_and_bars(self, session):
        _seed_snapshot(session, "AAA", 1, self.AS_OF, self.T)
        _seed_snapshot(session, "BBB", 2, self.AS_OF, self.T)
        # AAA: 60 priced bars ending at AS_OF, closes 1..60.
        days = _trading_days_ending(self.AS_OF, 60)
        _seed_bars(
            session, "AAA",
            [
                _bar(d, o=float(i + 1), h=float(i + 2), low=float(i),
                     c=float(i + 1), v=1000)
                for i, d in enumerate(days)
            ],
        )
        # BBB: only 3 bars (deep warm-up).
        _seed_bars(
            session, "BBB",
            [
                _bar(d, o=5.0, h=6.0, low=4.0, c=5.0, v=10)
                for d in _trading_days_ending(self.AS_OF, 3)
            ],
        )
        session.commit()

    def test_step_upserts_one_row_per_member_and_is_idempotent(
        self, pg_legacy_session
    ):
        self._seed_cohort_and_bars(pg_legacy_session)

        res1 = run_daily_feature_step(self.AS_OF)
        assert res1["computed"] == 2
        rows = _feature_rows(pg_legacy_session)
        assert [(r.symbol, r.rank) for r in rows] == [("AAA", 1), ("BBB", 2)]
        aaa = rows[0]
        assert aaa.data_date == self.AS_OF
        assert aaa.ranking_type == "top100"
        assert aaa.features["feature_version"] == 1
        assert aaa.features["sma60"] == pytest.approx(30.5)  # mean(1..60)
        assert aaa.features["vwap"] == pytest.approx((61 + 59 + 60) / 3)
        bbb = rows[1]
        assert bbb.features["sma5"] is None  # 3 bars → warm-up
        assert bbb.features["vwap"] == pytest.approx(5.0)

        # Re-run: same 2 rows (UNIQUE upsert), not 4.
        run_daily_feature_step(self.AS_OF)
        pg_legacy_session.expire_all()
        assert len(_feature_rows(pg_legacy_session)) == 2

    def test_step_per_symbol_failure_never_starves_the_rest(
        self, pg_legacy_session, monkeypatch
    ):
        """Inner isolation level (module docstring) — one bad symbol is caught
        and counted; every other member still gets its row."""
        import rainier.paper.features as features_mod

        self._seed_cohort_and_bars(pg_legacy_session)
        real_load = features_mod._load_window

        def exploding_load(session, symbol, data_date):
            if symbol == "AAA":
                raise RuntimeError("boom")
            return real_load(session, symbol, data_date)

        monkeypatch.setattr(features_mod, "_load_window", exploding_load)
        res = run_daily_feature_step(self.AS_OF)
        assert res == {"computed": 1, "recomputed": 0, "failed": 1}
        rows = _feature_rows(pg_legacy_session)
        assert [r.symbol for r in rows] == ["BBB"]  # AAA failed, BBB landed

    def test_step_recompute_failure_caught_and_counted(
        self, pg_legacy_session, monkeypatch
    ):
        """Inner isolation level, recompute loop — a failing changed pair is
        caught and counted, never raised; the stale row stays untouched."""
        import rainier.paper.features as features_mod

        self._seed_cohort_and_bars(pg_legacy_session)
        past = _trading_days_ending(self.AS_OF, 60)[-5]
        pg_legacy_session.add(
            QU100DailyFeatures(
                symbol="AAA", data_date=past, ranking_type="top100", rank=9,
                features={"feature_version": 1, "vwap": -1.0},
            )
        )
        pg_legacy_session.commit()
        real_load = features_mod._load_window

        def exploding_load(session, symbol, data_date):
            if data_date == past:
                raise RuntimeError("boom")
            return real_load(session, symbol, data_date)

        monkeypatch.setattr(features_mod, "_load_window", exploding_load)
        res = run_daily_feature_step(self.AS_OF, changed=[("AAA", past)])
        assert res == {"computed": 2, "recomputed": 0, "failed": 1}
        pg_legacy_session.expire_all()
        stale = next(
            r for r in _feature_rows(pg_legacy_session)
            if r.symbol == "AAA" and r.data_date == past
        )
        assert stale.features["vwap"] == -1.0  # untouched, not corrupted

    def test_step_null_close_bar_on_data_date_writes_gap_row(
        self, pg_legacy_session
    ):
        """A transient NULL-OHLC row ON the data_date is unusable — the
        priced-bar window filter reads it as a gap (same probe as ingest)."""
        _seed_snapshot(pg_legacy_session, "NNN", 4, self.AS_OF, self.T)
        days = _trading_days_ending(self.AS_OF, 5)
        _seed_bars(
            pg_legacy_session, "NNN",
            [_bar(d, o=1.0, h=2.0, low=0.5, c=1.0, v=10) for d in days[:-1]],
        )
        # The data_date bar exists but is NULL-priced (yfinance partial).
        _seed_bars(
            pg_legacy_session, "NNN",
            [{"date": days[-1], "open": None, "high": None, "low": None,
              "close": None, "volume": None}],
        )
        pg_legacy_session.commit()

        run_daily_feature_step(self.AS_OF)
        rows = _feature_rows(pg_legacy_session)
        assert len(rows) == 1
        assert rows[0].features["data_gap"] is True
        assert rows[0].features["vwap"] is None

    def test_step_writes_gap_row_for_missing_bar(self, pg_legacy_session):
        _seed_snapshot(pg_legacy_session, "GGG", 7, self.AS_OF, self.T)
        # Bars exist but none ON the data_date → gap row, not a skipped symbol.
        stale_days = _trading_days_ending(self.AS_OF - timedelta(days=7), 5)
        _seed_bars(
            pg_legacy_session, "GGG",
            [_bar(d, o=1.0, h=2.0, low=0.5, c=1.0, v=10) for d in stale_days],
        )
        pg_legacy_session.commit()

        run_daily_feature_step(self.AS_OF)
        rows = _feature_rows(pg_legacy_session)
        assert len(rows) == 1
        assert rows[0].features["data_gap"] is True
        assert rows[0].features["vwap"] is None

    def test_step_recomputes_changed_pairs(self, pg_legacy_session):
        self._seed_cohort_and_bars(pg_legacy_session)
        past = _trading_days_ending(self.AS_OF, 60)[-5]  # a priced past day
        # Stale feature row computed under pre-split prices.
        pg_legacy_session.add(
            QU100DailyFeatures(
                symbol="AAA", data_date=past, ranking_type="top100", rank=9,
                features={"feature_version": 1, "vwap": -1.0},
            )
        )
        pg_legacy_session.commit()

        res = run_daily_feature_step(
            self.AS_OF, changed=[("AAA", past), ("ZZZ", past)]
        )
        assert res["recomputed"] == 1  # ZZZ has no feature row → ignored
        pg_legacy_session.expire_all()
        row = next(
            r for r in _feature_rows(pg_legacy_session)
            if r.symbol == "AAA" and r.data_date == past
        )
        assert row.features["vwap"] != -1.0  # recomputed from current bars
        assert row.features["vwap"] == pytest.approx(
            (57.0 + 55.0 + 56.0) / 3  # bar i=55 (1-based 56): h=57, l=55, c=56
        )
        assert row.rank == 9  # rank untouched by recompute

    def test_step_changed_pair_already_in_cohort_not_double_counted(
        self, pg_legacy_session
    ):
        self._seed_cohort_and_bars(pg_legacy_session)
        res = run_daily_feature_step(
            self.AS_OF, changed=[("AAA", self.AS_OF)]
        )
        assert res["computed"] == 2
        assert res["recomputed"] == 0  # cohort pass already covered it
        assert len(_feature_rows(pg_legacy_session)) == 2


def _trading_days_ending(end: date, n: int) -> list[date]:
    """n weekdays ending at-or-before `end`, ascending."""
    out, cur = [], end
    while len(out) < n:
        if cur.weekday() < 5:
            out.append(cur)
        cur -= timedelta(days=1)
    return sorted(out)
