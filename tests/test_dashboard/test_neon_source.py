"""Unit tests for `rainier.dashboard.neon_source` — the Neon (canonical
``market.*``) data loaders that feed the dashboard renderers by default.

We avoid requiring a live Postgres by building an in-memory SQLite engine and
ATTACHing a database aliased ``market`` so the loaders' schema-qualified
``market.<table>`` SQL resolves. The loaders only need the engine + plain SQL,
so this faithfully exercises the column-projection + fail-loud + asof logic.

A separate ``requires_postgres`` integration test (skipped unless a live DB is
configured) renders a non-empty dashboard end-to-end from Neon.
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from rainier.dashboard import neon_source as ns

# ---------------------------------------------------------------------------
# In-memory SQLite engine with a `market` schema (via ATTACH)
# ---------------------------------------------------------------------------


@pytest.fixture
def market_engine():
    """SQLite engine with an attached ``market`` schema, seeded with small
    thematic_features / breadth / benchmark fixtures."""
    # A single shared in-memory connection (StaticPool) so the ATTACH + seed
    # persist for the loader's later connect() calls.
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(text("ATTACH DATABASE ':memory:' AS market"))
        conn.execute(
            text(
                """
                CREATE TABLE market.thematic_features_daily (
                    asof_date TEXT, symbol TEXT, sector_id INTEGER,
                    rank INTEGER, rank_delta_1d INTEGER, rank_delta_5d INTEGER,
                    r_5 INTEGER, r_10 INTEGER, r_20 INTEGER,
                    ret_ytd REAL, top15_streak INTEGER
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE market.breadth_indicator_daily (
                    asof_date TEXT, indicator TEXT, value REAL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE market.benchmark_ohlcv (
                    symbol TEXT, date TEXT, open REAL, high REAL, low REAL,
                    close REAL, volume INTEGER, fetched_at TEXT,
                    yfinance_version TEXT
                )
                """
            )
        )
    yield engine
    engine.dispose()


def _seed_features(engine, rows: list[dict]) -> None:
    with engine.begin() as conn:
        for r in rows:
            conn.execute(
                text(
                    "INSERT INTO market.thematic_features_daily "
                    "(asof_date, symbol, sector_id, rank, rank_delta_1d, "
                    "rank_delta_5d, r_5, r_10, r_20, ret_ytd, top15_streak) "
                    "VALUES (:asof_date, :symbol, :sector_id, :rank, "
                    ":rank_delta_1d, :rank_delta_5d, :r_5, :r_10, :r_20, "
                    ":ret_ytd, :top15_streak)"
                ),
                r,
            )


def _feat_row(asof: str, symbol: str, rank: int) -> dict:
    return {
        "asof_date": asof, "symbol": symbol, "sector_id": 1, "rank": rank,
        "rank_delta_1d": 0, "rank_delta_5d": 0, "r_5": 1, "r_10": 1,
        "r_20": 1, "ret_ytd": 0.1, "top15_streak": 0,
    }


# ---------------------------------------------------------------------------
# ETF features loader
# ---------------------------------------------------------------------------

REQUIRED_ETF_COLS = {
    "asof_date", "symbol", "sector_id", "rank", "rank_delta_1d",
    "rank_delta_5d", "r_5", "r_10", "r_20", "ret_ytd", "top15_streak",
}


def test_load_etf_features_returns_required_columns(market_engine):
    _seed_features(
        market_engine,
        [
            _feat_row("2026-06-04", "AAA", 90),
            _feat_row("2026-06-05", "AAA", 92),
            _feat_row("2026-06-05", "BBB", 50),
        ],
    )
    df = ns.load_etf_features_neon(market_engine, date(2026, 6, 5))
    assert REQUIRED_ETF_COLS.issubset(set(df.columns)), (
        f"missing required cols: {REQUIRED_ETF_COLS - set(df.columns)}"
    )
    # History is included (the 2026-06-04 AAA row), not just the asof slice.
    assert len(df) == 3
    assert {"AAA", "BBB"}.issubset(set(df["symbol"]))


def test_load_etf_features_renders_non_empty(market_engine):
    """The loaded frame drives the pure renderer to a non-empty page."""
    from rainier.dashboard.render_etf import render_etf_html

    _seed_features(
        market_engine,
        [_feat_row("2026-06-05", "AAA", 90), _feat_row("2026-06-05", "BBB", 50)],
    )
    df = ns.load_etf_features_neon(market_engine, date(2026, 6, 5))
    registry = pd.DataFrame({"sector_id": [1], "sector_name": ["Tech"]})
    html = render_etf_html(
        features=df, registry=registry, asof=date(2026, 6, 5),
        rendered_at_pt="12:40",
    )
    assert "Universe: 0" not in html
    assert "AAA" in html and "BBB" in html


def test_load_etf_features_raises_on_empty(market_engine):
    with pytest.raises(ns.EmptyNeonResultError):
        ns.load_etf_features_neon(market_engine, date(2026, 6, 5))


def test_load_etf_features_raises_when_asof_absent(market_engine):
    """History present but no row exactly at asof → fail loud (don't render an
    empty page from history-only rows)."""
    _seed_features(market_engine, [_feat_row("2026-06-01", "AAA", 90)])
    with pytest.raises(ns.EmptyNeonResultError):
        ns.load_etf_features_neon(market_engine, date(2026, 6, 5))


def test_latest_etf_asof(market_engine):
    _seed_features(
        market_engine,
        [_feat_row("2026-06-01", "AAA", 90), _feat_row("2026-06-05", "AAA", 91)],
    )
    assert ns.latest_etf_asof(market_engine) == date(2026, 6, 5)


def test_latest_etf_asof_raises_on_empty(market_engine):
    with pytest.raises(ns.EmptyNeonResultError):
        ns.latest_etf_asof(market_engine)


# ---------------------------------------------------------------------------
# Breadth + SPY loaders
# ---------------------------------------------------------------------------


def _seed_breadth(engine, rows: list[dict]) -> None:
    with engine.begin() as conn:
        for r in rows:
            conn.execute(
                text(
                    "INSERT INTO market.breadth_indicator_daily "
                    "(asof_date, indicator, value) "
                    "VALUES (:asof_date, :indicator, :value)"
                ),
                r,
            )


def test_load_breadth_returns_long_columns(market_engine):
    _seed_breadth(
        market_engine,
        [
            {"asof_date": "2026-06-04", "indicator": "pct_above_50ma", "value": 55.0},
            {"asof_date": "2026-06-05", "indicator": "pct_above_50ma", "value": 60.0},
        ],
    )
    df = ns.load_breadth_neon(market_engine, date(2026, 6, 5))
    assert set(df.columns) == {"asof_date", "indicator", "value"}
    assert len(df) == 2  # full history <= asof


def test_load_breadth_raises_on_empty(market_engine):
    with pytest.raises(ns.EmptyNeonResultError):
        ns.load_breadth_neon(market_engine, date(2026, 6, 5))


def test_latest_breadth_asof(market_engine):
    _seed_breadth(
        market_engine,
        [
            {"asof_date": "2026-06-02", "indicator": "x", "value": 1.0},
            {"asof_date": "2026-06-05", "indicator": "x", "value": 2.0},
        ],
    )
    assert ns.latest_breadth_asof(market_engine) == date(2026, 6, 5)


def test_load_spy_filters_symbol(market_engine):
    with market_engine.begin() as conn:
        for sym, c in [("SPY", 500.0), ("QQQ", 400.0)]:
            conn.execute(
                text(
                    "INSERT INTO market.benchmark_ohlcv "
                    "(symbol, date, open, high, low, close, volume, "
                    "fetched_at, yfinance_version) VALUES "
                    "(:s, '2026-06-05', :c, :c, :c, :c, 1000, '2026-06-05', '1')"
                ),
                {"s": sym, "c": c},
            )
    df = ns.load_spy_neon(market_engine)
    assert set(df["symbol"]) == {"SPY"}, "must filter to symbol='SPY'"
    assert "date" in df.columns and "close" in df.columns


def test_load_spy_empty_when_absent(market_engine):
    """No SPY rows → empty frame (NOT an error; the SPY pane is optional)."""
    df = ns.load_spy_neon(market_engine)
    assert df.empty


# ---------------------------------------------------------------------------
# ensure_env_loaded — the cron .env gotcha
# ---------------------------------------------------------------------------


def test_ensure_env_loaded_invokes_dotenv(monkeypatch):
    """ensure_env_loaded() must route through get_settings()'s load_dotenv() so
    the canonical engine (which reads os.environ directly, NO .env) finds
    DATABASE_URL on the cron path. We stub load_dotenv to set the var and
    assert ensure_env_loaded() makes it present — proving the .env hop happens
    before any get_engine() call."""
    import rainier.core.config as cfg

    monkeypatch.delenv("DATABASE_URL", raising=False)
    # get_settings() caches a process-lifetime singleton; clear it so our
    # stubbed load_dotenv is actually invoked (not short-circuited by a prior
    # test's cached Settings).
    monkeypatch.setattr(cfg, "_settings", None)

    called = {"n": 0}

    def fake_load_dotenv(*_a, **_k):
        called["n"] += 1
        os.environ["DATABASE_URL"] = "postgresql+psycopg://u:p@localhost/x"
        return True

    monkeypatch.setattr(cfg, "load_dotenv", fake_load_dotenv)

    try:
        ns.ensure_env_loaded()
        assert called["n"] >= 1, (
            "ensure_env_loaded must call load_dotenv via get_settings"
        )
        assert os.environ.get("DATABASE_URL") == "postgresql+psycopg://u:p@localhost/x"
    finally:
        # The stub wrote DATABASE_URL via raw os.environ (untracked by
        # monkeypatch); clear it so it can't leak into later tests.
        os.environ.pop("DATABASE_URL", None)


# ---------------------------------------------------------------------------
# requires_postgres — end-to-end from a real DB (skipped without one)
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_neon_render_end_to_end(monkeypatch):
    """Render a non-empty ETF dashboard from a live Postgres market schema.

    Skips unless DATABASE_URL points at a Postgres with seeded market.* data.
    This is the integration guard the task plan asks for; the SQLite tests
    above cover the loader logic deterministically in CI.
    """
    # Require an EXPLICIT test URL — never silently borrow the ambient
    # DATABASE_URL (which may point at production Neon or a stub set by another
    # test). The operator opts in via RAINIER_TEST_DATABASE_URL.
    url = os.environ.get("RAINIER_TEST_DATABASE_URL")
    if not url or not url.startswith("postgresql"):
        pytest.skip("no live Postgres configured (set RAINIER_TEST_DATABASE_URL)")

    from rainier.db.engine import get_engine

    monkeypatch.setenv("DATABASE_URL", url)
    engine = get_engine()
    try:
        asof = ns.latest_etf_asof(engine)
        df = ns.load_etf_features_neon(engine, asof)
    finally:
        engine.dispose()
    assert not df.empty
    assert REQUIRED_ETF_COLS.issubset(set(df.columns))
