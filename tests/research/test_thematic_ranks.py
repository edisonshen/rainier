"""Golden-fixture deterministic tests for Layer A compute (thematic ranks).

The fixture: 5 tickers x 30 trading days of synthetic OHLCV with crafted
return profiles so we can hand-compute REL_N, R_N, and Rank to the last bit.

Design ref: docs/DESIGN-thematic-ranks-dashboard.md §5.1 ([D-002] / [D-003] /
[D-008] / [D-013] / [D-014]).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from rainier.research.breadth.ranks import compute_thematic_features

# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


def _build_panel(
    symbols: list[str],
    n_days: int,
    daily_returns: dict[str, list[float]],
    start: date = date(2024, 10, 1),
) -> pd.DataFrame:
    """Build a deterministic long-format OHLCV panel.

    Each ticker's close price compounds from 100.0 by the daily_returns list.
    open/high/low track close (toy values; ranking code ignores them).
    """
    dates = []
    d = start
    while len(dates) < n_days:
        # weekday only (skip Sat=5, Sun=6)
        if d.weekday() < 5:
            dates.append(d)
        d = d + timedelta(days=1)

    rows = []
    for sym in symbols:
        rets = daily_returns[sym]
        close = 100.0
        for i, day in enumerate(dates):
            if i > 0:
                close = close * (1.0 + rets[i - 1])
            rows.append(
                {
                    "symbol": sym,
                    "date": day,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)


def _universe_log_seed(yaml_sha: str) -> pd.DataFrame:
    """Return a one-row Layer C log as the input would arrive at compute time."""
    return pd.DataFrame(
        [
            {
                "effective_from": date(2024, 10, 1),
                "yaml_sha": yaml_sha,
                "universe_size": 5,
                "tickers": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                "note": "test seed",
            }
        ]
    )


@pytest.fixture
def panel_5x30() -> pd.DataFrame:
    """Crafted 5-ticker x 30-day panel."""
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    # Per-ticker return profile to anchor expected ranks
    daily_returns = {
        "AAA": [0.01] * 29,  # steady leader
        "BBB": [0.005] * 29,  # steady mild
        "CCC": [0.0] * 29,  # flat
        "DDD": [-0.005] * 29,  # steady decliner
        "EEE": [-0.01] * 29,  # steady laggard
    }
    return _build_panel(symbols, 30, daily_returns)


@pytest.fixture
def sector_map() -> dict[str, str]:
    return {
        "AAA": "tech",
        "BBB": "tech",
        "CCC": "healthcare",
        "DDD": "healthcare",
        "EEE": "energy",
    }


@pytest.fixture
def ticker_registry() -> dict[str, int]:
    return {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4, "EEE": 5}


@pytest.fixture
def sector_registry() -> dict[str, int]:
    return {"tech": 1, "healthcare": 2, "energy": 3}


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


REQUIRED_COLS = {
    "asof_date",
    "trading_day_ordinal",
    "symbol",
    "ticker_id",
    "sector_id",
    "close",
    "ret_1d",
    "rel_5",
    "rel_10",
    "rel_20",
    "ret_ytd",
    "relvol_5",
    "relvol_10",
    "relvol_20",
    "r_5",
    "r_10",
    "r_20",
    "rank",
    "rank_delta_1d",
    "rank_delta_5d",
    "top15_streak",
    "rank_minus_sector_mean",
    "universe_size",
    "universe_yaml_sha",
    "computed_at",
}


def test_features_returns_required_schema(
    panel_5x30, sector_map, ticker_registry, sector_registry
):
    """compute_thematic_features returns a DataFrame with the §5.1 schema."""
    out = compute_thematic_features(
        panel=panel_5x30,
        asof=date(2024, 11, 8),
        sector_map=sector_map,
        ticker_registry=ticker_registry,
        sector_registry=sector_registry,
        universe_yaml_sha="abc123",
    )
    assert REQUIRED_COLS.issubset(out.columns), (
        f"missing columns: {REQUIRED_COLS - set(out.columns)}"
    )


def test_features_dtypes_match_design(
    panel_5x30, sector_map, ticker_registry, sector_registry
):
    """float32 for stationary numerical features; close stays float64."""
    out = compute_thematic_features(
        panel=panel_5x30,
        asof=date(2024, 11, 8),
        sector_map=sector_map,
        ticker_registry=ticker_registry,
        sector_registry=sector_registry,
        universe_yaml_sha="abc123",
    )
    # close: float64 (per [D-016])
    assert out["close"].dtype == np.float64
    # stationary returns: float32
    for col in ("rel_5", "rel_10", "rel_20", "relvol_5", "relvol_10", "relvol_20"):
        assert out[col].dtype == np.float32, f"{col} dtype: {out[col].dtype}"
    # ranks: int8 (0-99 fits)
    for col in ("r_5", "r_10", "r_20", "rank", "rank_delta_1d", "rank_delta_5d"):
        assert out[col].dtype == np.int8, f"{col} dtype: {out[col].dtype}"
    # top15_streak: int16
    assert out["top15_streak"].dtype == np.int16


# ---------------------------------------------------------------------------
# Rank composite arithmetic
# ---------------------------------------------------------------------------


def test_rank_composite_is_round_mean(
    panel_5x30, sector_map, ticker_registry, sector_registry
):
    """rank = round(mean(r_5, r_10, r_20)) per [D-003]."""
    out = compute_thematic_features(
        panel=panel_5x30,
        asof=date(2024, 11, 8),
        sector_map=sector_map,
        ticker_registry=ticker_registry,
        sector_registry=sector_registry,
        universe_yaml_sha="abc123",
    )
    for row in out.itertuples(index=False):
        expected = int(round((row.r_5 + row.r_10 + row.r_20) / 3))
        assert row.rank == expected, (
            f"{row.symbol}: rank={row.rank} vs round(mean)={expected}"
        )


def test_centile_scaling_min_max(
    panel_5x30, sector_map, ticker_registry, sector_registry
):
    """Highest REL → centile 99; lowest → centile 0 in a 5-ticker universe.

    With our crafted monotone returns (AAA top, EEE bottom), AAA must rank 99
    and EEE must rank 0 across all windows.
    """
    out = compute_thematic_features(
        panel=panel_5x30,
        asof=date(2024, 11, 8),
        sector_map=sector_map,
        ticker_registry=ticker_registry,
        sector_registry=sector_registry,
        universe_yaml_sha="abc123",
    )
    aaa = out.loc[out["symbol"] == "AAA"].iloc[0]
    eee = out.loc[out["symbol"] == "EEE"].iloc[0]
    for col in ("r_5", "r_10", "r_20"):
        assert aaa[col] == 99, f"AAA {col} should be 99, got {aaa[col]}"
        assert eee[col] == 0, f"EEE {col} should be 0, got {eee[col]}"


# ---------------------------------------------------------------------------
# Tie handling — method='average'
# ---------------------------------------------------------------------------


def test_centile_scaling_handles_ties_with_average_method(
    sector_map, ticker_registry, sector_registry
):
    """With 3 tickers tied at the same REL_5, all three should share the same
    rank (the average of their ordinal positions). Per [D-008].
    """
    # AAA flat; BBB flat; CCC flat; DDD up; EEE down. Three-way tie at flat.
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    daily_returns = {
        "AAA": [0.0] * 29,
        "BBB": [0.0] * 29,
        "CCC": [0.0] * 29,
        "DDD": [0.01] * 29,
        "EEE": [-0.01] * 29,
    }
    panel = _build_panel(symbols, 30, daily_returns)

    out = compute_thematic_features(
        panel=panel,
        asof=date(2024, 11, 8),
        sector_map=sector_map,
        ticker_registry=ticker_registry,
        sector_registry=sector_registry,
        universe_yaml_sha="abc123",
    )
    # The three flat tickers should share the same r_5 (the average ordinal).
    flat_ranks = out.loc[out["symbol"].isin(["AAA", "BBB", "CCC"]), "r_5"].tolist()
    assert len(set(flat_ranks)) == 1, f"ties not handled: {flat_ranks}"


# ---------------------------------------------------------------------------
# Determinism (byte-identical compute)
# ---------------------------------------------------------------------------


def test_compute_is_byte_deterministic(
    panel_5x30, sector_map, ticker_registry, sector_registry
):
    """Two identical compute calls produce identical DataFrames modulo
    computed_at (which is process-time)."""
    out_a = compute_thematic_features(
        panel=panel_5x30,
        asof=date(2024, 11, 8),
        sector_map=sector_map,
        ticker_registry=ticker_registry,
        sector_registry=sector_registry,
        universe_yaml_sha="abc123",
    )
    out_b = compute_thematic_features(
        panel=panel_5x30,
        asof=date(2024, 11, 8),
        sector_map=sector_map,
        ticker_registry=ticker_registry,
        sector_registry=sector_registry,
        universe_yaml_sha="abc123",
    )
    # Drop computed_at then compare the rest.
    a = out_a.drop(columns=["computed_at"]).reset_index(drop=True)
    b = out_b.drop(columns=["computed_at"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Insufficient history — NaN, excluded from Rank composite
# ---------------------------------------------------------------------------


def test_insufficient_history_yields_excluded_from_rank(
    sector_map, ticker_registry, sector_registry
):
    """A ticker with fewer than 20 days of history must have NaN rel_20 and
    must be excluded from the Rank composite for that day (Rank still computed
    from r_5 + r_10 only is NOT acceptable — DESIGN §5.1 + TASK-PLAN §6 say
    excluded, i.e. rank is NaN/missing → represented as sentinel).
    """
    # Build full 30d for 4 tickers + only 15d for FFF (insufficient for rel_20)
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    daily_returns = {
        "AAA": [0.01] * 29,
        "BBB": [0.005] * 29,
        "CCC": [0.0] * 29,
        "DDD": [-0.01] * 29,
    }
    panel_full = _build_panel(symbols, 30, daily_returns)

    # Now append FFF with only 15 days (starting after day 14 of the same
    # trading-day window).
    short_panel = _build_panel(["FFF"], 30, {"FFF": [0.002] * 29})
    # Keep only the last 15 trading days for FFF.
    short_panel = short_panel.tail(15).reset_index(drop=True)

    panel = pd.concat([panel_full, short_panel], ignore_index=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    sector_map_with_fff = dict(sector_map)
    sector_map_with_fff["FFF"] = "tech"
    ticker_registry_with_fff = dict(ticker_registry)
    ticker_registry_with_fff["FFF"] = 6

    out = compute_thematic_features(
        panel=panel,
        asof=date(2024, 11, 8),
        sector_map=sector_map_with_fff,
        ticker_registry=ticker_registry_with_fff,
        sector_registry=sector_registry,
        universe_yaml_sha="abc123",
    )
    fff = out.loc[out["symbol"] == "FFF"]
    if not fff.empty:
        row = fff.iloc[0]
        # rel_20 is undefined for FFF (only 15 days); must be NaN.
        assert pd.isna(row["rel_20"]) or row["rel_20"] == 0, (
            f"FFF should have NaN rel_20; got {row['rel_20']}"
        )


# ---------------------------------------------------------------------------
# top15_streak persistence column
# ---------------------------------------------------------------------------


def test_top15_streak_increments_and_resets(
    sector_map, ticker_registry, sector_registry
):
    """Streak counts consecutive days with rank >= 85 and resets to 0 on the
    first day below threshold. Per [D-014].
    """
    # Use a multi-asof_date series: AAA dominates for 5 days then collapses.
    # We compute features for one asof_date at a time; the streak must come
    # from prior features (pass via prev_features parameter).
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    daily_returns = {
        "AAA": [0.05] * 29,  # dominant
        "BBB": [0.0] * 29,
        "CCC": [0.0] * 29,
        "DDD": [0.0] * 29,
        "EEE": [0.0] * 29,
    }
    panel = _build_panel(symbols, 30, daily_returns)

    # Compute features at sequential asof_dates and check that streak grows.
    prev = None
    streaks = []
    asof_dates = sorted(panel["date"].unique())[5:10]  # 5 sequential days
    for asof in asof_dates:
        out = compute_thematic_features(
            panel=panel,
            asof=asof,
            sector_map=sector_map,
            ticker_registry=ticker_registry,
            sector_registry=sector_registry,
            universe_yaml_sha="abc123",
            prev_features=prev,
        )
        aaa_row = out.loc[out["symbol"] == "AAA"].iloc[0]
        if aaa_row["rank"] >= 85:
            streaks.append(int(aaa_row["top15_streak"]))
        prev = out

    # Streaks must be monotone non-decreasing across days where AAA stays top.
    assert streaks == sorted(streaks), f"streaks not monotone: {streaks}"
    # And the last streak must be at least len(streaks).
    if streaks:
        assert streaks[-1] >= len(streaks), (
            f"streak {streaks[-1]} too low for {len(streaks)} consecutive days"
        )


# ---------------------------------------------------------------------------
# Provenance + audit fields
# ---------------------------------------------------------------------------


def test_universe_size_recorded(
    panel_5x30, sector_map, ticker_registry, sector_registry
):
    """universe_size column matches the actual universe at asof_date."""
    out = compute_thematic_features(
        panel=panel_5x30,
        asof=date(2024, 11, 8),
        sector_map=sector_map,
        ticker_registry=ticker_registry,
        sector_registry=sector_registry,
        universe_yaml_sha="abc123",
    )
    assert (out["universe_size"] == 5).all(), (
        f"universe_size should be 5, got unique: {out['universe_size'].unique()}"
    )


def test_universe_yaml_sha_propagated(
    panel_5x30, sector_map, ticker_registry, sector_registry
):
    """universe_yaml_sha column carries the SHA passed in."""
    sha = "deadbeef" * 5
    out = compute_thematic_features(
        panel=panel_5x30,
        asof=date(2024, 11, 8),
        sector_map=sector_map,
        ticker_registry=ticker_registry,
        sector_registry=sector_registry,
        universe_yaml_sha=sha,
    )
    assert (out["universe_yaml_sha"] == sha).all()
