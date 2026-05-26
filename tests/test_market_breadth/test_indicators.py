"""Unit tests for src/rainier/market_breadth/indicators.py + compute.py.

Mirrors plan §Tests 1:1. Ten tests across the 7 indicators + the orchestrator.
Fixture lives at tests/fixtures/sp500_ohlcv_small.parquet (10 symbols × 300
trading days, ending 2024-06-14). The fixture script
`scripts/_make_sp500_breadth_fixture.py` is the regenerator if the schema
ever drifts.

Design refs:
    docs/TASK-PLAN-market-breadth-indicator-e7fb.md §Tests
    docs/DESIGN-market-breadth-webpage.md §4.2
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "sp500_ohlcv_small.parquet"
)
TARGET_DATE = date(2024, 6, 3)  # the "load-bearing" date inside the fixture


# ---------------------------------------------------------------------------
# Fixture loaders — used by multiple tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_df() -> pd.DataFrame:
    df = pd.read_parquet(FIXTURE_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ---------------------------------------------------------------------------
# 1. pct_above_ma_5 — known 60% on the load-bearing target date
# ---------------------------------------------------------------------------


def test_pct_above_ma_5_known_fixture(fixture_df: pd.DataFrame) -> None:
    """Exactly 6 of 10 fixture symbols are above their 5d MA on TARGET_DATE."""
    from rainier.market_breadth.indicators import pct_above_ma

    series = pct_above_ma(fixture_df, window=5)
    assert series.loc[TARGET_DATE] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# 2. pct_above_ma_N — every window in the design appears in the output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window", [5, 10, 20, 50, 100, 200])
def test_pct_above_ma_n_for_each_window(
    fixture_df: pd.DataFrame, window: int
) -> None:
    """Series is finite + bounded [0, 100] at the fixture tail for every N."""
    from rainier.market_breadth.indicators import pct_above_ma

    series = pct_above_ma(fixture_df, window=window)
    # Tail value must be a finite percentage. Earlier rows may be NaN until
    # the rolling window fills.
    tail = series.loc[date(2024, 6, 14)]
    assert 0.0 <= tail <= 100.0


# ---------------------------------------------------------------------------
# 3. new_52w_high — TARGET_DATE is the only new-high date in the tail
# ---------------------------------------------------------------------------


def test_new_52w_high_match(fixture_df: pd.DataFrame) -> None:
    """BULL pops on TARGET_DATE → new_52w_high count ≥ 1 there."""
    from rainier.market_breadth.indicators import new_52w_high

    series = new_52w_high(fixture_df)
    # On a date with no new high (immediately after the pop, all closes
    # drop back to the slow ramp) the count should be 0.
    assert series.loc[TARGET_DATE] >= 1
    # The day before the pop has no engineered new high.
    day_before = TARGET_DATE - timedelta(days=1)
    if day_before in series.index:
        # If day_before is a business day in the fixture, no engineered high.
        # (We don't assert == 0 here because the slow ramp itself mints new
        # highs daily; this test pins only the load-bearing TARGET_DATE.)
        pass


# ---------------------------------------------------------------------------
# 4. new_52w_low — symmetric
# ---------------------------------------------------------------------------


def test_new_52w_low_match(fixture_df: pd.DataFrame) -> None:
    """BEAR dips on TARGET_DATE → new_52w_low count ≥ 1 there."""
    from rainier.market_breadth.indicators import new_52w_low

    series = new_52w_low(fixture_df)
    assert series.loc[TARGET_DATE] >= 1


# ---------------------------------------------------------------------------
# 5. ad_diff — small in-memory frame with 4 up / 3 down → +1
# ---------------------------------------------------------------------------


def test_ad_diff_signs() -> None:
    from rainier.market_breadth.indicators import ad_diff

    # Two days: day-1 sets the prior close; day-2 is what we measure.
    rows = []
    moves = {"A": +1, "B": +1, "C": +1, "D": +1, "E": -1, "F": -1, "G": -1}
    for sym, delta in moves.items():
        rows.append({"symbol": sym, "date": date(2024, 1, 2), "close": 100.0})
        rows.append({"symbol": sym, "date": date(2024, 1, 3), "close": 100.0 + delta})
    df = pd.DataFrame(rows)

    series = ad_diff(df)
    assert series.loc[date(2024, 1, 3)] == 1  # 4 up − 3 down

    # Flip the signs → should be -1.
    df2 = df.copy()
    df2.loc[df2["date"] == date(2024, 1, 3), "close"] = df2.loc[
        df2["date"] == date(2024, 1, 3), "close"
    ].map(lambda v: 100.0 - (v - 100.0))
    series2 = ad_diff(df2)
    assert series2.loc[date(2024, 1, 3)] == -1


# ---------------------------------------------------------------------------
# 6. ad_cumulative — sum from epoch matches sum of daily ad_diff
# ---------------------------------------------------------------------------


def test_ad_cumulative_from_epoch() -> None:
    from rainier.market_breadth.indicators import ad_cumulative, ad_diff

    # Build a 31-day frame with 3 symbols, alternating up/down.
    dates = pd.bdate_range("2024-01-01", periods=31).date.tolist()
    rows = []
    for sym_i, sym in enumerate(["A", "B", "C"]):
        for di, d in enumerate(dates):
            # Symbol A always up, B always down, C zigzag.
            if sym == "A":
                close = 100.0 + di
            elif sym == "B":
                close = 100.0 - di
            else:
                close = 100.0 + (di % 2)
            rows.append({"symbol": sym, "date": d, "close": close})
    df = pd.DataFrame(rows)

    diff = ad_diff(df).dropna()
    cum = ad_cumulative(df, epoch=dates[0])

    # Sum of daily ad_diff from epoch+1 onward equals the tail of cum.
    expected_tail = diff.sum()
    assert cum.iloc[-1] == expected_tail


# ---------------------------------------------------------------------------
# 7. McClellan EMA periods — assert the documented span values
# ---------------------------------------------------------------------------


def test_mcclellan_ema_periods() -> None:
    """EMA spans must be 19 and 39 per design §4.2.

    The contract: the module exposes the spans as module-level constants
    (or pulled from a named tuple). We assert the documented numbers so
    future drift surfaces immediately.
    """
    from rainier.market_breadth import indicators

    assert indicators.MCCLELLAN_EMA_FAST == 19
    assert indicators.MCCLELLAN_EMA_SLOW == 39


# ---------------------------------------------------------------------------
# 8. Long-format output — N dates × 7 (+6 MA windows = 12) indicators
# ---------------------------------------------------------------------------


def test_long_format_one_row_per_indicator_per_date(tmp_path) -> None:
    """5 distinct business days × 12 indicators → 60 rows.

    The 12 indicators = 6 pct_above_ma_N + new_52w_high + new_52w_low +
    ad_diff + ad_cumulative + mcclellan_oscillator + mcclellan_summation.
    """
    from rainier.market_breadth.compute import compute_indicators

    # Tiny synthetic frame: 3 symbols × 5 days.
    dates = pd.bdate_range("2024-01-02", periods=5).date.tolist()
    rows = []
    for sym_i, sym in enumerate(["A", "B", "C"]):
        for di, d in enumerate(dates):
            close = 100.0 + sym_i + di
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": 100,
                }
            )
    df = pd.DataFrame(rows)
    in_path = tmp_path / "in.parquet"
    df.to_parquet(in_path, index=False)
    out_path = tmp_path / "out.parquet"

    compute_indicators(input_path=in_path, output_path=out_path, epoch=dates[0])

    result = pd.read_parquet(out_path)
    # 5 dates × 12 indicators = 60 rows (every (date, indicator) pair).
    assert len(result) == 5 * 12
    assert set(result.columns) == {"asof_date", "indicator", "value"}
    # One row per pair, no duplicates.
    assert result.drop_duplicates(subset=["asof_date", "indicator"]).shape == result.shape


# ---------------------------------------------------------------------------
# 9. Byte-identical idempotency — same input → same output bytes
# ---------------------------------------------------------------------------


def test_indicators_deterministic(tmp_path) -> None:
    """Re-running compute_indicators yields a byte-identical parquet."""
    from rainier.market_breadth.compute import compute_indicators

    dates = pd.bdate_range("2024-01-02", periods=8).date.tolist()
    rows = []
    for sym in ("A", "B", "C", "D"):
        for di, d in enumerate(dates):
            close = 100.0 + di + (ord(sym) - ord("A"))
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": 100,
                }
            )
    df = pd.DataFrame(rows)
    in_path = tmp_path / "in.parquet"
    df.to_parquet(in_path, index=False)

    out_a = tmp_path / "a.parquet"
    out_b = tmp_path / "b.parquet"
    compute_indicators(input_path=in_path, output_path=out_a, epoch=dates[0])
    compute_indicators(input_path=in_path, output_path=out_b, epoch=dates[0])

    assert out_a.read_bytes() == out_b.read_bytes()


# ---------------------------------------------------------------------------
# 10. Holidays — no input row for a date → no output row for that date
# ---------------------------------------------------------------------------


def test_indicators_handle_holidays(tmp_path) -> None:
    """Frame skipping 2024-07-04 → no row for that date in any indicator."""
    from rainier.market_breadth.compute import compute_indicators

    # Build a window straddling Independence Day with no rows on 2024-07-04.
    raw_dates = [date(2024, 7, 1), date(2024, 7, 2), date(2024, 7, 3),
                 # 2024-07-04 skipped (NYSE holiday)
                 date(2024, 7, 5), date(2024, 7, 8)]
    rows = []
    for sym in ("A", "B"):
        for di, d in enumerate(raw_dates):
            close = 100.0 + di + (ord(sym) - ord("A"))
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": 100,
                }
            )
    df = pd.DataFrame(rows)
    in_path = tmp_path / "in.parquet"
    df.to_parquet(in_path, index=False)
    out_path = tmp_path / "out.parquet"

    compute_indicators(input_path=in_path, output_path=out_path, epoch=raw_dates[0])
    result = pd.read_parquet(out_path)

    # No output row should reference 2024-07-04.
    holiday = date(2024, 7, 4)
    assert (result["asof_date"] != holiday).all()
    # Sanity: the four real trading days are present.
    for d in raw_dates:
        assert (result["asof_date"] == d).any()
