"""Market-breadth indicator math — 7 pure functions, one per indicator.

Inputs
------
All functions take a long-format DataFrame with columns
``symbol, date, [open, high, low,] close`` (extras tolerated). One row per
(symbol, date). The frame is what `data/cache/sp500_universe.parquet` ships
from ``ohlcv_backfill.py``.

Outputs
-------
Each function returns a ``pd.Series`` indexed by ``date`` (the breadth
universe-level value for that day). Earliest dates may be NaN where the
rolling lookback isn't yet filled — caller drops or fills as needed.

Indicators
----------
- ``pct_above_ma(df, window=N)`` — % of symbols with close > SMA_N(close)
  on date T. Returns a Series of percentages in [0, 100].
- ``new_52w_high(df)`` — count of symbols where high_T == max(high over
  trailing 252 trading rows including T). When `high` is absent, falls
  back to `close`.
- ``new_52w_low(df)`` — symmetric: count where low_T == min(low over
  trailing 252).
- ``ad_diff(df)`` — count(close_T > close_{T-1}) − count(close_T <
  close_{T-1}). Unchanged closes contribute 0 to both sides.
- ``ad_cumulative(df, epoch)`` — cumulative sum of ``ad_diff`` starting
  the first business day strictly AFTER ``epoch`` (the epoch row itself is
  NaN since A/D needs a prior close).
- ``mcclellan_oscillator(df)`` — ``EMA_19(ad_diff) − EMA_39(ad_diff)``.
  Pandas ``.ewm(span=N, adjust=False).mean()`` — standard recursive EMA
  matching breadth.app's published methodology.
- ``mcclellan_summation(df, epoch)`` — cumulative sum of
  ``mcclellan_oscillator`` from ``epoch`` (same epoch semantics as
  ``ad_cumulative``).

Determinism
-----------
All functions are pure (no I/O, no global state, no time.time). Same input
DataFrame → same output Series.

Design refs
-----------
    docs/DESIGN-market-breadth-webpage.md §4.2
    docs/TASK-PLAN-market-breadth-indicator-e7fb.md §Acceptance
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Module-level constants — load-bearing for test_mcclellan_ema_periods
# ---------------------------------------------------------------------------


MCCLELLAN_EMA_FAST = 19
MCCLELLAN_EMA_SLOW = 39
TRADING_DAYS_PER_YEAR = 252  # 52w-high / 52w-low lookback

# Set of breadth indicator names emitted by the long-format orchestrator.
# Kept here (next to the math) so the orchestrator can pull this enum
# without re-typing it.
MA_WINDOWS = (5, 10, 20, 50, 100, 200)


# ---------------------------------------------------------------------------
# Internal helpers — pivot long→wide once per call, reused across math
# ---------------------------------------------------------------------------


def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Return a (date × symbol) wide frame of ``value_col``.

    Long-format input has one row per (symbol, date). pivot() gives us
    the wide shape the rolling math wants. Holiday dates (no rows in
    input) stay absent — output index is the set of dates seen in input.
    """
    wide = df.pivot(index="date", columns="symbol", values=value_col).sort_index()
    return wide


# ---------------------------------------------------------------------------
# 1. pct_above_ma
# ---------------------------------------------------------------------------


def pct_above_ma(df: pd.DataFrame, window: int) -> pd.Series:
    """``count(close_T > SMA_N(close)) / universe_size_T × 100``.

    SMA is the simple moving average over the past ``window`` rows of the
    per-symbol close series (pandas ``.rolling(window).mean()``).

    On a given date T, the denominator is the number of symbols with a
    finite close AND a finite SMA — i.e., symbols that have at least
    ``window`` rows of history. Symbols still inside the warm-up window
    are excluded from both numerator and denominator.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")

    closes = _pivot(df, "close")
    sma = closes.rolling(window).mean()

    # `above` is True where close > sma AND both are finite.
    above = (closes > sma) & sma.notna() & closes.notna()
    # `defined` counts symbols where the SMA is computable for the date.
    defined = (sma.notna() & closes.notna()).sum(axis=1)

    pct = (above.sum(axis=1) / defined) * 100.0
    pct.name = f"pct_above_ma_{window}"
    return pct


# ---------------------------------------------------------------------------
# 2. new_52w_high — count of symbols at a new 252d high on day T
# ---------------------------------------------------------------------------


def new_52w_high(df: pd.DataFrame) -> pd.Series:
    """Count of symbols where ``high_T == max(high over trailing 252 rows)``.

    Falls back to ``close`` if the frame lacks a ``high`` column (e.g., a
    bare close-only test fixture).

    The "==" comparison is a strict tie at the per-row level: when today's
    high equals the prior 252-row max, we count it. NaN highs (warm-up
    period) are excluded.

    Warm-up caveat
    --------------
    ``min_periods=1`` means the rolling max is computed over whatever
    history is available, not strictly 252 rows. Until each symbol has
    252 trading days behind it, the "max-so-far" registers a "new high"
    on most rows (any within-history-so-far high counts). For an S&P 500
    universe loaded from a single snapshot, all symbols share the same
    data-start, so the unreliable window is the first ~252 trading days
    of the input. Downstream renderers should gate to dates ≥
    ``data_start + 252 trading days`` if presenting these series on their
    own. The default ``epoch=2020-01-01`` for ad_cumulative /
    mcclellan_summation already sits well past this warm-up boundary, so
    the chart anchors are safe.
    """
    col = "high" if "high" in df.columns else "close"
    highs = _pivot(df, col)
    rolling_max = highs.rolling(TRADING_DAYS_PER_YEAR, min_periods=1).max()
    at_high = (highs == rolling_max) & highs.notna()
    series = at_high.sum(axis=1).astype("int64")
    series.name = "new_52w_high"
    return series


# ---------------------------------------------------------------------------
# 3. new_52w_low — symmetric
# ---------------------------------------------------------------------------


def new_52w_low(df: pd.DataFrame) -> pd.Series:
    """Symmetric to :func:`new_52w_high` — same warm-up caveat applies."""
    col = "low" if "low" in df.columns else "close"
    lows = _pivot(df, col)
    rolling_min = lows.rolling(TRADING_DAYS_PER_YEAR, min_periods=1).min()
    at_low = (lows == rolling_min) & lows.notna()
    series = at_low.sum(axis=1).astype("int64")
    series.name = "new_52w_low"
    return series


# ---------------------------------------------------------------------------
# 4. ad_diff — advances minus declines on day T (close_T vs close_{T-1})
# ---------------------------------------------------------------------------


def ad_diff(df: pd.DataFrame) -> pd.Series:
    """``count(close_T > close_{T-1}) − count(close_T < close_{T-1})``.

    Symbols with unchanged close (delta == 0) contribute 0. The earliest
    date in the frame has no prior close → NaN for that row.
    """
    closes = _pivot(df, "close")
    delta = closes - closes.shift(1)
    advances = (delta > 0).sum(axis=1)
    declines = (delta < 0).sum(axis=1)
    # NaN-out the row where no symbol has a prior close.
    has_prior = delta.notna().any(axis=1)
    series = (advances - declines).astype("float64").where(has_prior)
    series.name = "ad_diff"
    return series


# ---------------------------------------------------------------------------
# 5. ad_cumulative — cumulative ad_diff from epoch+1 onward
# ---------------------------------------------------------------------------


def ad_cumulative(df: pd.DataFrame, epoch: date) -> pd.Series:
    """Cumulative ``ad_diff`` starting on the first row STRICTLY AFTER ``epoch``.

    Rows ≤ epoch are NOT included in the cumulative sum — they sit at NaN.
    The first post-epoch row's value equals that day's ``ad_diff``.
    """
    diff = ad_diff(df)
    # Filter to rows strictly > epoch. We compute cumsum on the kept slice
    # then reindex back to the full frame so dates ≤ epoch are NaN.
    kept = diff[diff.index > epoch]
    cum = kept.cumsum()
    series = cum.reindex(diff.index)
    series.name = "ad_cumulative"
    return series


# ---------------------------------------------------------------------------
# 6. mcclellan_oscillator — EMA_19(ad_diff) − EMA_39(ad_diff)
# ---------------------------------------------------------------------------


def mcclellan_oscillator(df: pd.DataFrame) -> pd.Series:
    """``EMA_{19}(ad_diff) − EMA_{39}(ad_diff)`` over the full history.

    Uses pandas' recursive EMA (``adjust=False``): each new point is
    ``alpha * x_T + (1 − alpha) * EMA_{T−1}`` with ``alpha = 2 / (span+1)``.
    This matches breadth.app's published methodology and is what most
    technical-analysis sources mean by "McClellan Oscillator".
    """
    diff = ad_diff(df)
    fast = diff.ewm(span=MCCLELLAN_EMA_FAST, adjust=False).mean()
    slow = diff.ewm(span=MCCLELLAN_EMA_SLOW, adjust=False).mean()
    series = fast - slow
    series.name = "mcclellan_oscillator"
    return series


# ---------------------------------------------------------------------------
# 7. mcclellan_summation — cumulative McClellan oscillator from epoch
# ---------------------------------------------------------------------------


def mcclellan_summation(df: pd.DataFrame, epoch: date) -> pd.Series:
    osc = mcclellan_oscillator(df)
    kept = osc[osc.index > epoch]
    cum = kept.cumsum()
    series = cum.reindex(osc.index)
    series.name = "mcclellan_summation"
    return series


# ---------------------------------------------------------------------------
# Convenience — enumerate every indicator the orchestrator will write
# ---------------------------------------------------------------------------


def indicator_names() -> Iterable[str]:
    """Stable enumeration of the 12 indicator names emitted into the long
    parquet (6 MA windows + 6 derived). Used by the orchestrator to lay
    out output rows and by callers that need to validate the
    ``indicator`` column.
    """
    for w in MA_WINDOWS:
        yield f"pct_above_ma_{w}"
    yield "new_52w_high"
    yield "new_52w_low"
    yield "ad_diff"
    yield "ad_cumulative"
    yield "mcclellan_oscillator"
    yield "mcclellan_summation"
