"""Directional bias determination using higher-timeframe structure."""

from __future__ import annotations

import pandas as pd

from rainier.core.types import Direction, Pivot


def determine_bias(
    df: pd.DataFrame,
    pivots: list[Pivot],
) -> Direction | None:
    """Determine directional bias from price structure.

    Rules:
    - Higher highs + higher lows → LONG bias
    - Lower highs + lower lows → SHORT bias
    - Mixed → None (no clear bias)

    Also considers: price relative to recent pivots, 50-bar SMA slope.
    """
    if len(pivots) < 4:
        return None

    # Get recent swing highs and lows (last 4 of each)
    swing_highs = sorted([p for p in pivots if p.is_high], key=lambda p: p.index)[-4:]
    swing_lows = sorted([p for p in pivots if not p.is_high], key=lambda p: p.index)[-4:]

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    # Check for higher highs / lower highs
    hh_count = sum(
        1 for i in range(1, len(swing_highs)) if swing_highs[i].price > swing_highs[i - 1].price
    )
    lh_count = sum(
        1 for i in range(1, len(swing_highs)) if swing_highs[i].price < swing_highs[i - 1].price
    )

    # Check for higher lows / lower lows
    hl_count = sum(
        1 for i in range(1, len(swing_lows)) if swing_lows[i].price > swing_lows[i - 1].price
    )
    ll_count = sum(
        1 for i in range(1, len(swing_lows)) if swing_lows[i].price < swing_lows[i - 1].price
    )

    # SMA slope as tiebreaker
    sma_bias = _sma_bias(df)

    bullish_score = hh_count + hl_count
    bearish_score = lh_count + ll_count

    if bullish_score > bearish_score:
        return Direction.LONG
    elif bearish_score > bullish_score:
        return Direction.SHORT
    else:
        return sma_bias


def _sma_bias(df: pd.DataFrame, period: int = 50) -> Direction | None:
    """Use slope of 50-bar SMA as a tiebreaker."""
    if len(df) < period + 10:
        return None

    sma = df["close"].rolling(window=period).mean()
    recent_slope = sma.iloc[-1] - sma.iloc[-10]

    if recent_slope > 0:
        return Direction.LONG
    elif recent_slope < 0:
        return Direction.SHORT
    return None
