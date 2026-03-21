"""Swing high/low (pivot) detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.core.config import PivotConfig
from quant.core.types import Pivot


def detect_pivots(df: pd.DataFrame, config: PivotConfig | None = None) -> list[Pivot]:
    """Detect swing highs and swing lows.

    A swing high at index i means: high[i] is the max of highs[i-lookback : i+lookback+1].
    A swing low at index i means: low[i] is the min of lows[i-lookback : i+lookback+1].
    """
    if config is None:
        config = PivotConfig()

    n = len(df)
    lb = config.lookback
    if n < 2 * lb + 1:
        return []

    highs = df["high"].values
    lows = df["low"].values
    timestamps = df["timestamp"].values

    pivots: list[Pivot] = []

    for i in range(lb, n - lb):
        window_highs = highs[i - lb : i + lb + 1]
        window_lows = lows[i - lb : i + lb + 1]

        # Swing high: current bar's high is strictly the max in the window
        if highs[i] == np.max(window_highs) and np.sum(window_highs == highs[i]) == 1:
            pivots.append(
                Pivot(
                    index=i,
                    price=float(highs[i]),
                    timestamp=pd.Timestamp(timestamps[i]).to_pydatetime(),
                    is_high=True,
                )
            )

        # Swing low: current bar's low is strictly the min in the window
        if lows[i] == np.min(window_lows) and np.sum(window_lows == lows[i]) == 1:
            pivots.append(
                Pivot(
                    index=i,
                    price=float(lows[i]),
                    timestamp=pd.Timestamp(timestamps[i]).to_pydatetime(),
                    is_high=False,
                )
            )

    return pivots


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()
