"""Price/volume indicators shared by analysis, signals and charts."""

from __future__ import annotations

import pandas as pd


def rolling_vwap(df: pd.DataFrame, period: int = 55) -> pd.Series:
    """Rolling VWAP of ``hlc3`` over the last ``period`` bars (TradingView ``VMA55``).

    ``sum(hlc3 * volume, n) / sum(volume, n)``; NaN until ``period`` bars exist or when the
    window has no volume.
    """
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].astype(float)
    pv = (hlc3 * vol).rolling(period).sum()
    v = vol.rolling(period).sum()
    return pv / v.where(v > 0)
