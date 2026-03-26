"""Market regime detection — classifies bars into trend/range/volatility states.

Uses ATR percentile, SMA slope, and ADX to determine the current market regime.
Pure analysis module — imports only from core/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rainier.analysis.pivots import compute_atr
from rainier.core.config import RegimeConfig
from rainier.core.types import MarketRegime


class RegimeDetector:
    """Classifies market regime using ATR percentile + SMA slope + ADX."""

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Classify each bar into a MarketRegime.

        Returns a Series of MarketRegime values with the same index as df.
        """
        cfg = self.config
        n = len(df)

        atr = compute_atr(df, period=cfg.atr_period)
        adx = compute_adx(df, period=cfg.adx_period)
        sma = df["close"].rolling(
            window=min(cfg.sma_period, n), min_periods=1
        ).mean()
        sma_slope = sma.diff().fillna(0.0)

        # ATR percentile rank over rolling window
        atr_pct = atr.rolling(
            window=min(cfg.atr_percentile_window, n), min_periods=1
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        atr_pct = atr_pct.fillna(0.5)

        regimes = []
        for i in range(n):
            atr_p = atr_pct.iloc[i] * 100  # 0-100 scale
            adx_val = adx.iloc[i]
            slope = sma_slope.iloc[i]

            if atr_p > cfg.high_vol_percentile and adx_val < cfg.adx_trend_threshold:
                regimes.append(MarketRegime.HIGH_VOLATILITY)
            elif adx_val >= cfg.adx_trend_threshold and slope > 0:
                regimes.append(MarketRegime.TRENDING_UP)
            elif adx_val >= cfg.adx_trend_threshold and slope <= 0:
                regimes.append(MarketRegime.TRENDING_DOWN)
            else:
                regimes.append(MarketRegime.RANGE_BOUND)

        return pd.Series(regimes, index=df.index)

    def detect_at(self, df: pd.DataFrame, bar_index: int) -> MarketRegime:
        """Classify regime at a specific bar index."""
        regimes = self.detect(df)
        return regimes.iloc[min(bar_index, len(regimes) - 1)]


# ---------------------------------------------------------------------------
# ADX computation
# ---------------------------------------------------------------------------


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index (ADX).

    ADX measures trend strength (0-100) regardless of direction.
    High ADX (>25) = strong trend, Low ADX (<20) = weak/no trend.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Directional movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    # True range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smoothed averages (Wilder's smoothing)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    smooth_plus = plus_dm.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    smooth_minus = minus_dm.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()

    # Directional indicators
    plus_di = 100 * smooth_plus / atr
    minus_di = 100 * smooth_minus / atr

    # DX and ADX
    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    dx = dx.fillna(0.0)

    adx = dx.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    adx = adx.fillna(0.0)

    return adx
