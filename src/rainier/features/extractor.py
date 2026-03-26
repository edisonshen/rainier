"""Feature extraction: transforms AnalysisResult + OHLCV into ML-ready feature vectors.

Consumes existing analysis pipeline output (pivots, S/R levels, pin bars,
inside bars, bias) and produces a DataFrame with one row per bar and
~50 numeric features suitable for ML model input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rainier.analysis.pivots import compute_atr
from rainier.analysis.regime import RegimeDetector, compute_adx
from rainier.core.types import (
    AnalysisResult,
    Direction,
    InsideBar,
    MarketRegime,
    PinBar,
    SRLevel,
    SRRole,
)


class FeatureExtractor:
    """Transforms AnalysisResult + OHLCV DataFrame into ML-ready features.

    Design decision (eng review 2026-03-22): consumes AnalysisResult from
    the existing analysis pipeline, NOT raw OHLCV directly. This avoids
    duplicating analysis logic.
    """

    def extract(self, result: AnalysisResult, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for every bar in *df*.

        Returns a DataFrame with the same index as *df*, one column per feature.
        NaN policy: fill with meaningful defaults, then assert no NaN.
        """
        n = len(df)
        features = pd.DataFrame(index=df.index)

        # --- Price action (per-bar) ---
        features = self._add_candle_features(features, df)

        # --- ATR / volatility ---
        features = self._add_volatility_features(features, df)

        # --- Pin bar flags ---
        features = self._add_pin_bar_features(features, df, result.pin_bars, n)

        # --- Inside bar flags ---
        features = self._add_inside_bar_features(features, result.inside_bars, n)

        # --- S/R distance features ---
        features = self._add_sr_features(features, df, result.sr_levels)

        # --- Multi-TF confluence ---
        features = self._add_confluence_features(features, df, result.sr_levels)

        # --- Trend / bias ---
        features = self._add_trend_features(features, df, result)

        # --- Rolling statistics ---
        features = self._add_rolling_features(features, df)

        # --- Regime features ---
        features = self._add_regime_features(features, df)

        # NaN policy: fill with defaults, then assert clean
        features = self._fill_defaults(features)
        nan_counts = features.isna().sum()
        bad = nan_counts[nan_counts > 0]
        if len(bad) > 0:
            raise ValueError(f"NaN detected in features after fill: {bad.to_dict()}")

        return features

    # ------------------------------------------------------------------
    # Candle features
    # ------------------------------------------------------------------

    @staticmethod
    def _add_candle_features(features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        high = df["high"]
        low = df["low"]
        opn = df["open"]
        close = df["close"]
        rng = high - low

        features["body_size"] = (close - opn).abs()
        features["range"] = rng
        features["body_range_ratio"] = features["body_size"] / rng
        features["upper_wick"] = high - np.maximum(close, opn)
        features["lower_wick"] = np.minimum(close, opn) - low
        features["close_position"] = (close - low) / rng  # 0=closed at low, 1=at high
        features["is_bullish"] = (close > opn).astype(np.float64)
        return features

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    @staticmethod
    def _add_volatility_features(
        features: pd.DataFrame, df: pd.DataFrame,
    ) -> pd.DataFrame:
        atr = compute_atr(df, period=14)
        features["atr_14"] = atr
        atr_mean_20 = atr.rolling(window=20, min_periods=1).mean()
        features["atr_ratio"] = atr / atr_mean_20
        features["range_vs_atr"] = features["range"] / atr
        return features

    # ------------------------------------------------------------------
    # Pin bar features
    # ------------------------------------------------------------------

    @staticmethod
    def _add_pin_bar_features(
        features: pd.DataFrame,
        df: pd.DataFrame,
        pin_bars: list[PinBar],
        n: int,
    ) -> pd.DataFrame:
        is_pinbar = np.zeros(n, dtype=np.float64)
        wick_ratio = np.zeros(n, dtype=np.float64)
        pinbar_direction = np.zeros(n, dtype=np.float64)
        sr_distance = np.full(n, 1.0)  # default: far from any level
        sr_strength = np.zeros(n, dtype=np.float64)

        for pb in pin_bars:
            if 0 <= pb.index < n:
                is_pinbar[pb.index] = 1.0
                wick_ratio[pb.index] = pb.wick_ratio
                pinbar_direction[pb.index] = (
                    1.0 if pb.direction == Direction.LONG else -1.0
                )
                sr_distance[pb.index] = pb.sr_distance_pct
                if pb.nearest_sr is not None:
                    sr_strength[pb.index] = pb.nearest_sr.strength

        features["is_pinbar"] = is_pinbar
        features["pinbar_wick_ratio"] = wick_ratio
        features["pinbar_direction"] = pinbar_direction
        features["pinbar_sr_distance"] = sr_distance
        features["pinbar_sr_strength"] = sr_strength
        return features

    # ------------------------------------------------------------------
    # Inside bar features
    # ------------------------------------------------------------------

    @staticmethod
    def _add_inside_bar_features(
        features: pd.DataFrame,
        inside_bars: list[InsideBar],
        n: int,
    ) -> pd.DataFrame:
        is_inside = np.zeros(n, dtype=np.float64)
        compression = np.zeros(n, dtype=np.float64)

        for ib in inside_bars:
            if 0 <= ib.index < n:
                is_inside[ib.index] = 1.0
                compression[ib.index] = ib.compression_ratio

        features["is_inside_bar"] = is_inside
        features["inside_bar_compression"] = compression
        return features

    # ------------------------------------------------------------------
    # S/R distance features (per bar, relative to all levels)
    # ------------------------------------------------------------------

    @staticmethod
    def _add_sr_features(
        features: pd.DataFrame,
        df: pd.DataFrame,
        sr_levels: list[SRLevel],
    ) -> pd.DataFrame:
        close = df["close"].values
        n = len(close)

        dist_support = np.full(n, np.inf)
        dist_resistance = np.full(n, np.inf)
        nearest_sr_strength_arr = np.zeros(n, dtype=np.float64)
        nearest_sr_touches_arr = np.zeros(n, dtype=np.float64)
        levels_within_1atr = np.zeros(n, dtype=np.float64)

        atr = compute_atr(df, period=14).values

        for level in sr_levels:
            for i in range(n):
                price = level.price_at(i)
                dist = abs(close[i] - price) / close[i] if close[i] != 0 else np.inf

                if level.role == SRRole.SUPPORT and close[i] >= price:
                    if dist < dist_support[i]:
                        dist_support[i] = dist
                        nearest_sr_strength_arr[i] = level.strength
                        nearest_sr_touches_arr[i] = level.touches
                elif level.role == SRRole.RESISTANCE and close[i] <= price:
                    if dist < dist_resistance[i]:
                        dist_resistance[i] = dist
                        nearest_sr_strength_arr[i] = level.strength
                        nearest_sr_touches_arr[i] = level.touches

                if atr[i] > 0 and abs(close[i] - price) <= atr[i]:
                    levels_within_1atr[i] += 1

        # Replace inf with a large default (10% of price)
        dist_support[dist_support == np.inf] = 0.1
        dist_resistance[dist_resistance == np.inf] = 0.1

        features["dist_nearest_support"] = dist_support
        features["dist_nearest_resistance"] = dist_resistance
        features["nearest_sr_strength"] = nearest_sr_strength_arr
        features["nearest_sr_touches"] = nearest_sr_touches_arr
        features["levels_within_1atr"] = levels_within_1atr
        return features

    # ------------------------------------------------------------------
    # Multi-TF confluence
    # ------------------------------------------------------------------

    @staticmethod
    def _add_confluence_features(
        features: pd.DataFrame,
        df: pd.DataFrame,
        sr_levels: list[SRLevel],
    ) -> pd.DataFrame:
        close = df["close"].values
        n = len(close)
        confluence = np.zeros(n, dtype=np.float64)

        for i in range(n):
            threshold = close[i] * 0.005
            tfs: set[str] = set()
            for level in sr_levels:
                price = level.price_at(i)
                if abs(close[i] - price) <= threshold:
                    tf_key = level.source_tf.value if level.source_tf else "_local"
                    tfs.add(tf_key)
            confluence[i] = len(tfs)

        features["num_tf_confluence"] = confluence
        return features

    # ------------------------------------------------------------------
    # Trend features
    # ------------------------------------------------------------------

    @staticmethod
    def _add_trend_features(
        features: pd.DataFrame,
        df: pd.DataFrame,
        result: AnalysisResult,
    ) -> pd.DataFrame:
        # Higher-TF bias as numeric
        if result.bias == Direction.LONG:
            features["higher_tf_bias"] = 1.0
        elif result.bias == Direction.SHORT:
            features["higher_tf_bias"] = -1.0
        else:
            features["higher_tf_bias"] = 0.0

        # Swing high/low sequence features from pivots
        close = df["close"].values
        n = len(close)

        bars_since_swing_high = np.full(n, n, dtype=np.float64)
        bars_since_swing_low = np.full(n, n, dtype=np.float64)

        high_indices = [p.index for p in result.pivots if p.is_high and p.index < n]
        low_indices = [p.index for p in result.pivots if not p.is_high and p.index < n]

        for i in range(n):
            past_highs = [h for h in high_indices if h <= i]
            if past_highs:
                bars_since_swing_high[i] = i - max(past_highs)

            past_lows = [lo for lo in low_indices if lo <= i]
            if past_lows:
                bars_since_swing_low[i] = i - max(past_lows)

        features["bars_since_swing_high"] = bars_since_swing_high
        features["bars_since_swing_low"] = bars_since_swing_low

        # Consecutive higher highs / lower lows
        high_prices = sorted(
            [(p.index, p.price) for p in result.pivots if p.is_high], key=lambda x: x[0]
        )
        low_prices = sorted(
            [(p.index, p.price) for p in result.pivots if not p.is_high], key=lambda x: x[0]
        )

        hh_count = _count_consecutive_higher(high_prices)
        ll_count = _count_consecutive_lower(low_prices)

        features["consecutive_higher_highs"] = float(hh_count)
        features["consecutive_lower_lows"] = float(ll_count)

        # Simple trend strength: SMA slope
        sma_50 = pd.Series(close).rolling(window=min(50, n), min_periods=1).mean()
        features["sma_50_slope"] = sma_50.diff().fillna(0.0).values

        return features

    # ------------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _add_rolling_features(features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        # Returns
        returns = close.pct_change().fillna(0.0)
        features["return_1bar"] = returns
        features["return_5bar"] = close.pct_change(periods=5).fillna(0.0)

        # Volume ratio
        if "volume" in df.columns:
            avg_vol = df["volume"].rolling(window=20, min_periods=1).mean()
            features["volume_ratio"] = df["volume"] / avg_vol
        else:
            features["volume_ratio"] = 0.5

        # Rolling volatility (std of returns)
        features["rolling_volatility_20"] = returns.rolling(
            window=20, min_periods=1
        ).std().fillna(0.0)

        return features

    # ------------------------------------------------------------------
    # Regime features
    # ------------------------------------------------------------------

    @staticmethod
    def _add_regime_features(
        features: pd.DataFrame, df: pd.DataFrame,
    ) -> pd.DataFrame:
        # Continuous features
        atr = compute_atr(df, period=14)
        atr_pct = atr.rolling(
            window=min(100, len(df)), min_periods=1
        ).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        features["atr_percentile"] = atr_pct.fillna(0.5)
        features["adx"] = compute_adx(df, period=14)

        # One-hot regime classification
        detector = RegimeDetector()
        regimes = detector.detect(df)
        for regime in MarketRegime:
            col = f"regime_{regime.value}"
            features[col] = (regimes == regime).astype(np.float64)

        return features

    # ------------------------------------------------------------------
    # NaN fill
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_defaults(features: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN with meaningful defaults per column type."""
        fill_map: dict[str, float] = {
            # Boolean flags default to 0 (absent)
            "is_pinbar": 0.0,
            "is_inside_bar": 0.0,
            "is_bullish": 0.0,
            # Ratios default to neutral
            "body_range_ratio": 0.5,
            "close_position": 0.5,
            "atr_ratio": 1.0,
            "range_vs_atr": 1.0,
            "volume_ratio": 0.5,
            # Distances default to far (0.1 = 10%)
            "dist_nearest_support": 0.1,
            "dist_nearest_resistance": 0.1,
        }
        for col, default in fill_map.items():
            if col in features.columns:
                features[col] = features[col].fillna(default)

        # Remaining NaN: fill with 0.0
        features = features.fillna(0.0)
        return features


def _count_consecutive_higher(indexed_prices: list[tuple[int, float]]) -> int:
    """Count consecutive higher values from the end of the series."""
    if len(indexed_prices) < 2:
        return 0
    count = 0
    for i in range(len(indexed_prices) - 1, 0, -1):
        if indexed_prices[i][1] > indexed_prices[i - 1][1]:
            count += 1
        else:
            break
    return count


def _count_consecutive_lower(indexed_prices: list[tuple[int, float]]) -> int:
    """Count consecutive lower values from the end of the series."""
    if len(indexed_prices) < 2:
        return 0
    count = 0
    for i in range(len(indexed_prices) - 1, 0, -1):
        if indexed_prices[i][1] < indexed_prices[i - 1][1]:
            count += 1
        else:
            break
    return count
