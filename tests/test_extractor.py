"""Tests for feature extraction."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from rainier.core.types import (
    AnalysisResult,
    Candle,
    Direction,
    InsideBar,
    PinBar,
    Pivot,
    SRLevel,
    SRRole,
    SRType,
    Timeframe,
)
from rainier.features.extractor import FeatureExtractor


def _make_df(n: int = 50) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with a mild uptrend."""
    rows = []
    price = 100.0
    for i in range(n):
        o = price
        h = price + 2.0
        lo = price - 1.0
        c = price + 0.5
        rows.append({
            "timestamp": datetime(2025, 1, 1) + timedelta(hours=i),
            "open": o, "high": h, "low": lo, "close": c,
            "volume": 1000.0 + i * 10,
        })
        price = c
    return pd.DataFrame(rows)


def _make_analysis(n: int = 50) -> AnalysisResult:
    """Minimal AnalysisResult with a pin bar, inside bar, pivots, and S/R."""
    sr = SRLevel(
        price=100.0, sr_type=SRType.HORIZONTAL, role=SRRole.SUPPORT,
        strength=0.8, touches=3, source_tf=Timeframe.H1,
    )
    sr2 = SRLevel(
        price=120.0, sr_type=SRType.HORIZONTAL, role=SRRole.RESISTANCE,
        strength=0.7, touches=2, source_tf=Timeframe.H4,
    )
    candle = Candle(
        timestamp=datetime(2025, 1, 1, 10), open=104.0, high=105.0,
        low=99.5, close=104.5, volume=2000.0, symbol="NQ", timeframe=Timeframe.H1,
    )
    pin_bar = PinBar(
        candle=candle, index=10, direction=Direction.LONG,
        wick_ratio=3.5, nearest_sr=sr, sr_distance_pct=0.005,
    )
    mother = Candle(
        timestamp=datetime(2025, 1, 1, 20), open=110.0, high=115.0,
        low=108.0, close=114.0, volume=1500.0, symbol="NQ", timeframe=Timeframe.H1,
    )
    inside = Candle(
        timestamp=datetime(2025, 1, 1, 21), open=112.0, high=113.0,
        low=109.0, close=112.5, volume=800.0, symbol="NQ", timeframe=Timeframe.H1,
    )
    inside_bar = InsideBar(
        candle=inside, index=21, mother_candle=mother,
        mother_index=20, compression_ratio=0.57,
    )
    pivots = [
        Pivot(index=5, price=99.0, timestamp=datetime(2025, 1, 1, 5), is_high=False),
        Pivot(index=15, price=112.0, timestamp=datetime(2025, 1, 1, 15), is_high=True),
        Pivot(index=25, price=105.0, timestamp=datetime(2025, 1, 2, 1), is_high=False),
        Pivot(index=35, price=118.0, timestamp=datetime(2025, 1, 2, 11), is_high=True),
    ]

    return AnalysisResult(
        symbol="NQ", timeframe=Timeframe.H1,
        sr_levels=[sr, sr2],
        pin_bars=[pin_bar],
        inside_bars=[inside_bar],
        pivots=pivots,
        bias=Direction.LONG,
    )


class TestFeatureExtractor:
    def test_output_shape(self):
        df = _make_df(50)
        result = _make_analysis(50)
        features = FeatureExtractor().extract(result, df)
        assert len(features) == len(df)
        assert features.shape[1] > 20  # at least 20 feature columns

    def test_no_nan(self):
        df = _make_df(50)
        result = _make_analysis(50)
        features = FeatureExtractor().extract(result, df)
        assert features.isna().sum().sum() == 0

    def test_candle_features_present(self):
        df = _make_df(30)
        result = _make_analysis(30)
        features = FeatureExtractor().extract(result, df)
        expected = [
            "body_size", "range", "body_range_ratio", "upper_wick",
            "lower_wick", "close_position", "is_bullish",
        ]
        for col in expected:
            assert col in features.columns, f"Missing candle feature: {col}"

    def test_pin_bar_flags(self):
        df = _make_df(50)
        result = _make_analysis(50)
        features = FeatureExtractor().extract(result, df)
        # Pin bar at index 10
        assert features.loc[10, "is_pinbar"] == 1.0
        assert features.loc[10, "pinbar_wick_ratio"] == 3.5
        assert features.loc[10, "pinbar_direction"] == 1.0  # LONG
        # Non-pin bar
        assert features.loc[0, "is_pinbar"] == 0.0

    def test_inside_bar_flags(self):
        df = _make_df(50)
        result = _make_analysis(50)
        features = FeatureExtractor().extract(result, df)
        assert features.loc[21, "is_inside_bar"] == 1.0
        assert features.loc[21, "inside_bar_compression"] == pytest.approx(0.57)
        assert features.loc[0, "is_inside_bar"] == 0.0

    def test_sr_distances(self):
        df = _make_df(50)
        result = _make_analysis(50)
        features = FeatureExtractor().extract(result, df)
        assert "dist_nearest_support" in features.columns
        assert "dist_nearest_resistance" in features.columns
        # All distances should be non-negative
        assert (features["dist_nearest_support"] >= 0).all()
        assert (features["dist_nearest_resistance"] >= 0).all()

    def test_trend_features(self):
        df = _make_df(50)
        result = _make_analysis(50)
        features = FeatureExtractor().extract(result, df)
        # Bias is LONG → 1.0
        assert features.loc[0, "higher_tf_bias"] == 1.0
        assert "bars_since_swing_high" in features.columns
        assert "consecutive_higher_highs" in features.columns

    def test_volatility_features(self):
        df = _make_df(50)
        result = _make_analysis(50)
        features = FeatureExtractor().extract(result, df)
        assert "atr_14" in features.columns
        assert "atr_ratio" in features.columns
        # ATR should be positive for non-flat data
        assert (features["atr_14"].iloc[14:] > 0).all()

    def test_confluence_feature(self):
        df = _make_df(50)
        result = _make_analysis(50)
        features = FeatureExtractor().extract(result, df)
        assert "num_tf_confluence" in features.columns

    def test_rolling_features(self):
        df = _make_df(50)
        result = _make_analysis(50)
        features = FeatureExtractor().extract(result, df)
        assert "return_1bar" in features.columns
        assert "volume_ratio" in features.columns
        assert "rolling_volatility_20" in features.columns

    def test_empty_analysis(self):
        """Extraction with no pin bars, no inside bars, no pivots."""
        df = _make_df(30)
        result = AnalysisResult(
            symbol="NQ", timeframe=Timeframe.H1,
            sr_levels=[], pin_bars=[], inside_bars=[], pivots=[], bias=None,
        )
        features = FeatureExtractor().extract(result, df)
        assert len(features) == 30
        assert features.isna().sum().sum() == 0
        assert (features["is_pinbar"] == 0.0).all()
        assert features.loc[0, "higher_tf_bias"] == 0.0
