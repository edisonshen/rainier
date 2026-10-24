"""Tests for regime detection, filtering, and feature extraction."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from rainier.analysis.regime import RegimeDetector, compute_adx
from rainier.core.types import MarketRegime, Timeframe

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_trending_up(n: int = 200) -> pd.DataFrame:
    """Strong uptrend with steadily rising prices."""
    base = datetime(2025, 1, 1)
    rows = []
    price = 100.0
    for i in range(n):
        move = 1.5 + np.random.uniform(0, 0.5)
        o = price
        h = price + move + 0.5
        low = price - 0.3
        c = price + move
        rows.append({
            "timestamp": base + timedelta(hours=i),
            "open": o, "high": h, "low": low, "close": c,
            "volume": 1000.0,
        })
        price = c
    return pd.DataFrame(rows)


def _make_trending_down(n: int = 200) -> pd.DataFrame:
    """Strong downtrend with steadily falling prices."""
    base = datetime(2025, 1, 1)
    rows = []
    price = 500.0
    for i in range(n):
        move = 1.5 + np.random.uniform(0, 0.5)
        o = price
        h = price + 0.3
        low = price - move - 0.5
        c = price - move
        rows.append({
            "timestamp": base + timedelta(hours=i),
            "open": o, "high": h, "low": low, "close": c,
            "volume": 1000.0,
        })
        price = c
    return pd.DataFrame(rows)


def _make_range_bound(n: int = 200) -> pd.DataFrame:
    """Sideways choppy market oscillating around 100."""
    base = datetime(2025, 1, 1)
    rows = []
    price = 100.0
    for i in range(n):
        move = np.sin(i * 0.3) * 0.5
        o = price
        h = price + 0.5
        low = price - 0.5
        c = price + move
        rows.append({
            "timestamp": base + timedelta(hours=i),
            "open": o, "high": h, "low": low, "close": c,
            "volume": 1000.0,
        })
        price = 100.0 + np.sin(i * 0.3) * 2  # mean-revert to 100
    return pd.DataFrame(rows)


def _make_high_volatility(n: int = 200) -> pd.DataFrame:
    """Expanding range with no clear direction."""
    base = datetime(2025, 1, 1)
    rows = []
    price = 100.0
    for i in range(n):
        # Volatility expands over time, direction alternates
        vol = 0.5 + (i / n) * 5.0
        move = vol * (1 if i % 2 == 0 else -1)
        o = price
        h = price + abs(move) + vol
        low = price - abs(move) - vol
        c = price + move * 0.1  # close near open (no trend)
        rows.append({
            "timestamp": base + timedelta(hours=i),
            "open": o, "high": h, "low": low, "close": c,
            "volume": 1000.0,
        })
        price = c
    return pd.DataFrame(rows)


class TestADX:
    def test_adx_returns_series(self):
        df = _make_trending_up(100)
        adx = compute_adx(df, period=14)
        assert isinstance(adx, pd.Series)
        assert len(adx) == len(df)

    def test_adx_range_0_to_100(self):
        df = _make_trending_up(200)
        adx = compute_adx(df, period=14)
        assert adx.min() >= 0
        assert adx.max() <= 100

    def test_adx_trending_higher_than_range(self):
        trend_df = _make_trending_up(200)
        range_df = _make_range_bound(200)
        adx_trend = compute_adx(trend_df).iloc[-1]
        adx_range = compute_adx(range_df).iloc[-1]
        assert adx_trend > adx_range


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------


class TestRegimeDetector:
    def test_detect_returns_series(self):
        df = _make_trending_up(100)
        detector = RegimeDetector()
        regimes = detector.detect(df)
        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(df)

    def test_all_values_are_market_regime(self):
        df = _make_trending_up(100)
        detector = RegimeDetector()
        regimes = detector.detect(df)
        for r in regimes:
            assert isinstance(r, MarketRegime)

    def test_trending_up_detected(self):
        df = _make_trending_up(200)
        detector = RegimeDetector()
        regimes = detector.detect(df)
        # Last bars should be trending up
        last_regimes = regimes.iloc[-20:]
        trending_up_count = (last_regimes == MarketRegime.TRENDING_UP).sum()
        assert trending_up_count > 10, (
            f"Expected mostly TRENDING_UP in uptrend tail, "
            f"got {trending_up_count}/20"
        )

    def test_trending_down_detected(self):
        df = _make_trending_down(200)
        detector = RegimeDetector()
        regimes = detector.detect(df)
        last_regimes = regimes.iloc[-20:]
        trending_down_count = (
            last_regimes == MarketRegime.TRENDING_DOWN
        ).sum()
        assert trending_down_count > 10, (
            f"Expected mostly TRENDING_DOWN in downtrend tail, "
            f"got {trending_down_count}/20"
        )

    def test_detect_at_returns_single_regime(self):
        df = _make_trending_up(100)
        detector = RegimeDetector()
        regime = detector.detect_at(df, len(df) - 1)
        assert isinstance(regime, MarketRegime)


# ---------------------------------------------------------------------------
# Regime filter
# ---------------------------------------------------------------------------


class TestRegimeFeatures:
    def test_extractor_includes_regime_columns(self):
        from rainier.analysis.analyzer import analyze
        from rainier.features.extractor import FeatureExtractor

        df = _make_trending_up(100)
        result = analyze(df, "TEST", Timeframe.H1)
        extractor = FeatureExtractor()
        features = extractor.extract(result, df)

        # Check regime columns exist
        assert "atr_percentile" in features.columns
        assert "adx" in features.columns
        assert "regime_trending_up" in features.columns
        assert "regime_trending_down" in features.columns
        assert "regime_range_bound" in features.columns
        assert "regime_high_volatility" in features.columns

    def test_regime_one_hot_sums_to_one(self):
        from rainier.analysis.analyzer import analyze
        from rainier.features.extractor import FeatureExtractor

        df = _make_trending_up(100)
        result = analyze(df, "TEST", Timeframe.H1)
        extractor = FeatureExtractor()
        features = extractor.extract(result, df)

        regime_cols = [
            "regime_trending_up", "regime_trending_down",
            "regime_range_bound", "regime_high_volatility",
        ]
        row_sums = features[regime_cols].sum(axis=1)
        assert (row_sums == 1.0).all(), "Each bar should be exactly one regime"

    def test_no_nan_in_regime_features(self):
        from rainier.analysis.analyzer import analyze
        from rainier.features.extractor import FeatureExtractor

        df = _make_trending_up(100)
        result = analyze(df, "TEST", Timeframe.H1)
        extractor = FeatureExtractor()
        features = extractor.extract(result, df)

        assert features["atr_percentile"].isna().sum() == 0
        assert features["adx"].isna().sum() == 0
