"""Tests for HMM regime detector."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from rainier.core.types import MarketRegime
from rainier.ml.regime import HMMRegimeDetector


def _make_trending_up(n: int = 300) -> pd.DataFrame:
    """Synthetic uptrend data."""
    price = 100.0
    rows = []
    for i in range(n):
        move = np.random.normal(0.3, 0.5)  # positive drift
        o = price
        h = price + abs(move) + 0.5
        lo = price - 0.3
        c = price + move
        rows.append({
            "timestamp": pd.Timestamp("2023-01-01") + timedelta(days=i),
            "open": o, "high": h, "low": lo, "close": max(c, lo + 0.01),
            "volume": float(1_000_000 + np.random.randint(-100000, 100000)),
        })
        price = max(c, lo + 0.01)
    return pd.DataFrame(rows)


def _make_mixed_regimes(n: int = 600) -> pd.DataFrame:
    """Synthetic data with regime changes: uptrend → sideways → downtrend."""
    rows = []
    price = 100.0
    segment = n // 3

    for i in range(n):
        if i < segment:
            # Uptrend
            move = np.random.normal(0.4, 0.5)
        elif i < 2 * segment:
            # Sideways
            move = np.random.normal(0.0, 0.3)
        else:
            # Downtrend
            move = np.random.normal(-0.4, 0.5)

        o = price
        h = price + abs(np.random.normal(0, 1))
        lo = price - abs(np.random.normal(0, 1))
        c = price + move
        c = max(c, lo + 0.01)
        rows.append({
            "timestamp": pd.Timestamp("2023-01-01") + timedelta(days=i),
            "open": o, "high": h, "low": lo, "close": c,
            "volume": float(1_000_000),
        })
        price = c

    return pd.DataFrame(rows)


class TestHMMRegimeDetector:
    def test_fit_predict_returns_series(self):
        df = _make_trending_up(200)
        detector = HMMRegimeDetector(n_states=3)
        regimes = detector.fit_predict(df)
        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(df)

    def test_all_regimes_are_valid(self):
        df = _make_mixed_regimes(600)
        detector = HMMRegimeDetector(n_states=3)
        regimes = detector.fit_predict(df)
        valid = set(MarketRegime)
        for r in regimes:
            assert r in valid

    def test_regime_stability(self):
        """Regimes should not flip every single bar."""
        np.random.seed(123)
        df = _make_mixed_regimes(600)
        detector = HMMRegimeDetector(n_states=3, random_state=123)
        regimes = detector.fit_predict(df)

        # Count regime changes — shouldn't change on every bar
        changes = sum(1 for i in range(1, len(regimes)) if regimes.iloc[i] != regimes.iloc[i-1])
        # At minimum, should have fewer transitions than bars
        assert changes < len(regimes) * 0.8, (
            f"Too many regime changes: {changes}/{len(regimes)} "
            f"({changes/len(regimes):.0%} of bars)"
        )

    def test_summary_has_expected_keys(self):
        df = _make_trending_up(200)
        detector = HMMRegimeDetector(n_states=3)
        regimes = detector.fit_predict(df)
        summary = detector.regime_summary(regimes)
        assert "distribution" in summary
        assert "avg_duration" in summary
        assert "total_bars" in summary
        assert summary["total_bars"] == len(df)

    def test_save_load_roundtrip(self, tmp_path):
        df = _make_trending_up(200)
        detector = HMMRegimeDetector(n_states=3)
        detector.fit(df)
        regimes_before = detector.predict(df)

        model_path = tmp_path / "hmm_model.pkl"
        detector.save(model_path)

        detector2 = HMMRegimeDetector()
        detector2.load(model_path)
        regimes_after = detector2.predict(df)

        assert (regimes_before == regimes_after).all()

    def test_mixed_regimes_detects_multiple_states(self):
        """Mixed data should produce at least 2 distinct regimes."""
        np.random.seed(42)
        df = _make_mixed_regimes(600)
        detector = HMMRegimeDetector(n_states=3, random_state=42)
        regimes = detector.fit_predict(df)

        unique_regimes = regimes.unique()
        assert len(unique_regimes) >= 2, (
            f"Expected at least 2 regimes, got {len(unique_regimes)}: {unique_regimes}"
        )
