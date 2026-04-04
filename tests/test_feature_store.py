"""Tests for ML feature store."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from rainier.ml.feature_store import (
    LABEL_HORIZONS,
    LABEL_THRESHOLD,
    compute_forward_labels,
    export_symbol_features,
    validate_parquet,
)


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    """Synthetic daily OHLCV with a mild uptrend."""
    rows = []
    price = 100.0
    for i in range(n):
        o = price
        h = price + 2.0
        lo = price - 1.0
        c = price + 0.5
        rows.append({
            "timestamp": pd.Timestamp("2023-01-01", tz="UTC") + timedelta(days=i),
            "open": o, "high": h, "low": lo, "close": c,
            "volume": float(1_000_000 + i * 1000),
        })
        price = c
    return pd.DataFrame(rows)


class TestComputeForwardLabels:
    def test_produces_all_horizons(self):
        df = _make_ohlcv(200)
        labels = compute_forward_labels(df)
        for h in LABEL_HORIZONS:
            assert f"fwd_return_{h}d" in labels.columns
            assert f"label_{h}d" in labels.columns

    def test_last_bars_are_nan(self):
        df = _make_ohlcv(200)
        labels = compute_forward_labels(df)
        # Last N bars should be NaN for horizon N
        for h in LABEL_HORIZONS:
            assert labels[f"fwd_return_{h}d"].iloc[-1] != labels[f"fwd_return_{h}d"].iloc[-1]  # NaN

    def test_positive_return_gets_label_1(self):
        df = _make_ohlcv(200)  # mild uptrend → forward returns should be positive
        labels = compute_forward_labels(df)
        # Most early bars should have positive 5d returns in an uptrend
        valid = labels["label_5d"].dropna()
        assert valid.mean() > 0.5  # mostly bullish in an uptrend

    def test_shape_matches_input(self):
        df = _make_ohlcv(150)
        labels = compute_forward_labels(df)
        assert len(labels) == len(df)


class TestExportSymbolFeatures:
    """Integration test using DB — only runs if DB is available."""

    @pytest.fixture
    def db_available(self):
        try:
            from rainier.core.database import get_session
            with get_session() as db:
                db.execute(db.get_bind().dialect.server_version_info if hasattr(
                    db.get_bind().dialect, 'server_version_info') else None)
            return True
        except Exception:
            return False

    def test_export_known_symbol(self, db_available):
        if not db_available:
            pytest.skip("Database not available")

        from datetime import date
        result = export_symbol_features("AAPL", date(2023, 1, 1), date(2024, 1, 1))
        if result is None:
            pytest.skip("AAPL not in DB")

        assert "symbol" in result.columns
        assert "date" in result.columns
        assert "label_5d" in result.columns
        assert result["symbol"].iloc[0] == "AAPL"
        # No NaN in feature columns (excluding label columns which can be NaN)
        feature_cols = [c for c in result.columns if not c.startswith("fwd_return_")
                       and not c.startswith("label_") and c not in ("symbol", "date", "close", "volume")]
        assert result[feature_cols].isna().sum().sum() == 0


class TestValidateParquet:
    def test_validates_exported_file(self, tmp_path):
        # Create a minimal Parquet file
        df = pd.DataFrame({
            "body_size": [1.0, 2.0, 3.0],
            "range": [2.0, 3.0, 4.0],
            "fwd_return_5d": [0.01, -0.02, np.nan],
            "label_5d": [1.0, 0.0, np.nan],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "close": [150.0, 151.0, 152.0],
            "volume": [1000000.0, 1100000.0, 1200000.0],
        })
        path = tmp_path / "test.parquet"
        df.to_parquet(path, index=False)

        stats = validate_parquet(path)
        assert stats["rows"] == 3
        assert stats["symbols"] == 1
        assert stats["features"] == 2  # body_size, range
        assert stats["feature_nan_total"] == 0
        assert "label_5d" in stats["label_positive_rate"]
