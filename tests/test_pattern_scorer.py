"""Tests for XGBoost pattern scorer training pipeline."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from rainier.ml.pattern_scorer import (
    TrainConfig,
    get_feature_columns,
    train_model,
    walk_forward_split,
)


def _make_training_data(n: int = 500, n_symbols: int = 2) -> pd.DataFrame:
    """Create synthetic feature store data for testing."""
    rows = []
    for sym_idx in range(n_symbols):
        symbol = f"SYM{sym_idx}"
        for i in range(n):
            # Simple features with some predictive signal
            body_size = np.random.exponential(1.0)
            vol_ratio = np.random.lognormal(0, 0.3)
            atr_ratio = np.random.lognormal(0, 0.2)
            return_5bar = np.random.normal(0, 0.02)

            # Label with some correlation to features
            prob = 0.5 + 0.1 * (vol_ratio > 1) + 0.05 * (return_5bar > 0)
            label = float(np.random.random() < prob)

            rows.append({
                "body_size": body_size,
                "range": body_size * 1.5,
                "body_range_ratio": 0.6,
                "upper_wick": body_size * 0.2,
                "lower_wick": body_size * 0.3,
                "close_position": np.random.uniform(0.2, 0.8),
                "is_bullish": float(np.random.random() > 0.5),
                "atr_14": np.random.exponential(2.0),
                "atr_ratio": atr_ratio,
                "range_vs_atr": np.random.uniform(0.5, 2.0),
                "is_pinbar": float(np.random.random() > 0.9),
                "pinbar_wick_ratio": 0.0,
                "pinbar_direction": 0.0,
                "pinbar_sr_distance": 0.1,
                "pinbar_sr_strength": 0.0,
                "is_inside_bar": float(np.random.random() > 0.95),
                "inside_bar_compression": 0.0,
                "dist_nearest_support": np.random.exponential(0.03),
                "dist_nearest_resistance": np.random.exponential(0.03),
                "nearest_sr_strength": np.random.uniform(0, 1),
                "nearest_sr_touches": float(np.random.randint(0, 5)),
                "levels_within_1atr": float(np.random.randint(0, 3)),
                "num_tf_confluence": float(np.random.randint(0, 3)),
                "higher_tf_bias": np.random.choice([-1, 0, 1]),
                "bars_since_swing_high": float(np.random.randint(1, 50)),
                "bars_since_swing_low": float(np.random.randint(1, 50)),
                "consecutive_higher_highs": float(np.random.randint(0, 5)),
                "consecutive_lower_lows": float(np.random.randint(0, 5)),
                "sma_50_slope": np.random.normal(0, 0.5),
                "return_1bar": np.random.normal(0, 0.01),
                "return_5bar": return_5bar,
                "volume_ratio": vol_ratio,
                "rolling_volatility_20": np.random.exponential(0.01),
                "atr_percentile": np.random.uniform(0, 1),
                "adx": np.random.uniform(10, 50),
                "regime_trending_up": float(np.random.random() > 0.7),
                "regime_trending_down": float(np.random.random() > 0.8),
                "regime_range_bound": float(np.random.random() > 0.6),
                "regime_high_volatility": float(np.random.random() > 0.9),
                "fwd_return_5d": np.random.normal(0, 0.03),
                "label_5d": label,
                "symbol": symbol,
                "date": pd.Timestamp("2023-01-01") + timedelta(days=i),
                "close": 100 + np.random.normal(0, 5),
                "volume": float(np.random.randint(500000, 2000000)),
            })
    return pd.DataFrame(rows)


class TestGetFeatureColumns:
    def test_excludes_meta_and_labels(self):
        df = _make_training_data(50, 1)
        cols = get_feature_columns(df)
        assert "symbol" not in cols
        assert "date" not in cols
        assert "close" not in cols
        assert "volume" not in cols
        assert "label_5d" not in cols
        assert "fwd_return_5d" not in cols
        assert "body_size" in cols
        assert "atr_ratio" in cols


class TestWalkForwardSplit:
    def test_produces_correct_number_of_folds(self):
        df = _make_training_data(500, 1)
        folds = walk_forward_split(df, n_folds=3, test_ratio=0.2)
        assert len(folds) == 3

    def test_train_grows_test_slides(self):
        df = _make_training_data(500, 1)
        folds = walk_forward_split(df, n_folds=3, test_ratio=0.2)
        # Each fold should have larger training set
        for i in range(1, len(folds)):
            assert len(folds[i][0]) >= len(folds[i-1][0])

    def test_no_overlap(self):
        df = _make_training_data(500, 1)
        folds = walk_forward_split(df, n_folds=3, test_ratio=0.2)
        for train, test in folds:
            assert train.index[-1] < test.index[0]


class TestTrainModel:
    def test_train_produces_model_and_results(self, tmp_path):
        df = _make_training_data(500, 2)
        parquet_path = tmp_path / "test_features.parquet"
        df.to_parquet(parquet_path, index=False)

        config = TrainConfig(label_col="label_5d", n_folds=2)
        model, result = train_model(parquet_path, config, output_dir=tmp_path)

        assert model is not None
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.precision <= 1.0
        assert result.n_test > 0
        assert len(result.fold_scores) == 2
        assert len(result.feature_importance) > 0

    def test_saves_model_and_meta(self, tmp_path):
        df = _make_training_data(500, 1)
        parquet_path = tmp_path / "test_features.parquet"
        df.to_parquet(parquet_path, index=False)

        train_model(parquet_path, TrainConfig(n_folds=2), output_dir=tmp_path)

        assert (tmp_path / "pattern_scorer.json").exists()
        assert (tmp_path / "pattern_scorer_meta.json").exists()

    def test_raises_on_missing_label(self, tmp_path):
        df = _make_training_data(100, 1)
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path, index=False)

        with pytest.raises(ValueError, match="Label column"):
            train_model(parquet_path, TrainConfig(label_col="label_99d"))
