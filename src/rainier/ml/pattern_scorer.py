"""XGBoost pattern scorer — train, evaluate, and predict.

Trains an XGBoost classifier on feature store Parquet data with
walk-forward cross-validation. Produces a model that scores pattern
setups as probability of hitting target.

Design decisions (eng review 2026-03-22):
- Single model with regime as feature initially
- Split to per-regime models only at 200+ samples/regime
- Walk-forward cross-validation mandatory (no random splits)
- NaN policy: fill with meaningful defaults + assert no NaN before input
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

# Features to exclude from model input
META_COLS = {"symbol", "date", "close", "volume"}
LABEL_PREFIX = "label_"
FWD_RETURN_PREFIX = "fwd_return_"


@dataclass
class TrainConfig:
    """Configuration for XGBoost training."""
    label_col: str = "label_5d"
    test_ratio: float = 0.2
    n_folds: int = 3
    early_stopping_rounds: int = 20
    xgb_params: dict | None = None

    def get_xgb_params(self) -> dict:
        if self.xgb_params:
            return self.xgb_params
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbosity": 0,
        }


@dataclass
class EvalResult:
    """Evaluation metrics from a trained model."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    profit_factor: float
    n_test: int
    n_positive: int
    feature_importance: dict[str, float]
    fold_scores: list[float]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Extract feature column names from a training DataFrame."""
    return [
        c for c in df.columns
        if c not in META_COLS
        and not c.startswith(LABEL_PREFIX)
        and not c.startswith(FWD_RETURN_PREFIX)
    ]


def walk_forward_split(
    df: pd.DataFrame,
    n_folds: int = 3,
    test_ratio: float = 0.2,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Walk-forward cross-validation splits (anchored).

    Each fold expands the training window and slides the test window:
    Fold 1: [==train==][test]
    Fold 2: [====train====][test]
    Fold 3: [======train======][test]
    """
    n = len(df)
    test_size = int(n * test_ratio / n_folds)
    if test_size < 10:
        test_size = min(50, n // (n_folds + 1))

    folds = []
    for i in range(n_folds):
        test_end = n - (n_folds - 1 - i) * test_size
        test_start = test_end - test_size
        train_end = test_start

        if train_end < 50:  # minimum training size
            continue

        train = df.iloc[:train_end]
        test = df.iloc[test_start:test_end]
        folds.append((train, test))

    return folds


def train_model(
    parquet_path: Path,
    config: TrainConfig | None = None,
    output_dir: Path | None = None,
) -> tuple[xgb.XGBClassifier, EvalResult]:
    """Train XGBoost model on feature store data with walk-forward CV.

    Args:
        parquet_path: Path to feature store Parquet file.
        config: Training configuration.
        output_dir: Directory to save model + metadata.

    Returns:
        Tuple of (trained model, evaluation results).
    """
    config = config or TrainConfig()
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d rows from %s", len(df), parquet_path)

    # Validate label column exists
    if config.label_col not in df.columns:
        raise ValueError(
            f"Label column '{config.label_col}' not found. "
            f"Available: {[c for c in df.columns if c.startswith('label_')]}"
        )

    # Drop rows without labels
    df = df.dropna(subset=[config.label_col]).reset_index(drop=True)
    logger.info("After dropping NaN labels: %d rows", len(df))

    feature_cols = get_feature_columns(df)
    logger.info("Features: %d columns", len(feature_cols))

    X = df[feature_cols].values
    y = df[config.label_col].values.astype(int)

    # Assert no NaN in features
    nan_mask = np.isnan(X)
    if nan_mask.any():
        nan_cols = [feature_cols[i] for i in range(len(feature_cols)) if nan_mask[:, i].any()]
        raise ValueError(f"NaN in features: {nan_cols}")

    # Walk-forward CV
    folds = walk_forward_split(df, n_folds=config.n_folds, test_ratio=config.test_ratio)
    fold_scores: list[float] = []
    xgb_params = config.get_xgb_params()

    for fold_idx, (train_df, test_df) in enumerate(folds):
        X_train = train_df[feature_cols].values
        y_train = train_df[config.label_col].values.astype(int)
        X_test = test_df[feature_cols].values
        y_test = test_df[config.label_col].values.astype(int)

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        fold_acc = accuracy_score(y_test, y_pred)
        fold_scores.append(fold_acc)
        logger.info(
            "Fold %d: train=%d, test=%d, accuracy=%.3f",
            fold_idx + 1, len(train_df), len(test_df), fold_acc,
        )

    # Final model: train on all data except last test_ratio
    split_idx = int(len(df) * (1 - config.test_ratio))
    X_train_final = df.iloc[:split_idx][feature_cols].values
    y_train_final = df.iloc[:split_idx][config.label_col].values.astype(int)
    X_test_final = df.iloc[split_idx:][feature_cols].values
    y_test_final = df.iloc[split_idx:][config.label_col].values.astype(int)

    final_model = xgb.XGBClassifier(**xgb_params)
    final_model.fit(
        X_train_final, y_train_final,
        eval_set=[(X_test_final, y_test_final)],
        verbose=False,
    )

    # Evaluate final model
    y_pred_final = final_model.predict(X_test_final)
    y_proba_final = final_model.predict_proba(X_test_final)

    # Profit factor: sum of predicted-positive returns / sum of predicted-positive losses
    test_df_final = df.iloc[split_idx:].copy()
    fwd_col = f"fwd_return_{config.label_col.replace('label_', '').replace('d', '')}d"
    profit_factor = _compute_profit_factor(
        test_df_final, y_pred_final, fwd_col
    )

    # Feature importance
    importance = dict(zip(feature_cols, final_model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    eval_result = EvalResult(
        accuracy=float(accuracy_score(y_test_final, y_pred_final)),
        precision=float(precision_score(y_test_final, y_pred_final, zero_division=0)),
        recall=float(recall_score(y_test_final, y_pred_final, zero_division=0)),
        f1=float(f1_score(y_test_final, y_pred_final, zero_division=0)),
        profit_factor=profit_factor,
        n_test=len(y_test_final),
        n_positive=int(y_test_final.sum()),
        feature_importance=importance,
        fold_scores=fold_scores,
    )

    logger.info(
        "Final model: accuracy=%.3f, precision=%.3f, recall=%.3f, f1=%.3f, pf=%.2f",
        eval_result.accuracy, eval_result.precision, eval_result.recall,
        eval_result.f1, eval_result.profit_factor,
    )

    # Save model + metadata
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "pattern_scorer.json"
        final_model.save_model(str(model_path))

        meta = {
            "label_col": config.label_col,
            "feature_cols": feature_cols,
            "n_train": split_idx,
            "n_test": len(y_test_final),
            "accuracy": eval_result.accuracy,
            "precision": eval_result.precision,
            "recall": eval_result.recall,
            "f1": eval_result.f1,
            "profit_factor": eval_result.profit_factor,
            "fold_scores": eval_result.fold_scores,
            "top_features": dict(list(importance.items())[:15]),
        }
        meta_path = output_dir / "pattern_scorer_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))

        logger.info("Model saved to %s", model_path)

    return final_model, eval_result


def _compute_profit_factor(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    fwd_col: str,
) -> float:
    """Compute profit factor: gross wins / gross losses for predicted-positive trades."""
    if fwd_col not in test_df.columns:
        return 0.0

    predicted_buy = y_pred == 1
    if not predicted_buy.any():
        return 0.0

    returns = test_df[fwd_col].values[predicted_buy]
    returns = returns[~np.isnan(returns)]

    gross_wins = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())

    if gross_losses == 0:
        return float("inf") if gross_wins > 0 else 0.0
    return float(gross_wins / gross_losses)


def explain_model(
    model: xgb.XGBClassifier,
    parquet_path: Path,
    output_path: Path | None = None,
) -> dict:
    """Generate SHAP explanations for the trained model."""
    import shap

    df = pd.read_parquet(parquet_path)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].dropna()

    # Use a sample for SHAP (can be slow on large datasets)
    sample_size = min(1000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample[feature_cols])

    # Mean absolute SHAP values per feature
    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_cols,
    ).sort_values(ascending=False)

    result = {
        "mean_shap": mean_shap.to_dict(),
        "top_10": list(mean_shap.head(10).index),
        "sample_size": sample_size,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

    return result
