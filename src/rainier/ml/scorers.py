"""Scoring strategy implementations — BookScorer (rule-based) and MLScorer (XGBoost).

Both implement the ScoringStrategy protocol from core/protocols.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rainier.analysis.pattern_primitives import VolumePriceSignal, analyze_volume_price
from rainier.core.types import PatternSignal


class BookScorer:
    """Rule-based pattern scorer using weighted sub-scores.

    Refactored from stock_patterns.py::score_pattern() into a class
    implementing the ScoringStrategy protocol.

    Breakdown:
        35% — pattern weight (from config)
        20% — volume confirmed (volume breakout) + 5% no divergence
        15% — pattern clarity (neckline defined, key_points present)
        15% — risk-reward ratio
        10% — status (confirmed vs forming)
    """

    def __init__(self, pattern_weights: dict[str, float] | None = None):
        self.pattern_weights = pattern_weights or {
            "false_breakdown": 0.9,
            "false_breakdown_w_bottom": 0.85,
            "false_breakout": 0.9,
            "false_breakout_hs_top": 0.85,
            "w_bottom": 0.7,
            "m_top": 0.7,
            "bull_flag": 0.65,
            "bear_flag": 0.65,
            "hs_bottom": 0.6,
            "hs_top": 0.6,
            "sym_triangle_bottom": 0.55,
            "sym_triangle_top": 0.55,
        }

    def score(self, pattern: PatternSignal, features: pd.DataFrame) -> float:
        """Score pattern setup. Features DataFrame is unused by BookScorer
        but required by the ScoringStrategy protocol for ML scorers."""
        # Compute volume-price signal from features if available
        vol_divergence = False
        if "volume_ratio" in features.columns and len(features) > 0:
            last_row = features.iloc[-1]
            vol_divergence = (
                last_row.get("is_bullish", 0) > 0 and last_row.get("volume_ratio", 1) < 1
            ) or (
                last_row.get("is_bullish", 0) == 0 and last_row.get("volume_ratio", 1) < 1
            )

        # 35% pattern weight
        weight = self.pattern_weights.get(pattern.pattern_type, 0.5)
        score_weight = 0.35 * weight

        # 20% volume + 5% divergence
        score_volume = 0.20 if pattern.volume_confirmed else 0.0
        score_divergence = 0.05 if not vol_divergence else 0.0

        # 15% pattern clarity
        clarity = 0.0
        if pattern.neckline > 0:
            clarity += 0.075
        if pattern.key_points:
            clarity += 0.075
        score_clarity = clarity

        # 15% risk-reward
        if pattern.rr_ratio >= 3.0:
            score_rr = 0.15
        elif pattern.rr_ratio >= 2.0:
            score_rr = 0.10
        elif pattern.rr_ratio >= 1.5:
            score_rr = 0.05
        else:
            score_rr = 0.0

        # 10% status
        score_status = 0.10 if pattern.status == "confirmed" else 0.0

        total = score_weight + score_volume + score_divergence + score_clarity + score_rr + score_status
        return round(min(total, 1.0), 4)


class MLScorer:
    """XGBoost-based pattern scorer.

    Wraps a trained XGBoost model to score pattern setups.
    Input: PatternSignal + feature DataFrame from FeatureExtractor.
    Output: probability [0, 1] that the trade hits target.
    """

    def __init__(self, model_path: Path | None = None):
        self.model = None
        self.feature_names: list[str] = []
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: Path) -> None:
        """Load a trained XGBoost model from JSON."""
        import xgboost as xgb

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))
        self.feature_names = self.model.get_booster().feature_names or []

    def score(self, pattern: PatternSignal, features: pd.DataFrame) -> float:
        """Score using the trained XGBoost model.

        Uses the last row of the features DataFrame (current bar)
        plus pattern metadata as input.
        """
        if self.model is None:
            raise RuntimeError("No model loaded — call load() or train() first")

        # Build input row from features + pattern metadata
        row = self._build_input(pattern, features)
        proba = self.model.predict_proba(row)[0]
        # Return probability of positive class (label=1)
        return float(proba[1]) if len(proba) > 1 else float(proba[0])

    def _build_input(
        self, pattern: PatternSignal, features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine features + pattern metadata into model input."""
        if features.empty:
            raise ValueError("Features DataFrame is empty")

        # Use last row of features (current bar)
        row = features.iloc[[-1]].copy()

        # Add pattern metadata as features
        row["pattern_rr_ratio"] = pattern.rr_ratio
        row["pattern_volume_confirmed"] = float(pattern.volume_confirmed)
        row["pattern_status_confirmed"] = float(pattern.status == "confirmed")
        row["pattern_risk_pct"] = pattern.risk_pct
        row["pattern_reward_pct"] = pattern.reward_pct

        # Align columns to what the model expects
        if self.feature_names:
            missing = set(self.feature_names) - set(row.columns)
            for col in missing:
                row[col] = 0.0
            row = row[self.feature_names]

        return row
