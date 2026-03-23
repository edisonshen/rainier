"""Confidence scoring for trade signals — weighted sub-scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rainier.core.config import ScorerConfig
from rainier.core.types import Direction, PinBar, SRLevel


def score_setup(
    pin_bar: PinBar,
    df: pd.DataFrame,
    bias: Direction | None,
    config: ScorerConfig | None = None,
    sr_levels: list[SRLevel] | None = None,
) -> float:
    """Compute a confidence score (0.0 - 1.0) for a pin bar setup.

    Sub-scores:
    - S/R strength: how strong is the nearby S/R level
    - Wick ratio: larger wick → stronger rejection
    - Volume spike: volume at pin bar vs average
    - Trend alignment: pin bar direction matches bias
    - Multi-TF confluence: distinct timeframes among nearby S/R levels
    """
    if config is None:
        config = ScorerConfig()

    sr_strength = _sr_strength_score(pin_bar)
    wick_score = _wick_ratio_score(pin_bar)
    volume_score = _volume_spike_score(pin_bar, df)
    trend_score = _trend_alignment_score(pin_bar, bias)
    confluence_score = _multi_tf_confluence_score(pin_bar, sr_levels)

    total = (
        config.weight_sr_strength * sr_strength
        + config.weight_wick_ratio * wick_score
        + config.weight_volume_spike * volume_score
        + config.weight_trend_alignment * trend_score
        + config.weight_multi_tf_confluence * confluence_score
    )

    return float(np.clip(total, 0.0, 1.0))


def _sr_strength_score(pin_bar: PinBar) -> float:
    if pin_bar.nearest_sr is None:
        return 0.0
    return pin_bar.nearest_sr.strength


def _wick_ratio_score(pin_bar: PinBar) -> float:
    """Higher wick ratio → stronger rejection → higher score. Cap at ratio=5."""
    return float(np.clip(pin_bar.wick_ratio / 5.0, 0.0, 1.0))


def _volume_spike_score(pin_bar: PinBar, df: pd.DataFrame) -> float:
    """Pin bar volume relative to average."""
    if "volume" not in df.columns or df["volume"].sum() == 0:
        return 0.5
    avg_vol = df["volume"].mean()
    if avg_vol == 0:
        return 0.5
    ratio = pin_bar.candle.volume / avg_vol
    return float(np.clip(ratio / 2.0, 0.0, 1.0))  # 2x avg volume = 1.0


def _trend_alignment_score(pin_bar: PinBar, bias: Direction | None) -> float:
    """Counter-trend at S/R is the strategy, but alignment with higher TF bias is a plus."""
    if bias is None:
        return 0.5
    # Pin bar direction matching bias = good (counter-trend reversal aligning with bigger picture)
    if pin_bar.direction == bias:
        return 0.8
    return 0.3


def _multi_tf_confluence_score(
    pin_bar: PinBar,
    sr_levels: list[SRLevel] | None,
    proximity_pct: float = 0.005,
) -> float:
    """Count distinct timeframes among S/R levels near the pin bar.

    Levels within ``proximity_pct`` of the pin bar's price are considered "nearby".
    Score mapping: 0 or 1 TF → 0.3, 2 TFs → 0.6, 3+ TFs → 1.0.
    Falls back to 0.5 when no sr_levels are provided (backward-compat).
    """
    if sr_levels is None:
        return 0.5

    ref_price = pin_bar.candle.close
    threshold = ref_price * proximity_pct

    nearby_tfs: set[str] = set()
    for level in sr_levels:
        price = level.price_at(pin_bar.index)
        if abs(price - ref_price) <= threshold:
            tf_key = level.source_tf.value if level.source_tf is not None else "_local"
            nearby_tfs.add(tf_key)

    n = len(nearby_tfs)
    if n <= 1:
        return 0.3
    if n == 2:
        return 0.6
    return 1.0
