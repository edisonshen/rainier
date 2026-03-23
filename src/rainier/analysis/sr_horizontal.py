"""Horizontal S/R detection: pivot clustering, zone merging, strength scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rainier.core.config import SRHorizontalConfig
from rainier.core.types import Pivot, SRLevel, SRRole, SRType


def detect_horizontal_sr(
    pivots: list[Pivot],
    df: pd.DataFrame,
    atr: pd.Series,
    config: SRHorizontalConfig | None = None,
) -> list[SRLevel]:
    """Cluster pivots into horizontal S/R zones and score them.

    Algorithm:
    1. Sort pivots by price
    2. Cluster pivots within cluster_atr_mult * ATR of each other
    3. Merge each cluster into a single S/R level (average price)
    4. Score by: touches, recency, volume, round number
    """
    if config is None:
        config = SRHorizontalConfig()

    if len(pivots) < config.min_touches:
        return []

    avg_atr = float(atr.mean()) if len(atr) > 0 else 1.0
    cluster_dist = config.cluster_atr_mult * avg_atr

    # Sort by price
    sorted_pivots = sorted(pivots, key=lambda p: p.price)

    # Greedy clustering
    clusters: list[list[Pivot]] = []
    current_cluster: list[Pivot] = [sorted_pivots[0]]

    for pivot in sorted_pivots[1:]:
        if pivot.price - current_cluster[-1].price <= cluster_dist:
            current_cluster.append(pivot)
        else:
            clusters.append(current_cluster)
            current_cluster = [pivot]
    clusters.append(current_cluster)

    # Filter clusters with enough touches
    clusters = [c for c in clusters if len(c) >= config.min_touches]

    if not clusters:
        return []

    # Build S/R levels
    max_index = df.index[-1] if len(df) > 0 else 0
    levels: list[SRLevel] = []

    for cluster in clusters:
        price = float(np.mean([p.price for p in cluster]))
        touches = len(cluster)

        # Determine role: majority of pivots are highs → resistance, lows → support
        n_highs = sum(1 for p in cluster if p.is_high)
        role = SRRole.RESISTANCE if n_highs > len(cluster) / 2 else SRRole.SUPPORT

        # Score components
        touch_score = min(touches / 5.0, 1.0)  # cap at 5 touches
        recency_score = _recency_score(cluster, max_index)
        volume_score = _volume_score(cluster, df)
        round_score = _round_number_score(price)

        strength = (
            config.weight_touches * touch_score
            + config.weight_recency * recency_score
            + config.weight_volume * volume_score
            + config.weight_round_number * round_score
        )

        levels.append(
            SRLevel(
                price=price,
                sr_type=SRType.HORIZONTAL,
                role=role,
                strength=float(np.clip(strength, 0.0, 1.0)),
                touches=touches,
                first_seen=min(p.timestamp for p in cluster),
                last_tested=max(p.timestamp for p in cluster),
            )
        )

    return sorted(levels, key=lambda l: l.strength, reverse=True)


def _recency_score(cluster: list[Pivot], max_index: int) -> float:
    """More recent pivots → higher score."""
    if max_index == 0:
        return 0.5
    most_recent = max(p.index for p in cluster)
    return most_recent / max_index


def _volume_score(cluster: list[Pivot], df: pd.DataFrame) -> float:
    """Higher volume at pivot points → higher score."""
    if "volume" not in df.columns or df["volume"].sum() == 0:
        return 0.5  # neutral if no volume data
    avg_vol = df["volume"].mean()
    if avg_vol == 0:
        return 0.5
    pivot_vols = [df.iloc[p.index]["volume"] for p in cluster if p.index < len(df)]
    if not pivot_vols:
        return 0.5
    return float(np.clip(np.mean(pivot_vols) / avg_vol, 0.0, 2.0) / 2.0)


def _round_number_score(price: float) -> float:
    """Prices near round numbers (00, 50, 000) get a bonus."""
    # Check proximity to nearest round number
    remainder_100 = price % 100
    dist_100 = min(remainder_100, 100 - remainder_100)

    remainder_50 = price % 50
    dist_50 = min(remainder_50, 50 - remainder_50)

    # Normalize: 0 distance = score 1.0, distance >= 25 = score 0.0
    score_100 = max(0.0, 1.0 - dist_100 / 25.0)
    score_50 = max(0.0, 1.0 - dist_50 / 12.5) * 0.5  # half weight for 50s

    return min(score_100 + score_50, 1.0)
