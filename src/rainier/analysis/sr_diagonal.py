"""Diagonal S/R (trendline) detection via swing-point regression."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from rainier.core.config import SRDiagonalConfig
from rainier.core.types import Pivot, SRLevel, SRRole, SRType


def detect_diagonal_sr(
    pivots: list[Pivot],
    df: pd.DataFrame,
    atr: pd.Series,
    config: SRDiagonalConfig | None = None,
) -> list[SRLevel]:
    """Detect diagonal trendlines by fitting lines through pairs of swing points.

    Algorithm:
    1. Separate swing highs and swing lows
    2. For each pair of swing highs → candidate resistance line
    3. For each pair of swing lows → candidate support line
    4. Validate: count how many other pivots touch/react to the line
    5. Deduplicate by slope/intercept similarity
    6. Return scored levels
    """
    if config is None:
        config = SRDiagonalConfig()

    avg_atr = float(atr.mean()) if len(atr) > 0 else 1.0
    tolerance = config.price_tolerance_atr_mult * avg_atr

    swing_highs = [p for p in pivots if p.is_high]
    swing_lows = [p for p in pivots if not p.is_high]

    levels: list[SRLevel] = []

    # Resistance lines from swing highs
    if len(swing_highs) >= config.min_swing_points:
        candidates = _fit_lines(swing_highs, df, tolerance, config)
        for level in candidates:
            level.role = SRRole.RESISTANCE
        levels.extend(candidates)

    # Support lines from swing lows
    if len(swing_lows) >= config.min_swing_points:
        candidates = _fit_lines(swing_lows, df, tolerance, config)
        for level in candidates:
            level.role = SRRole.SUPPORT
        levels.extend(candidates)

    # Deduplicate
    levels = _deduplicate(levels, avg_atr, config)

    return sorted(levels, key=lambda l: l.strength, reverse=True)


def _fit_lines(
    swing_points: list[Pivot],
    df: pd.DataFrame,
    tolerance: float,
    config: SRDiagonalConfig,
) -> list[SRLevel]:
    """Fit lines through all pairs of swing points, validate, return candidates."""
    candidates: list[SRLevel] = []

    # Limit to most recent swing points to avoid O(n^2) blowup
    points = sorted(swing_points, key=lambda p: p.index)[-20:]

    for p1, p2 in combinations(points, 2):
        if p1.index == p2.index:
            continue

        # Line: price = p1.price + slope * (bar_index - p1.index)
        slope = (p2.price - p1.price) / (p2.index - p1.index)

        # Count touches: how many other pivot points are near this line
        touches = 0
        for p in swing_points:
            expected = p1.price + slope * (p.index - p1.index)
            if abs(p.price - expected) <= tolerance:
                touches += 1

        if touches < config.min_touches:
            continue

        # Score: more touches + more recent = better
        max_idx = len(df) - 1 if len(df) > 0 else 1
        recency = max(p1.index, p2.index) / max_idx if max_idx > 0 else 0.5
        touch_score = min(touches / 5.0, 1.0)
        strength = 0.6 * touch_score + 0.4 * recency

        # Use the later point as anchor for price_at() calculations
        anchor = max(p1.index, p2.index)
        anchor_price = p1.price + slope * (anchor - p1.index)

        candidates.append(
            SRLevel(
                price=anchor_price,
                sr_type=SRType.DIAGONAL,
                role=SRRole.SUPPORT,  # will be overwritten by caller
                strength=float(np.clip(strength, 0.0, 1.0)),
                touches=touches,
                slope=slope,
                anchor_index=anchor,
                first_seen=min(p1.timestamp, p2.timestamp),
                last_tested=max(p1.timestamp, p2.timestamp),
            )
        )

    return candidates


def _deduplicate(
    levels: list[SRLevel],
    avg_atr: float,
    config: SRDiagonalConfig,
) -> list[SRLevel]:
    """Remove near-duplicate trendlines by slope/intercept similarity."""
    if not levels:
        return []

    intercept_tol = config.intercept_similarity_atr_mult * avg_atr
    slope_tol = config.slope_similarity_threshold

    # Sort by strength descending — keep the stronger one
    sorted_levels = sorted(levels, key=lambda l: l.strength, reverse=True)
    kept: list[SRLevel] = []

    for level in sorted_levels:
        is_dup = False
        for existing in kept:
            if existing.role != level.role:
                continue
            if (
                abs(existing.slope - level.slope) < slope_tol
                and abs(existing.price - level.price) < intercept_tol
            ):
                is_dup = True
                break
        if not is_dup:
            kept.append(level)

    return kept
