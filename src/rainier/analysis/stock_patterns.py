"""Caisen pattern detectors — all 12 chart patterns from "Erta Reversal" methodology.

Detects W Bottom, M Top, False Breakdown, False Breakout, False Breakdown W Bottom,
False Breakout H&S Top, Bull Flag, Bear Flag, H&S Bottom, H&S Top,
Sym Triangle Bottom, Sym Triangle Top.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from rainier.analysis.pattern_primitives import (
    Breakout,
    SwingPoint,
    VolumePriceSignal,
    analyze_volume_price,
    detect_breakout,
    find_swing_points,
)
from rainier.analysis.target_calculator import (
    compute_double_bottom_targets,
    compute_double_top_targets,
    compute_false_breakdown_targets,
    compute_false_breakout_targets,
    compute_flag_targets,
    compute_hs_targets,
    compute_triangle_targets,
)
from rainier.core.config import StockScreenerConfig
from rainier.core.types import PatternSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_patterns(
    symbol: str,
    df: pd.DataFrame,
    config: StockScreenerConfig,
) -> list[PatternSignal]:
    """Detect all 12 Caisen patterns on daily OHLCV data.

    Args:
        symbol: Stock ticker
        df: DataFrame with columns: open, high, low, close, volume (daily bars)
        config: Screener configuration

    Returns:
        List of detected patterns, sorted by confidence descending.
    """
    if len(df) < config.min_pattern_bars:
        return []

    swing_points = find_swing_points(df, lookback=config.swing_lookback)
    if len(swing_points) < 2:
        return []

    vol_price = analyze_volume_price(df)
    detectors = [
        _detect_w_bottom,
        _detect_m_top,
        _detect_false_breakdown,
        _detect_false_breakout,
        _detect_false_breakdown_w,
        _detect_false_breakout_hs,
        _detect_bull_flag,
        _detect_bear_flag,
        _detect_hs_bottom,
        _detect_hs_top,
        _detect_sym_triangle_bottom,
        _detect_sym_triangle_top,
    ]

    results: list[PatternSignal] = []
    for detector in detectors:
        try:
            patterns = detector(symbol, df, swing_points, config)
            # Score each pattern
            for p in patterns:
                confidence = score_pattern(p, vol_price, config.pattern_weights)
                # Rebuild with computed confidence (frozen dataclass)
                scored = PatternSignal(
                    symbol=p.symbol,
                    pattern_type=p.pattern_type,
                    direction=p.direction,
                    status=p.status,
                    confidence=confidence,
                    entry_price=p.entry_price,
                    stop_loss=p.stop_loss,
                    target_wave1=p.target_wave1,
                    target_wave2=p.target_wave2,
                    risk_pct=p.risk_pct,
                    reward_pct=p.reward_pct,
                    rr_ratio=p.rr_ratio,
                    neckline=p.neckline,
                    key_points=p.key_points,
                    volume_confirmed=p.volume_confirmed,
                    pattern_start_idx=p.pattern_start_idx,
                    pattern_end_idx=p.pattern_end_idx,
                    breakout_idx=p.breakout_idx,
                )
                results.append(scored)
        except Exception:
            logger.exception("Error in %s for %s", detector.__name__, symbol)

    results.sort(key=lambda p: p.confidence, reverse=True)
    logger.info("%s: detected %d patterns", symbol, len(results))
    return results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_pattern(
    pattern: PatternSignal,
    vol_price: VolumePriceSignal,
    pattern_weights: dict[str, float],
) -> float:
    """Score a pattern signal (0-1) based on design doc section 3.5.

    Breakdown:
        35% — pattern weight (from config)
        20% — volume confirmed (volume breakout) + 5% no divergence
        15% — pattern clarity (neckline defined, key_points present)
        15% — risk-reward ratio
        10% — status (confirmed vs forming)
    """
    # 35% pattern weight
    weight = pattern_weights.get(pattern.pattern_type, 0.5)
    score_weight = 0.35 * weight

    # 20% volume + 5% divergence
    score_volume = 0.20 if pattern.volume_confirmed else 0.0
    score_divergence = 0.05 if not vol_price.divergence else 0.0

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _swing_lows(swing_points: list[SwingPoint]) -> list[SwingPoint]:
    return [sp for sp in swing_points if sp.type == "low"]


def _swing_highs(swing_points: list[SwingPoint]) -> list[SwingPoint]:
    return [sp for sp in swing_points if sp.type == "high"]


def _within_tolerance(a: float, b: float, tolerance_pct: float) -> bool:
    """Check if two prices are within tolerance_pct of their average."""
    avg = (a + b) / 2
    if avg == 0:
        return False
    return abs(a - b) / avg <= tolerance_pct


def _span_ok(idx_start: int, idx_end: int, config: StockScreenerConfig) -> bool:
    span = idx_end - idx_start
    return config.min_pattern_bars <= span <= config.max_pattern_bars


def _check_breakout(
    df: pd.DataFrame,
    level: float,
    direction: str,
    start_idx: int,
    config: StockScreenerConfig,
) -> Breakout | None:
    return detect_breakout(
        df,
        level=level,
        direction=direction,
        start_idx=start_idx,
        vol_multiplier=config.volume_breakout_multiplier,
    )


def _find_swing_high_between(
    swing_points: list[SwingPoint], idx_start: int, idx_end: int
) -> SwingPoint | None:
    """Find the highest swing high between two bar indices."""
    candidates = [
        sp for sp in swing_points
        if sp.type == "high" and idx_start < sp.index < idx_end
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda sp: sp.price)


def _find_swing_low_between(
    swing_points: list[SwingPoint], idx_start: int, idx_end: int
) -> SwingPoint | None:
    """Find the lowest swing low between two bar indices."""
    candidates = [
        sp for sp in swing_points
        if sp.type == "low" and idx_start < sp.index < idx_end
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda sp: sp.price)


# ---------------------------------------------------------------------------
# Pattern 1: W Bottom
# ---------------------------------------------------------------------------


def _detect_w_bottom(
    symbol: str,
    df: pd.DataFrame,
    swing_points: list[SwingPoint],
    config: StockScreenerConfig,
) -> list[PatternSignal]:
    results: list[PatternSignal] = []
    lows = _swing_lows(swing_points)

    for i in range(len(lows) - 1):
        left = lows[i]
        right = lows[i + 1]

        if not _within_tolerance(left.price, right.price, config.neckline_tolerance_pct):
            continue
        if not _span_ok(left.index, right.index, config):
            continue

        # Neckline = swing high between the two bottoms
        neck_sp = _find_swing_high_between(swing_points, left.index, right.index)
        if neck_sp is None:
            continue
        neckline = neck_sp.price

        # Check breakout
        breakout = _check_breakout(df, neckline, "up", right.index, config)
        status = "confirmed" if breakout and not breakout.false_breakout else "forming"
        vol_confirmed = breakout.with_volume if breakout else False

        targets = compute_double_bottom_targets(
            neckline, left.price, right.price, config.stop_buffer_pct
        )

        results.append(PatternSignal(
            symbol=symbol,
            pattern_type="w_bottom",
            direction="bullish",
            status=status,
            confidence=0.0,  # scored later
            entry_price=targets.entry,
            stop_loss=targets.stop_loss,
            target_wave1=targets.target_wave1,
            target_wave2=targets.target_wave2,
            risk_pct=targets.risk_pct,
            reward_pct=targets.reward_pct,
            rr_ratio=targets.rr_ratio,
            neckline=neckline,
            key_points={
                "left_bottom": left.price,
                "right_bottom": right.price,
                "neckline": neckline,
            },
            volume_confirmed=vol_confirmed,
            pattern_start_idx=left.index,
            pattern_end_idx=right.index,
            breakout_idx=breakout.bar_index if breakout else None,
        ))
        logger.debug("%s: W bottom at bars %d-%d", symbol, left.index, right.index)

    return results


# ---------------------------------------------------------------------------
# Pattern 2: M Top
# ---------------------------------------------------------------------------


def _detect_m_top(
    symbol: str,
    df: pd.DataFrame,
    swing_points: list[SwingPoint],
    config: StockScreenerConfig,
) -> list[PatternSignal]:
    results: list[PatternSignal] = []
    highs = _swing_highs(swing_points)

    for i in range(len(highs) - 1):
        left = highs[i]
        right = highs[i + 1]

        if not _within_tolerance(left.price, right.price, config.neckline_tolerance_pct):
            continue
        if not _span_ok(left.index, right.index, config):
            continue

        # Neckline = swing low between the two tops
        neck_sp = _find_swing_low_between(swing_points, left.index, right.index)
        if neck_sp is None:
            continue
        neckline = neck_sp.price

        breakout = _check_breakout(df, neckline, "down", right.index, config)
        status = "confirmed" if breakout and not breakout.false_breakout else "forming"
        vol_confirmed = breakout.with_volume if breakout else False

        targets = compute_double_top_targets(
            neckline, left.price, right.price, config.stop_buffer_pct
        )

        results.append(PatternSignal(
            symbol=symbol,
            pattern_type="m_top",
            direction="bearish",
            status=status,
            confidence=0.0,
            entry_price=targets.entry,
            stop_loss=targets.stop_loss,
            target_wave1=targets.target_wave1,
            target_wave2=targets.target_wave2,
            risk_pct=targets.risk_pct,
            reward_pct=targets.reward_pct,
            rr_ratio=targets.rr_ratio,
            neckline=neckline,
            key_points={
                "left_top": left.price,
                "right_top": right.price,
                "neckline": neckline,
            },
            volume_confirmed=vol_confirmed,
            pattern_start_idx=left.index,
            pattern_end_idx=right.index,
            breakout_idx=breakout.bar_index if breakout else None,
        ))

    return results


# ---------------------------------------------------------------------------
# Pattern 3: False Breakdown
# ---------------------------------------------------------------------------


def _detect_false_breakdown(
    symbol: str,
    df: pd.DataFrame,
    swing_points: list[SwingPoint],
    config: StockScreenerConfig,
) -> list[PatternSignal]:
    results: list[PatternSignal] = []
    lows = _swing_lows(swing_points)
    closes = df["close"].to_numpy(dtype=np.float64)

    for i in range(1, len(lows)):
        prior = lows[i - 1]
        current = lows[i]

        # Current low must break below prior low
        if current.price >= prior.price:
            continue

        # Price must recover above prior low within 1-5 bars
        recovery_found = False
        recovery_bar = None
        check_end = min(current.index + 6, len(closes))
        for j in range(current.index + 1, check_end):
            if closes[j] > prior.price:
                recovery_found = True
                recovery_bar = j
                break

        if not recovery_found:
            continue

        # Volume check on recovery bar
        volumes = df["volume"].to_numpy(dtype=np.float64)
        vol_start = max(0, recovery_bar - 20)
        if vol_start < recovery_bar:
            avg_vol = float(np.mean(volumes[vol_start:recovery_bar]))
        else:
            avg_vol = float(volumes[recovery_bar])
        vol_confirmed = bool(
            volumes[recovery_bar] > config.volume_breakout_multiplier * avg_vol
        ) if avg_vol > 0 else False

        targets = compute_false_breakdown_targets(
            prior.price, current.price, stop_buffer_pct=config.stop_buffer_pct
        )

        status = "confirmed"

        results.append(PatternSignal(
            symbol=symbol,
            pattern_type="false_breakdown",
            direction="bullish",
            status=status,
            confidence=0.0,
            entry_price=targets.entry,
            stop_loss=targets.stop_loss,
            target_wave1=targets.target_wave1,
            target_wave2=targets.target_wave2,
            risk_pct=targets.risk_pct,
            reward_pct=targets.reward_pct,
            rr_ratio=targets.rr_ratio,
            neckline=prior.price,
            key_points={
                "prior_low": prior.price,
                "false_low": current.price,
                "recovery_bar": recovery_bar,
            },
            volume_confirmed=vol_confirmed,
            pattern_start_idx=prior.index,
            pattern_end_idx=recovery_bar,
            breakout_idx=recovery_bar,
        ))

    return results


# ---------------------------------------------------------------------------
# Pattern 4: False Breakout
# ---------------------------------------------------------------------------


def _detect_false_breakout(
    symbol: str,
    df: pd.DataFrame,
    swing_points: list[SwingPoint],
    config: StockScreenerConfig,
) -> list[PatternSignal]:
    results: list[PatternSignal] = []
    highs = _swing_highs(swing_points)
    closes = df["close"].to_numpy(dtype=np.float64)

    for i in range(1, len(highs)):
        prior = highs[i - 1]
        current = highs[i]

        # Current high must break above prior high
        if current.price <= prior.price:
            continue

        # Price must fall back below prior high within 1-5 bars
        rejection_found = False
        rejection_bar = None
        check_end = min(current.index + 6, len(closes))
        for j in range(current.index + 1, check_end):
            if closes[j] < prior.price:
                rejection_found = True
                rejection_bar = j
                break

        if not rejection_found:
            continue

        volumes = df["volume"].to_numpy(dtype=np.float64)
        vol_start = max(0, rejection_bar - 20)
        if vol_start < rejection_bar:
            avg_vol = float(np.mean(volumes[vol_start:rejection_bar]))
        else:
            avg_vol = float(volumes[rejection_bar])
        vol_confirmed = bool(
            volumes[rejection_bar] > config.volume_breakout_multiplier * avg_vol
        ) if avg_vol > 0 else False

        targets = compute_false_breakout_targets(
            prior.price, current.price, stop_buffer_pct=config.stop_buffer_pct
        )

        results.append(PatternSignal(
            symbol=symbol,
            pattern_type="false_breakout",
            direction="bearish",
            status="confirmed",
            confidence=0.0,
            entry_price=targets.entry,
            stop_loss=targets.stop_loss,
            target_wave1=targets.target_wave1,
            target_wave2=targets.target_wave2,
            risk_pct=targets.risk_pct,
            reward_pct=targets.reward_pct,
            rr_ratio=targets.rr_ratio,
            neckline=prior.price,
            key_points={
                "prior_high": prior.price,
                "false_high": current.price,
                "rejection_bar": rejection_bar,
            },
            volume_confirmed=vol_confirmed,
            pattern_start_idx=prior.index,
            pattern_end_idx=rejection_bar,
            breakout_idx=rejection_bar,
        ))

    return results


# ---------------------------------------------------------------------------
# Pattern 5: False Breakdown W Bottom
# ---------------------------------------------------------------------------


def _detect_false_breakdown_w(
    symbol: str,
    df: pd.DataFrame,
    swing_points: list[SwingPoint],
    config: StockScreenerConfig,
) -> list[PatternSignal]:
    """W bottom where the second bottom breaks below the first, then recovers.

    Combination of false breakdown + W bottom: the second low undercuts the
    first, creating a false breakdown, then price rallies through the neckline.
    """
    results: list[PatternSignal] = []
    lows = _swing_lows(swing_points)
    closes = df["close"].to_numpy(dtype=np.float64)

    for i in range(1, len(lows)):
        first = lows[i - 1]
        second = lows[i]

        # Second bottom must be BELOW first (false breakdown component)
        if second.price >= first.price:
            continue
        if not _span_ok(first.index, second.index, config):
            continue

        # Must recover above first low within 1-5 bars
        recovery_bar = None
        check_end = min(second.index + 6, len(closes))
        for j in range(second.index + 1, check_end):
            if closes[j] > first.price:
                recovery_bar = j
                break
        if recovery_bar is None:
            continue

        # Find neckline = swing high between the two lows
        neck_sp = _find_swing_high_between(swing_points, first.index, second.index)
        if neck_sp is None:
            continue
        neckline = neck_sp.price

        # Check breakout above neckline
        breakout = _check_breakout(df, neckline, "up", recovery_bar, config)
        status = "confirmed" if breakout and not breakout.false_breakout else "forming"
        vol_confirmed = breakout.with_volume if breakout else False

        targets = compute_false_breakdown_targets(
            first.price, second.price, neckline=neckline,
            stop_buffer_pct=config.stop_buffer_pct,
        )

        results.append(PatternSignal(
            symbol=symbol,
            pattern_type="false_breakdown_w_bottom",
            direction="bullish",
            status=status,
            confidence=0.0,
            entry_price=targets.entry,
            stop_loss=targets.stop_loss,
            target_wave1=targets.target_wave1,
            target_wave2=targets.target_wave2,
            risk_pct=targets.risk_pct,
            reward_pct=targets.reward_pct,
            rr_ratio=targets.rr_ratio,
            neckline=neckline,
            key_points={
                "first_bottom": first.price,
                "false_low": second.price,
                "neckline": neckline,
                "recovery_bar": recovery_bar,
            },
            volume_confirmed=vol_confirmed,
            pattern_start_idx=first.index,
            pattern_end_idx=second.index,
            breakout_idx=breakout.bar_index if breakout else None,
        ))

    return results


# ---------------------------------------------------------------------------
# Pattern 6: False Breakout H&S Top
# ---------------------------------------------------------------------------


def _detect_false_breakout_hs(
    symbol: str,
    df: pd.DataFrame,
    swing_points: list[SwingPoint],
    config: StockScreenerConfig,
) -> list[PatternSignal]:
    """H&S top where the head is a false breakout above the left shoulder.

    The head breaks above the left shoulder's high (false breakout), then
    price reverses and forms the right shoulder, eventually breaking the neckline.
    """
    results: list[PatternSignal] = []
    highs = _swing_highs(swing_points)
    closes = df["close"].to_numpy(dtype=np.float64)

    for i in range(2, len(highs)):
        left_sh = highs[i - 2]
        head = highs[i - 1]
        right_sh = highs[i]

        # Head must be highest (and above left shoulder = false breakout)
        if head.price <= left_sh.price or head.price <= right_sh.price:
            continue

        # Shoulders approximately equal
        if not _within_tolerance(left_sh.price, right_sh.price, 0.05):
            continue

        if not _span_ok(left_sh.index, right_sh.index, config):
            continue

        # Head must reverse back below left shoulder within 1-5 bars
        # (false breakout component)
        false_bo_confirmed = False
        check_end = min(head.index + 6, len(closes))
        for j in range(head.index + 1, check_end):
            if closes[j] < left_sh.price:
                false_bo_confirmed = True
                break

        if not false_bo_confirmed:
            continue

        # Find neckline from swing lows between shoulders
        low1 = _find_swing_low_between(swing_points, left_sh.index, head.index)
        low2 = _find_swing_low_between(swing_points, head.index, right_sh.index)
        if low1 is None or low2 is None:
            continue
        neckline = (low1.price + low2.price) / 2

        breakout = _check_breakout(df, neckline, "down", right_sh.index, config)
        status = "confirmed" if breakout and not breakout.false_breakout else "forming"
        vol_confirmed = breakout.with_volume if breakout else False

        targets = compute_hs_targets(
            neckline, head.price, "bearish", config.stop_buffer_pct
        )

        results.append(PatternSignal(
            symbol=symbol,
            pattern_type="false_breakout_hs",
            direction="bearish",
            status=status,
            confidence=0.0,
            entry_price=targets.entry,
            stop_loss=targets.stop_loss,
            target_wave1=targets.target_wave1,
            target_wave2=targets.target_wave2,
            risk_pct=targets.risk_pct,
            reward_pct=targets.reward_pct,
            rr_ratio=targets.rr_ratio,
            neckline=neckline,
            key_points={
                "left_shoulder": left_sh.price,
                "head": head.price,
                "right_shoulder": right_sh.price,
                "neckline": neckline,
            },
            volume_confirmed=vol_confirmed,
            pattern_start_idx=left_sh.index,
            pattern_end_idx=right_sh.index,
            breakout_idx=breakout.bar_index if breakout else None,
        ))

    return results


# ---------------------------------------------------------------------------
# Pattern 7: Bull Flag
# ---------------------------------------------------------------------------


def _detect_bull_flag(
    symbol: str,
    df: pd.DataFrame,
    swing_points: list[SwingPoint],
    config: StockScreenerConfig,
) -> list[PatternSignal]:
    results: list[PatternSignal] = []
    highs_arr = df["high"].to_numpy(dtype=np.float64)
    lows_arr = df["low"].to_numpy(dtype=np.float64)

    lows = _swing_lows(swing_points)
    highs = _swing_highs(swing_points)

    for i in range(len(lows) - 1):
        pole_bottom = lows[i]
        # Find the next swing high after pole_bottom = pole top
        pole_tops = [h for h in highs if h.index > pole_bottom.index]
        if not pole_tops:
            continue
        pole_top = pole_tops[0]

        pole_height = pole_top.price - pole_bottom.price
        if pole_height <= 0:
            continue

        # Look for pullback: 3+ bars after pole_top with lower highs + lower lows
        flag_start = pole_top.index + 1
        flag_bars = 0
        for j in range(flag_start, min(flag_start + 30, len(highs_arr))):
            if j >= len(highs_arr):
                break
            flag_bars += 1
            if flag_bars < 3:
                continue
            # Check descending channel: compare highs and lows
            flag_highs = highs_arr[flag_start: j + 1]
            flag_lows = lows_arr[flag_start: j + 1]
            if len(flag_highs) < 3:
                continue

            # Lower highs: each high <= previous
            lower_highs = all(
                flag_highs[k] <= flag_highs[k - 1] for k in range(1, len(flag_highs))
            )
            # Lower lows: each low <= previous
            lower_lows = all(
                flag_lows[k] <= flag_lows[k - 1] for k in range(1, len(flag_lows))
            )
            if not (lower_highs and lower_lows):
                continue

            # Retracement check: 30-60% of pole
            pullback_low = float(np.min(flag_lows))
            retracement = (pole_top.price - pullback_low) / pole_height
            if not (0.30 <= retracement <= 0.60):
                continue

            # Check breakout above flag upper trendline or pole_top
            breakout_level = float(flag_highs[-1])  # flag's upper edge
            breakout = _check_breakout(df, breakout_level, "up", j, config)
            status = "confirmed" if breakout and not breakout.false_breakout else "forming"
            vol_confirmed = breakout.with_volume if breakout else False

            targets = compute_flag_targets(
                pole_bottom.price, pole_top.price, pullback_low,
                "bullish", config.stop_buffer_pct,
            )

            results.append(PatternSignal(
                symbol=symbol,
                pattern_type="bull_flag",
                direction="bullish",
                status=status,
                confidence=0.0,
                entry_price=targets.entry,
                stop_loss=targets.stop_loss,
                target_wave1=targets.target_wave1,
                target_wave2=targets.target_wave2,
                risk_pct=targets.risk_pct,
                reward_pct=targets.reward_pct,
                rr_ratio=targets.rr_ratio,
                neckline=breakout_level,
                key_points={
                    "pole_bottom": pole_bottom.price,
                    "pole_top": pole_top.price,
                    "pullback_low": pullback_low,
                    "flag_start": flag_start,
                    "flag_end": j,
                },
                volume_confirmed=vol_confirmed,
                pattern_start_idx=pole_bottom.index,
                pattern_end_idx=j,
                breakout_idx=breakout.bar_index if breakout else None,
            ))
            break  # One flag per pole

    return results


# ---------------------------------------------------------------------------
# Pattern 8: Bear Flag
# ---------------------------------------------------------------------------


def _detect_bear_flag(
    symbol: str,
    df: pd.DataFrame,
    swing_points: list[SwingPoint],
    config: StockScreenerConfig,
) -> list[PatternSignal]:
    results: list[PatternSignal] = []
    highs_arr = df["high"].to_numpy(dtype=np.float64)
    lows_arr = df["low"].to_numpy(dtype=np.float64)

    lows = _swing_lows(swing_points)
    highs = _swing_highs(swing_points)

    for i in range(len(highs) - 1):
        pole_top = highs[i]
        # Find the next swing low after pole_top = pole bottom
        pole_bottoms = [lo for lo in lows if lo.index > pole_top.index]
        if not pole_bottoms:
            continue
        pole_bottom = pole_bottoms[0]

        pole_height = pole_top.price - pole_bottom.price
        if pole_height <= 0:
            continue

        # Look for pullback: 3+ bars after pole_bottom with higher highs + higher lows
        flag_start = pole_bottom.index + 1
        flag_bars = 0
        for j in range(flag_start, min(flag_start + 30, len(lows_arr))):
            if j >= len(lows_arr):
                break
            flag_bars += 1
            if flag_bars < 3:
                continue

            flag_highs = highs_arr[flag_start: j + 1]
            flag_lows = lows_arr[flag_start: j + 1]
            if len(flag_highs) < 3:
                continue

            # Higher highs and higher lows (ascending channel = bear flag)
            higher_highs = all(
                flag_highs[k] >= flag_highs[k - 1] for k in range(1, len(flag_highs))
            )
            higher_lows = all(
                flag_lows[k] >= flag_lows[k - 1] for k in range(1, len(flag_lows))
            )
            if not (higher_highs and higher_lows):
                continue

            # Retracement check: 30-60% of pole
            pullback_high = float(np.max(flag_highs))
            retracement = (pullback_high - pole_bottom.price) / pole_height
            if not (0.30 <= retracement <= 0.60):
                continue

            breakout_level = float(flag_lows[-1])  # flag's lower edge
            breakout = _check_breakout(df, breakout_level, "down", j, config)
            status = "confirmed" if breakout and not breakout.false_breakout else "forming"
            vol_confirmed = breakout.with_volume if breakout else False

            targets = compute_flag_targets(
                pole_top.price, pole_bottom.price, pullback_high,
                "bearish", config.stop_buffer_pct,
            )

            results.append(PatternSignal(
                symbol=symbol,
                pattern_type="bear_flag",
                direction="bearish",
                status=status,
                confidence=0.0,
                entry_price=targets.entry,
                stop_loss=targets.stop_loss,
                target_wave1=targets.target_wave1,
                target_wave2=targets.target_wave2,
                risk_pct=targets.risk_pct,
                reward_pct=targets.reward_pct,
                rr_ratio=targets.rr_ratio,
                neckline=breakout_level,
                key_points={
                    "pole_top": pole_top.price,
                    "pole_bottom": pole_bottom.price,
                    "pullback_high": pullback_high,
                    "flag_start": flag_start,
                    "flag_end": j,
                },
                volume_confirmed=vol_confirmed,
                pattern_start_idx=pole_top.index,
                pattern_end_idx=j,
                breakout_idx=breakout.bar_index if breakout else None,
            ))
            break

    return results


# ---------------------------------------------------------------------------
# Pattern 9: H&S Bottom
# ---------------------------------------------------------------------------


def _detect_hs_bottom(
    symbol: str,
    df: pd.DataFrame,
    swing_points: list[SwingPoint],
    config: StockScreenerConfig,
) -> list[PatternSignal]:
    results: list[PatternSignal] = []
    lows = _swing_lows(swing_points)

    for i in range(2, len(lows)):
        left_sh = lows[i - 2]
        head = lows[i - 1]
        right_sh = lows[i]

        # Head must be the lowest
        if head.price >= left_sh.price or head.price >= right_sh.price:
            continue

        # Shoulders approximately equal (within 5%)
        if not _within_tolerance(left_sh.price, right_sh.price, 0.05):
            continue

        if not _span_ok(left_sh.index, right_sh.index, config):
            continue

        # Find neckline from swing highs between shoulders
        high1 = _find_swing_high_between(swing_points, left_sh.index, head.index)
        high2 = _find_swing_high_between(swing_points, head.index, right_sh.index)
        if high1 is None or high2 is None:
            continue
        neckline = (high1.price + high2.price) / 2

        breakout = _check_breakout(df, neckline, "up", right_sh.index, config)
        status = "confirmed" if breakout and not breakout.false_breakout else "forming"
        vol_confirmed = breakout.with_volume if breakout else False

        targets = compute_hs_targets(
            neckline, head.price, "bullish", config.stop_buffer_pct
        )

        results.append(PatternSignal(
            symbol=symbol,
            pattern_type="hs_bottom",
            direction="bullish",
            status=status,
            confidence=0.0,
            entry_price=targets.entry,
            stop_loss=targets.stop_loss,
            target_wave1=targets.target_wave1,
            target_wave2=targets.target_wave2,
            risk_pct=targets.risk_pct,
            reward_pct=targets.reward_pct,
            rr_ratio=targets.rr_ratio,
            neckline=neckline,
            key_points={
                "left_shoulder": left_sh.price,
                "head": head.price,
                "right_shoulder": right_sh.price,
                "neckline": neckline,
            },
            volume_confirmed=vol_confirmed,
            pattern_start_idx=left_sh.index,
            pattern_end_idx=right_sh.index,
            breakout_idx=breakout.bar_index if breakout else None,
        ))

    return results


# ---------------------------------------------------------------------------
# Pattern 10: H&S Top
# ---------------------------------------------------------------------------


def _detect_hs_top(
    symbol: str,
    df: pd.DataFrame,
    swing_points: list[SwingPoint],
    config: StockScreenerConfig,
) -> list[PatternSignal]:
    results: list[PatternSignal] = []
    highs = _swing_highs(swing_points)

    for i in range(2, len(highs)):
        left_sh = highs[i - 2]
        head = highs[i - 1]
        right_sh = highs[i]

        # Head must be the highest
        if head.price <= left_sh.price or head.price <= right_sh.price:
            continue

        # Shoulders approximately equal
        if not _within_tolerance(left_sh.price, right_sh.price, 0.05):
            continue

        if not _span_ok(left_sh.index, right_sh.index, config):
            continue

        # Find neckline from swing lows between shoulders
        low1 = _find_swing_low_between(swing_points, left_sh.index, head.index)
        low2 = _find_swing_low_between(swing_points, head.index, right_sh.index)
        if low1 is None or low2 is None:
            continue
        neckline = (low1.price + low2.price) / 2

        breakout = _check_breakout(df, neckline, "down", right_sh.index, config)
        status = "confirmed" if breakout and not breakout.false_breakout else "forming"
        vol_confirmed = breakout.with_volume if breakout else False

        targets = compute_hs_targets(
            neckline, head.price, "bearish", config.stop_buffer_pct
        )

        results.append(PatternSignal(
            symbol=symbol,
            pattern_type="hs_top",
            direction="bearish",
            status=status,
            confidence=0.0,
            entry_price=targets.entry,
            stop_loss=targets.stop_loss,
            target_wave1=targets.target_wave1,
            target_wave2=targets.target_wave2,
            risk_pct=targets.risk_pct,
            reward_pct=targets.reward_pct,
            rr_ratio=targets.rr_ratio,
            neckline=neckline,
            key_points={
                "left_shoulder": left_sh.price,
                "head": head.price,
                "right_shoulder": right_sh.price,
                "neckline": neckline,
            },
            volume_confirmed=vol_confirmed,
            pattern_start_idx=left_sh.index,
            pattern_end_idx=right_sh.index,
            breakout_idx=breakout.bar_index if breakout else None,
        ))

    return results


# ---------------------------------------------------------------------------
# Pattern 11: Sym Triangle Bottom
# ---------------------------------------------------------------------------


def _detect_sym_triangle_bottom(
    symbol: str,
    df: pd.DataFrame,
    swing_points: list[SwingPoint],
    config: StockScreenerConfig,
) -> list[PatternSignal]:
    return _detect_sym_triangle(symbol, df, swing_points, config, direction="bullish")


# ---------------------------------------------------------------------------
# Pattern 12: Sym Triangle Top
# ---------------------------------------------------------------------------


def _detect_sym_triangle_top(
    symbol: str,
    df: pd.DataFrame,
    swing_points: list[SwingPoint],
    config: StockScreenerConfig,
) -> list[PatternSignal]:
    return _detect_sym_triangle(symbol, df, swing_points, config, direction="bearish")


def _detect_sym_triangle(
    symbol: str,
    df: pd.DataFrame,
    swing_points: list[SwingPoint],
    config: StockScreenerConfig,
    direction: str,
) -> list[PatternSignal]:
    """Detect symmetrical triangle with breakout in given direction.

    Requires converging trendlines (descending upper + ascending lower) with
    at least 2 touch points on each line.  Breakout should occur at 1/2 to 3/4
    of the triangle's horizontal length.
    """
    results: list[PatternSignal] = []
    highs = _swing_highs(swing_points)
    lows = _swing_lows(swing_points)

    if len(highs) < 2 or len(lows) < 2:
        return results

    # Try consecutive pairs of swing highs for upper trendline
    for hi in range(len(highs) - 1):
        h1 = highs[hi]
        h2 = highs[hi + 1]

        # Upper line must be descending
        if h2.price >= h1.price:
            continue

        # Find at least 2 swing lows between/around these highs for lower line
        matching_lows = [
            lo for lo in lows
            if h1.index <= lo.index <= h2.index + config.swing_lookback
        ]
        if len(matching_lows) < 2:
            continue

        l1 = matching_lows[0]
        l2 = matching_lows[-1]

        # Lower line must be ascending
        if l2.price <= l1.price:
            continue

        if not _span_ok(h1.index, max(h2.index, l2.index), config):
            continue

        # Triangle must converge: upper descending, lower ascending
        # Compute intersection point (apex)
        upper_slope = (h2.price - h1.price) / (h2.index - h1.index) if h2.index != h1.index else 0
        lower_slope = (l2.price - l1.price) / (l2.index - l1.index) if l2.index != l1.index else 0

        if upper_slope >= lower_slope:
            continue  # Not converging

        # Apex bar (where lines intersect)
        # h1.price + upper_slope * (apex - h1.index) = l1.price + lower_slope * (apex - l1.index)
        denom = lower_slope - upper_slope
        if denom == 0:
            continue
        apex_idx = (
            (h1.price - l1.price + lower_slope * l1.index - upper_slope * h1.index) / denom
        )
        triangle_length = apex_idx - h1.index
        if triangle_length <= 0:
            continue

        # Breakout should happen at 1/2 to 3/4 of triangle length
        breakout_zone_start = int(h1.index + triangle_length * 0.5)
        breakout_zone_end = int(h1.index + triangle_length * 0.75)

        # Determine breakout level at the midpoint of the breakout zone
        mid_zone = (breakout_zone_start + breakout_zone_end) // 2
        upper_at_mid = h1.price + upper_slope * (mid_zone - h1.index)
        lower_at_mid = l1.price + lower_slope * (mid_zone - l1.index)

        triangle_high = max(h1.price, h2.price)
        triangle_low = min(l1.price, l2.price)

        if direction == "bullish":
            breakout_level = upper_at_mid
            breakout = _check_breakout(df, breakout_level, "up", breakout_zone_start, config)
        else:
            breakout_level = lower_at_mid
            breakout = _check_breakout(df, breakout_level, "down", breakout_zone_start, config)

        status = "confirmed" if breakout and not breakout.false_breakout else "forming"
        vol_confirmed = breakout.with_volume if breakout else False

        targets = compute_triangle_targets(
            triangle_high, triangle_low, breakout_level, direction, config.stop_buffer_pct
        )

        pattern_type = (
            "sym_triangle_bottom" if direction == "bullish" else "sym_triangle_top"
        )

        results.append(PatternSignal(
            symbol=symbol,
            pattern_type=pattern_type,
            direction=direction,
            status=status,
            confidence=0.0,
            entry_price=targets.entry,
            stop_loss=targets.stop_loss,
            target_wave1=targets.target_wave1,
            target_wave2=targets.target_wave2,
            risk_pct=targets.risk_pct,
            reward_pct=targets.reward_pct,
            rr_ratio=targets.rr_ratio,
            neckline=breakout_level,
            key_points={
                "upper_start": h1.price,
                "upper_end": h2.price,
                "lower_start": l1.price,
                "lower_end": l2.price,
                "triangle_high": triangle_high,
                "triangle_low": triangle_low,
            },
            volume_confirmed=vol_confirmed,
            pattern_start_idx=h1.index,
            pattern_end_idx=max(h2.index, l2.index),
            breakout_idx=breakout.bar_index if breakout else None,
        ))

    return results
