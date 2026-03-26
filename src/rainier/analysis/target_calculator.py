"""Price target calculator — measured move for all patterns.

Implements measured-move price targets from Caisen's methodology.
Used by all pattern detectors to compute entry, stop-loss, and target levels.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TargetLevels:
    """Computed price levels for a pattern trade setup."""

    entry: float
    stop_loss: float
    target_wave1: float
    target_wave2: float | None
    risk_pct: float  # abs(entry - stop_loss) / entry
    reward_pct: float  # abs(target_wave1 - entry) / entry
    rr_ratio: float  # reward / risk


def _safe_rr(risk_pct: float, reward_pct: float) -> float:
    """Compute reward/risk ratio, returning 0 when risk is zero."""
    if risk_pct == 0:
        return 0.0
    return reward_pct / risk_pct


def _pct(a: float, b: float) -> float:
    """Compute abs(a - b) / b, returning 0 when b is zero."""
    if b == 0:
        return 0.0
    return abs(a - b) / abs(b)


def compute_double_bottom_targets(
    neckline: float,
    bottom1: float,
    bottom2: float,
    stop_buffer_pct: float = 0.02,
) -> TargetLevels:
    """W bottom / double bottom — bullish measured move.

    distance = neckline - avg(bottom1, bottom2)
    target_wave1 = neckline + distance
    target_wave2 = target_wave1 + distance
    """
    avg_bottom = (bottom1 + bottom2) / 2
    distance = neckline - avg_bottom
    entry = neckline
    stop_loss = neckline * (1 - stop_buffer_pct)
    target_wave1 = neckline + distance
    target_wave2 = target_wave1 + distance
    risk_pct = _pct(stop_loss, entry)
    reward_pct = _pct(target_wave1, entry)
    return TargetLevels(
        entry=entry,
        stop_loss=stop_loss,
        target_wave1=target_wave1,
        target_wave2=target_wave2,
        risk_pct=risk_pct,
        reward_pct=reward_pct,
        rr_ratio=_safe_rr(risk_pct, reward_pct),
    )


def compute_double_top_targets(
    neckline: float,
    top1: float,
    top2: float,
    stop_buffer_pct: float = 0.02,
) -> TargetLevels:
    """M top / double top — bearish measured move (mirror of double bottom).

    distance = avg(top1, top2) - neckline
    target_wave1 = neckline - distance
    target_wave2 = target_wave1 - distance
    """
    avg_top = (top1 + top2) / 2
    distance = avg_top - neckline
    entry = neckline
    stop_loss = neckline * (1 + stop_buffer_pct)
    target_wave1 = neckline - distance
    target_wave2 = target_wave1 - distance
    risk_pct = _pct(stop_loss, entry)
    reward_pct = _pct(target_wave1, entry)
    return TargetLevels(
        entry=entry,
        stop_loss=stop_loss,
        target_wave1=target_wave1,
        target_wave2=target_wave2,
        risk_pct=risk_pct,
        reward_pct=reward_pct,
        rr_ratio=_safe_rr(risk_pct, reward_pct),
    )


def compute_hs_targets(
    neckline: float,
    head_price: float,
    direction: str,
    stop_buffer_pct: float = 0.02,
) -> TargetLevels:
    """Head & Shoulders — bullish (inverse H&S) or bearish (H&S top).

    distance = abs(neckline - head_price)
    Bullish: targets above neckline, SL below neckline.
    Bearish: targets below neckline, SL above neckline.
    """
    distance = abs(neckline - head_price)
    entry = neckline

    if direction == "bullish":
        stop_loss = neckline * (1 - stop_buffer_pct)
        target_wave1 = neckline + distance
        target_wave2 = target_wave1 + distance
    elif direction == "bearish":
        stop_loss = neckline * (1 + stop_buffer_pct)
        target_wave1 = neckline - distance
        target_wave2 = target_wave1 - distance
    else:
        raise ValueError(f"direction must be 'bullish' or 'bearish', got {direction!r}")

    risk_pct = _pct(stop_loss, entry)
    reward_pct = _pct(target_wave1, entry)
    return TargetLevels(
        entry=entry,
        stop_loss=stop_loss,
        target_wave1=target_wave1,
        target_wave2=target_wave2,
        risk_pct=risk_pct,
        reward_pct=reward_pct,
        rr_ratio=_safe_rr(risk_pct, reward_pct),
    )


def compute_flag_targets(
    pole_start: float,
    pole_end: float,
    pullback_low: float,
    direction: str,
    stop_buffer_pct: float = 0.02,
) -> TargetLevels:
    """Bull/bear flag — single measured move equal to the pole height.

    Bullish: target = pullback_low + pole_height, SL below pullback_low.
    Bearish: pullback_low acts as pullback_high; target = pullback_high - pole_height,
             SL above pullback_high.
    target_wave2 is always None (single measured move for flags).
    """
    pole_height = abs(pole_end - pole_start)

    if direction == "bullish":
        entry = pole_end
        stop_loss = pullback_low * (1 - stop_buffer_pct)
        target_wave1 = pullback_low + pole_height
    elif direction == "bearish":
        # pullback_low is the pullback_high in bearish context
        entry = pole_end
        stop_loss = pullback_low * (1 + stop_buffer_pct)
        target_wave1 = pullback_low - pole_height
    else:
        raise ValueError(f"direction must be 'bullish' or 'bearish', got {direction!r}")

    risk_pct = _pct(stop_loss, entry)
    reward_pct = _pct(target_wave1, entry)
    return TargetLevels(
        entry=entry,
        stop_loss=stop_loss,
        target_wave1=target_wave1,
        target_wave2=None,
        risk_pct=risk_pct,
        reward_pct=reward_pct,
        rr_ratio=_safe_rr(risk_pct, reward_pct),
    )


def compute_triangle_targets(
    triangle_high: float,
    triangle_low: float,
    breakout_level: float,
    direction: str,
    stop_buffer_pct: float = 0.02,
) -> TargetLevels:
    """Triangle breakout — target equals the widest side of the triangle.

    triangle_side = triangle_high - triangle_low
    Bullish: target = breakout_level + triangle_side, SL below breakout_level.
    Bearish: target = breakout_level - triangle_side, SL above breakout_level.
    target_wave2 is always None.
    """
    triangle_side = triangle_high - triangle_low
    entry = breakout_level

    if direction == "bullish":
        stop_loss = breakout_level * (1 - stop_buffer_pct)
        target_wave1 = breakout_level + triangle_side
    elif direction == "bearish":
        stop_loss = breakout_level * (1 + stop_buffer_pct)
        target_wave1 = breakout_level - triangle_side
    else:
        raise ValueError(f"direction must be 'bullish' or 'bearish', got {direction!r}")

    risk_pct = _pct(stop_loss, entry)
    reward_pct = _pct(target_wave1, entry)
    return TargetLevels(
        entry=entry,
        stop_loss=stop_loss,
        target_wave1=target_wave1,
        target_wave2=None,
        risk_pct=risk_pct,
        reward_pct=reward_pct,
        rr_ratio=_safe_rr(risk_pct, reward_pct),
    )


def compute_false_breakdown_targets(
    support_level: float,
    false_low: float,
    neckline: float | None = None,
    stop_buffer_pct: float = 0.02,
) -> TargetLevels:
    """False breakdown — bullish reversal.

    Entry at support_level recovery, SL below false_low.
    If neckline provided: uses W-bottom measured move from neckline.
    Otherwise: target = support_level + (support_level - false_low) * 2.
    """
    entry = support_level
    stop_loss = false_low * (1 - stop_buffer_pct)

    if neckline is not None:
        distance = neckline - false_low
        target_wave1 = neckline + distance
        target_wave2 = target_wave1 + distance
    else:
        bounce = (support_level - false_low) * 2
        target_wave1 = support_level + bounce
        target_wave2 = None

    risk_pct = _pct(stop_loss, entry)
    reward_pct = _pct(target_wave1, entry)
    return TargetLevels(
        entry=entry,
        stop_loss=stop_loss,
        target_wave1=target_wave1,
        target_wave2=target_wave2,
        risk_pct=risk_pct,
        reward_pct=reward_pct,
        rr_ratio=_safe_rr(risk_pct, reward_pct),
    )


def compute_false_breakout_targets(
    resistance_level: float,
    false_high: float,
    neckline: float | None = None,
    stop_buffer_pct: float = 0.02,
) -> TargetLevels:
    """False breakout — bearish reversal (mirror of false breakdown).

    Entry at resistance_level rejection, SL above false_high.
    If neckline provided: uses M-top measured move from neckline.
    Otherwise: target = resistance_level - (false_high - resistance_level) * 2.
    """
    entry = resistance_level
    stop_loss = false_high * (1 + stop_buffer_pct)

    if neckline is not None:
        distance = false_high - neckline
        target_wave1 = neckline - distance
        target_wave2 = target_wave1 - distance
    else:
        drop = (false_high - resistance_level) * 2
        target_wave1 = resistance_level - drop
        target_wave2 = None

    risk_pct = _pct(stop_loss, entry)
    reward_pct = _pct(target_wave1, entry)
    return TargetLevels(
        entry=entry,
        stop_loss=stop_loss,
        target_wave1=target_wave1,
        target_wave2=target_wave2,
        risk_pct=risk_pct,
        reward_pct=reward_pct,
        rr_ratio=_safe_rr(risk_pct, reward_pct),
    )
