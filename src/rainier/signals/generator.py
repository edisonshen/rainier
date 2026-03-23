"""Generate trade signals from analysis results: entry, SL, TP, confidence."""

from __future__ import annotations

import pandas as pd

from rainier.core.config import SignalConfig
from rainier.core.types import (
    AnalysisResult,
    Direction,
    PinBar,
    Signal,
    SignalStatus,
    SRLevel,
)

from .scorer import score_setup


def generate_signals(
    result: AnalysisResult,
    df: pd.DataFrame,
    config: SignalConfig | None = None,
) -> list[Signal]:
    """Convert pin bars from analysis into actionable trade signals.

    For each pin bar:
    1. Entry = S/R level price (limit order)
    2. Stop loss = beyond pin bar wick
    3. Take profit = next S/R level in trade direction, or default R:R
    4. Score the setup
    5. Filter by min confidence
    """
    if config is None:
        config = SignalConfig()

    all_signals: list[Signal] = []

    for pin_bar in result.pin_bars:
        confidence = score_setup(
            pin_bar, df, result.bias, config.scorer, sr_levels=result.sr_levels,
        )

        if confidence < config.scorer.min_confidence:
            continue

        entry, sl, tp = _compute_levels(pin_bar, result.sr_levels, config)
        if entry is None:
            continue

        # Check minimum R:R
        risk = abs(entry - sl)
        if risk > 0:
            rr = abs(tp - entry) / risk
            if rr < config.min_rr_ratio:
                continue

        all_signals.append(
            Signal(
                symbol=result.symbol,
                timeframe=result.timeframe,
                direction=pin_bar.direction,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                confidence=confidence,
                timestamp=pin_bar.candle.timestamp,
                status=SignalStatus.PENDING,
                pin_bar=pin_bar,
                sr_level=pin_bar.nearest_sr,
            )
        )

    # Deduplicate: at the same S/R level + direction, keep only the highest confidence signal
    return _dedup_signals(all_signals)


def _compute_levels(
    pin_bar: PinBar,
    sr_levels: list[SRLevel],
    config: SignalConfig,
) -> tuple[float | None, float, float]:
    """Compute entry, stop loss, and take profit.

    Returns (entry, stop_loss, take_profit) or (None, 0, 0) if invalid.
    """
    if pin_bar.nearest_sr is None:
        return None, 0.0, 0.0

    sr_price = pin_bar.nearest_sr.price_at(pin_bar.index)
    candle = pin_bar.candle

    if pin_bar.direction == Direction.LONG:
        # Bullish pin bar near support → buy
        entry = sr_price
        sl = candle.low - candle.range * 0.1  # just below the wick
        risk = entry - sl
        if risk <= 0:
            return None, 0.0, 0.0

        # TP: next resistance above entry, or default R:R
        tp = _find_next_sr(entry, Direction.LONG, pin_bar.index, sr_levels)
        if tp is None:
            tp = entry + risk * config.default_rr_target

    else:
        # Bearish pin bar near resistance → sell
        entry = sr_price
        sl = candle.high + candle.range * 0.1  # just above the wick
        risk = sl - entry
        if risk <= 0:
            return None, 0.0, 0.0

        # TP: next support below entry, or default R:R
        tp = _find_next_sr(entry, Direction.SHORT, pin_bar.index, sr_levels)
        if tp is None:
            tp = entry - risk * config.default_rr_target

    return entry, sl, tp


def _find_next_sr(
    entry: float,
    direction: Direction,
    bar_index: int,
    sr_levels: list[SRLevel],
) -> float | None:
    """Find the next S/R level in the trade direction for take profit."""
    candidates: list[tuple[float, float]] = []

    for level in sr_levels:
        price = level.price_at(bar_index)

        if direction == Direction.LONG and price > entry:
            # Looking for resistance above for TP
            candidates.append((price, level.strength))
        elif direction == Direction.SHORT and price < entry:
            # Looking for support below for TP
            candidates.append((price, level.strength))

    if not candidates:
        return None

    # Pick the nearest S/R level in the trade direction
    if direction == Direction.LONG:
        candidates.sort(key=lambda x: x[0])
    else:
        candidates.sort(key=lambda x: -x[0])

    return candidates[0][0]


def _dedup_signals(signals: list[Signal]) -> list[Signal]:
    """At the same entry price + direction, keep only the highest confidence signal."""
    best: dict[tuple[float, str], Signal] = {}

    for sig in signals:
        key = (round(sig.entry_price, 2), sig.direction.value)
        if key not in best or sig.confidence > best[key].confidence:
            best[key] = sig

    return sorted(best.values(), key=lambda s: s.confidence, reverse=True)
