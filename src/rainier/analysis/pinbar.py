"""Pin bar detection and pin bar line (S/R) derivation.

Xiaojiang methodology:
1. Find candles with significant wicks showing rejection (relaxed criteria)
2. Draw lines at EXACT wick tip prices (high or low)
3. Cluster nearby wick tips — the price with the most touches wins
4. Do on 1D first (major levels), then 1H (minor levels)
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from rainier.core.config import PinBarConfig
from rainier.core.types import Candle, Direction, PinBar, SRLevel, SRRole, SRType, Timeframe


def detect_pin_bars_raw(
    df: pd.DataFrame,
    config: PinBarConfig | None = None,
    symbol: str = "",
    timeframe: Timeframe = Timeframe.H1,
) -> list[PinBar]:
    """Detect pin bars per Xiaojiang book methodology.

    Book rules (strict):
    1. Dominant wick > 2/3 of total range
    2. Secondary wick < 1/3 of dominant wick (not a spinning top)
    3. Dominant wick >= 2x body
    4. Body in the opposite half from dominant wick
    5. Visually prominent: amplitude >= median of recent bars ("nose vs face")
    """
    if config is None:
        config = PinBarConfig()

    pin_bars: list[PinBar] = []

    # Pre-compute recent bar ranges for "visually prominent" check
    ranges = (df["high"] - df["low"]).values

    for i in range(1, len(df)):
        row = df.iloc[i]
        candle = Candle(
            timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0)),
            symbol=symbol,
            timeframe=timeframe,
        )

        if candle.range == 0:
            continue

        upper = candle.upper_wick
        lower = candle.lower_wick
        body = candle.body_size if candle.body_size > 0 else candle.range * 0.01

        dominant_wick = max(upper, lower)
        secondary_wick = min(upper, lower)

        # Rule 1: Dominant wick > 2/3 of range
        if dominant_wick / candle.range < config.min_dominant_wick_ratio:
            continue

        # Rule 2: Secondary wick < 1/3 of dominant (no spinning tops)
        if dominant_wick > 0 and secondary_wick / dominant_wick > config.max_secondary_wick_ratio:
            continue

        # Rule 3: Wick/body ratio
        is_bearish_pin = upper > lower and upper / body >= config.min_wick_body_ratio
        is_bullish_pin = lower > upper and lower / body >= config.min_wick_body_ratio

        if not is_bearish_pin and not is_bullish_pin:
            continue

        # Rule 4: Body in opposite half from dominant wick
        mid = (candle.high + candle.low) / 2
        if is_bullish_pin and candle.body_bottom < mid:
            continue
        if is_bearish_pin and candle.body_top > mid:
            continue

        # Rule 5: Visually prominent — amplitude >= median of recent bars
        lookback_start = max(0, i - config.min_amplitude_lookback)
        if i - lookback_start >= 5:  # need enough history
            recent_ranges = ranges[lookback_start:i]
            median_range = float(np.median(recent_ranges))
            if candle.range < median_range:
                continue

        direction = Direction.LONG if is_bullish_pin else Direction.SHORT
        wick_ratio = (lower / body) if is_bullish_pin else (upper / body)

        pin_bars.append(
            PinBar(
                candle=candle,
                index=i,
                direction=direction,
                wick_ratio=wick_ratio,
            )
        )

    return pin_bars


def derive_pin_bar_lines(
    pin_bars: list[PinBar],
    atr: float,
    cluster_atr_mult: float = 0.15,
    tick_size: float = 0.25,
    min_touches: int = 3,
) -> list[SRLevel]:
    """Derive horizontal S/R levels from pin bar wick tips.

    Algorithm:
    1. Extract exact wick tip price from each pin bar
       - Bullish pin → low (lower wick tip)
       - Bearish pin → high (upper wick tip)
    2. Cluster nearby tips (within cluster_atr_mult * ATR)
    3. Within each cluster, the line price = the wick tip that has the most
       touches (mode), NOT the average. Keeps the line at an exact price.
    4. More touches = stronger level
    """
    if not pin_bars:
        return []

    # Extract wick tip prices
    tips: list[tuple[float, PinBar]] = []
    for pb in pin_bars:
        if pb.direction == Direction.LONG:
            tips.append((pb.candle.low, pb))
        else:
            tips.append((pb.candle.high, pb))

    tips.sort(key=lambda t: t[0])

    cluster_dist = cluster_atr_mult * atr

    # Greedy clustering
    clusters: list[list[tuple[float, PinBar]]] = []
    current: list[tuple[float, PinBar]] = [tips[0]]

    for tip in tips[1:]:
        if tip[0] - current[-1][0] <= cluster_dist:
            current.append(tip)
        else:
            clusters.append(current)
            current = [tip]
    clusters.append(current)

    # Filter: need min_touches pin bar wick tips to form a valid level
    clusters = [c for c in clusters if len(c) >= min_touches]

    # Build S/R levels
    levels: list[SRLevel] = []
    for cluster in clusters:
        prices = [t[0] for t in cluster]
        bars = [t[1] for t in cluster]
        touches = len(cluster)

        # Line price = the most common wick tip price (mode), not average
        # Round to tick_size for counting
        rounded = [round(p / tick_size) * tick_size for p in prices]
        price_counts = Counter(rounded)
        line_price = price_counts.most_common(1)[0][0]

        # Determine role by majority direction
        n_bullish = sum(1 for pb in bars if pb.direction == Direction.LONG)
        role = SRRole.SUPPORT if n_bullish >= len(bars) - n_bullish else SRRole.RESISTANCE

        # Strength: more touches = stronger
        strength = min(0.3 + touches * 0.15, 1.0)

        levels.append(
            SRLevel(
                price=line_price,
                sr_type=SRType.HORIZONTAL,
                role=role,
                strength=strength,
                touches=touches,
                first_seen=min(pb.candle.timestamp for pb in bars),
                last_tested=max(pb.candle.timestamp for pb in bars),
            )
        )

    return sorted(levels, key=lambda l: l.strength, reverse=True)


def match_pin_bars_to_levels(
    pin_bars: list[PinBar],
    sr_levels: list[SRLevel],
    proximity_pct: float = 0.005,
) -> list[PinBar]:
    """Match raw pin bars to S/R levels, returning only those near a level."""
    matched: list[PinBar] = []

    for pb in pin_bars:
        nearest, dist_pct = _find_nearest_sr(pb, sr_levels, proximity_pct)
        if nearest is not None:
            matched.append(
                PinBar(
                    candle=pb.candle,
                    index=pb.index,
                    direction=pb.direction,
                    wick_ratio=pb.wick_ratio,
                    nearest_sr=nearest,
                    sr_distance_pct=dist_pct,
                )
            )

    return matched


def _find_nearest_sr(
    pb: PinBar,
    sr_levels: list[SRLevel],
    proximity_pct: float,
) -> tuple[SRLevel | None, float]:
    """Find the nearest S/R level to a pin bar's wick tip."""
    if not sr_levels or pb.candle.close == 0:
        return None, 0.0

    if pb.direction == Direction.LONG:
        tip_price = pb.candle.low
    else:
        tip_price = pb.candle.high

    best_level: SRLevel | None = None
    best_dist_pct = float("inf")

    for level in sr_levels:
        level_price = level.price_at(pb.index)
        dist = abs(tip_price - level_price)
        dist_pct = dist / pb.candle.close

        if dist_pct <= proximity_pct and dist_pct < best_dist_pct:
            best_dist_pct = dist_pct
            best_level = level

    if best_level is None:
        return None, 0.0

    return best_level, best_dist_pct
