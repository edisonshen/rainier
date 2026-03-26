"""Orchestrator: runs all detectors and produces an AnalysisResult.

Xiaojiang methodology flow:
1. Find pin bars on each timeframe (1D, 4H, 1H)
2. Derive "pin bar lines" from wick tip clusters on each TF
3. Merge all lines — 1D lines are major, 1H lines are minor
4. Apply all lines to the trading timeframe (5m) chart
"""

from __future__ import annotations

import pandas as pd

from rainier.core.config import AnalysisConfig
from rainier.core.types import AnalysisResult, SRLevel, Timeframe

from .bias import determine_bias
from .inside_bar import detect_inside_bars
from .pinbar import derive_pin_bar_lines, detect_pin_bars_raw, match_pin_bars_to_levels
from .pivots import compute_atr, detect_pivots
from .sr_diagonal import detect_diagonal_sr
from .sr_horizontal import detect_horizontal_sr


def analyze(
    df: pd.DataFrame,
    symbol: str,
    timeframe: Timeframe,
    config: AnalysisConfig | None = None,
    higher_tf_levels: list[SRLevel] | None = None,
    min_touches: int = 3,
    skip_own_levels: bool = False,
) -> AnalysisResult:
    """Run the full analysis pipeline on a single timeframe.

    Args:
        higher_tf_levels: S/R levels from higher timeframes to include.
        min_touches: minimum pin bar wick tips to form a level (per-symbol).
        skip_own_levels: if True, don't derive S/R from this TF (use only
            higher_tf_levels). Used when this is the trading TF in multi-TF
            analysis — the trading TF shows data, higher TFs provide levels.
    """
    if config is None:
        config = AnalysisConfig()

    result = AnalysisResult(symbol=symbol, timeframe=timeframe)

    if len(df) < 2:
        return result

    # 1. Pivots + ATR
    pivots = detect_pivots(df, config.pivot)
    result.pivots = pivots
    atr = compute_atr(df)
    avg_atr = float(atr.mean())

    # 2. Detect raw pin bars on this timeframe (always — needed for matching)
    raw_pin_bars = detect_pin_bars_raw(df, config.pin_bar, symbol, timeframe)

    if skip_own_levels:
        # Trading TF: only use higher TF levels, don't derive own S/R
        own_levels: list[SRLevel] = []
        d_levels: list[SRLevel] = []
    else:
        # 3. Derive pin bar lines from this timeframe's pin bars
        pb_lines = derive_pin_bar_lines(raw_pin_bars, avg_atr, min_touches=min_touches)

        # 4. Traditional horizontal S/R from pivot clusters (secondary)
        h_levels = detect_horizontal_sr(pivots, df, atr, config.sr_horizontal)

        # 5. Merge: pin bar lines + pivot-based, dedup
        own_levels = _merge_sr_levels(pb_lines, h_levels, avg_atr * 0.3)
        own_levels = sorted(own_levels, key=lambda l: l.strength, reverse=True)[:config.max_sr_levels]

        # 6. Diagonal trendlines (secondary, capped)
        d_levels = detect_diagonal_sr(pivots, df, atr, config.sr_diagonal)
        d_levels = sorted(d_levels, key=lambda l: l.strength, reverse=True)[:config.max_diagonal_levels]

    # 7. Combine with higher TF levels, filter to visible price range
    if higher_tf_levels:
        all_h = _merge_sr_levels(higher_tf_levels, own_levels, avg_atr * 0.3)
    else:
        all_h = own_levels

    # Clip to trading chart's price range ± 10%
    price_high = df["high"].max()
    price_low = df["low"].min()
    margin = (price_high - price_low) * 0.10
    upper_bound = price_high + margin
    lower_bound = price_low - margin

    all_h = [l for l in all_h if lower_bound <= l.price <= upper_bound]
    d_levels = [l for l in d_levels if lower_bound <= l.price <= upper_bound]

    # Final dedup: lines within 0.15% of each other are the same level — keep most touches
    all_h = sorted(all_h, key=lambda l: l.touches, reverse=True)
    mid_price = (price_high + price_low) / 2
    final_dedup_dist = mid_price * 0.0015  # 0.15% of price
    all_h = _dedup_levels(all_h, final_dedup_dist)

    result.sr_levels = all_h + d_levels

    # 8. Match pin bars to final S/R levels
    result.pin_bars = match_pin_bars_to_levels(
        raw_pin_bars, result.sr_levels, config.pin_bar.sr_proximity_pct
    )

    # 9. Inside bars + bias
    result.inside_bars = detect_inside_bars(df, config.inside_bar, symbol, timeframe)
    result.bias = determine_bias(df, pivots)

    return result


def analyze_multi_tf(
    data: dict[Timeframe, pd.DataFrame],
    symbol: str,
    trading_tf: Timeframe,
    config: AnalysisConfig | None = None,
    min_touches: int = 3,
) -> AnalysisResult:
    """Multi-timeframe analysis: derive pin bar lines from higher TFs,
    apply them to the trading timeframe.

    Args:
        data: dict of timeframe → DataFrame. Must include trading_tf.
        symbol: instrument symbol
        trading_tf: the timeframe to trade on (e.g., 5m)
        config: analysis config

    Example:
        data = {Timeframe.D1: df_daily, Timeframe.H1: df_hourly, Timeframe.M5: df_5m}
        result = analyze_multi_tf(data, "MES", Timeframe.M5)
    """
    if config is None:
        config = AnalysisConfig()

    # Timeframe hierarchy: higher TFs first
    tf_order = [Timeframe.W1, Timeframe.D1, Timeframe.H4, Timeframe.H1,
                Timeframe.M30, Timeframe.M15, Timeframe.M5, Timeframe.M1]

    higher_tfs = [tf for tf in tf_order if tf in data and tf != trading_tf
                  and tf_order.index(tf) < tf_order.index(trading_tf)]

    # Collect pin bar lines from all higher TFs
    all_higher_levels: list[SRLevel] = []

    for tf in higher_tfs:
        tf_df = data[tf]
        if len(tf_df) < 2:
            continue

        atr = compute_atr(tf_df)
        avg_atr = float(atr.mean())

        raw_pbs = detect_pin_bars_raw(tf_df, config.pin_bar, symbol, tf)
        pb_lines = derive_pin_bar_lines(raw_pbs, avg_atr, min_touches=min_touches)

        # Also get pivot-based levels
        pivots = detect_pivots(tf_df, config.pivot)
        h_levels = detect_horizontal_sr(pivots, tf_df, atr, config.sr_horizontal)

        tf_levels = _merge_sr_levels(pb_lines, h_levels, avg_atr * 0.3)
        # Keep only top levels per TF
        tf_levels = sorted(tf_levels, key=lambda l: l.strength, reverse=True)[:config.max_sr_levels]

        # Tag the source timeframe in the level for chart labeling
        for level in tf_levels:
            level.source_tf = tf

        all_higher_levels.extend(tf_levels)

    # Merge across timeframes: when a lower-TF level is near a higher-TF level,
    # absorb it — use the lower-TF price for precision (more data points),
    # keep the higher-TF label, boost strength with confluence.
    if all_higher_levels:
        trading_df = data[trading_tf]
        trading_atr = float(compute_atr(trading_df).mean()) if len(trading_df) > 1 else 1.0
        all_higher_levels = _merge_multi_tf_levels(
            all_higher_levels, tf_order, trading_atr * 0.5,
        )

    # Run analysis on the trading TF with higher TF levels injected.
    # skip_own_levels=True: trading TF only shows data, doesn't derive its own S/R.
    # Book: use higher TF for key levels, trading TF only for signals.
    trading_df = data[trading_tf]
    return analyze(
        trading_df, symbol, trading_tf, config,
        higher_tf_levels=all_higher_levels, min_touches=min_touches,
        skip_own_levels=True,
    )


def _merge_multi_tf_levels(
    levels: list[SRLevel],
    tf_order: list[Timeframe],
    merge_dist: float,
) -> list[SRLevel]:
    """Merge levels across timeframes: higher TF absorbs nearby lower TF levels.

    When a 1D level and a 1H level are at similar prices, keep the 1D level
    but refine its price using the lower-TF level (more data = more precision).
    Boost strength to reflect multi-TF confluence.
    """
    def _tf_rank(tf: Timeframe | None) -> int:
        if tf is None:
            return len(tf_order)
        return tf_order.index(tf) if tf in tf_order else len(tf_order)

    # Sort by TF rank (highest TF first), then by strength
    levels = sorted(levels, key=lambda l: (_tf_rank(l.source_tf), -l.strength))

    merged: list[SRLevel] = []
    for level in levels:
        absorbed = False
        for existing in merged:
            if abs(existing.price - level.price) <= merge_dist:
                # Lower-TF level is near a higher-TF level — absorb it
                # Refine price: use lower-TF price (more precise)
                if _tf_rank(level.source_tf) > _tf_rank(existing.source_tf):
                    existing.price = level.price
                # Boost strength for multi-TF confluence
                existing.strength = min(existing.strength + 0.1, 1.0)
                existing.touches += level.touches
                absorbed = True
                break
        if not absorbed:
            merged.append(level)

    return merged


def _merge_sr_levels(
    primary: list[SRLevel],
    secondary: list[SRLevel],
    dedup_dist: float,
) -> list[SRLevel]:
    """Merge two lists, primary takes precedence for dedup."""
    merged = list(primary)
    for sec in secondary:
        is_dup = any(abs(pri.price - sec.price) <= dedup_dist for pri in merged)
        if not is_dup:
            merged.append(sec)
    return merged


def _dedup_levels(levels: list[SRLevel], dedup_dist: float) -> list[SRLevel]:
    """Deduplicate a list of levels by price proximity. Keeps stronger ones."""
    kept: list[SRLevel] = []
    for level in levels:
        is_dup = any(abs(k.price - level.price) <= dedup_dist for k in kept)
        if not is_dup:
            kept.append(level)
    return kept
