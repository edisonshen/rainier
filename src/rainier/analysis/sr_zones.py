"""Multi-timeframe S/R zones from touch-and-reaction points (1D -> 4H -> 1H, then consolidate).

Per timeframe:
  1. Reaction points: swing wick tips whose prominence (the move away from them) is
     >= ``prom_atr_mult`` x ATR. Pin-bar tips get extra weight; older points decay with
     a half-life.
  2. Lines: weighted Gaussian KDE over the reaction prices (bandwidth = ``bandwidth_atr_mult``
     x ATR). Each density peak with >= ``min_touches`` points becomes a line priced at the
     weighted median of its points (a real wick price), with a 20-80% quantile zone.

Across timeframes (highest first), with distances in trading-reference ATR (1H):
  * within ``tight_atr_mult`` -> tuned into one line (higher-TF price kept)
  * within ``band_atr_mult``  -> widened into a band

Validated on MES (tune Jan-Jun 2026, holdout Aug-Oct 2026): zones confirmed on >= 2
timeframes, touched by a 5m rejection candle, beat randomly shifted zones in both periods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from rainier.core.config import SRZoneConfig, SRZoneTimeframeConfig
from rainier.core.types import SRLevel, SRRole, SRType, Timeframe

from .pivots import compute_atr

TF_ORDER = [Timeframe.W1, Timeframe.D1, Timeframe.H4, Timeframe.H1,
            Timeframe.M30, Timeframe.M15, Timeframe.M5, Timeframe.M1]


@dataclass(frozen=True, slots=True)
class ReactionPoint:
    price: float
    index: int
    prominence_atr: float
    is_pin: bool


@dataclass(slots=True)
class Zone:
    price: float
    low: float
    high: float
    score: float
    touches: int
    pins: int
    timeframes: list[Timeframe] = field(default_factory=list)


def reaction_points(df: pd.DataFrame, atr: float, prom_atr_mult: float) -> list[ReactionPoint]:
    """Wick tips that price touched and then moved away from by >= prom_atr_mult x ATR."""
    hi, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)
    op, cl = df["open"].to_numpy(float), df["close"].to_numpy(float)
    n = len(df)
    pts: list[ReactionPoint] = []
    for series, is_high in ((hi, True), (-lo, False)):
        peaks, props = find_peaks(series, prominence=prom_atr_mult * atr)
        for i, prom in zip(peaks, props["prominences"], strict=True):
            if i > n - 3:  # reaction not confirmed yet
                continue
            rng = hi[i] - lo[i]
            wick = hi[i] - max(op[i], cl[i]) if is_high else min(op[i], cl[i]) - lo[i]
            pts.append(ReactionPoint(
                price=float(hi[i] if is_high else lo[i]), index=int(i),
                prominence_atr=min(float(prom) / atr, 3.0),
                is_pin=bool(rng > 0 and wick / rng >= 2 / 3),
            ))
    return pts


def timeframe_zones(
    df: pd.DataFrame, tf_cfg: SRZoneTimeframeConfig, cfg: SRZoneConfig,
) -> list[Zone]:
    """KDE lines (with quantile zones) from one timeframe's reaction points."""
    df = df.iloc[-tf_cfg.lookback:]
    if len(df) < 20:
        return []
    atr = float(compute_atr(df).iloc[-1])
    if not atr > 0:
        return []
    pts = reaction_points(df, atr, cfg.prom_atr_mult)
    if len(pts) < cfg.min_touches:
        return []
    n = len(df)
    prices = np.array([p.price for p in pts])
    weights = np.array([
        p.prominence_atr * 0.5 ** ((n - 1 - p.index) / tf_cfg.half_life)
        * (cfg.pin_weight if p.is_pin else 1.0)
        for p in pts
    ])
    bw = cfg.bandwidth_atr_mult * atr
    grid = np.arange(prices.min() - bw, prices.max() + bw, bw / 5)
    density = (weights[None, :] * np.exp(-0.5 * ((grid[:, None] - prices[None, :]) / bw) ** 2)).sum(1)
    peaks, _ = find_peaks(density)
    zones: list[Zone] = []
    for g in grid[peaks]:
        member = np.abs(prices - g) <= 1.5 * bw
        if member.sum() < cfg.min_touches:
            continue
        p, w = prices[member], weights[member]
        order = np.argsort(p)
        cum = np.cumsum(w[order])
        price = float(p[order][np.searchsorted(cum, cum[-1] / 2)])
        q_lo, q_hi = np.quantile(p, [0.2, 0.8])
        zones.append(Zone(
            price=price, low=float(min(q_lo, price)), high=float(max(q_hi, price)),
            score=float(w.sum()), touches=int(member.sum()),
            pins=int(sum(pts[i].is_pin for i in np.flatnonzero(member))),
        ))
    return sorted(zones, key=lambda z: -z.score)[:tf_cfg.top]


def consolidate_zones(
    zones_by_tf: dict[Timeframe, list[Zone]], ref_atr: float, cfg: SRZoneConfig,
) -> list[Zone]:
    """Highest TF first: near lines tune into one; farther-but-close lines widen into a band."""
    candidates = [
        Zone(z.price, z.low, z.high, z.score * cfg.timeframes[tf].weight,
             z.touches, z.pins, [tf])
        for tf, zs in zones_by_tf.items() for z in zs
    ]
    candidates.sort(key=lambda z: (TF_ORDER.index(z.timeframes[0]), -z.score))
    tight, band = cfg.tight_atr_mult * ref_atr, cfg.band_atr_mult * ref_atr
    merged: list[Zone] = []
    for z in candidates:
        near = [m for m in merged if abs(m.price - z.price) <= band]
        if not near:
            merged.append(z)
            continue
        m = min(near, key=lambda m: abs(m.price - z.price))
        if abs(m.price - z.price) > tight:
            m.low = min(m.low, z.low, z.price)
            m.high = max(m.high, z.high, z.price)
        m.score += z.score
        m.touches += z.touches
        m.pins += z.pins
        m.timeframes.extend(z.timeframes)
    for m in merged:
        m.low = max(m.low, m.price - band)
        m.high = min(m.high, m.price + band)
    return merged


def detect_mtf_zones(
    data: dict[Timeframe, pd.DataFrame], cfg: SRZoneConfig, current_price: float | None = None,
) -> list[Zone]:
    """Draw zones on every configured timeframe present in ``data`` and consolidate them.

    Callers must pass only closed bars (point-in-time); nothing here looks ahead.
    """
    zones_by_tf = {
        tf: timeframe_zones(data[tf], tf_cfg, cfg)
        for tf, tf_cfg in cfg.timeframes.items() if tf in data
    }
    ref_tf = cfg.reference_tf if cfg.reference_tf in data else next(iter(zones_by_tf), None)
    if ref_tf is None:
        return []
    ref_df = data[ref_tf].iloc[-cfg.timeframes[ref_tf].lookback:] \
        if ref_tf in cfg.timeframes else data[ref_tf]
    ref_atr = float(compute_atr(ref_df).iloc[-1])
    zones = consolidate_zones(zones_by_tf, ref_atr, cfg)
    if current_price is not None:
        zones = [z for z in zones if abs(z.price - current_price) <= cfg.max_distance_atr_mult * ref_atr]
    return zones


def zones_to_levels(zones: list[Zone], current_price: float) -> list[SRLevel]:
    """Convert zones to SRLevels. Role is by side of current price; strength is score / max."""
    top = max((z.score for z in zones), default=1.0) or 1.0
    levels = []
    for z in zones:
        tfs = tuple(dict.fromkeys(z.timeframes))
        levels.append(SRLevel(
            price=z.price,
            sr_type=SRType.HORIZONTAL,
            role=SRRole.SUPPORT if z.price <= current_price else SRRole.RESISTANCE,
            strength=z.score / top,
            touches=z.touches,
            source_tf=min(tfs, key=TF_ORDER.index),
            zone_low=z.low,
            zone_high=z.high,
            timeframes=tfs,
        ))
    return levels
