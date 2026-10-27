"""Xiaojiang-style trend channels and trend lines on closed 4H/1D bars (chart overlay only).

Channels: a zigzag of +-``leg_atr`` x ATR splits price into trend legs. Each confirmed leg's
turning point anchors the base line through the leg's later major swings (highs in a down leg,
lows in an up leg). A later swing that pierces the line re-anchors it, if price is back inside
once that swing is known, so lines grow longer and shallower. The parallel line goes through
the opposite swing that the most other swings touch; the midline sits halfway.
``swing_bars`` closes in a row more than ``break_atr`` x ATR through the base line break it.

Trend lines: falling lines through swing highs and rising lines through swing lows, tested
``line_min_tests`` times within ``line_tol_pts`` and never closed through since the anchor.

Everything is in bar-index space (slope per bar of the frame's own timeframe), from the bars
passed in only, so callers pass closed bars for a point-in-time result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rainier.core.config import TrendTimeframeConfig
from rainier.core.types import Timeframe

from .sr_eval import BAR_DURATION

ATR_BARS = 250
MIN_LEG_SWINGS = 6
MAX_OUTSIDE_CLOSES = 0.05
SAME_LINE_PTS = 4.0


@dataclass(frozen=True, slots=True)
class TrendChannel:
    direction: int  # +1 up channel (base line through lows), -1 down (through highs)
    anchor: int
    anchor_price: float
    slope: float
    width: float  # parallel line = base + width
    base_touches: tuple[int, ...]
    parallel_touches: tuple[int, ...]
    broken: int | None

    def base(self, index: np.ndarray | int) -> np.ndarray | float:
        return self.anchor_price + self.slope * (np.asarray(index) - self.anchor)

    def parallel(self, index: np.ndarray | int) -> np.ndarray | float:
        return self.base(index) + self.width


@dataclass(frozen=True, slots=True)
class TrendLine:
    direction: int  # +1 rising support through lows, -1 falling resistance through highs
    anchor: int
    anchor_price: float
    slope: float
    touches: tuple[int, ...]

    def value(self, index: np.ndarray | int) -> np.ndarray | float:
        return self.anchor_price + self.slope * (np.asarray(index) - self.anchor)


def swing_points(x: np.ndarray, k: int, highs: bool) -> list[int]:
    """Bars that are the extreme of the ``k`` bars on each side."""
    if len(x) < 2 * k + 1:
        return []
    win = np.lib.stride_tricks.sliding_window_view(x, 2 * k + 1)
    ext = win.max(1) if highs else win.min(1)
    return (np.nonzero(ext == x[k:len(x) - k])[0] + k).tolist()


def zigzag(hi: np.ndarray, lo: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Turning points ``(bar, +1 high | -1 low)``; the last one is the unconfirmed extreme."""
    points: list[tuple[int, int]] = []
    trend, ext = 0, 0
    for j in range(1, len(hi)):
        if (trend == 1 and hi[j] >= hi[ext]) or (trend == -1 and lo[j] <= lo[ext]):
            ext = j
        if trend == 0:
            if hi[j] - lo[ext] >= threshold or (
                    hi[j] >= hi[ext] and hi[j] - lo[:j + 1].min() >= threshold):
                points.append((int(np.argmin(lo[:j + 1])), -1))
                trend, ext = 1, j
            elif hi[:j + 1].max() - lo[j] >= threshold:
                points.append((int(np.argmax(hi[:j + 1])), 1))
                trend, ext = -1, j
        elif trend == 1 and hi[ext] - lo[j] >= threshold:
            points.append((ext, 1))
            trend, ext = -1, j
        elif trend == -1 and hi[j] - lo[ext] >= threshold:
            points.append((ext, -1))
            trend, ext = 1, j
    points.append((ext, trend))
    return points


def _base_line(p, c, d, anchor, leg_end, swings, k, tol, brk, min_gap):
    """(last base swing, slope, break bar | None) of the line from ``anchor`` through the leg."""
    n = len(c)
    last = slope = None
    for j in swings:
        if not anchor + min_gap <= j <= leg_end or j + k >= n or (p[j] - p[anchor]) * d <= 0:
            continue
        sj = (p[j] - p[anchor]) / (j - anchor)
        known = j + k  # a k-bar swing is confirmed k bars later
        if last is None:
            last, slope = j, sj
        elif ((p[j] - (p[anchor] + slope * (j - anchor))) * d < tol
              and (c[known] - (p[anchor] + sj * (known - anchor))) * d > -brk):
            last, slope = j, sj
    if last is None:
        return None, None, None
    after = np.arange(last + 1, n)
    outside = (c[last + 1:] - (p[anchor] + slope * (after - anchor))) * d < -brk
    run = 0
    for i, out in enumerate(outside):
        run = run + 1 if out else 0
        if run >= k:
            return last, slope, int(last + 1 + i - k + 1)
    return last, slope, None


def _parallel(q, c, d, line, start, end, swings, touch_swings, tol):
    """(width, touches) of the opposite-side line touched by the most swings, or None."""
    best = None
    for o in swings:
        if not start <= o <= end:
            continue
        width = q[o] - line[o]
        if width * d <= 0:
            continue
        par = line + width
        outside = np.sum((c[start:end + 1] - par[start:end + 1]) * d > tol)
        if outside > MAX_OUTSIDE_CLOSES * (end - start):
            continue
        touches = tuple(j for j in touch_swings if start <= j <= end and abs(q[j] - par[j]) <= tol)
        if best is None or (len(touches), -abs(width)) > (len(best[1]), -abs(best[0])):
            best = (float(width), touches)
    return best


def _atr(hi: np.ndarray, lo: np.ndarray) -> float:
    return float(np.mean(hi[-ATR_BARS:] - lo[-ATR_BARS:]))


def find_channels(df: pd.DataFrame, cfg: TrendTimeframeConfig) -> list[TrendChannel]:
    """The last ``cfg.legs`` leg channels that start within the last ``cfg.lookback`` bars."""
    hi, lo, c = (df[x].to_numpy(float) for x in ("high", "low", "close"))
    n = len(c)
    if n < 2 * cfg.swing_bars + 2:
        return []
    atr = _atr(hi, lo)
    tol, brk, k = cfg.tol_atr * atr, cfg.break_atr * atr, cfg.swing_bars
    start = max(0, n - cfg.lookback)
    turns = zigzag(hi, lo, cfg.leg_atr * atr)[:-1]
    out: list[TrendChannel] = []
    for (anchor, kind), (leg_end, _) in zip(turns, [*turns[1:], (n - 1, 0)], strict=True):
        if anchor < start:
            continue
        d = -kind  # a leg from a high runs down
        p, q = (lo, hi) if d == 1 else (hi, lo)
        last, slope, broken = _base_line(p, c, d, anchor, leg_end, swing_points(p, k, d == -1),
                                         k, tol, brk, 2 * k)
        if last is None:
            continue
        end = broken if broken is not None else n - 1
        line = p[anchor] + slope * (np.arange(n) - anchor)
        touch_p = swing_points(p, cfg.touch_bars, d == -1)
        base_touches = tuple(j for j in touch_p
                             if anchor <= j <= end and abs(p[j] - line[j]) <= tol)
        par = _parallel(q, c, d, line, anchor, end, swing_points(q, k, d == 1),
                        swing_points(q, cfg.touch_bars, d == 1), tol)
        if (par is None or len(par[1]) < 2 or len(base_touches) < 2
                or end - anchor < MIN_LEG_SWINGS * k):
            continue
        out.append(TrendChannel(d, int(anchor), float(p[anchor]), float(slope), par[0],
                                base_touches, par[1], broken))
    return out[-cfg.legs:]


def find_trend_lines(df: pd.DataFrame, cfg: TrendTimeframeConfig) -> list[TrendLine]:
    """Unbroken lines through the last ``line_swings`` swings, most-tested per price."""
    hi, lo, c = (df[x].to_numpy(float) for x in ("high", "low", "close"))
    n = len(c)
    a = max(0, n - cfg.line_lookback)
    tol = cfg.line_tol_pts
    found: list[TrendLine] = []
    for highs, p in ((True, hi), (False, lo)):
        d = -1 if highs else 1
        piv = [j for j in swing_points(p[a:], cfg.line_swing_bars, highs)][-cfg.line_swings:]
        for x, j1 in enumerate(piv):
            for j2 in piv[x + 1:]:
                if (p[a + j2] - p[a + j1]) * d <= 0:
                    continue  # falling resistance / rising support only
                slope = (p[a + j2] - p[a + j1]) / (j2 - j1)
                idx = np.arange(a + j1, n)
                line = p[a + j1] + slope * (idx - a - j1)
                if ((c[a + j1:] - line) * d < -tol).any():
                    continue
                touches = idx[np.abs(p[a + j1:] - line) <= tol]
                if len(touches) >= cfg.line_min_tests:
                    found.append(TrendLine(d, a + j1, float(p[a + j1]), float(slope),
                                           tuple(int(t) for t in touches)))
    kept: list[TrendLine] = []
    for ln in sorted(found, key=lambda ln: -len(ln.touches)):
        if all(ln.direction != o.direction or abs(ln.value(n - 1) - o.value(n - 1)) >= SAME_LINE_PTS
               for o in kept):
            kept.append(ln)
    return sorted(kept, key=lambda ln: ln.anchor)


def trend_overlays(
    data: dict[Timeframe, pd.DataFrame], cfg: dict[Timeframe, TrendTimeframeConfig],
    asof: pd.Timestamp,
) -> dict[Timeframe, tuple[list[TrendChannel], list[TrendLine]]]:
    """Channels + lines per configured timeframe from the bars closed by the end of the 5m bar
    at ``asof``; bar indices stay valid for ``data[tf]`` (closed bars are its prefix)."""
    now = pd.Timestamp(asof) + BAR_DURATION[Timeframe.M5]
    out = {}
    for tf, tf_cfg in cfg.items():
        if tf not in data:
            continue
        df = data[tf]
        closed = df[pd.to_datetime(df["timestamp"]) + BAR_DURATION[tf] <= now]
        closed = closed.reset_index(drop=True)
        out[tf] = (find_channels(closed, tf_cfg), find_trend_lines(closed, tf_cfg))
    return out
