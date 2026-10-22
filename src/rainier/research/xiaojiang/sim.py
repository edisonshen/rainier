"""Buy/exit signals from the line book and a 5m bar-by-bar MES simulation of them."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from rainier.core.types import Timeframe
from rainier.research.xiaojiang.lines import SESSION_SHIFT, LineBook, Lines

POINT_USD = 5.0  # MES
COMMISSION_USD = 1.24  # round trip, assumed
SLIP_PTS = 0.25  # one tick on stop / market fills; limits fill at their price
TICK = 0.25
GAP = pd.Timedelta("3h")


@dataclass(frozen=True)
class XJParams:
    """Defaults = the doc's rules with its own filters (thick lines, confluence, early touches)."""

    entry: str = "left"  # left: resting limit at the line; right: enter on rejection close
    stop_pts: float = 12.0  # doc: ES 6, doubled for MES
    exit: str = "scale"  # next | next_thick | line2 | scale | rr1 .. rr5
    min_rank: int = 3  # thickest source needed: 1=15m, 2=1H/prior day, 3=4H/prior week, 4=1D
    min_conf: int = 2  # distinct sources merged into the line (confluence)
    max_touches: int = 2  # skip lines tested more often than this (doc: many touches -> break)
    k_near: int = 3  # doc: the 2-3 lines above/below price are the day's range
    bias: str = "none"  # none | trend (inferred: 20-day mean +- 0.5 ATR decides the side)
    per_line: int = 1  # trades per line per session (1 = first touch only)
    confirm_bars: int = 3  # right side: bars after the touch allowed to confirm
    breakeven: bool = False  # not in the doc: her hard stop stays put; True moves it after TP1

    def label(self) -> str:
        return " ".join(f"{k}={v}" for k, v in asdict(self).items())


@dataclass(frozen=True, slots=True)
class Order:
    direction: int  # +1 buy, -1 sell
    line: float
    stop: float


def plan_orders(lines: Lines, price: float, p: XJParams, bias: int = 0,
                used: list[float] | None = None, tol: float = 3.0) -> list[Order]:
    """The day's resting orders: buy eligible lines below ``price``, sell eligible lines above."""
    i = lines.below(price)
    out: list[Order] = []
    for d, idx in ((1, range(i - 1, max(-1, i - 1 - p.k_near), -1)),
                   (-1, range(i, min(len(lines.price), i + p.k_near)))):
        if bias and d != bias:
            continue
        for j in idx:
            px = lines.price[j]
            if px == price or lines.rank[j] < p.min_rank or lines.nconf[j] < p.min_conf \
                    or lines.touches[j] > p.max_touches:
                continue
            if used and sum(abs(u - px) <= tol for u in used) >= p.per_line:
                continue
            out.append(Order(d, px, px - d * p.stop_pts))
    return out


def targets(lines: Lines, d: int, entry: float, stop: float, p: XJParams) -> list[float]:
    """Take-profit prices (each >=1R away), nearest first; ``scale`` exits half at each of two."""
    risk = abs(entry - stop)
    if p.exit.startswith("rr"):
        return [entry + d * risk * float(p.exit[2:])]
    ahead = [(px, r) for px, r in zip(lines.price, lines.rank, strict=True)
             if d * (px - entry) >= risk]
    ahead.sort(key=lambda x: d * x[0])
    if p.exit == "next_thick":
        ahead = [a for a in ahead if a[1] >= max(2, p.min_rank)]
    if p.exit == "line2":  # hold through the next line to the one after
        ahead = ahead[1:]
    tps = [a[0] for a in ahead[: 2 if p.exit == "scale" else 1]]
    return tps or [entry + d * 2 * risk]


@dataclass(slots=True)
class _Pos:
    d: int
    entry: float
    stop: float
    tps: list[float]
    size: float
    t_in: int
    line: float
    pnl: float = 0.0  # realised points * size so far


class Market:
    """5m bars plus everything the simulation needs per bar (precomputed once)."""

    def __init__(self, data: dict[Timeframe, pd.DataFrame], book: LineBook):
        m5 = data[Timeframe.M5].reset_index(drop=True)
        self.t = m5["timestamp"]
        self.o, self.h, self.l, self.c = (m5[k].to_numpy(float) for k in
                                          ("open", "high", "low", "close"))
        self.session = (self.t + SESSION_SHIFT).dt.normalize().to_numpy()
        nxt = self.t.shift(-1)
        self.last_before_gap = ((nxt - self.t) > GAP).to_numpy() | nxt.isna().to_numpy()
        self.book = book
        self.bias = _trend_bias(data.get(Timeframe.D1), self.t)


def _trend_bias(d1: pd.DataFrame | None, t: pd.Series) -> np.ndarray:
    if d1 is None or d1.empty:
        return np.zeros(len(t), dtype=int)
    d = d1.reset_index(drop=True)
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(),
                    (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    sma, atr = d.close.rolling(20).mean(), tr.rolling(14).mean()
    b = np.where(d.close > sma + 0.5 * atr, 1, np.where(d.close < sma - 0.5 * atr, -1, 0))
    known = (d.timestamp + pd.Timedelta("1D")).to_numpy("datetime64[ns]")
    k = np.searchsorted(known, t.to_numpy("datetime64[ns]"), side="right") - 1
    return np.where(k >= 0, b[np.clip(k, 0, None)], 0)


def simulate(mkt: Market, p: XJParams, start: int = 0, end: int | None = None,
             rng: np.random.Generator | None = None, shift=(6.0, 20.0)) -> pd.DataFrame:
    """Trade the plan bar by bar (one MES position at a time). ``rng`` = random-line control.

    Conservative fills: a limit needs one tick of trade-through, a bar touching both stop and
    target is a loss, the fill bar can stop out but not take profit, flat before session gaps.
    """
    end = len(mkt.c) if end is None else end
    o, h, l, c = mkt.o, mkt.h, mkt.l, mkt.c
    trades: list[tuple] = []
    pos: _Pos | None = None
    used: list[float] = []
    armed: list[tuple[Order, int]] = []
    sess = None
    shifted: dict[int, Lines] = {}

    def lines_for(i: int) -> Lines:
        ln = mkt.book.for_bar(i)
        if rng is None:
            return ln
        k = int(mkt.book.hour_of_bar[i])
        if k not in shifted:
            shifted[k] = ln.shifted(rng, *shift)
        return shifted[k]

    def close(i: int, px: float, why: str) -> None:
        nonlocal pos
        pts = pos.pnl + pos.d * (px - pos.entry) * pos.size
        trades.append((mkt.t[pos.t_in], mkt.t[i], pos.d, pos.line, pos.entry, why,
                       pts, pts * POINT_USD - COMMISSION_USD, abs(pos.entry - pos.stop)))
        pos = None

    for i in range(max(start, 1), end):
        if mkt.session[i] != sess:
            sess, used, armed = mkt.session[i], [], []
        if pos is not None:
            hit_stop = (pos.d > 0 and l[i] <= pos.stop) or (pos.d < 0 and h[i] >= pos.stop)
            if hit_stop:
                px = min(o[i], pos.stop) if pos.d > 0 else max(o[i], pos.stop)
                close(i, px - pos.d * SLIP_PTS, "stop" if pos.pnl == 0 else "stop_after_tp")
            else:
                while pos.tps and pos.d * (h[i] if pos.d > 0 else l[i]) >= \
                        pos.d * pos.tps[0] + TICK:
                    tp = pos.tps.pop(0)
                    part = pos.size if not pos.tps else pos.size / 2
                    pos.pnl += pos.d * (tp - pos.entry) * part
                    pos.size -= part
                    if p.breakeven:
                        pos.stop = pos.entry
                    if pos.size <= 1e-9:
                        pos.size, pos.tps = 0.0, []
                        close(i, tp, "target")
                        break
            if pos is not None and mkt.last_before_gap[i]:
                close(i, c[i] - pos.d * SLIP_PTS, "session_end")
            continue
        if mkt.last_before_gap[i]:
            continue
        lines = lines_for(i)
        bias = int(mkt.bias[i]) if p.bias == "trend" else 0
        if p.entry == "right":
            for od, k in armed:
                ok = od.direction * (c[i] - od.line) > 0 and od.direction * (c[i] - o[i]) > 0
                if 0 < i - k <= p.confirm_bars and ok:
                    entry = c[i] + od.direction * SLIP_PTS
                    pos = _Pos(od.direction, entry, od.stop,
                               targets(lines, od.direction, entry, od.stop, p), 1.0,
                               i, od.line)
                    used.append(od.line)
                    break
            armed = [(od, k) for od, k in armed if i - k < p.confirm_bars and
                     not (od.direction * (od.stop - (l[i] if od.direction > 0 else h[i])) >= 0)]
            if pos is not None:
                armed = []
                continue
        orders = plan_orders(lines, c[i - 1], p, bias, used)
        fills = [od for od in orders if (od.direction > 0 and l[i] <= od.line - TICK) or
                 (od.direction < 0 and h[i] >= od.line + TICK)]
        if not fills:
            continue
        # A bar opening beyond the stop means the line already broke: no trade.
        fills = [od for od in fills if od.direction * (o[i] - od.stop) > 0]
        if not fills:
            continue
        od = min(fills, key=lambda x: abs(x.line - o[i]))
        if p.entry == "right":
            if all(a[0].line != od.line for a in armed):
                armed.append((od, i))
            continue
        entry = min(o[i], od.line) if od.direction > 0 else max(o[i], od.line)
        pos = _Pos(od.direction, entry, od.stop, targets(lines, od.direction, entry, od.stop, p),
                   1.0, i, od.line)
        used.append(od.line)
        if (od.direction > 0 and l[i] <= od.stop) or (od.direction < 0 and h[i] >= od.stop):
            close(i, od.stop - od.direction * SLIP_PTS, "stop")
        elif mkt.last_before_gap[i]:
            close(i, c[i] - od.direction * SLIP_PTS, "session_end")
    return pd.DataFrame(trades, columns=["t_in", "t_out", "dir", "line", "entry", "exit",
                                         "pts", "usd", "risk"])


def metrics(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"trades": 0, "win": np.nan, "avg_usd": np.nan, "pf": np.nan, "total_usd": 0.0,
                "max_dd_usd": 0.0, "avg_r": np.nan}
    eq = tr["usd"].cumsum()
    gain, loss = tr.usd[tr.usd > 0].sum(), -tr.usd[tr.usd < 0].sum()
    return {"trades": len(tr), "win": float((tr.usd > 0).mean()), "avg_usd": float(tr.usd.mean()),
            "pf": float(gain / loss) if loss else np.inf, "total_usd": float(tr.usd.sum()),
            "max_dd_usd": float((eq.cummax().clip(lower=0) - eq).max()),
            "avg_r": float((tr.pts / tr.risk.clip(lower=TICK)).mean())}
