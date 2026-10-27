"""Xiaojiang transaction rules on the chart's S/R zones: when to open, hard stop, when to close.

Levels come only from ``analysis.sr_zones`` (via ``ZoneBook``). Rules from the doc:
- buy support below price, sell resistance above;
- left side: a limit rests at the zone; right side: enter on a rejection close after the touch;
- fixed hard stop in points, never moved;
- trade the 1st..Nth touch of a zone, skip later ones; the trend can pick the side;
- take profit at the next zone or at a fixed R.

Fills are conservative: a limit needs one tick through, a bar touching both stop and target is
a loss, the fill bar can stop out but not take profit, and the position is flat before session
gaps (weekends). One MES at a time, round-trip commission, 0.25 pt slippage on stops.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rainier.analysis.pivots import compute_atr
from rainier.core.config import XJSetupConfig
from rainier.core.types import Timeframe

from .zones import SESSION_SHIFT, Lines, ZoneBook

POINT_USD = 5.0
COMMISSION_USD = 1.24
SLIP_PTS = 0.25
TICK = 0.25
GAP = pd.Timedelta("3h")
ZONE_TOL = 3.0  # same zone across hourly snapshots if its center moved <= this
AWAY_PTS = 10.0  # price must leave a zone by this much before the next touch counts
COLS = ["t_in", "t_out", "dir", "line", "entry", "exit", "px_out", "pts", "usd", "risk",
        "stop", "tp", "zone_lo", "zone_hi", "touch", "bias"]


def _week_close(t: pd.Timestamp) -> bool:
    """The Friday 13:55 PT bar is the last 5m bar of the week."""
    return t.weekday() == 4 and (t.hour, t.minute) >= (13, 55)


@dataclass(frozen=True, slots=True)
class Order:
    direction: int
    line: float  # entry level: zone edge or center
    stop: float
    zone: int  # zone id (stable across hourly snapshots)
    lo: float
    hi: float


class Market:
    """5m bars + point-in-time zones + per-zone touch history, precomputed once per book."""

    def __init__(self, data: dict[Timeframe, pd.DataFrame], book: ZoneBook):
        m5 = data[Timeframe.M5].reset_index(drop=True)
        self.t = m5["timestamp"]
        self.o, self.hi, self.lo, self.c = (m5[k].to_numpy(float) for k in
                                            ("open", "high", "low", "close"))
        self.session = (self.t + SESSION_SHIFT).dt.normalize().to_numpy()
        self.sess_idx = np.unique(self.session, return_inverse=True)[1]
        gap = ((self.t.shift(-1) - self.t) > GAP).to_numpy()
        gap[-1] = _week_close(self.t.iloc[-1])  # data end mid-week: the position stays open
        self.last_before_gap = gap
        self.book = book
        self._d1 = data.get(Timeframe.D1)
        self._bias: dict[str, np.ndarray] = {}
        self._centers: list[float] = []
        self._center_ids: list[int] = []
        self.snap_ids = [self.ids_for(ln, add=True) for ln in book.snapshots]
        self._touches()

    def ids_for(self, lines: Lines, add: bool = False) -> list[int]:
        """Zone ids by center proximity; unseen zones get new ids (``add``) or negative ones."""
        row = []
        cs, ids = self._centers, self._center_ids
        for z in lines.price:
            k = bisect_left(cs, z)
            best = min((j for j in (k - 1, k) if 0 <= j < len(cs)),
                       key=lambda j: abs(cs[j] - z), default=None)
            if best is not None and abs(cs[best] - z) <= ZONE_TOL:
                row.append(ids[best])
            elif add:
                cs.insert(k, z)
                ids.insert(k, len(ids))
                row.append(ids[k])
            else:
                row.append(-1 - len(row))
        return row

    def _touches(self) -> None:
        armed: dict[int, bool] = {}
        tb: dict[int, list[int]] = defaultdict(list)
        hob = self.book.hour_of_bar
        for i in range(len(self.c)):
            k = hob[i]
            ln = self.book.snapshots[k]
            lo_i, hi_i = self.lo[i], self.hi[i]
            for zid, zl, zh in zip(self.snap_ids[k], ln.lo, ln.hi, strict=True):
                inter = lo_i <= zh and hi_i >= zl
                if zid not in armed:
                    armed[zid] = not inter
                if inter:
                    if armed[zid]:
                        tb[zid].append(i)
                        armed[zid] = False
                elif lo_i >= zh + AWAY_PTS or hi_i <= zl - AWAY_PTS:
                    armed[zid] = True
        self.touch_bars = {z: np.asarray(v) for z, v in tb.items()}

    def prior_touches(self, zid: int, i: int, days: int) -> int:
        """Touches of zone ``zid`` on bars before ``i`` within the last ``days`` sessions."""
        v = self.touch_bars.get(zid)
        if v is None or not len(v):
            return 0
        n = int(np.searchsorted(v, i))
        if n == 0:
            return 0
        now = self.sess_idx[min(i, len(self.sess_idx) - 1)]
        return int((self.sess_idx[v[:n]] > now - days).sum())

    def bias(self, name: str) -> np.ndarray:
        if name == "none":
            return np.zeros(len(self.t), dtype=int)
        if name not in self._bias:
            no_d1 = self._d1 is None or self._d1.empty
            self._bias[name] = (np.zeros(len(self.t), dtype=int) if no_d1
                                else _trend_d1(self._d1, self.t))
        return self._bias[name]


def _trend_d1(d1: pd.DataFrame, t: pd.Series) -> np.ndarray:
    """Daily close vs its 20-day mean with a +-0.5 ATR dead band (range), as of each 5m bar."""
    d = d1.reset_index(drop=True)
    sma, atr = d.close.rolling(20).mean(), compute_atr(d)
    b = np.where(d.close > sma + 0.5 * atr, 1, np.where(d.close < sma - 0.5 * atr, -1, 0))
    known = (d.timestamp + pd.Timedelta("1D")).to_numpy("datetime64[ns]")
    k = np.searchsorted(known, t.to_numpy("datetime64[ns]"), side="right") - 1
    return np.where(k >= 0, b[np.clip(k, 0, None)], 0).astype(int)


def plan_orders(mkt: Market, i: int, lines: Lines, ids: list[int], price: float,
                p: XJSetupConfig, bias: int, used: dict[int, int]) -> list[Order]:
    """Orders for bar ``i``: the nearest eligible zones below (buy) and above (sell) ``price``."""
    k = lines.below(price)
    out: list[Order] = []
    for d, idx in ((1, range(k - 1, max(-1, k - 1 - p.k_near), -1)),
                   (-1, range(k, min(len(lines.price), k + p.k_near)))):
        if bias and d != bias:
            continue
        for j in idx:
            if lines.rank[j] < p.min_rank or lines.nconf[j] < p.min_conf:
                continue
            px = lines.price[j] if p.level == "center" else (lines.hi[j] if d > 0 else lines.lo[j])
            if d * (price - px) <= 0 or used.get(ids[j], 0) >= p.per_line:
                continue
            if mkt.prior_touches(ids[j], i, p.test_days) >= p.max_tests:
                continue
            out.append(Order(d, px, px - d * p.stop_pts, ids[j], lines.lo[j], lines.hi[j]))
    return out


def target(lines: Lines, ids: list[int], d: int, entry: float, stop: float, zone: int,
           p: XJSetupConfig) -> float:
    """Fixed R for rrN; else the near edge of the next zone >= 1R away (2R if there is none)."""
    risk = abs(entry - stop)
    if p.exit.startswith("rr"):
        return entry + d * risk * float(p.exit[2:])
    ahead = [lo if d > 0 else hi for lo, hi, zid in zip(lines.lo, lines.hi, ids, strict=True)
             if zid != zone and d * ((lo if d > 0 else hi) - entry) >= risk]
    return min(ahead, key=lambda x: d * x) if ahead else entry + d * 2 * risk


@dataclass(slots=True)
class _Pos:
    od: Order
    entry: float
    tp: float
    i: int
    touch: int
    bias: int


@dataclass
class SimResult:
    trades: pd.DataFrame
    open: dict | None = None  # position still held at the last bar
    used: dict[int, int] = field(default_factory=dict)
    armed: list[tuple[Order, int]] = field(default_factory=list)


def simulate(mkt: Market, p: XJSetupConfig, start: int = 0, end: int | None = None) -> SimResult:
    """Bar-by-bar over ``[start, end)``, one MES at a time."""
    end = len(mkt.c) if end is None else end
    o, h, lo, c = mkt.o, mkt.hi, mkt.lo, mkt.c
    trades: list[tuple] = []
    pos: _Pos | None = None
    used: dict[int, int] = {}
    armed: list[tuple[Order, int]] = []
    sess = None
    bias_arr = mkt.bias(p.bias)

    def close(i: int, px: float, why: str) -> None:
        nonlocal pos
        od, pts = pos.od, pos.od.direction * (px - pos.entry)
        trades.append((mkt.t[pos.i], mkt.t[i], od.direction, od.line, pos.entry, why, px, pts,
                       pts * POINT_USD - COMMISSION_USD, abs(pos.entry - od.stop), od.stop,
                       pos.tp, od.lo, od.hi, pos.touch, pos.bias))
        pos = None

    def open_(i: int, od: Order, entry: float, lines: Lines, ids: list[int], bias: int) -> None:
        nonlocal pos
        pos = _Pos(od, entry, target(lines, ids, od.direction, entry, od.stop, od.zone, p), i,
                   mkt.prior_touches(od.zone, i, p.test_days) + 1, bias)
        used[od.zone] = used.get(od.zone, 0) + 1

    for i in range(max(start, 1), end):
        if mkt.session[i] != sess:
            sess, used, armed = mkt.session[i], {}, []
        if pos is not None:
            d, stop = pos.od.direction, pos.od.stop
            if (d > 0 and lo[i] <= stop) or (d < 0 and h[i] >= stop):
                close(i, (min(o[i], stop) if d > 0 else max(o[i], stop)) - d * SLIP_PTS, "stop")
            elif d * (h[i] if d > 0 else lo[i]) >= d * pos.tp + TICK:
                close(i, pos.tp, "target")
            elif mkt.last_before_gap[i]:
                close(i, c[i] - d * SLIP_PTS, "session_end")
            continue
        if mkt.last_before_gap[i]:
            continue
        k = mkt.book.hour_of_bar[i]
        lines, ids = mkt.book.snapshots[k], mkt.snap_ids[k]
        bias = int(bias_arr[i])
        if p.entry == "right":
            for od, j in armed:
                ok = od.direction * (c[i] - od.line) > 0 and od.direction * (c[i] - o[i]) > 0
                if 0 < i - j <= p.confirm_bars and ok:
                    open_(i, od, c[i] + od.direction * SLIP_PTS, lines, ids, bias)
                    break
            armed = [(od, j) for od, j in armed if i - j < p.confirm_bars and
                     od.direction * (od.stop - (lo[i] if od.direction > 0 else h[i])) < 0]
            if pos is not None:
                armed = []
                continue
        orders = plan_orders(mkt, i, lines, ids, c[i - 1], p, bias, used)
        fills = [od for od in orders if (od.direction > 0 and lo[i] <= od.line - TICK) or
                 (od.direction < 0 and h[i] >= od.line + TICK)]
        fills = [od for od in fills if od.direction * (o[i] - od.stop) > 0]
        if not fills:
            continue
        od = min(fills, key=lambda x: abs(x.line - o[i]))
        if p.entry == "right":
            if all(a[0].zone != od.zone for a in armed):
                armed.append((od, i))
            continue
        open_(i, od, min(o[i], od.line) if od.direction > 0 else max(o[i], od.line),
              lines, ids, bias)
        if (od.direction > 0 and lo[i] <= od.stop) or (od.direction < 0 and h[i] >= od.stop):
            close(i, od.stop - od.direction * SLIP_PTS, "stop")
        elif mkt.last_before_gap[i]:
            close(i, c[i] - od.direction * SLIP_PTS, "session_end")
    held = None
    if pos is not None:
        od = pos.od
        held = {"t_in": mkt.t[pos.i], "dir": od.direction, "line": od.line, "entry": pos.entry,
                "stop": od.stop, "tp": pos.tp, "zone_lo": od.lo, "zone_hi": od.hi}
    return SimResult(pd.DataFrame(trades, columns=COLS), held, used, armed)


def next_orders(mkt: Market, res: SimResult, p: XJSetupConfig, lines: Lines) -> list[dict]:
    """What the setup is waiting for on the next bar, on ``lines`` (the zones the chart draws).

    Nothing while a position is open (one at a time) or after the week's last bar. Left side:
    resting limits with stop and take-profit. Right side: zones already touched and waiting for
    the rejection close, then zones where a touch would arm one (the take-profit depends on that
    close, so it is ``None``).
    """
    if res.open is not None or mkt.last_before_gap[-1]:
        return []
    n = len(mkt.c)
    ids = mkt.ids_for(lines)
    out: list[dict] = []
    pending = [od for od, _ in res.armed] if p.entry == "right" else []
    for od in pending:
        k = next((k for k in range(len(lines.price))
                  if lines.lo[k] - TICK <= od.line <= lines.hi[k] + TICK), None)
        if k is not None:
            out.append({"dir": od.direction, "price": od.line, "stop": od.stop, "tp": None,
                        "kind": "rejection", "zone_lo": lines.lo[k], "zone_hi": lines.hi[k]})
    seen = {(o["dir"], o["zone_lo"]) for o in out}
    for od in plan_orders(mkt, n, lines, ids, float(mkt.c[-1]), p, int(mkt.bias(p.bias)[-1]),
                          res.used):
        if (od.direction, od.lo) in seen:
            continue
        tp = (target(lines, ids, od.direction, od.line, od.stop, od.zone, p)
              if p.entry == "left" else None)
        out.append({"dir": od.direction, "price": od.line, "stop": od.stop, "tp": tp,
                    "kind": "limit" if p.entry == "left" else "rejection",
                    "zone_lo": od.lo, "zone_hi": od.hi})
    return out


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
