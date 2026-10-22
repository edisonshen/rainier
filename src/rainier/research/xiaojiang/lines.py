"""Point-in-time horizontal line book: multi-TF swing lines plus the prior session/week box."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rainier.core.types import Timeframe

# Thickness rank of each line source; "PD"/"PW" are the prior session / prior week box edges.
TF_RANK = {"15m": 1, "1H": 2, "4H": 3, "1D": 4, "PD": 2, "PW": 3}
_DUR = {"15m": pd.Timedelta("15min"), "1H": pd.Timedelta("1h"), "4H": pd.Timedelta("4h"),
        "1D": pd.Timedelta("1D")}
SESSION_SHIFT = pd.Timedelta("9h")  # PT: the 15:00 open becomes midnight of the session date
_HOUR = np.int64(3_600_000_000_000)


@dataclass(frozen=True)
class LineConfig:
    pivot_lookback: dict[str, int] = field(
        default_factory=lambda: {"15m": 6, "1H": 5, "4H": 3, "1D": 3})
    window_days: dict[str, int] = field(
        default_factory=lambda: {"15m": 10, "1H": 45, "4H": 180, "1D": 730})
    merge_frac: float = 0.0008  # lines closer than this fraction of price are one line
    box: bool = True


@dataclass(slots=True)
class Lines:
    """Lines known at one instant, sorted by price."""
    price: list[float]
    rank: list[int]
    nconf: list[int]
    touches: list[int]

    def below(self, x: float) -> int:
        """Index of the first line >= x (lines [0, i) are below x)."""
        return bisect_left(self.price, x)

    def shifted(self, rng: np.random.Generator, lo: float, hi: float) -> Lines:
        """Same lines moved by a random +-[lo, hi] points each (random-line control)."""
        d = rng.uniform(lo, hi, len(self.price)) * rng.choice([-1.0, 1.0], len(self.price))
        order = np.argsort(np.asarray(self.price) + d)
        p = (np.asarray(self.price) + d)[order]
        return Lines([float(round(v * 4) / 4) for v in p],
                     [self.rank[k] for k in order], [self.nconf[k] for k in order],
                     [self.touches[k] for k in order])


def resample_15m(m5: pd.DataFrame) -> pd.DataFrame:
    g = m5.set_index("timestamp").resample("15min", label="left", closed="left")
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna(subset=["open"]).reset_index()


def _pivots(df: pd.DataFrame, lb: int, dur: pd.Timedelta) -> pd.DataFrame:
    """Swing highs/lows with the time they become known (lb bars later, at that bar's close)."""
    w = 2 * lb + 1
    hi = df["high"].to_numpy() == df["high"].rolling(w, center=True).max().to_numpy()
    lo = df["low"].to_numpy() == df["low"].rolling(w, center=True).min().to_numpy()
    known = df["timestamp"].shift(-lb) + dur
    rows = [pd.DataFrame({"t": df["timestamp"][m], "known": known[m], "price": df[col][m]})
            for m, col in ((hi, "high"), (lo, "low"))]
    p = pd.concat(rows).dropna()
    return p.sort_values("known").reset_index(drop=True)


def session_boxes(m5: pd.DataFrame) -> pd.DataFrame:
    """Per 5m bar: high/low of the previous completed session and previous completed week."""
    t = m5["timestamp"]
    sdate = (t + SESSION_SHIFT).dt.normalize()
    s = m5.groupby(sdate).agg(high=("high", "max"), low=("low", "min"))
    s["pd_high"], s["pd_low"] = s["high"].shift(1), s["low"].shift(1)
    week = s.index.to_period("W-SAT")  # week runs Sun 15:00 -> Fri close
    w = s.groupby(week).agg(high=("high", "max"), low=("low", "min"))
    w = w.shift(1).rename(columns={"high": "pw_high", "low": "pw_low"})
    s = s.join(w, on=week)
    return s.loc[sdate, ["pd_high", "pd_low", "pw_high", "pw_low"]].reset_index(drop=True)


def _cluster(prices: np.ndarray, tol: float) -> list[tuple[float, int]]:
    if len(prices) == 0:
        return []
    p = np.sort(prices)
    cuts = np.flatnonzero(np.diff(p) > tol) + 1
    return [(float(g.mean()), len(g)) for g in np.split(p, cuts)]


class LineBook:
    """Hourly snapshots of the line set, built only from bars closed before each hour."""

    def __init__(self, data: dict[Timeframe, pd.DataFrame], cfg: LineConfig | None = None):
        self.cfg = cfg or LineConfig()
        m5 = data[Timeframe.M5]
        frames = {"15m": resample_15m(m5), "1H": data.get(Timeframe.H1),
                  "4H": data.get(Timeframe.H4), "1D": data.get(Timeframe.D1)}
        self._piv = {}
        for tf, df in frames.items():
            if df is not None and len(df):
                p = _pivots(df, self.cfg.pivot_lookback[tf], _DUR[tf])
                self._piv[tf] = (p["known"].to_numpy("datetime64[ns]").astype(np.int64),
                                 p["t"].to_numpy("datetime64[ns]").astype(np.int64),
                                 p["price"].to_numpy(float))
        self._boxes = session_boxes(m5) if self.cfg.box else None
        t = m5["timestamp"].to_numpy("datetime64[ns]").astype(np.int64)
        hours = t - t % _HOUR
        uniq, first_idx, self.hour_of_bar = np.unique(hours, return_index=True,
                                                      return_inverse=True)
        self.snapshots = [self.at(int(h), int(i)) for h, i in zip(uniq, first_idx, strict=True)]

    def at(self, t: int, bar: int | None = None) -> Lines:
        """Lines known at time ``t`` (ns, naive PT); ``bar`` is the 5m bar index for the box."""
        raw: list[tuple[float, int, int, int]] = []  # price, rank, tf bit, touches
        price_ref = None
        for bit, (tf, (known, ts, price)) in enumerate(self._piv.items()):
            win = np.int64(self.cfg.window_days[tf]) * 24 * _HOUR
            k = np.searchsorted(known, t, side="right")
            sel = price[:k][ts[:k] >= t - win]
            if price_ref is None and len(sel):
                price_ref = float(sel[-1])
            tol = self.cfg.merge_frac * (price_ref or 1.0)
            raw += [(p, TF_RANK[tf], 1 << bit, n) for p, n in _cluster(sel, tol)]
        if self._boxes is not None and bar is not None:
            row = self._boxes.iloc[bar]
            for bit, name in ((8, "PD"), (9, "PW")):
                for col in (f"{name.lower()}_high", f"{name.lower()}_low"):
                    if row[col] == row[col]:
                        raw.append((float(row[col]), TF_RANK[name], 1 << bit, 1))
        if not raw:
            return Lines([], [], [], [])
        raw.sort()
        tol = self.cfg.merge_frac * raw[len(raw) // 2][0]
        out = Lines([], [], [], [])
        grp = [raw[0]]
        for r in raw[1:] + [None]:
            if r is not None and r[0] - grp[-1][0] <= tol:
                grp.append(r)
                continue
            w = sum(g[1] for g in grp)
            bits = 0
            for g in grp:
                bits |= g[2]
            out.price.append(round(sum(g[0] * g[1] for g in grp) / w * 4) / 4)
            out.rank.append(max(g[1] for g in grp))
            out.nconf.append(bin(bits).count("1"))
            out.touches.append(sum(g[3] for g in grp))
            grp = [r] if r is not None else []
        return out

    def for_bar(self, i: int) -> Lines:
        return self.snapshots[self.hour_of_bar[i]]
