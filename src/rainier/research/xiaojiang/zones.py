"""The levels Xiaojiang trades: the chart's S/R zones (``analysis.sr_zones``), point in time.

There is one S/R finder. The simulator gets each hour's zones from bars closed by then, so
every order rests on a zone drawn on /trading/mes; Xiaojiang's rules only decide the side,
the entry, the hard stop and the exit.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rainier.analysis.sr_zones import Zone, detect_mtf_zones
from rainier.core.config import SRZoneConfig
from rainier.core.types import Timeframe

SESSION_SHIFT = pd.Timedelta("9h")  # PT: the 15:00 open becomes midnight of the session date
_HOUR = np.int64(3_600_000_000_000)
# Thickness of a zone = its highest timeframe (min_rank 2 = 1H, 3 = 4H, 4 = 1D).
TF_RANK = {Timeframe.H1: 2, Timeframe.H4: 3, Timeframe.D1: 4}
_DUR = {Timeframe.H1: pd.Timedelta("1h"), Timeframe.H4: pd.Timedelta("4h"),
        Timeframe.D1: pd.Timedelta("1D")}


@dataclass(slots=True)
class Lines:
    """Zone prices known at one instant, sorted by price."""
    price: list[float]
    rank: list[int]
    nconf: list[int]  # timeframes that drew the zone (confluence)
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


def zones_to_lines(zones: list[Zone]) -> Lines:
    rows = sorted((round(z.price * 4) / 4, max(TF_RANK.get(tf, 1) for tf in z.timeframes),
                   len(set(z.timeframes)), z.touches) for z in zones)
    return Lines(*(list(c) for c in zip(*rows, strict=True))) if rows else Lines([], [], [], [])


class ZoneBook:
    """Hourly snapshots of the chart's zones, each from 1D/4H/1H bars fully closed by then."""

    def __init__(self, data: dict[Timeframe, pd.DataFrame], cfg: SRZoneConfig | None = None):
        self.cfg = cfg or SRZoneConfig()
        self._htf = {}
        for tf in self.cfg.timeframes:
            df = data.get(tf)
            if df is not None and len(df):
                df = df.reset_index(drop=True)
                known = (df.timestamp + _DUR[tf]).to_numpy("datetime64[ns]").astype(np.int64)
                self._htf[tf] = (df, known)
        self._cache: dict[tuple[int, ...], Lines] = {}
        t = data[Timeframe.M5]["timestamp"].to_numpy("datetime64[ns]").astype(np.int64)
        uniq, self.hour_of_bar = np.unique(t - t % _HOUR, return_inverse=True)
        self.snapshots = [self.at(int(h)) for h in uniq]

    def at(self, t: int, bar: int | None = None) -> Lines:
        """Zones known at ``t`` (ns, naive PT). ``bar`` is unused (kept for the book interface)."""
        counts = tuple(int(np.searchsorted(k, t, side="right")) for _, k in self._htf.values())
        if counts not in self._cache:
            past = {tf: df.iloc[:n] for (tf, (df, _)), n in
                    zip(self._htf.items(), counts, strict=True) if n}
            self._cache[counts] = zones_to_lines(detect_mtf_zones(past, self.cfg) if past else [])
        return self._cache[counts]

    def for_bar(self, i: int) -> Lines:
        return self.snapshots[self.hour_of_bar[i]]
