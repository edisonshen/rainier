"""Point-in-time snapshots of the chart's S/R zones (``analysis.sr_zones``). No other levels."""

from __future__ import annotations

import copy
from bisect import bisect_left
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rainier.analysis.sr_zones import Zone, detect_mtf_zones
from rainier.core.config import SRZoneConfig
from rainier.core.types import Timeframe

SESSION_SHIFT = pd.Timedelta("9h")  # PT wall clock: the 15:00 PT open becomes midnight
_HOUR = np.int64(3_600_000_000_000)
TF_RANK = {Timeframe.H1: 2, Timeframe.H4: 3, Timeframe.D1: 4}
_DUR = {Timeframe.H1: pd.Timedelta("1h"), Timeframe.H4: pd.Timedelta("4h"),
        Timeframe.D1: pd.Timedelta("1D")}


def _q(x: float) -> float:
    return round(x * 4) / 4


@dataclass(slots=True)
class Lines:
    """Zones sorted by price: center, band, highest TF rank, TF count, zone-finder touches."""

    price: list[float]
    lo: list[float]
    hi: list[float]
    rank: list[int]
    nconf: list[int]
    touches: list[int]

    def below(self, x: float) -> int:
        return bisect_left(self.price, x)

    def shifted(self, offsets: list[float]) -> Lines:
        d = np.asarray(offsets, dtype=float)
        o = np.argsort(np.asarray(self.price) + d)

        def q(v: list[float]) -> list[float]:
            return [_q(float(x)) for x in (np.asarray(v) + d)[o]]

        return Lines(q(self.price), q(self.lo), q(self.hi), [self.rank[k] for k in o],
                     [self.nconf[k] for k in o], [self.touches[k] for k in o])


def zones_to_lines(zones: list[Zone]) -> Lines:
    rows = sorted((_q(z.price), _q(z.low), _q(z.high),
                   max(TF_RANK.get(tf, 1) for tf in z.timeframes), len(set(z.timeframes)),
                   z.touches) for z in zones)
    if not rows:
        return Lines([], [], [], [], [], [])
    return Lines(*(list(c) for c in zip(*rows, strict=True)))


class ZoneBook:
    """Hourly snapshots of the chart's zones, each from 1D/4H/1H bars closed by that hour."""

    def __init__(self, data: dict[Timeframe, pd.DataFrame], cfg: SRZoneConfig | None = None):
        self.cfg = cfg or SRZoneConfig()
        self._htf: dict[Timeframe, tuple[pd.DataFrame, np.ndarray]] = {}
        for tf in self.cfg.timeframes:
            df = data.get(tf)
            if df is not None and len(df):
                df = df.reset_index(drop=True)
                known = (df.timestamp + _DUR[tf]).to_numpy("datetime64[ns]").astype(np.int64)
                self._htf[tf] = (df, known)
        self._cache: dict[tuple[int, ...], Lines] = {}
        t = data[Timeframe.M5]["timestamp"].to_numpy("datetime64[ns]").astype(np.int64)
        self.hours, self.hour_of_bar = np.unique(t - t % _HOUR, return_inverse=True)
        self.snapshots = [self.at(int(h)) for h in self.hours]

    def at(self, t: int) -> Lines:
        counts = tuple(int(np.searchsorted(k, t, side="right")) for _, k in self._htf.values())
        if counts not in self._cache:
            past = {tf: df.iloc[:n] for (tf, (df, _)), n in
                    zip(self._htf.items(), counts, strict=True) if n}
            self._cache[counts] = zones_to_lines(detect_mtf_zones(past, self.cfg) if past else [])
        return self._cache[counts]

    def shifted(self, seed: int, a: float = 6.0, b: float = 20.0) -> ZoneBook:
        """Random-zone control: each zone moved by its own fixed +-[a, b] points in every hour."""
        rng = np.random.default_rng(seed)
        offset: dict[float, float] = {}
        moved: dict[int, Lines] = {}

        def move(s: Lines) -> Lines:
            for z in s.price:
                if z not in offset:
                    offset[z] = float(rng.uniform(a, b) * rng.choice([-1.0, 1.0]))
            return s.shifted([offset[z] for z in s.price])

        out = copy.copy(self)
        out.snapshots = [moved[id(s)] if id(s) in moved else moved.setdefault(id(s), move(s))
                         for s in self.snapshots]
        return out
