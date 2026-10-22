"""Chronological tuning: grid on the train window, judge on the later holdout, vs random lines."""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from rainier.research.xiaojiang.sim import Market, XJParams, metrics, simulate

GRID = {
    "entry": ["left", "right"],
    "stop_pts": [5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 24.0],  # fixed hard stop, MES points
    "exit": ["next", "next_thick", "line2", "scale", "rr1", "rr2", "rr3", "rr5"],
    "min_rank": [1, 2, 3],
    "min_conf": [1, 2],
    "max_touches": [2, 5, 99],
    "k_near": [2, 3],
    "bias": ["none"],  # strict trend filters lost to "none"; trend_sides=up only matched it
    "per_line": [1, 2],
}


def grid(space: dict[str, list] | None = None) -> list[XJParams]:
    space = space or GRID
    return [XJParams(**dict(zip(space, v, strict=True))) for v in itertools.product(*space.values())]


def split(tr: pd.DataFrame, periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> dict:
    out = {}
    for name, (a, b) in periods.items():
        part = tr[(tr.t_in >= a) & (tr.t_in < b)]
        out[name] = metrics(part)
        months = part.groupby(part.t_in.dt.to_period("M")).usd.sum()
        out[name]["months_pos"] = f"{int((months > 0).sum())}/{len(months)}"
    return out


def evaluate(mkt: Market, p: XJParams, periods: dict, seed: int | None = None) -> dict:
    rng = None if seed is None else np.random.default_rng(seed)
    return split(simulate(mkt, p, rng=rng), periods)


_MKT: Market | None = None
_PERIODS: dict = {}


def _work(p: XJParams) -> tuple[XJParams, dict]:
    return p, evaluate(_MKT, p, _PERIODS)


def run_grid(mkt: Market, params: list[XJParams], periods: dict, workers: int = 8) -> pd.DataFrame:
    """One row per config with ``<period>_<metric>`` columns (forked workers share ``mkt``)."""
    global _MKT, _PERIODS
    _MKT, _PERIODS = mkt, periods
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_work, params, chunksize=16))
    rows = []
    for p, r in res:
        row = {k: getattr(p, k) for k in GRID}
        for per, m in r.items():
            row |= {f"{per}_{k}": v for k, v in m.items()}
        rows.append(row)
    return pd.DataFrame(rows)


def random_control(mkt: Market, p: XJParams, periods: dict, seeds: int = 30) -> pd.DataFrame:
    """Same rules on randomly shifted lines; the real lines should beat most of these."""
    rows = []
    for s in range(seeds):
        r = evaluate(mkt, p, periods, seed=s)
        rows.append({f"{per}_{k}": v for per, m in r.items() for k, v in m.items()} | {"seed": s})
    return pd.DataFrame(rows)
