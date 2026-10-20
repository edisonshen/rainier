"""Walk-forward S/R scorecard: do higher-TF zones get more 5m touch-and-reactions than random?

At each step ``t`` on the trading (5m) series, zones are drawn only from higher-TF bars that
had closed by ``t``. Over the following bars, every touch of a zone by a rejection candle
(closes back on the approach side with a >= 50% wick) is resolved as a bounce (+1, price
moves ``up`` x ATR away) or a break (-1, price moves ``dn`` x ATR through) within ``horizon``
bars. The same zones shifted by uniform random offsets (+-``max_shift_atr`` x reference ATR)
are the control. Edge = bounce rate (real) - bounce rate (random).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rainier.core.types import Timeframe

from .pivots import compute_atr
from .sr_zones import Zone

BAR_DURATION = {
    Timeframe.M1: pd.Timedelta(minutes=1), Timeframe.M5: pd.Timedelta(minutes=5),
    Timeframe.M15: pd.Timedelta(minutes=15), Timeframe.M30: pd.Timedelta(minutes=30),
    Timeframe.H1: pd.Timedelta(hours=1), Timeframe.H4: pd.Timedelta(hours=4),
    Timeframe.D1: pd.Timedelta(days=1), Timeframe.W1: pd.Timedelta(weeks=1),
}


@dataclass(frozen=True)
class ReactionTest:
    """One resolution rule: bounce at +up x ATR, break at -dn x ATR, within horizon bars."""
    up: float
    dn: float
    horizon: int


DEFAULT_TESTS = (ReactionTest(2.0, 2.0, 36), ReactionTest(3.0, 3.0, 72), ReactionTest(3.0, 1.5, 72))


def _utc(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True)


def closed_bars(data: dict[Timeframe, pd.DataFrame], now: pd.Timestamp) -> dict[Timeframe, pd.DataFrame]:
    """Point-in-time view: only bars whose close time (start + duration) is <= now."""
    return {tf: df[_utc(df["timestamp"]) + BAR_DURATION[tf] <= now].reset_index(drop=True)
            for tf, df in data.items()}


def touch_outcomes(
    prices: list[float], fut: pd.DataFrame, atr: float, test: ReactionTest, touch_atr: float = 0.5,
) -> list[int]:
    """Bounce(+1)/break(-1)/unresolved(0) for each rejection-candle touch of each price."""
    o, c = fut["open"].to_numpy(float), fut["close"].to_numpy(float)
    h, lo = fut["high"].to_numpy(float), fut["low"].to_numpy(float)
    w = touch_atr * atr
    out: list[int] = []
    for p in prices:
        i = 1
        while i < len(fut):
            side = 1 if c[i - 1] > p + w else (-1 if c[i - 1] < p - w else 0)
            touched = (side == 1 and lo[i] <= p + w) or (side == -1 and h[i] >= p - w)
            rng = h[i] - lo[i]
            wick = (min(o[i], c[i]) - lo[i]) if side == 1 else (h[i] - max(o[i], c[i]))
            if not (touched and rng > 0 and (c[i] - p) * side > 0 and wick / rng >= 0.5):
                i += 1
                continue
            r, j = 0, i + 1
            for j in range(i + 1, min(i + 1 + test.horizon, len(fut))):
                if (c[j] - p) * side >= test.up * atr:
                    r = 1
                    break
                if (c[j] - p) * side <= -test.dn * atr:
                    r = -1
                    break
            out.append(r)
            i = j + 1
    return out


def walk_forward(
    data: dict[Timeframe, pd.DataFrame],
    trading: pd.DataFrame,
    draw: Callable[[dict[Timeframe, pd.DataFrame], float], list[Zone]],
    holdout_start: pd.Timestamp,
    *,
    reference_tf: Timeframe = Timeframe.H1,
    step: int = 48,
    tests: tuple[ReactionTest, ...] = DEFAULT_TESTS,
    n_random: int = 10,
    max_shift_atr: float = 4.0,
    max_gap_days: int = 4,
    warmup: int = 100,
) -> pd.DataFrame:
    """One row per resolved-or-not touch: period, test, n_timeframes, control, result.

    ``draw(closed_data, price)`` returns zones; the random control uses a per-step seed so
    every method evaluated on the same data sees the same shifts.
    """
    trading = trading.reset_index(drop=True)
    ts = _utc(trading["timestamp"])
    atr_t = compute_atr(trading).to_numpy()
    longest = max(t.horizon for t in tests)
    rows = []
    for t in range(warmup, len(trading) - longest, step):
        end = min(t + step + longest, len(trading) - 1)
        if (ts.iloc[end] - ts.iloc[t - 1]).days > max_gap_days:
            continue
        now = ts.iloc[t]
        past = closed_bars(data, now)
        if reference_tf not in past or len(past[reference_tf]) < 20:
            continue
        ref_atr = float(compute_atr(past[reference_tf]).iloc[-1])
        price = float(trading["close"].iloc[t - 1])
        zones = draw(past, price)
        period = "holdout" if now >= holdout_start else "tune"
        fut = trading.iloc[t - 1:end + 1].reset_index(drop=True)
        shifts = np.random.default_rng(t).uniform(
            -max_shift_atr * ref_atr, max_shift_atr * ref_atr, (n_random, len(zones)))
        for ti, test in enumerate(tests):
            for z in zones:
                ntf = len(set(z.timeframes))
                for r in touch_outcomes([z.price], fut, float(atr_t[t - 1]), test):
                    rows.append((period, ti, ntf, False, r))
            for s in shifts:
                for z, d in zip(zones, s, strict=True):
                    ntf = len(set(z.timeframes))
                    for r in touch_outcomes([z.price + d], fut, float(atr_t[t - 1]), test):
                        rows.append((period, ti, ntf, True, r))
    return pd.DataFrame(rows, columns=["period", "test", "n_timeframes", "control", "result"])


def scorecard(outcomes: pd.DataFrame, min_timeframes: int = 2) -> pd.DataFrame:
    """Bounce rate real vs random per (period, test), with edge and its standard error in points."""
    d = outcomes[(outcomes.result != 0) & (outcomes.n_timeframes >= min_timeframes)]
    d = d.assign(bounce=(d.result == 1).astype(float))
    g = d.groupby(["period", "test", "control"]).bounce.agg(["mean", "size"]).unstack("control")
    out = pd.DataFrame({
        "touches": g[("size", False)], "bounce_rate": g[("mean", False)],
        "random_rate": g[("mean", True)],
    })
    out["edge_pts"] = (out.bounce_rate - out.random_rate) * 100
    out["se_pts"] = np.sqrt(out.bounce_rate * (1 - out.bounce_rate) / out.touches) * 100
    return out
