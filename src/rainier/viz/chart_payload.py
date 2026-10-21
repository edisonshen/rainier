"""JSON payload for the fengshen.dev S/R chart: recent candles per TF + current zones."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from rainier.core.types import SRLevel, Timeframe
from rainier.data.csv_provider import CSV_TZ


def _utc(ts: pd.Series) -> pd.Series:
    """Naive timestamps are CSVProvider wall-clock times; the chart needs real UTC."""
    ts = pd.to_datetime(ts)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(CSV_TZ, ambiguous=False, nonexistent="shift_forward")
    return ts.dt.tz_convert("UTC")


def sr_chart_payload(
    symbol: str,
    asof: datetime,
    price: float,
    levels: list[SRLevel],
    data: dict[Timeframe, pd.DataFrame],
    bars: dict[Timeframe, int],
) -> dict:
    """``candles[tf]`` rows are ``[unix_seconds, open, high, low, close]``; ``asof`` is UTC ISO."""
    candles = {}
    for tf, n in bars.items():
        if tf not in data:
            continue
        df = data[tf].tail(n)
        t = _utc(df["timestamp"]).astype("int64") // 10**9
        candles[tf.value] = [
            [int(ts), round(o, 2), round(h, 2), round(lo, 2), round(c, 2)]
            for ts, o, h, lo, c in zip(t, df["open"], df["high"], df["low"], df["close"],
                                       strict=True)
        ]
    return {
        "symbol": symbol,
        "asof": _utc(pd.Series([asof])).iloc[0].isoformat(),
        "price": price,
        "zones": [{
            "price": lv.price, "low": lv.zone_low, "high": lv.zone_high,
            "role": lv.role.value, "strength": round(lv.strength, 4),
            "touches": lv.touches, "timeframes": [t.value for t in lv.timeframes],
        } for lv in levels],
        "candles": candles,
    }
