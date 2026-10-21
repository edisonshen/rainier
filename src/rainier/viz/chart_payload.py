"""JSON payload for the fengshen.dev S/R chart: recent candles per TF + current zones."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from rainier.core.types import SRLevel, Timeframe


def sr_chart_payload(
    symbol: str,
    asof: datetime,
    price: float,
    levels: list[SRLevel],
    data: dict[Timeframe, pd.DataFrame],
    bars: dict[Timeframe, int],
) -> dict:
    """``candles[tf]`` rows are ``[unix_seconds, open, high, low, close]``."""
    candles = {}
    for tf, n in bars.items():
        if tf not in data:
            continue
        df = data[tf].tail(n)
        t = pd.to_datetime(df["timestamp"], utc=True).astype("int64") // 10**9
        candles[tf.value] = [
            [int(ts), round(o, 2), round(h, 2), round(lo, 2), round(c, 2)]
            for ts, o, h, lo, c in zip(t, df["open"], df["high"], df["low"], df["close"],
                                       strict=True)
        ]
    return {
        "symbol": symbol,
        "asof": asof.isoformat(),
        "price": price,
        "zones": [{
            "price": lv.price, "low": lv.zone_low, "high": lv.zone_high,
            "role": lv.role.value, "strength": round(lv.strength, 4),
            "touches": lv.touches, "timeframes": [t.value for t in lv.timeframes],
        } for lv in levels],
        "candles": candles,
    }
