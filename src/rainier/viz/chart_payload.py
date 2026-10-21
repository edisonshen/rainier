"""JSON payload for the fengshen.dev S/R chart: recent candles per TF + current zones."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from rainier.analysis.indicators import rolling_vwap
from rainier.core.types import CSV_TZ, SRLevel, Timeframe


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
    vma_bars: int = 55,
    vma_offset: int = 2,
) -> dict:
    """``candles[tf]`` rows are ``[unix_seconds, open, high, low, close]``; ``asof`` is UTC ISO.

    ``vma.series[tf]`` rows are ``[unix_seconds, value]``: the rolling VWAP drawn
    ``vma_offset`` bars to the right like TradingView's ``offset=``, so the last points sit
    past the last candle (times extrapolated by the last bar spacing).
    """
    candles, vma = {}, {}
    for tf, n in bars.items():
        if tf not in data:
            continue
        df = data[tf].tail(n)
        t = _utc(df["timestamp"]).astype("int64") // 10**9
        vma[tf.value] = _shifted(t.tolist(), rolling_vwap(data[tf], vma_bars).tail(n).tolist(),
                                 vma_offset)
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
        "vma": {"bars": vma_bars, "offset": vma_offset, "series": vma},
    }


def _shifted(times: list[int], values: list[float], offset: int) -> list[list]:
    if len(times) < 2:
        return []
    step = times[-1] - times[-2]
    plot_t = times[offset:] + [times[-1] + step * k for k in range(1, offset + 1)]
    return [[int(pt), round(v, 2)] for pt, v in zip(plot_t, values, strict=True) if v == v]
