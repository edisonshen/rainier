"""JSON payload for the fengshen.dev S/R chart: recent candles per TF + current zones."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from rainier.analysis.indicators import rolling_vwap
from rainier.analysis.trend_channels import TrendChannel, TrendLine
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
    xj: list[dict] | None = None,
    trend: dict[Timeframe, tuple[list[TrendChannel], list[TrendLine]]] | None = None,
    trend_extend_bars: int = 10,
) -> dict:
    """``candles[tf]`` rows are ``[unix_seconds, open, high, low, close]``; ``asof`` is UTC ISO.

    ``vma.series[tf]`` rows are ``[unix_seconds, value]``: the rolling VWAP drawn
    ``vma_offset`` bars to the right like TradingView's ``offset=``, so the last points sit
    past the last candle (times extrapolated by the last bar spacing).

    ``xj.setups`` (from ``research.xiaojiang.chart.chart_setups``), one per A/B toggle:
    ``trades`` rows ``[t_in, t_out, dir, level, entry, exit, stop, tp, reason, usd, zone_lo,
    zone_hi]``, ``open`` ``[t_in, dir, level, entry, stop, tp, zone_lo, zone_hi]`` or null,
    ``orders`` rows ``[dir, price, stop, tp|null, kind, zone_lo, zone_hi]``.

    ``trend[tf]`` (from ``analysis.trend_channels``, bar indices into ``data[tf]``):
    ``channels`` ``{dir, broken, rows: [[unix_seconds, base, parallel], ...]}`` and ``lines``
    ``{dir, tests, rows: [[unix_seconds, value], ...]}``. Rows are the first in-window bar, the
    last bar (or the break), then one row per projected bar for unbroken ones; the page draws
    straight segments between them, which is the line in bar space.
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
    out = {
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
    if xj:
        out["xj"] = {"setups": [_xj_setup(s) for s in xj]}
    if trend:
        out["trend"] = {tf.value: _trend(data[tf], bars[tf], *found, trend_extend_bars)
                        for tf, found in trend.items() if tf in data and tf in bars}
    return out


def _trend(df: pd.DataFrame, n: int, channels: list[TrendChannel], lines: list[TrendLine],
           extend: int) -> dict:
    t = (_utc(df["timestamp"]).astype("int64") // 10**9).tolist()
    last, first = len(t) - 1, max(0, len(t) - n)
    step = t[-1] - t[-2] if len(t) > 1 else 0
    future = list(range(last + 1, last + 1 + extend)) if step > 0 else []

    def bars(anchor: int, broken: int | None) -> list[int]:
        start, end = max(anchor, first), last if broken is None else min(broken, last)
        if end <= start:
            return []
        return [start, end, *(future if broken is None else [])]

    def at(i: int) -> int:
        return int(t[i] if i <= last else t[last] + step * (i - last))

    chans = []
    for ch in channels:
        idx = bars(ch.anchor, ch.broken)
        if idx:
            chans.append({"dir": ch.direction, "broken": ch.broken is not None,
                          "rows": [[at(i), _px(ch.base(i)), _px(ch.parallel(i))] for i in idx]})
    trend_lines = []
    for ln in lines:
        idx = bars(ln.anchor, None)
        if idx:
            trend_lines.append({"dir": ln.direction, "tests": len(ln.touches),
                                "rows": [[at(i), _px(ln.value(i))] for i in idx]})
    return {"channels": chans, "lines": trend_lines}


def _unix(ts) -> list[int]:
    s = pd.Series(list(ts))
    return (_utc(s).astype("int64") // 10**9).tolist() if len(s) else []


def _px(x) -> float | None:
    return None if x is None or x != x else round(float(x), 2)


def _xj_setup(s: dict) -> dict:
    tr = s["trades"]
    trades = [
        [a, b, int(r.dir), _px(r.line), _px(r.entry), _px(r.px_out), _px(r.stop), _px(r.tp),
         r.exit, round(float(r.usd), 2), _px(r.zone_lo), _px(r.zone_hi)]
        for a, b, r in zip(_unix(tr.t_in), _unix(tr.t_out), tr.itertuples(), strict=True)
    ]
    op = s["open"]
    held = None if op is None else [
        _unix([op["t_in"]])[0], int(op["dir"]),
        *(_px(op[k]) for k in ("line", "entry", "stop", "tp", "zone_lo", "zone_hi"))]
    orders = [[int(o["dir"]), _px(o["price"]), _px(o["stop"]), _px(o["tp"]), o["kind"],
               _px(o["zone_lo"]), _px(o["zone_hi"])] for o in s["orders"]]
    return {"key": s["key"], "label": s["label"], "stop_pts": s["stop_pts"], "trades": trades,
            "open": held, "orders": orders}


def _shifted(times: list[int], values: list[float], offset: int) -> list[list]:
    if len(times) < max(2, offset + 1):
        return []
    step = times[-1] - times[-2]
    plot_t = times[offset:] + [times[-1] + step * k for k in range(1, offset + 1)]
    return [[int(pt), round(v, 2)] for pt, v in zip(plot_t, values, strict=True) if v == v]
