"""Xiaojiang setups on the chart's 5m window: past trades, the open position, next-bar orders."""

from __future__ import annotations

import pandas as pd

from rainier.analysis.sr_zones import Zone
from rainier.analysis.trend_channels import BAR_DURATION, overlay_levels
from rainier.core.config import SRZoneConfig, TrendTimeframeConfig, XJSetupConfig
from rainier.core.types import Timeframe

from .sim import Market, next_orders, simulate
from .zones import ZoneBook, zones_to_lines

# Bars before the window feed the touch counts (test_days sessions) and a position opened there.
WARMUP = pd.Timedelta(days=16)


def chart_setups(data: dict[Timeframe, pd.DataFrame], setups: dict[str, XJSetupConfig],
                 chart_zones: list[Zone], since: pd.Timestamp,
                 cfg: SRZoneConfig | None = None,
                 trend: dict[Timeframe, TrendTimeframeConfig] | None = None) -> list[dict]:
    """Each setup traded on point-in-time zones from ``since``; orders use ``chart_zones``."""
    if not setups or Timeframe.M5 not in data:
        return []
    m5 = data[Timeframe.M5]
    m5 = m5[m5.timestamp >= since - WARMUP].reset_index(drop=True)
    if m5.empty:
        return []
    d = {**data, Timeframe.M5: m5}
    # One extra row: the lines as of the next bar, for next_orders.
    nxt = pd.concat([m5.timestamp, m5.timestamp.tail(1) + BAR_DURATION[Timeframe.M5]])
    levels = (overlay_levels(d, trend, nxt)
              if trend and any(p.overlap_pts for p in setups.values()) else None)
    mkt = Market(d, ZoneBook(d, cfg), levels)
    lines = zones_to_lines(chart_zones)
    out = []
    for key, p in setups.items():
        res = simulate(mkt, p)
        out.append({"key": key, "label": p.label or key, "stop_pts": p.stop_pts,
                    "trades": res.trades[res.trades.t_in >= since].reset_index(drop=True),
                    "open": res.open, "orders": next_orders(mkt, res, p, lines)})
    return out
