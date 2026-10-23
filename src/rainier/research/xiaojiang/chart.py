"""One Xiaojiang setup's backtest trades + resting orders, for the /trading/mes chart."""

from __future__ import annotations

import pandas as pd

from rainier.analysis.sr_zones import Zone
from rainier.core.config import SRZoneConfig
from rainier.core.types import Timeframe
from rainier.research.xiaojiang.sim import Market, XJParams, next_bar_plan, simulate, targets
from rainier.research.xiaojiang.zones import ZoneBook, zones_to_lines

WARMUP = pd.Timedelta("1D")  # 5m bars simulated before ``since`` so open positions carry over


def chart_signals(data: dict[Timeframe, pd.DataFrame], params: dict, since: pd.Timestamp,
                  cfg: SRZoneConfig | None = None, zones: list[Zone] | None = None) -> dict:
    """Trades entered at or after ``since`` (naive PT, like the CSV loader) at each hour's
    point-in-time chart zones, and the next bar's orders, each ``(direction, limit, hard stop,
    take-profit)``, placed on ``zones`` (the zones the chart draws) when given."""
    p = XJParams(**params)
    m5 = data[Timeframe.M5]
    data = data | {Timeframe.M5: m5[m5.timestamp >= since - WARMUP].reset_index(drop=True)}
    mkt = Market(data, ZoneBook(data, cfg))
    tr = simulate(mkt, p)
    _, _, lines, _, orders = next_bar_plan(mkt, p, None if zones is None else zones_to_lines(zones))
    return {
        "label": f"{p.stop_pts:g} pt stop, {p.exit} exit",
        "trades": tr[tr.t_in >= since].reset_index(drop=True),
        "orders": [(od.direction, od.line, od.stop,
                    targets(lines, od.direction, od.line, od.stop, p)[0]) for od in orders],
    }
