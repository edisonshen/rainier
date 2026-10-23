"""One Xiaojiang setup's backtest trades + resting orders, for the /trading/mes chart."""

from __future__ import annotations

import pandas as pd

from rainier.core.types import Timeframe
from rainier.research.xiaojiang.lines import LineBook
from rainier.research.xiaojiang.sim import Market, XJParams, next_bar_plan, simulate, targets


def chart_signals(data: dict[Timeframe, pd.DataFrame], params: dict,
                  since: pd.Timestamp) -> dict:
    """Trades entered at or after ``since`` (naive PT, like the CSV loader) and the next bar's
    orders, each ``(direction, limit, hard stop, take-profit)``."""
    p = XJParams(**params)
    mkt = Market(data, LineBook(data))
    tr = simulate(mkt, p)
    _, _, lines, _, orders = next_bar_plan(mkt, p)
    return {
        "label": f"{p.stop_pts:g} pt stop, {p.exit} exit",
        "trades": tr[tr.t_in >= since].reset_index(drop=True),
        "orders": [(od.direction, od.line, od.stop,
                    targets(lines, od.direction, od.line, od.stop, p)[0]) for od in orders],
    }
