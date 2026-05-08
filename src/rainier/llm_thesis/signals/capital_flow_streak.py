"""capital_flow_streak signal — consecutive same-direction days + history."""

from __future__ import annotations

import logging
from datetime import timedelta

from rainier.core.database import get_session
from rainier.core.models import StockCapitalFlow

from .base import SignalContext, SignalValue

log = logging.getLogger(__name__)


class CapitalFlowStreakSignal:
    name = "capital_flow_streak"
    version = "v1"
    cost_estimate_ms = 5

    def compute(self, ctx: SignalContext) -> SignalValue | None:
        days = int(ctx.params.get("days", 10))
        since = ctx.scan_date - timedelta(days=days)

        try:
            with get_session() as session:
                rows = (
                    session.query(
                        StockCapitalFlow.flow_date,
                        StockCapitalFlow.capital_flow_direction,
                    )
                    .filter(
                        StockCapitalFlow.symbol == ctx.symbol,
                        StockCapitalFlow.period_type == "daily",
                        StockCapitalFlow.flow_date >= since,
                    )
                    .order_by(StockCapitalFlow.flow_date.desc())
                    .all()
                )
        except Exception:
            log.exception("capital_flow_streak_db_error symbol=%s", ctx.symbol)
            return None

        if not rows:
            return {"streak_days": 0, "direction": "N", "history": []}

        history: list[tuple[str, str]] = [
            (r.flow_date.isoformat(), r.capital_flow_direction or "N") for r in rows
        ]
        # Most-recent day defines the active direction.
        direction = history[0][1]
        streak = 0
        for _, d in history:
            if d == direction and direction in {"+", "-"}:
                streak += 1
            else:
                break
        if direction not in {"+", "-"}:
            streak = 0

        return {
            "streak_days": streak,
            "direction": direction,
            "history": history,
        }

    def render_for_prompt(self, value: SignalValue) -> str:
        streak = int(value.get("streak_days", 0) or 0)
        direction = value.get("direction", "N")
        if not value.get("history"):
            return "Capital flow streak: no recent flow data."
        if direction not in {"+", "-"} or streak == 0:
            return "Capital flow streak: neutral / mixed in the lookback window."
        label = "inflow" if direction == "+" else "outflow"
        return f"Capital flow streak: {streak} consecutive day(s) of net {label} ({direction})."
