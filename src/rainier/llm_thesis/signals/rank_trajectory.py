"""rank_trajectory signal — N-day QU100 rank history + delta + trend label."""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone
from typing import Any

from sqlalchemy import func

from rainier.core.database import get_session
from rainier.core.models import MoneyFlowSnapshot

from .base import SignalContext, SignalValue

log = logging.getLogger(__name__)


class RankTrajectorySignal:
    name = "rank_trajectory"
    version = "v1"
    cost_estimate_ms = 5

    def compute(self, ctx: SignalContext) -> SignalValue | None:
        days = int(ctx.params.get("days", 10))
        since = datetime.combine(ctx.scan_date, time.min, tzinfo=timezone.utc) - timedelta(
            days=days
        )

        try:
            with get_session() as session:
                # Pick best (smallest) rank per day so multi-session days collapse
                # to a single point on the trajectory.
                rows = (
                    session.query(
                        MoneyFlowSnapshot.data_date,
                        func.min(MoneyFlowSnapshot.rank).label("best_rank"),
                    )
                    .filter(
                        MoneyFlowSnapshot.symbol == ctx.symbol,
                        MoneyFlowSnapshot.captured_at >= since,
                        MoneyFlowSnapshot.ranking_type == "top100",
                    )
                    .group_by(MoneyFlowSnapshot.data_date)
                    .order_by(MoneyFlowSnapshot.data_date.asc())
                    .all()
                )
        except Exception:
            log.exception("rank_trajectory_db_error symbol=%s", ctx.symbol)
            return None

        if not rows:
            return {"points": [], "delta_10d": None, "trend": "flat"}

        points: list[tuple[str, int]] = [
            (r.data_date.isoformat(), int(r.best_rank)) for r in rows
        ]

        delta: int | None
        trend: str
        if len(points) >= 2:
            # Lower rank is better → delta = old - new, positive means improving.
            delta = points[0][1] - points[-1][1]
            if delta >= 5:
                trend = "rising"
            elif delta <= -5:
                trend = "falling"
            else:
                trend = "flat"
        else:
            delta = None
            trend = "flat"

        return {"points": points, "delta_10d": delta, "trend": trend}

    def render_for_prompt(self, value: SignalValue) -> str:
        points: list[Any] = value.get("points") or []
        delta = value.get("delta_10d")
        trend = value.get("trend", "flat")
        if not points:
            return "Rank trajectory: no recent QU100 history."
        first_date, first_rank = points[0]
        last_date, last_rank = points[-1]
        delta_str = "n/a" if delta is None else f"{delta:+d}"
        return (
            f"Rank trajectory ({len(points)}d): {first_date} #{first_rank} "
            f"→ {last_date} #{last_rank} (Δ {delta_str}, trend={trend})."
        )
