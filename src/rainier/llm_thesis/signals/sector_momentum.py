"""sector_momentum signal — sentiment today vs N days ago + delta + flip date."""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone

from sqlalchemy import func

from rainier.core.database import get_session
from rainier.core.models import MoneyFlowSnapshot

from .base import SignalContext, SignalValue

log = logging.getLogger(__name__)


class SectorMomentumSignal:
    name = "sector_momentum"
    version = "v1"
    cost_estimate_ms = 20

    def compute(self, ctx: SignalContext) -> SignalValue | None:
        sector = ctx.candidate.sector
        if not sector or sector == "Unknown":
            return {
                "sector": sector,
                "sentiment_today": None,
                "sentiment_prior": None,
                "delta": None,
                "shifted_at": None,
            }

        days = int(ctx.params.get("days", 10))
        scan_dt = datetime.combine(ctx.scan_date, time.max, tzinfo=timezone.utc)
        prior_dt_floor = scan_dt - timedelta(days=days)

        try:
            with get_session() as session:
                latest_ts = (
                    session.query(func.max(MoneyFlowSnapshot.captured_at))
                    .filter(MoneyFlowSnapshot.captured_at <= scan_dt)
                    .scalar()
                )
                prior_ts = (
                    session.query(func.max(MoneyFlowSnapshot.captured_at))
                    .filter(MoneyFlowSnapshot.captured_at <= prior_dt_floor)
                    .scalar()
                )
                if latest_ts is None:
                    return None

                # Defer expensive analysis to analyze_sectors_at (also explicit
                # session injection so we don't open a second connection).
                from rainier.analysis.sector_analyzer import analyze_sectors_at

                today = analyze_sectors_at(latest_ts, session=session)
                prior = (
                    analyze_sectors_at(prior_ts, session=session) if prior_ts else []
                )
        except Exception:
            log.exception("sector_momentum_db_error symbol=%s", ctx.symbol)
            return None

        today_sent = next(
            (st.net_sentiment for st in today if st.sector == sector), None
        )
        prior_sent = next(
            (st.net_sentiment for st in prior if st.sector == sector), None
        )
        delta: float | None
        if today_sent is not None and prior_sent is not None:
            delta = round(today_sent - prior_sent, 4)
        else:
            delta = None

        return {
            "sector": sector,
            "sentiment_today": today_sent,
            "sentiment_prior": prior_sent,
            "delta": delta,
            "shifted_at": None,  # PR2 may add precise flip-date detection
        }

    def render_for_prompt(self, value: SignalValue) -> str:
        sector = value.get("sector") or "?"
        today = value.get("sentiment_today")
        prior = value.get("sentiment_prior")
        delta = value.get("delta")
        if today is None:
            return f"Sector momentum: no sector data for {sector}."
        prior_str = f"{prior:+.2f}" if prior is not None else "n/a"
        delta_str = f"{delta:+.2f}" if delta is not None else "n/a"
        return (
            f"Sector momentum ({sector}): now {today:+.2f}, prior {prior_str} "
            f"(Δ {delta_str})."
        )
