"""fear_greed signal — CNN Fear & Greed market mood as of the prior close.

Market-wide context (identical for every ticker in a scan). Composite only: the
per-component ratings stored in ``raw`` are rated on CNN's raw component series
and disagree with CNN's own headline component ratings, so they are not sent.
Point-in-time safe:
reads only dates strictly before ``scan_date`` (the scan runs before the day's
15:10 PT capture), taking each date's newest observation.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import NamedTuple

from rainier.core.database import get_session
from rainier.core.models import FearGreedIndex

from .base import SignalContext, SignalValue

log = logging.getLogger(__name__)

class FearGreedSignal:
    name = "fear_greed"
    version = "v1"
    cost_estimate_ms = 10

    def compute(self, ctx: SignalContext) -> SignalValue | None:
        lookback = int(ctx.params.get("lookback_days", 45))
        try:
            with get_session() as session:
                rows = (
                    session.query(FearGreedIndex)
                    .filter(
                        FearGreedIndex.date < ctx.scan_date,
                        FearGreedIndex.date >= ctx.scan_date - timedelta(days=lookback),
                    )
                    .order_by(
                        FearGreedIndex.date.desc(),
                        FearGreedIndex.observed_at.desc(),
                        FearGreedIndex.id.desc(),  # same tie-break as the ingest
                    )
                    .all()
                )
                series = _newest_per_date([_Point(r.date, r.score, r.rating) for r in rows])
        except Exception:
            log.exception("fear_greed_db_error symbol=%s", ctx.symbol)
            return None
        if not series:
            return None

        latest = series[0]
        return {
            "as_of": latest.date.isoformat(),
            "score": round(latest.score, 1),
            "rating": latest.rating,
            "week_ago": _score_on_or_before(series, latest.date - timedelta(days=7)),
            "month_ago": _score_on_or_before(series, latest.date - timedelta(days=30)),
        }

    def render_for_prompt(self, value: SignalValue) -> str:
        def fmt(score):
            return f"{score:.1f}" if score is not None else "n/a"

        return (
            f"Market Fear & Greed (CNN, close {value['as_of']}): {fmt(value['score'])} "
            f"{value.get('rating') or ''}".rstrip()
            + f"; 1w ago {fmt(value.get('week_ago'))}, 1m ago {fmt(value.get('month_ago'))}."
        )


class _Point(NamedTuple):
    date: date
    score: float
    rating: str | None


def _newest_per_date(rows: list[_Point]) -> list[_Point]:
    """Rows arrive date desc, observed_at desc; keep the first (newest) per date."""
    seen: set = set()
    return [r for r in rows if not (r.date in seen or seen.add(r.date))]


def _score_on_or_before(series: list[_Point], day: date) -> float | None:
    row = next((r for r in series if r.date <= day), None)
    return round(row.score, 1) if row else None
