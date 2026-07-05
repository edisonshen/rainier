"""Sector trend analysis — Layer 2 of QU100 stock screener."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date as date_type

from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from rainier.core.database import get_session
from rainier.core.models import MoneyFlowSnapshot
from rainier.core.types import SectorTrend

log = logging.getLogger(__name__)

_BULLISH_THRESHOLD = 0.3
_BEARISH_THRESHOLD = -0.3
_TOP_STOCKS_LIMIT = 5

# QU100 sentiment is built from the two QU100 books ONLY (top100 + bottom100).
# money_flow_snapshots may one day also hold auxiliary ranking types (e.g.
# concept/etf boards); those must NOT be folded into QU100 sector sentiment or the
# screener / SectorMomentumSignal would silently skew. The generation filter is
# restricted to these two so a stray ranking type can never leak in.
_QU100_RANKING_TYPES = ("top100", "bottom100")


# ---------------------------------------------------------------------------
# Latest-generation selection (per ranking_type)
# ---------------------------------------------------------------------------
#
# Under the rebuild-the-day fix a day holds ONE captured_at per
# (data_date, ranking_type). A partial-failure scrape can leave top100 and
# bottom100 on DIFFERENT (data_date, captured_at) generations. A single global
# max(captured_at) reads only one ranking type's rows -> half-book. So we resolve
# the latest generation INDEPENDENTLY per ranking_type and union the batches.


def _latest_generation_filter(session: Session, as_of: date_type | None):
    """Build a WHERE predicate selecting the latest snapshot generation PER
    ranking_type (optionally bounded to ``data_date <= as_of``).

    Returns ``None`` when no rows match (caller returns an empty result).
    """
    date_q = session.query(
        MoneyFlowSnapshot.ranking_type,
        func.max(MoneyFlowSnapshot.data_date).label("max_date"),
    ).filter(MoneyFlowSnapshot.ranking_type.in_(_QU100_RANKING_TYPES))
    if as_of is not None:
        date_q = date_q.filter(MoneyFlowSnapshot.data_date <= as_of)
    latest_dates = date_q.group_by(MoneyFlowSnapshot.ranking_type).all()
    if not latest_dates:
        return None

    # For each (ranking_type, its latest data_date), find that generation's
    # captured_at (one per (data_date, ranking_type) under the rebuild fix; max()
    # is a safe belt-and-suspenders for any legacy multi-capture day).
    clauses = []
    for ranking_type, max_date in latest_dates:
        max_cap = (
            session.query(func.max(MoneyFlowSnapshot.captured_at))
            .filter(
                MoneyFlowSnapshot.ranking_type == ranking_type,
                MoneyFlowSnapshot.data_date == max_date,
            )
            .scalar()
        )
        clauses.append(
            and_(
                MoneyFlowSnapshot.ranking_type == ranking_type,
                MoneyFlowSnapshot.data_date == max_date,
                MoneyFlowSnapshot.captured_at == max_cap,
            )
        )
    return or_(*clauses)


def _build_sector_trends(
    rows: list[tuple[str | None, str | None, int, str]],
    *,
    normalize_casing: bool = False,
) -> list[SectorTrend]:
    """Group rows ``(sector, long_short, rank, symbol)`` into ranked SectorTrends.

    ``normalize_casing=True`` canonicalizes ``long_short`` at ingestion
    ('Long In' -> 'Long in', 'Short In' -> 'Short in') so the exact-match
    counts below admit the 406 historical title-cased days. Default ``False``
    preserves live behavior byte-identically.
    """
    sector_data: dict[str, list[tuple[str, str | None, int]]] = defaultdict(list)
    for sector, long_short, rank, symbol in rows:
        sector_key = sector or "Unknown"
        if normalize_casing and long_short is not None:
            long_short = long_short.capitalize()
        sector_data[sector_key].append((symbol, long_short, rank))

    sector_trends: list[SectorTrend] = []
    for sector, stocks in sector_data.items():
        long_count = sum(1 for _, ls, _ in stocks if ls == "Long in")
        short_count = sum(1 for _, ls, _ in stocks if ls == "Short in")
        total = len(stocks)
        net_sentiment = (long_count - short_count) / total if total > 0 else 0.0
        if net_sentiment > _BULLISH_THRESHOLD:
            trend_direction = "bullish"
        elif net_sentiment < _BEARISH_THRESHOLD:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"
        long_stocks = [(symbol, rank) for symbol, ls, rank in stocks if ls == "Long in"]
        long_stocks.sort(key=lambda x: x[1])
        top_stocks = [symbol for symbol, _ in long_stocks[:_TOP_STOCKS_LIMIT]]
        sector_trends.append(
            SectorTrend(
                sector=sector,
                long_in_count=long_count,
                short_in_count=short_count,
                net_sentiment=round(net_sentiment, 4),
                top_stocks=top_stocks,
                trend_direction=trend_direction,
                sector_rank=0,  # assigned after sorting
            )
        )

    sector_trends.sort(key=lambda st: st.net_sentiment, reverse=True)
    return [
        SectorTrend(
            sector=st.sector,
            long_in_count=st.long_in_count,
            short_in_count=st.short_in_count,
            net_sentiment=st.net_sentiment,
            top_stocks=st.top_stocks,
            trend_direction=st.trend_direction,
            sector_rank=i,
        )
        for i, st in enumerate(sector_trends, start=1)
    ]


def analyze_sectors(session: Session | None = None) -> list[SectorTrend]:
    """Analyze sector trends from latest QU100 money flow data.

    Groups ALL stocks (top100 + bottom100) by sector, computes:
    - long_in_count: Number of "Long in" stocks
    - short_in_count: Number of "Short in" stocks
    - net_sentiment: (long - short) / total
    - trend_direction: "bullish" if net_sentiment > 0.3,
      "bearish" if < -0.3, else "neutral"
    - top_stocks: Best ranked "Long in" stocks in sector (up to 5)

    Returns list of SectorTrend sorted by net_sentiment descending.
    """
    if session is not None:
        return _analyze_sectors(session)

    with get_session() as s:
        return _analyze_sectors(s)


def _analyze_sectors(session: Session) -> list[SectorTrend]:
    """Core implementation with an explicit session — latest generation per
    ranking_type (no global captured_at half-book)."""
    predicate = _latest_generation_filter(session, as_of=None)
    if predicate is None:
        log.warning("No money flow snapshots found — returning empty sector list")
        return []

    rows = (
        session.query(
            MoneyFlowSnapshot.sector,
            MoneyFlowSnapshot.long_short,
            MoneyFlowSnapshot.rank,
            MoneyFlowSnapshot.symbol,
        )
        .filter(predicate)
        .all()
    )
    ranked = _build_sector_trends(rows)
    log.info(
        "Sector analysis complete: %d sectors, top=%s",
        len(ranked),
        ranked[0].sector if ranked else "N/A",
    )
    return ranked


def analyze_sectors_at(
    as_of: date_type,
    session: Session | None = None,
    *,
    normalize_casing: bool = False,
) -> list[SectorTrend]:
    """Analyze sector trends as of a given data DATE (latest generation on or
    before ``as_of``, resolved INDEPENDENTLY per ranking_type).

    Used by the LLM-thesis sector_momentum signal to compare today's sentiment
    against N days ago. Keyed on ``data_date`` (not a single global
    ``captured_at``): under the rebuild-the-day fix a day holds one
    ``captured_at`` per ``(data_date, ranking_type)``, so a single timestamp
    could only match ONE ranking type. We pick each ranking type's latest
    generation with ``data_date <= as_of`` and union the books.

    ``normalize_casing=True`` (A/B replay only) canonicalizes ``long_short``
    casing before the sentiment counts, admitting the 406 historical
    'Long In'/'Short In' days. The sector leg queries MoneyFlowSnapshot rows
    itself, so it needs the flag directly — accessor-level normalization in
    the screener cannot reach it.

    The default ``False`` keeps live behavior byte-identical — including the
    recorded design-§9.5 decision: `sector_momentum` live-consumes this
    function over PRIOR data_dates, so historical 'Long In' days already go
    uncounted in one live output today. That accepted exposure is resolved
    later through the substrate's designated first experiment, not here.
    """
    if session is not None:
        return _analyze_sectors_as_of(session, as_of, normalize_casing=normalize_casing)
    with get_session() as s:
        return _analyze_sectors_as_of(s, as_of, normalize_casing=normalize_casing)


def _analyze_sectors_as_of(
    session: Session,
    as_of: date_type,
    *,
    normalize_casing: bool = False,
) -> list[SectorTrend]:
    """Latest generation per ranking_type with ``data_date <= as_of``."""
    predicate = _latest_generation_filter(session, as_of=as_of)
    if predicate is None:
        log.warning("No snapshots on or before as_of=%s", as_of)
        return []
    rows = (
        session.query(
            MoneyFlowSnapshot.sector,
            MoneyFlowSnapshot.long_short,
            MoneyFlowSnapshot.rank,
            MoneyFlowSnapshot.symbol,
        )
        .filter(predicate)
        .all()
    )
    return _build_sector_trends(rows, normalize_casing=normalize_casing)


def get_sector_boost(sector: str, sector_trends: list[SectorTrend]) -> float:
    """Get the sector boost for a stock's sector.

    Returns 0.1 for bullish sectors, 0.0 otherwise.
    """
    for st in sector_trends:
        if st.sector == sector:
            return 0.1 if st.trend_direction == "bullish" else 0.0
    return 0.0
