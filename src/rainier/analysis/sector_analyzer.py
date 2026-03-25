"""Sector trend analysis — Layer 2 of QU100 stock screener."""

from __future__ import annotations

import logging
from collections import defaultdict

from sqlalchemy import func
from sqlalchemy.orm import Session

from rainier.core.database import get_session
from rainier.core.models import MoneyFlowSnapshot, Stock
from rainier.core.types import SectorTrend

log = logging.getLogger(__name__)

_BULLISH_THRESHOLD = 0.3
_BEARISH_THRESHOLD = -0.3
_TOP_STOCKS_LIMIT = 5


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
    """Core implementation with an explicit session."""
    # 1. Get latest captured_at timestamp
    latest_ts = session.query(func.max(MoneyFlowSnapshot.captured_at)).scalar()
    if latest_ts is None:
        log.warning("No money flow snapshots found — returning empty sector list")
        return []

    # 2. Query all snapshots at that timestamp (both top100 and bottom100)
    rows = (
        session.query(
            MoneyFlowSnapshot.sector,
            MoneyFlowSnapshot.long_short,
            MoneyFlowSnapshot.rank,
            Stock.symbol,
        )
        .join(Stock, MoneyFlowSnapshot.stock_id == Stock.id)
        .filter(MoneyFlowSnapshot.captured_at == latest_ts)
        .all()
    )

    if not rows:
        log.warning("No snapshots at latest timestamp %s", latest_ts)
        return []

    # 3. Group by sector
    sector_data: dict[str, list[tuple[str, str | None, int]]] = defaultdict(list)
    for sector, long_short, rank, symbol in rows:
        sector_key = sector or "Unknown"
        sector_data[sector_key].append((symbol, long_short, rank))

    # 4. Compute metrics per sector
    sector_trends: list[SectorTrend] = []
    for sector, stocks in sector_data.items():
        long_count = sum(1 for _, ls, _ in stocks if ls == "Long in")
        short_count = sum(1 for _, ls, _ in stocks if ls == "Short in")
        total = len(stocks)

        net_sentiment = (long_count - short_count) / total if total > 0 else 0.0

        # Classify trend direction
        if net_sentiment > _BULLISH_THRESHOLD:
            trend_direction = "bullish"
        elif net_sentiment < _BEARISH_THRESHOLD:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"

        # Top 5 "Long in" stocks by rank (ascending = best first)
        long_stocks = [
            (symbol, rank)
            for symbol, ls, rank in stocks
            if ls == "Long in"
        ]
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
                sector_rank=0,  # placeholder, assigned after sorting
            )
        )

    # 5. Sort by net_sentiment descending, assign ranks
    sector_trends.sort(key=lambda st: st.net_sentiment, reverse=True)
    ranked: list[SectorTrend] = []
    for i, st in enumerate(sector_trends, start=1):
        ranked.append(
            SectorTrend(
                sector=st.sector,
                long_in_count=st.long_in_count,
                short_in_count=st.short_in_count,
                net_sentiment=st.net_sentiment,
                top_stocks=st.top_stocks,
                trend_direction=st.trend_direction,
                sector_rank=i,
            )
        )

    log.info(
        "Sector analysis complete: %d sectors, top=%s",
        len(ranked),
        ranked[0].sector if ranked else "N/A",
    )
    return ranked


def get_sector_boost(sector: str, sector_trends: list[SectorTrend]) -> float:
    """Get the sector boost for a stock's sector.

    Returns 0.1 for bullish sectors, 0.0 otherwise.
    """
    for st in sector_trends:
        if st.sector == sector:
            return 0.1 if st.trend_direction == "bullish" else 0.0
    return 0.0
