"""WS A — real-engine replay harness for the shadow WATCH-buy measurement.

`paper/sweep.py` is missed-winner attribution, NOT position replay. This is a
new day-by-day driver: it walks historical theses over a window, opens SHADOW
positions for WATCH verdicts at each candidate threshold T, runs them through
the REAL fill/exit engine (`positions.create_shadow_positions_for_theses` ->
`fill_pending_positions` -> `update_open_positions`), and reports the shadow
book's return per T vs cash and vs a benchmark.

The harness is config-/DB-driven; the per-T return MATH is factored into pure
functions (`shadow_book_return`, `benchmark_return`) so it is unit-testable
without a database. The orchestration loop is exercised by the postgres-lane
integration test.

ASCII flow (per threshold T, per trading day d in window):
    theses(d) ──► create_shadow_positions_for_theses(conf>=T, long-shape)
                       │
                       ▼
              fill_pending_positions(as_of=d)
                       │
                       ▼
              update_open_positions(as_of=d)
    end of window ──► shadow_book_return() vs cash(0) vs benchmark_return()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplayArm:
    """One threshold arm's measured outcome over a window."""

    threshold: int
    fired: int            # shadow positions opened
    book_return_pct: float  # shadow book return over the window
    benchmark_return_pct: float
    cash_return_pct: float  # always 0.0 (the do-nothing baseline)


def shadow_book_return(trade_returns: list[float]) -> float:
    """Pure: equal-weight mean return across the shadow book's resolved trades.

    Each entry is a single trade's fractional return (e.g. 0.05 = +5%). With no
    trades the book sat in cash → 0.0. Equal-weight matches the fixed-$10k-per-
    position sizing the live engine uses.
    """
    if not trade_returns:
        return 0.0
    return sum(trade_returns) / len(trade_returns)


def benchmark_return(start_close: float | None, end_close: float | None) -> float:
    """Pure: benchmark (e.g. SPY) point-to-point return over the window.

    Returns 0.0 when either endpoint is missing or the start is non-positive
    (no data → no measurable benchmark move).
    """
    if start_close is None or end_close is None or start_close <= 0:
        return 0.0
    return (end_close - start_close) / start_close


def _theses_on(session, scan_date: date) -> list[dict]:
    """Historical theses scanned on `scan_date`, shaped for the position fns."""
    from sqlalchemy import select

    from rainier.core.models import LLMAnalysisRecord, ScreenedStockRecord

    rows = session.execute(
        select(
            ScreenedStockRecord.thesis_id,
            ScreenedStockRecord.symbol,
            ScreenedStockRecord.session_name,
            LLMAnalysisRecord.recommendation,
            ScreenedStockRecord.llm_confidence,
        )
        .join(
            LLMAnalysisRecord,
            LLMAnalysisRecord.id == ScreenedStockRecord.thesis_id,
        )
        .where(
            ScreenedStockRecord.scan_date == scan_date,
            ScreenedStockRecord.thesis_id.is_not(None),
        )
    ).all()
    return [
        {
            "thesis_id": int(tid),
            "symbol": symbol,
            "verdict": rec,
            "llm_confidence": conf,
            "session_name": session_name,
        }
        for tid, symbol, session_name, rec, conf in rows
    ]


def _benchmark_endpoints(
    session, benchmark_symbol: str, start: date, end: date
) -> tuple[float | None, float | None]:
    from sqlalchemy import select

    from rainier.core.models import StockPrice
    from rainier.paper.ingest import canonical_instant

    def _close_on(d: date) -> float | None:
        row = session.execute(
            select(StockPrice.close)
            .where(
                StockPrice.symbol == benchmark_symbol,
                StockPrice.date <= canonical_instant(d),
                StockPrice.close.is_not(None),
            )
            .order_by(StockPrice.date.desc())
            .limit(1)
        ).first()
        return float(row[0]) if row else None

    return _close_on(start), _close_on(end)


def replay_threshold(
    *,
    threshold: int,
    trading_days: list[date],
    benchmark_symbol: str = "SPY",
) -> ReplayArm:
    """Replay one threshold T over `trading_days`, returning the arm outcome.

    Drives the REAL engine day by day. Shadow rows opened here are flagged and
    excluded from every live read, so this never touches the live book. Caller
    is responsible for isolating runs (e.g. one throwaway schema per T) so arms
    don't contaminate each other's shadow book.
    """
    from rainier.core.database import get_session
    from rainier.paper import positions

    if not trading_days:
        return ReplayArm(threshold, 0, 0.0, 0.0, 0.0)

    fired = 0
    for d in trading_days:
        with get_session() as session:
            theses = _theses_on(session, d)
        if theses:
            res = positions.create_shadow_positions_for_theses(
                theses, scan_date=d, min_confidence=threshold
            )
            fired += res["created"]
        positions.fill_pending_positions(as_of=d)
        positions.update_open_positions(as_of=d)

    # Collect the shadow book's resolved trade returns.
    from sqlalchemy import select

    from rainier.core.models import PaperTrade

    with get_session() as session:
        rows = session.execute(
            select(PaperTrade.return_pct).where(
                PaperTrade.shadow.is_(True),
                PaperTrade.return_pct.is_not(None),
            )
        ).all()
        trade_returns = [float(r[0]) for r in rows]

        start_close, end_close = _benchmark_endpoints(
            session, benchmark_symbol, trading_days[0], trading_days[-1]
        )

    return ReplayArm(
        threshold=threshold,
        fired=fired,
        book_return_pct=shadow_book_return(trade_returns),
        benchmark_return_pct=benchmark_return(start_close, end_close),
        cash_return_pct=0.0,
    )
