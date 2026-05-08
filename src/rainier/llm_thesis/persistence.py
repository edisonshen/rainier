"""Persistence helpers for ScreenedStockRecord rows."""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rainier.core.database import get_session
from rainier.core.models import ScreenedStockRecord
from rainier.core.types import StockCandidate

log = logging.getLogger(__name__)


def _candidate_row(
    candidate: StockCandidate,
    rule_rank: int,
    scan_date: date,
    session_name: str,
) -> dict[str, Any]:
    # Codex P1: composite_score holds the post-boost composite from
    # `_to_candidate` (i.e. `StockCandidate.signal_strength`); money_flow_score
    # holds the RAW Layer-1 money-flow value from `MoneyFlowSignal` so SQL
    # backfills can compare each layer independently. Falling back to
    # signal_strength would silently duplicate the column, which biases the
    # shadow-validation analysis PR2 will run on top of these rows.
    return {
        "scan_date": scan_date,
        "session_name": session_name,
        "symbol": candidate.symbol,
        "rule_rank": rule_rank,
        "composite_score": float(candidate.signal_strength or 0.0),
        "money_flow_score": (
            float(candidate.money_flow_score)
            if candidate.money_flow_score is not None
            else None
        ),
        "sector": candidate.sector,
        "pattern_type": candidate.pattern_type,
        "pattern_confidence": (
            float(candidate.pattern_confidence) if candidate.pattern_confidence else None
        ),
    }


def persist_screened_stocks(
    candidates: list[StockCandidate],
    scan_date: date,
    session_name: str,
) -> int:
    """Bulk insert (idempotent via ON CONFLICT DO NOTHING).

    Returns row-count attempted. Re-runs on the same (scan_date, session_name)
    are no-ops thanks to the UNIQUE constraint on (scan_date, session_name, symbol).
    """
    if not candidates:
        return 0
    rows = [
        _candidate_row(c, rank, scan_date, session_name)
        for rank, c in enumerate(candidates, start=1)
    ]
    with get_session() as session:
        stmt = pg_insert(ScreenedStockRecord).values(rows).on_conflict_do_nothing(
            index_elements=["scan_date", "session_name", "symbol"]
        )
        session.execute(stmt)
    return len(rows)


def update_with_thesis(
    *,
    symbol: str,
    scan_date: date,
    session_name: str,
    llm_confidence: int | None,
    shadow_combined_score: float | None,
    would_be_combined_rank: int | None,
    thesis_id: int | None,
    patterns_count: int | None,
) -> bool:
    """Patch the matching ScreenedStockRecord row with LLM-thesis fields.

    Returns True iff one row was updated. Logs a warning on zero rows affected
    (likely means persist_screened_stocks failed earlier for this ticker).
    """
    with get_session() as session:
        stmt = (
            update(ScreenedStockRecord)
            .where(
                ScreenedStockRecord.scan_date == scan_date,
                ScreenedStockRecord.session_name == session_name,
                ScreenedStockRecord.symbol == symbol,
            )
            .values(
                llm_confidence=llm_confidence,
                shadow_combined_score=shadow_combined_score,
                would_be_combined_rank=would_be_combined_rank,
                thesis_id=thesis_id,
                patterns_in_chart_not_in_indicators_count=patterns_count,
            )
        )
        result = session.execute(stmt)

    affected = int(result.rowcount or 0)
    if affected == 0:
        log.warning(
            "update_with_thesis_no_row_affected symbol=%s scan_date=%s session=%s",
            symbol,
            scan_date,
            session_name,
        )
        return False
    return True


def candidate_to_input_summary(candidate: StockCandidate) -> dict[str, Any]:
    """Frozen-dataclass → dict for evidence-pack inclusion (and idempotency hash)."""
    return asdict(candidate)


def utc_now_naive() -> datetime:
    """Helper used by `rainier thesis log` so action times are stable."""
    return datetime.now(timezone.utc)
