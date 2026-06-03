"""Step 3 — screened-stock level persistence (TEST-SPEC §C).

`persist_screened_stocks` now writes the trade levels and upserts (DO UPDATE)
the level columns, backfilling NULLs only. Postgres-dialect ON CONFLICT → these
run under `requires_postgres`.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select, text

from rainier.core.models import ScreenedStockRecord
from rainier.core.types import StockCandidate
from rainier.llm_thesis.persistence import persist_screened_stocks

pytestmark = pytest.mark.requires_postgres


def _cand(symbol, entry=None, stop=None, target=None, rr=None):
    return StockCandidate(
        symbol=symbol, rank=1, rank_change=0, long_short="Long in",
        capital_flow_direction="+", sector="Tech", signal_strength=0.8,
        pattern_type="bull_flag", entry_price=entry, stop_loss=stop,
        target_price=target, rr_ratio=rr,
    )


def _row(session, symbol):
    return session.execute(
        select(ScreenedStockRecord).where(ScreenedStockRecord.symbol == symbol)
    ).scalars().one()


def test_c1_levels_populated(pg_legacy_session):
    persist_screened_stocks(
        [_cand("AAA", entry=100, stop=90, target=120, rr=2.0)],
        date(2026, 1, 9), "close",
    )
    pg_legacy_session.expire_all()
    r = _row(pg_legacy_session, "AAA")
    assert (r.entry_price, r.stop_loss, r.target_price, r.rr_ratio) == (100, 90, 120, 2.0)


def test_c2_do_update_backfills_null(pg_legacy_session):
    # Pre-existing row with NULL levels (e.g. an earlier persist without levels).
    pg_legacy_session.execute(
        text(
            "INSERT INTO screened_stocks (scan_date, session_name, symbol, "
            "rule_rank, composite_score) VALUES ('2026-01-09','close','AAA',1,0.8)"
        )
    )
    pg_legacy_session.commit()
    persist_screened_stocks(
        [_cand("AAA", entry=100, stop=90, target=120, rr=2.0)],
        date(2026, 1, 9), "close",
    )
    pg_legacy_session.expire_all()
    r = _row(pg_legacy_session, "AAA")
    assert (r.entry_price, r.stop_loss, r.target_price) == (100, 90, 120)


def test_c3_no_clobber_of_non_null(pg_legacy_session):
    persist_screened_stocks(
        [_cand("AAA", entry=100, stop=90, target=120, rr=2.0)],
        date(2026, 1, 9), "close",
    )
    # Re-persist with different levels — must NOT overwrite the populated ones.
    persist_screened_stocks(
        [_cand("AAA", entry=55, stop=50, target=70, rr=3.0)],
        date(2026, 1, 9), "close",
    )
    pg_legacy_session.expire_all()
    r = _row(pg_legacy_session, "AAA")
    assert (r.entry_price, r.stop_loss, r.target_price, r.rr_ratio) == (100, 90, 120, 2.0)
