"""Step (I) — end-to-end happy path (TEST-SPEC §I1). DB-backed."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import text

from rainier.core.models import ScreenedStockRecord
from rainier.paper.ingest import canonical_instant
from rainier.paper.positions import (
    create_positions_for_theses,
    fill_pending_positions,
    update_open_positions,
)
from rainier.paper.report import compute_daily_payload, persist_daily_snapshot

pytestmark = pytest.mark.requires_postgres

# Trading days: Mon 2026-01-05 .. Fri 2026-01-09.
T0 = date(2026, 1, 5)   # scan/thesis day
T1 = date(2026, 1, 6)   # fill
T2 = date(2026, 1, 7)
T3 = date(2026, 1, 8)   # target hit
T4 = date(2026, 1, 9)


def _price(s, sym, d, o, h, low, c):
    s.execute(text("INSERT INTO stocks (symbol) VALUES (:s) ON CONFLICT DO NOTHING"),
              {"s": sym})
    s.execute(
        text("INSERT INTO stock_prices (symbol, date, open, high, low, close, volume) "
             "VALUES (:s,:d,:o,:h,:l,:c,1000)"),
        {"s": sym, "d": canonical_instant(d), "o": o, "h": h, "l": low, "c": c},
    )
    s.commit()


def test_i1_happy_path(pg_legacy_session):
    s = pg_legacy_session
    # Seed thesis + screened row (afternoon, conf 7), levels entry 100 / SL 90 / TP 120.
    s.execute(text("INSERT INTO analysis_results (id, llm_model, prompt_template) "
                   "VALUES (1,'m','t')"))
    s.add(ScreenedStockRecord(
        scan_date=T0, session_name="close", symbol="AAA", rule_rank=1,
        composite_score=0.9, pattern_type="bull_flag", entry_price=100,
        stop_loss=90, target_price=120, rr_ratio=2.0,
    ))
    s.commit()

    thesis = {"thesis_id": 1, "symbol": "AAA", "verdict": "setup_long",
              "llm_confidence": 7, "session_name": "close"}
    create_positions_for_theses([thesis], scan_date=T0)

    # Price path: flat through T2, target on T3.
    _price(s, "AAA", T1, 100, 105, 99, 102)
    _price(s, "AAA", T2, 102, 106, 100, 103)
    _price(s, "AAA", T3, 103, 121, 102, 120)  # target
    _price(s, "AAA", T4, 120, 122, 118, 121)

    # Run the daily loop across T1..T4 with as_of advancing.
    snapshots = 0
    for as_of, expect_status in [
        (T1, "open"), (T2, "open"), (T3, "closed"), (T4, "closed")
    ]:
        fill_pending_positions(as_of=as_of)
        update_open_positions(as_of=as_of)
        payload = compute_daily_payload(as_of)
        persist_daily_snapshot(as_of, payload)
        snapshots += 1
        s.expire_all()
        p = s.execute(
            text("SELECT status, entry_date, exit_reason, exit_date, pnl "
                 "FROM paper_trade WHERE symbol='AAA'")
        ).one()
        assert p.status == expect_status, f"as_of={as_of}"

    # Final state: filled T1, closed target T3 with exact pnl.
    s.expire_all()
    pos = s.execute(
        text("SELECT * FROM paper_trade WHERE symbol='AAA'")
    ).mappings().one()
    assert pos["entry_date"] == T1 and pos["entry_price"] == 100
    assert pos["exit_reason"] == "target" and pos["exit_date"] == T3
    assert pos["exit_price"] == 120 and pos["pnl"] == pytest.approx(100 * (120 - 100))

    # One snapshot per day.
    rows = s.execute(
        text("SELECT count(*) FROM paper_report_snapshot WHERE report_type='daily'")
    ).scalar()
    assert rows == 4
    # The T3 snapshot booked the exit on its as_of date.
    t3_payload = s.execute(
        text("SELECT payload FROM paper_report_snapshot WHERE as_of_date=:d"),
        {"d": T3},
    ).scalar()
    assert [e["symbol"] for e in t3_payload["todays_exits"]] == ["AAA"]
