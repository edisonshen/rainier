"""Phase 2 — future-fills-only time-stop adoption (verify-only regression) +
weekly paper_lessons insight.

Future-fills-only invariant (D6/E5, the fill wiring shipped in #117): adopting
a learned time-stop snapshots it onto NEW fills at fill time; already-open
positions filled before adoption keep their prior NULL — never retroactively
mutated. This test PROVES that invariant; it adds no fill-wiring code.

Postgres-only (paper_trade JSONB/partial-unique + thesis_evaluations).
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select, text

from rainier.core.models import PaperTrade, StockPrice
from rainier.paper.ingest import canonical_instant
from rainier.paper.positions import fill_pending_positions

pytestmark = pytest.mark.requires_postgres


def _seed_thesis(session, tid):
    session.execute(
        text("INSERT INTO analysis_results (id, llm_model, prompt_template) "
             "VALUES (:i,'m','t') ON CONFLICT DO NOTHING"),
        {"i": tid},
    )


def _mk_pending(session, tid, symbol, scan_date):
    _seed_thesis(session, tid)
    session.add(PaperTrade(
        thesis_id=tid, symbol=symbol, scan_date=scan_date, session_name="close",
        status="pending", planned_entry_price=100.0, stop_loss=90.0,
        target_price=120.0,
    ))
    session.commit()


def _price(session, symbol, d, o, h, low, c):
    session.execute(
        text("INSERT INTO stocks (symbol) VALUES (:s) ON CONFLICT DO NOTHING"),
        {"s": symbol},
    )
    session.add(StockPrice(symbol=symbol, date=canonical_instant(d), open=o,
                           high=h, low=low, close=c, volume=1000))
    session.commit()


def _get(session, tid):
    session.expire_all()
    return session.execute(
        select(PaperTrade).where(PaperTrade.thesis_id == tid)
    ).scalars().first()


def test_future_fills_only_time_stop_adoption(pg_legacy_session):
    """A position filled BEFORE adoption keeps NULL time_stop_days; one filled
    AFTER snapshots the learned k. No retroactive mutation."""
    s = pg_legacy_session

    # --- Pre-adoption: fill BEFORE the time-stop was learned (k=None). ---
    before_scan = date(2026, 3, 2)  # Monday
    _mk_pending(s, 1, "BEFORE", before_scan)
    # T+1 = 2026-03-03; provide a clean open between stop/target.
    _price(s, "BEFORE", date(2026, 3, 3), 100, 101, 99, 100)
    res1 = fill_pending_positions(as_of=date(2026, 3, 3), learned_time_stop_days=None)
    assert res1["filled"] == 1
    before_pos = _get(s, 1)
    assert before_pos.status == "open"
    assert before_pos.time_stop_days is None  # filled before adoption → NULL

    # --- Adoption happens (config now learned_time_stop_days=5). ---
    # A NEW pending filled AFTER adoption snapshots k=5.
    after_scan = date(2026, 3, 4)
    _mk_pending(s, 2, "AFTER", after_scan)
    _price(s, "AFTER", date(2026, 3, 5), 100, 101, 99, 100)
    res2 = fill_pending_positions(as_of=date(2026, 3, 5), learned_time_stop_days=5)
    assert res2["filled"] == 1
    after_pos = _get(s, 2)
    assert after_pos.status == "open"
    assert after_pos.time_stop_days == 5  # filled after adoption → snapshots k

    # The already-open BEFORE position is NEVER retroactively mutated, even
    # though the later fill ran with k=5.
    before_again = _get(s, 1)
    assert before_again.time_stop_days is None


# ---------------------------------------------------------------------------
# weekly paper_lessons insight (D7c) from closed paper trades
# ---------------------------------------------------------------------------


def _add_closed(session, tid, symbol, *, exit_date, return_pct, pnl, reason):
    _seed_thesis(session, tid)
    session.add(PaperTrade(
        thesis_id=tid, symbol=symbol, scan_date=date(2026, 3, 1),
        session_name="close", status="closed", planned_entry_price=100.0,
        stop_loss=90.0, target_price=120.0, entry_date=date(2026, 3, 2),
        entry_price=100.0, shares=100, residual_cash=0.0, exit_date=exit_date,
        exit_price=100 * (1 + return_pct), exit_reason=reason,
        return_pct=return_pct, pnl=pnl,
    ))
    session.commit()


def test_paper_lessons_insight_from_closed_trades(pg_legacy_session):
    s = pg_legacy_session
    from rainier.llm_thesis.research import check_paper_lessons

    as_of = date(2026, 3, 20)
    _add_closed(s, 1, "WIN", exit_date=date(2026, 3, 10), return_pct=0.10,
                pnl=1000.0, reason="target")
    _add_closed(s, 2, "LOS", exit_date=date(2026, 3, 12), return_pct=-0.05,
                pnl=-500.0, reason="stop_loss")
    _add_closed(s, 3, "TIM", exit_date=date(2026, 3, 15), return_pct=0.02,
                pnl=200.0, reason="time_stop")

    out = check_paper_lessons(eval_date=as_of, days=30)
    assert len(out) == 1
    ins = out[0]
    assert ins.kind == "paper_lessons"
    assert ins.severity == "info"
    assert ins.action["kind"] == "noop"
    ev = ins.evidence
    assert ev["n_closed"] == 3
    assert abs(ev["win_rate"] - 2 / 3) < 1e-9
    assert ev["exit_reason_mix"] == {"target": 1, "stop_loss": 1, "time_stop": 1}
    assert ev["best"]["symbol"] == "WIN"
    assert ev["worst"]["symbol"] == "LOS"


def test_paper_lessons_no_closed_trades_emits_nothing(pg_legacy_session):
    s = pg_legacy_session  # noqa: F841
    from rainier.llm_thesis.research import check_paper_lessons

    assert check_paper_lessons(eval_date=date(2026, 3, 20), days=30) == []
