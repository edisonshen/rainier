"""WS B — reconsider invalidated bull-traps (`false_breakout` only).

Lanes:
  * pure-logic (no DB): the reclaim predicate + audit gate math;
  * postgres-lane (`pg_legacy_engine`, skips without PG): detect+enqueue
    idempotency / K-day whipsaw dedup / scope exclusion / queue processing.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import text

from rainier.core.database import get_session
from rainier.paper.reclaim import (
    compute_audit_pass,
    detect_and_enqueue_reclaims,
    is_reclaimed,
    process_reclaim_queue,
)

# --------------------------------------------------------------------------
# Pure-logic lane
# --------------------------------------------------------------------------


def test_is_reclaimed_true_when_close_exceeds_level():
    assert is_reclaimed([90.0, 95.0, 101.0], 100.0) is True


def test_is_reclaimed_false_when_no_close_exceeds():
    assert is_reclaimed([90.0, 99.99, 100.0], 100.0) is False


def test_is_reclaimed_equality_does_not_reclaim():
    # A close exactly AT the trap high does not clear it.
    assert is_reclaimed([100.0], 100.0) is False


def test_audit_gate_fails_below_min_events():
    res = compute_audit_pass([0.1, 0.2], min_events=5)
    assert res.events == 2
    assert res.passes is False


def test_audit_gate_fails_on_nonpositive_mean():
    res = compute_audit_pass([0.1, -0.3, 0.1], min_events=2)
    assert res.mean_forward_return is not None
    assert res.mean_forward_return < 0
    assert res.passes is False


def test_audit_gate_passes_with_enough_positive_events():
    res = compute_audit_pass([0.05, 0.1, 0.2], min_events=3)
    assert res.events == 3
    assert res.passes is True


def test_audit_gate_fails_with_no_events():
    res = compute_audit_pass([], min_events=1)
    assert res.events == 0
    assert res.mean_forward_return is None
    assert res.passes is False


def test_migration_0012_files_exist():
    from tests.test_paper.conftest import MIGRATION_0012_DOWN, MIGRATION_0012_UP

    assert MIGRATION_0012_UP.exists()
    assert MIGRATION_0012_DOWN.exists()


# --------------------------------------------------------------------------
# Postgres-lane — detect+enqueue, idempotency, dedup, scope. Skips without PG.
# --------------------------------------------------------------------------


def _seed_thesis(session, *, symbol: str, recommendation: str) -> int:
    res = session.execute(
        text(
            "INSERT INTO analysis_results (llm_model, prompt_template, "
            "recommendation) VALUES ('test', 'test', :rec) RETURNING id"
        ),
        {"rec": recommendation},
    )
    return int(res.scalar())


def _seed_screened(
    session,
    *,
    symbol: str,
    scan_date: date,
    thesis_id: int,
    pattern_type: str,
    invalidation_level: float | None,
) -> None:
    session.execute(
        text(
            "INSERT INTO screened_stocks "
            "(scan_date, session_name, symbol, rule_rank, composite_score, "
            " pattern_type, thesis_id, bearish_invalidation_level) "
            "VALUES (:sd, 'close', :sym, 1, 0.5, :pt, :tid, :lvl)"
        ),
        {
            "sd": scan_date,
            "sym": symbol,
            "pt": pattern_type,
            "tid": thesis_id,
            "lvl": invalidation_level,
        },
    )


def _seed_price(session, *, symbol: str, d: date, close: float) -> None:
    session.execute(
        text(
            "INSERT INTO stock_prices (symbol, date, open, high, low, close, "
            "volume) VALUES (:sym, :d, :c, :c, :c, :c, 1000)"
        ),
        {"sym": symbol, "d": datetime(d.year, d.month, d.day, tzinfo=timezone.utc),
         "c": close},
    )


@pytest.fixture
def reclaim_db(pg_legacy_engine):
    """A seeded DB: one declined false_breakout (trap=100) whose later close
    reclaimed it (105), plus a control no-reclaim symbol."""
    eval_date = date(2026, 6, 12)
    scan_date = date(2026, 6, 2)
    with get_session() as session:
        # Reclaimed: trap 100, later close 105.
        tid = _seed_thesis(session, symbol="CRDO", recommendation="watch")
        _seed_screened(
            session, symbol="CRDO", scan_date=scan_date, thesis_id=tid,
            pattern_type="false_breakout", invalidation_level=100.0,
        )
        _seed_price(session, symbol="CRDO", d=scan_date, close=95.0)
        _seed_price(session, symbol="CRDO", d=date(2026, 6, 10), close=105.0)

        # Not reclaimed: trap 200, later close 150.
        tid2 = _seed_thesis(session, symbol="TTMI", recommendation="watch")
        _seed_screened(
            session, symbol="TTMI", scan_date=scan_date, thesis_id=tid2,
            pattern_type="false_breakout", invalidation_level=200.0,
        )
        _seed_price(session, symbol="TTMI", d=scan_date, close=180.0)
        _seed_price(session, symbol="TTMI", d=date(2026, 6, 10), close=150.0)
    return {"eval_date": eval_date, "scan_date": scan_date}


def _queue_rows(symbol: str | None = None) -> list[dict]:
    with get_session() as session:
        sql = "SELECT symbol, status, invalidation_level FROM paper_reclaim_queue"
        if symbol:
            sql += " WHERE symbol = :sym"
            rows = session.execute(text(sql), {"sym": symbol}).mappings().all()
        else:
            rows = session.execute(text(sql)).mappings().all()
    return [dict(r) for r in rows]


def test_detect_enqueues_reclaimed_not_unreclaimed(reclaim_db):
    res = detect_and_enqueue_reclaims(
        eval_date=reclaim_db["eval_date"], window_days=30, dedup_days=5
    )
    assert res["enqueued"] == 1
    rows = _queue_rows()
    symbols = {r["symbol"] for r in rows}
    assert symbols == {"CRDO"}
    assert rows[0]["status"] == "queued"


def test_detect_is_idempotent(reclaim_db):
    detect_and_enqueue_reclaims(
        eval_date=reclaim_db["eval_date"], window_days=30, dedup_days=5
    )
    # Second run on the same state must NOT create a second row.
    res2 = detect_and_enqueue_reclaims(
        eval_date=reclaim_db["eval_date"], window_days=30, dedup_days=5
    )
    assert res2["enqueued"] == 0
    assert len(_queue_rows("CRDO")) == 1


def test_false_breakout_hs_never_enters_queue(pg_legacy_engine):
    """Scope regression: false_breakout_hs has no false_high → never persists a
    bearish_invalidation_level → never enters the reclaim queue."""
    scan_date = date(2026, 6, 2)
    with get_session() as session:
        tid = _seed_thesis(session, symbol="SGHC", recommendation="watch")
        # NULL invalidation level (mirrors what the screener persists for _hs).
        _seed_screened(
            session, symbol="SGHC", scan_date=scan_date, thesis_id=tid,
            pattern_type="false_breakout_hs", invalidation_level=None,
        )
        _seed_price(session, symbol="SGHC", d=scan_date, close=95.0)
        _seed_price(session, symbol="SGHC", d=date(2026, 6, 10), close=999.0)
    res = detect_and_enqueue_reclaims(
        eval_date=date(2026, 6, 12), window_days=30, dedup_days=5
    )
    assert res["enqueued"] == 0
    assert _queue_rows("SGHC") == []


def test_setup_long_thesis_not_reconsidered(pg_legacy_engine):
    """A false_breakout whose thesis was setup_long is already acted on the
    bullish side — no bear trap to reconsider."""
    scan_date = date(2026, 6, 2)
    with get_session() as session:
        tid = _seed_thesis(session, symbol="XYZ", recommendation="setup_long")
        _seed_screened(
            session, symbol="XYZ", scan_date=scan_date, thesis_id=tid,
            pattern_type="false_breakout", invalidation_level=100.0,
        )
        _seed_price(session, symbol="XYZ", d=scan_date, close=95.0)
        _seed_price(session, symbol="XYZ", d=date(2026, 6, 10), close=120.0)
    res = detect_and_enqueue_reclaims(
        eval_date=date(2026, 6, 12), window_days=30, dedup_days=5
    )
    assert res["enqueued"] == 0


def test_process_queue_detection_only_when_gate_disabled(reclaim_db):
    """auto_thesis False → rows stay queued, nothing spawned (the default)."""
    detect_and_enqueue_reclaims(
        eval_date=reclaim_db["eval_date"], window_days=30, dedup_days=5
    )
    spawned = []

    def _spawn(symbol, queue_id):  # must NOT be called when gate is off
        spawned.append(symbol)
        return 999

    res = process_reclaim_queue(
        eval_date=reclaim_db["eval_date"],
        auto_thesis=False,
        audit_horizon_days=10,
        audit_min_events=1,
        spawn_fn=_spawn,
    )
    assert res["gate_passed"] == 0
    assert res["spawned"] == 0
    assert spawned == []
    assert _queue_rows("CRDO")[0]["status"] == "queued"
