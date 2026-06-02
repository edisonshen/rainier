"""Step 4 — thesis-time position creation (TEST-SPEC §D)."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select, text

from rainier.core.models import PaperSkip, PaperTrade, ScreenedStockRecord
from rainier.paper.positions import create_positions_for_theses

pytestmark = pytest.mark.requires_postgres

SCAN = date(2026, 1, 9)


def _mk_thesis(session, thesis_id, symbol, verdict="setup_long", conf=7,
               session_name="close"):
    session.execute(
        text("INSERT INTO analysis_results (id, llm_model, prompt_template) "
             "VALUES (:i,'m','t') ON CONFLICT DO NOTHING"),
        {"i": thesis_id},
    )
    return {
        "thesis_id": thesis_id, "symbol": symbol, "verdict": verdict,
        "llm_confidence": conf, "session_name": session_name,
    }


def _mk_screened(session, symbol, session_name="close", entry=100, stop=90,
                 target=120, rr=2.0):
    # Conflict-safe: some cases (d2/d4) call this repeatedly for the same
    # (scan_date, session_name, symbol) inside one test; the UNIQUE constraint
    # would otherwise raise. DO NOTHING keeps the first row's levels.
    from sqlalchemy.dialects.postgresql import insert as _pg_insert

    stmt = (
        _pg_insert(ScreenedStockRecord)
        .values(
            scan_date=SCAN, session_name=session_name, symbol=symbol,
            rule_rank=1, composite_score=0.8, pattern_type="bull_flag",
            entry_price=entry, stop_loss=stop, target_price=target, rr_ratio=rr,
        )
        .on_conflict_do_nothing(
            index_elements=["scan_date", "session_name", "symbol"]
        )
    )
    session.execute(stmt)
    session.commit()


def _trades(session, symbol=None):
    q = select(PaperTrade)
    if symbol:
        q = q.where(PaperTrade.symbol == symbol)
    return session.execute(q).scalars().all()


def _skips(session, symbol=None):
    q = select(PaperSkip)
    if symbol:
        q = q.where(PaperSkip.symbol == symbol)
    return session.execute(q).scalars().all()


def test_d1_setup_long_creates_pending_with_null_fill(pg_legacy_session):
    t = _mk_thesis(pg_legacy_session, 1, "AAA", conf=7)
    _mk_screened(pg_legacy_session, "AAA")
    create_positions_for_theses([t], scan_date=SCAN)
    pg_legacy_session.expire_all()
    rows = _trades(pg_legacy_session, "AAA")
    assert len(rows) == 1
    r = rows[0]
    assert r.status == "pending"
    assert r.planned_entry_price == 100 and r.stop_loss == 90 and r.target_price == 120
    assert r.time_stop_days is None
    # All fill fields NULL while pending.
    assert r.entry_date is None and r.entry_price is None and r.shares is None
    assert r.allocated_amount is None and r.residual_cash is None
    assert r.price_basis is None


def test_d2_non_setup_long_skipped(pg_legacy_session):
    for v in ("watch", "no_setup"):
        t = _mk_thesis(pg_legacy_session, hash(v) % 10000, "AAA", verdict=v)
        _mk_screened(pg_legacy_session, "AAA")
        create_positions_for_theses([t], scan_date=SCAN)
    pg_legacy_session.expire_all()
    assert len(_trades(pg_legacy_session, "AAA")) == 0
    assert len(_skips(pg_legacy_session, "AAA")) == 0  # not a buy signal != skip


def test_d3_confidence_gate(pg_legacy_session):
    t5 = _mk_thesis(pg_legacy_session, 5, "FIV", conf=5)
    _mk_screened(pg_legacy_session, "FIV")
    create_positions_for_theses([t5], scan_date=SCAN)
    t6 = _mk_thesis(pg_legacy_session, 6, "SIX", conf=6)
    _mk_screened(pg_legacy_session, "SIX")
    create_positions_for_theses([t6], scan_date=SCAN)
    pg_legacy_session.expire_all()
    assert len(_trades(pg_legacy_session, "FIV")) == 0
    assert len(_trades(pg_legacy_session, "SIX")) == 1


def test_d4_session_gate(pg_legacy_session):
    for sess in ("morning", "midday"):
        t = _mk_thesis(pg_legacy_session, hash(sess) % 10000, "AAA",
                       session_name=sess)
        _mk_screened(pg_legacy_session, "AAA", session_name=sess)
        create_positions_for_theses([t], scan_date=SCAN)
    pg_legacy_session.expire_all()
    assert len(_trades(pg_legacy_session, "AAA")) == 0
    assert len(_skips(pg_legacy_session, "AAA")) == 0


def test_d5_idempotent_on_thesis_id(pg_legacy_session):
    t = _mk_thesis(pg_legacy_session, 1, "AAA")
    _mk_screened(pg_legacy_session, "AAA")
    create_positions_for_theses([t], scan_date=SCAN)
    create_positions_for_theses([t], scan_date=SCAN)
    pg_legacy_session.expire_all()
    assert len(_trades(pg_legacy_session, "AAA")) == 1


def test_d6_levels_from_persisted_row_not_candidate(pg_legacy_session):
    """D6 — levels come from the persisted screened row. The creation API takes
    only thesis metadata (no in-memory levels), so a Tier-1 cache-replayed
    thesis can't inject stale levels: the position mirrors the DB row exactly."""
    t = _mk_thesis(pg_legacy_session, 1, "AAA")
    _mk_screened(pg_legacy_session, "AAA", entry=42, stop=40, target=55, rr=3.0)
    create_positions_for_theses([t], scan_date=SCAN)
    pg_legacy_session.expire_all()
    r = _trades(pg_legacy_session, "AAA")[0]
    assert (r.planned_entry_price, r.stop_loss, r.target_price, r.rr_ratio) == (
        42, 40, 55, 3.0
    )


def test_d7_one_active_per_symbol_dedupe_and_skip(pg_legacy_session):
    # AAA already open via thesis 1.
    t1 = _mk_thesis(pg_legacy_session, 1, "AAA")
    _mk_screened(pg_legacy_session, "AAA")
    create_positions_for_theses([t1], scan_date=SCAN)
    # New thesis 2 for AAA → no second position + a paper_skip(symbol_already_active).
    t2 = _mk_thesis(pg_legacy_session, 2, "AAA")
    create_positions_for_theses([t2], scan_date=SCAN)
    pg_legacy_session.expire_all()
    assert len(_trades(pg_legacy_session, "AAA")) == 1
    skips = _skips(pg_legacy_session, "AAA")
    assert len(skips) == 1 and skips[0].reason == "symbol_already_active"
    assert skips[0].thesis_id == 2


def test_d7b_skip_ledger_idempotent(pg_legacy_session):
    t1 = _mk_thesis(pg_legacy_session, 1, "AAA")
    _mk_screened(pg_legacy_session, "AAA")
    create_positions_for_theses([t1], scan_date=SCAN)
    t2 = _mk_thesis(pg_legacy_session, 2, "AAA")
    create_positions_for_theses([t2], scan_date=SCAN)
    create_positions_for_theses([t2], scan_date=SCAN)  # re-run
    pg_legacy_session.expire_all()
    assert len(_skips(pg_legacy_session, "AAA")) == 1


def test_d7c_forced_db_conflict_batch_continues(pg_legacy_session):
    # An open AAA exists. Batch [AAA(thesis2), BBB(thesis3)] → AAA skip
    # (clean rollback), BBB inserts.
    t1 = _mk_thesis(pg_legacy_session, 1, "AAA")
    _mk_screened(pg_legacy_session, "AAA")
    create_positions_for_theses([t1], scan_date=SCAN)
    t2 = _mk_thesis(pg_legacy_session, 2, "AAA")
    t3 = _mk_thesis(pg_legacy_session, 3, "BBB")
    _mk_screened(pg_legacy_session, "BBB")
    create_positions_for_theses([t2, t3], scan_date=SCAN)
    pg_legacy_session.expire_all()
    assert len(_trades(pg_legacy_session, "AAA")) == 1
    assert len(_trades(pg_legacy_session, "BBB")) == 1
    skips = _skips(pg_legacy_session, "AAA")
    assert len(skips) == 1 and skips[0].reason == "symbol_already_active"


@pytest.mark.parametrize("null_field", ["entry_price", "stop_loss", "target_price"])
def test_d8_null_levels_zero_rows_then_retry(pg_legacy_session, null_field):
    t = _mk_thesis(pg_legacy_session, 1, "AAA")
    kwargs = dict(entry=100, stop=90, target=120)
    mapping = {"entry_price": "entry", "stop_loss": "stop", "target_price": "target"}
    kwargs[mapping[null_field]] = None
    _mk_screened(pg_legacy_session, "AAA", **kwargs)
    create_positions_for_theses([t], scan_date=SCAN)
    pg_legacy_session.expire_all()
    assert len(_trades(pg_legacy_session, "AAA")) == 0  # count==0, not just no-open
    skips = _skips(pg_legacy_session, "AAA")
    assert len(skips) == 1 and skips[0].reason == "missing_levels"

    # Retry: backfill the level, re-run → position created (skip didn't poison).
    pg_legacy_session.execute(
        text(f"UPDATE screened_stocks SET {null_field} = 100 WHERE symbol='AAA'")
    )
    pg_legacy_session.commit()
    create_positions_for_theses([t], scan_date=SCAN)
    pg_legacy_session.expire_all()
    assert len(_trades(pg_legacy_session, "AAA")) == 1
    # codex iter-6: the stale missing_levels skip must be CLEARED on the
    # successful retry — else the thesis double-counts as opened AND skipped.
    assert len(_skips(pg_legacy_session, "AAA")) == 0


def test_d9_absent_screened_row_skip(pg_legacy_session):
    t = _mk_thesis(pg_legacy_session, 1, "AAA")
    pg_legacy_session.commit()
    # No screened row for AAA.
    create_positions_for_theses([t], scan_date=SCAN)
    pg_legacy_session.expire_all()
    assert len(_trades(pg_legacy_session, "AAA")) == 0
    skips = _skips(pg_legacy_session, "AAA")
    assert len(skips) == 1 and skips[0].reason == "missing_screened_record"


def test_d10_same_symbol_day_highest_conviction(pg_legacy_session):
    # afternoon conf 6, close conf 8 → conf-8 wins; conf-6 → lower_conviction skip.
    ta = _mk_thesis(pg_legacy_session, 1, "AAA", conf=6, session_name="afternoon")
    tc = _mk_thesis(pg_legacy_session, 2, "AAA", conf=8, session_name="close")
    _mk_screened(pg_legacy_session, "AAA", session_name="afternoon")
    _mk_screened(pg_legacy_session, "AAA", session_name="close")
    # input order shouldn't matter:
    create_positions_for_theses([ta, tc], scan_date=SCAN)
    pg_legacy_session.expire_all()
    rows = _trades(pg_legacy_session, "AAA")
    assert len(rows) == 1 and rows[0].llm_confidence == 8 and rows[0].thesis_id == 2
    skips = _skips(pg_legacy_session, "AAA")
    assert len(skips) == 1 and skips[0].reason == "same_symbol_lower_conviction"
    assert skips[0].thesis_id == 1


def test_d10_tie_earliest_session_wins(pg_legacy_session):
    ta = _mk_thesis(pg_legacy_session, 1, "AAA", conf=7, session_name="afternoon")
    tc = _mk_thesis(pg_legacy_session, 2, "AAA", conf=7, session_name="close")
    _mk_screened(pg_legacy_session, "AAA", session_name="afternoon")
    _mk_screened(pg_legacy_session, "AAA", session_name="close")
    create_positions_for_theses([tc, ta], scan_date=SCAN)  # close first in input
    pg_legacy_session.expire_all()
    rows = _trades(pg_legacy_session, "AAA")
    assert len(rows) == 1 and rows[0].thesis_id == 1  # afternoon wins tie


def test_d10_cross_session_higher_conf_close_displaces_pending(pg_legacy_session):
    """codex iter-1 regression — production calls create_positions_for_theses
    once PER SESSION. An afternoon conf-6 pending must be DISPLACED by a later
    SEPARATE close conf-8 call for the same symbol (D10 cross-session), not
    permanently win via symbol_already_active."""
    # Afternoon call: conf-6 AAA → pending.
    ta = _mk_thesis(pg_legacy_session, 1, "AAA", conf=6, session_name="afternoon")
    _mk_screened(pg_legacy_session, "AAA", session_name="afternoon")
    create_positions_for_theses([ta], scan_date=SCAN)
    pg_legacy_session.expire_all()
    assert len(_trades(pg_legacy_session, "AAA")) == 1

    # Close call (separate invocation): conf-8 AAA → displaces the afternoon row.
    tc = _mk_thesis(pg_legacy_session, 2, "AAA", conf=8, session_name="close")
    _mk_screened(pg_legacy_session, "AAA", session_name="close")
    create_positions_for_theses([tc], scan_date=SCAN)
    pg_legacy_session.expire_all()

    rows = _trades(pg_legacy_session, "AAA")
    active = [r for r in rows if r.status in ("pending", "open")]
    assert len(active) == 1 and active[0].thesis_id == 2 and active[0].llm_confidence == 8
    # The displaced afternoon row is expired (frees the partial-unique slot).
    expired = [r for r in rows if r.status == "expired"]
    assert len(expired) == 1 and expired[0].thesis_id == 1
    # The displaced (lower-conviction) afternoon thesis is recorded.
    skips = _skips(pg_legacy_session, "AAA")
    assert len(skips) == 1 and skips[0].reason == "same_symbol_lower_conviction"
    assert skips[0].thesis_id == 1


def test_d10_cross_session_lower_conf_close_skipped(pg_legacy_session):
    """Symmetric guard — a LATER lower-conf close call must NOT displace the
    higher-conf afternoon pending; it's skipped same_symbol_lower_conviction."""
    ta = _mk_thesis(pg_legacy_session, 1, "AAA", conf=8, session_name="afternoon")
    _mk_screened(pg_legacy_session, "AAA", session_name="afternoon")
    create_positions_for_theses([ta], scan_date=SCAN)
    tc = _mk_thesis(pg_legacy_session, 2, "AAA", conf=6, session_name="close")
    _mk_screened(pg_legacy_session, "AAA", session_name="close")
    create_positions_for_theses([tc], scan_date=SCAN)
    pg_legacy_session.expire_all()
    active = [r for r in _trades(pg_legacy_session, "AAA")
              if r.status in ("pending", "open")]
    assert len(active) == 1 and active[0].thesis_id == 1  # afternoon conf-8 keeps slot
    skips = _skips(pg_legacy_session, "AAA")
    assert len(skips) == 1 and skips[0].thesis_id == 2
    assert skips[0].reason == "same_symbol_lower_conviction"


def test_d7_prior_day_pending_not_displaced(pg_legacy_session):
    """A pending from a DIFFERENT (prior) scan_date is NOT displaced by a new
    day's higher-conf pick — cross-day stays symbol_already_active (D7)."""
    prior = date(2026, 1, 8)
    ta = _mk_thesis(pg_legacy_session, 1, "AAA", conf=6)
    pg_legacy_session.add(
        ScreenedStockRecord(
            scan_date=prior, session_name="close", symbol="AAA", rule_rank=1,
            composite_score=0.8, pattern_type="bull_flag", entry_price=100,
            stop_loss=90, target_price=120, rr_ratio=2.0,
        )
    )
    pg_legacy_session.commit()
    create_positions_for_theses([ta], scan_date=prior)
    # New day, higher conf — must NOT displace the prior-day pending.
    tc = _mk_thesis(pg_legacy_session, 2, "AAA", conf=9)
    _mk_screened(pg_legacy_session, "AAA")  # SCAN = 2026-01-09
    create_positions_for_theses([tc], scan_date=SCAN)
    pg_legacy_session.expire_all()
    active = [r for r in _trades(pg_legacy_session, "AAA")
              if r.status in ("pending", "open")]
    assert len(active) == 1 and active[0].thesis_id == 1  # prior-day pending kept
    skips = _skips(pg_legacy_session, "AAA")
    assert len(skips) == 1 and skips[0].thesis_id == 2
    assert skips[0].reason == "symbol_already_active"


def test_d11_gate_reject_is_not_a_skip(pg_legacy_session):
    # morning setup_long + afternoon conf-4 setup_long → no position, no skip.
    tm = _mk_thesis(pg_legacy_session, 1, "AAA", session_name="morning")
    ta = _mk_thesis(pg_legacy_session, 2, "BBB", conf=4, session_name="afternoon")
    _mk_screened(pg_legacy_session, "AAA", session_name="morning")
    _mk_screened(pg_legacy_session, "BBB", session_name="afternoon")
    create_positions_for_theses([tm, ta], scan_date=SCAN)
    pg_legacy_session.expire_all()
    assert len(_trades(pg_legacy_session)) == 0
    assert len(_skips(pg_legacy_session)) == 0
