"""Step 5 + 6 (DB-backed) — fill (§E) + update_open_positions (§F12/F13/F16)."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select, text

from rainier.core.models import PaperSkip, PaperTrade, StockPrice
from rainier.paper.ingest import canonical_instant
from rainier.paper.positions import (
    fill_pending_positions,
    update_open_positions,
)

pytestmark = pytest.mark.requires_postgres


def _seed_thesis(session, tid):
    session.execute(
        text("INSERT INTO analysis_results (id, llm_model, prompt_template) "
             "VALUES (:i,'m','t') ON CONFLICT DO NOTHING"),
        {"i": tid},
    )


def _mk_pending(session, tid, symbol, scan_date, *, stop=90.0, target=120.0,
                time_stop_days=None):
    _seed_thesis(session, tid)
    p = PaperTrade(
        thesis_id=tid, symbol=symbol, scan_date=scan_date, session_name="close",
        status="pending", planned_entry_price=100.0, stop_loss=stop,
        target_price=target, time_stop_days=time_stop_days,
    )
    session.add(p)
    session.commit()
    return p.id


def _mk_open(session, tid, symbol, entry_date, *, entry=100.0, stop=90.0,
             target=120.0, shares=100, time_stop_days=None):
    _seed_thesis(session, tid)
    p = PaperTrade(
        thesis_id=tid, symbol=symbol, scan_date=entry_date, session_name="close",
        status="open", planned_entry_price=entry, stop_loss=stop,
        target_price=target, entry_date=entry_date, entry_price=entry,
        shares=shares, allocated_amount=shares * entry,
        residual_cash=10000 - shares * entry, price_basis="adjusted",
        time_stop_days=time_stop_days,
    )
    session.add(p)
    session.commit()
    return p.id


def _price(session, symbol, d, o, h, low, c, v=1000):
    session.execute(
        text("INSERT INTO stocks (symbol) VALUES (:s) ON CONFLICT DO NOTHING"),
        {"s": symbol},
    )
    session.add(
        StockPrice(symbol=symbol, date=canonical_instant(d), open=o, high=h,
                   low=low, close=c, volume=v)
    )
    session.commit()


def _get(session, pid):
    session.expire_all()
    return session.get(PaperTrade, pid)


# --------------------------- Fill (E) ---------------------------


def test_e1_t1_open_fill(pg_legacy_session):
    s = pg_legacy_session
    scan = date(2026, 1, 5)  # Monday
    pid = _mk_pending(s, 1, "AAA", scan)
    _price(s, "AAA", date(2026, 1, 6), 100, 105, 99, 102)  # T+1 Tuesday
    fill_pending_positions(as_of=date(2026, 1, 6))
    p = _get(s, pid)
    assert p.status == "open" and p.entry_price == 100 and p.shares == 100
    assert p.allocated_amount == 10000 and p.residual_cash == 0
    assert p.entry_date == date(2026, 1, 6)
    assert p.price_basis == "adjusted"


def test_e1b_weekend_t1_is_monday(pg_legacy_session):
    s = pg_legacy_session
    scan = date(2026, 1, 9)  # Friday
    pid = _mk_pending(s, 1, "AAA", scan)
    _price(s, "AAA", date(2026, 1, 12), 100, 105, 99, 102)  # Monday
    fill_pending_positions(as_of=date(2026, 1, 12))
    p = _get(s, pid)
    assert p.status == "open" and p.entry_date == date(2026, 1, 12)


def test_e1c_fill_validity_guard_gap_past_target(pg_legacy_session):
    s = pg_legacy_session
    scan = date(2026, 1, 5)
    pid = _mk_pending(s, 1, "AAA", scan, stop=90, target=120)
    _price(s, "AAA", date(2026, 1, 6), 125, 130, 124, 128)  # open gapped past target
    fill_pending_positions(as_of=date(2026, 1, 6))
    p = _get(s, pid)
    assert p.status == "expired"
    skips = s.execute(select(PaperSkip).where(PaperSkip.symbol == "AAA")).scalars().all()
    assert len(skips) == 1 and skips[0].reason == "gap_invalidated"


def test_e1c_fill_validity_guard_gap_past_stop(pg_legacy_session):
    s = pg_legacy_session
    pid = _mk_pending(s, 1, "AAA", date(2026, 1, 5), stop=90, target=120)
    _price(s, "AAA", date(2026, 1, 6), 88, 92, 85, 89)  # open <= stop
    fill_pending_positions(as_of=date(2026, 1, 6))
    assert _get(s, pid).status == "expired"


def test_e2_sizing_residual(pg_legacy_session):
    s = pg_legacy_session
    pid = _mk_pending(s, 1, "AAA", date(2026, 1, 5), stop=20, target=50)
    _price(s, "AAA", date(2026, 1, 6), 33.0, 35, 32, 34)
    fill_pending_positions(as_of=date(2026, 1, 6))
    p = _get(s, pid)
    assert p.shares == 303
    assert p.allocated_amount == pytest.approx(9999.0)
    assert p.residual_cash == pytest.approx(1.0)


def test_e3_pending_until_priced_then_expire(pg_legacy_session):
    s = pg_legacy_session
    scan = date(2026, 1, 5)  # Mon; T+1 = Tue 1/6
    pid = _mk_pending(s, 1, "AAA", scan)
    # No T+1 open. as_of = T+1 → stays pending (still within window).
    fill_pending_positions(as_of=date(2026, 1, 6))
    assert _get(s, pid).status == "pending"
    # A price appears at T+2 (1/7) → fills at T+2.
    _price(s, "AAA", date(2026, 1, 7), 100, 105, 99, 102)
    fill_pending_positions(as_of=date(2026, 1, 7))
    p = _get(s, pid)
    assert p.status == "open" and p.entry_date == date(2026, 1, 7)


def test_e3_fills_first_priced_session_when_t1_missing(pg_legacy_session):
    """codex iter-1 regression — a missing T+1 open must NOT block a fill at a
    later in-window session. Monday scan, NO Tuesday(T+1) bar, Wednesday(T+2)
    open present, single fill pass at as_of=Wed → fills at Wed (was: queried only
    T+1, found nothing, and would expire/stay pending)."""
    s = pg_legacy_session
    scan = date(2026, 1, 5)  # Mon; T+1 = Tue 1/6 (no bar), T+2 = Wed 1/7
    pid = _mk_pending(s, 1, "AAA", scan)
    _price(s, "AAA", date(2026, 1, 7), 100, 105, 99, 102)  # only T+2 priced
    # ONE fill pass that first sees the window — must fill at T+2, not expire.
    fill_pending_positions(as_of=date(2026, 1, 7))
    p = _get(s, pid)
    assert p.status == "open" and p.entry_date == date(2026, 1, 7)
    assert p.entry_price == 100


def test_e3_expire_after_two_sessions_no_price(pg_legacy_session):
    s = pg_legacy_session
    scan = date(2026, 1, 5)  # Mon; T+1 = Tue 1/6
    pid = _mk_pending(s, 1, "AAA", scan)
    # No prices at all. After 2 trading sessions past T+1 with no open → expired.
    # T+1 = 1/6; deadline session = 1/7; expires once as_of >= next session 1/8.
    fill_pending_positions(as_of=date(2026, 1, 7))
    assert _get(s, pid).status == "pending"  # not one session early
    fill_pending_positions(as_of=date(2026, 1, 8))
    assert _get(s, pid).status == "expired"


def test_concurrency_guard_fill_does_not_resurrect_closed(pg_legacy_session):
    """codex iter-4 regression — the fill/expire UPDATEs guard expected_status,
    so a stale pass can't flip a terminal (closed/expired) row back to open. A
    fill pass over an already-closed position is a no-op (rowcount 0)."""
    s = pg_legacy_session
    pid = _mk_open(s, 1, "AAA", date(2026, 1, 6), entry=100, stop=80, target=120)
    _price(s, "AAA", date(2026, 1, 6), 100, 121, 99, 120)  # target same day
    update_open_positions(as_of=date(2026, 1, 6))
    assert _get(s, pid).status == "closed"
    booked = _get(s, pid)
    booked_tuple = (booked.exit_price, booked.exit_reason, booked.pnl, booked.status)
    # A fill pass (which only targets pending) must not touch the closed row.
    fill_pending_positions(as_of=date(2026, 1, 8))
    after = _get(s, pid)
    assert (after.exit_price, after.exit_reason, after.pnl, after.status) == booked_tuple


def test_e4_no_double_fill(pg_legacy_session):
    s = pg_legacy_session
    pid = _mk_open(s, 1, "AAA", date(2026, 1, 6), entry=100)
    # Re-run fill on an already-open position → no change.
    fill_pending_positions(as_of=date(2026, 1, 8))
    p = _get(s, pid)
    assert p.status == "open" and p.entry_price == 100


def test_e6_expired_never_resurrects(pg_legacy_session):
    s = pg_legacy_session
    pid = _mk_pending(s, 1, "AAA", date(2026, 1, 5))
    fill_pending_positions(as_of=date(2026, 1, 8))  # expire (no price)
    assert _get(s, pid).status == "expired"
    # Now the T+1 open is back-filled by ingest → must stay expired.
    _price(s, "AAA", date(2026, 1, 6), 100, 105, 99, 102)
    fill_pending_positions(as_of=date(2026, 1, 8))
    assert _get(s, pid).status == "expired"


def test_e5a_time_stop_snapshot_value_then_frozen(pg_legacy_session):
    """E5a — config learned_time_stop_days=10 at fill → position stores 10;
    a later config change does NOT mutate the already-open position."""
    s = pg_legacy_session
    pid = _mk_pending(s, 1, "AAA", date(2026, 1, 5))
    _price(s, "AAA", date(2026, 1, 6), 100, 105, 99, 102)
    fill_pending_positions(as_of=date(2026, 1, 6), learned_time_stop_days=10)
    assert _get(s, pid).time_stop_days == 10
    # A later fill pass with a different config value must not touch the open row.
    fill_pending_positions(as_of=date(2026, 1, 8), learned_time_stop_days=20)
    assert _get(s, pid).time_stop_days == 10


def test_e5b_time_stop_snapshot_null_then_frozen(pg_legacy_session):
    """E5b (the dangerous one) — filled while config is NULL stores NULL; config
    later adopting 10 must NOT retroactively time-stop the already-open
    position (future-fills-only invariant)."""
    s = pg_legacy_session
    pid = _mk_pending(s, 1, "AAA", date(2026, 1, 5))
    _price(s, "AAA", date(2026, 1, 6), 100, 105, 99, 102)
    fill_pending_positions(as_of=date(2026, 1, 6), learned_time_stop_days=None)
    assert _get(s, pid).time_stop_days is None
    fill_pending_positions(as_of=date(2026, 1, 8), learned_time_stop_days=10)
    assert _get(s, pid).time_stop_days is None


# --------------------------- Update / exit (F db) ---------------------------


def test_f12_db_no_lookahead_after_as_of(pg_legacy_session):
    s = pg_legacy_session
    pid = _mk_open(s, 1, "AAA", date(2026, 1, 5), entry=100, stop=80, target=120)
    _price(s, "AAA", date(2026, 1, 5), 100, 101, 99, 100)
    _price(s, "AAA", date(2026, 1, 6), 100, 102, 99, 100)
    _price(s, "AAA", date(2026, 1, 7), 100, 103, 99, 100)
    _price(s, "AAA", date(2026, 1, 8), 100, 121, 99, 120)   # target bar
    _price(s, "AAA", date(2026, 1, 9), 100, 101, 79, 82)    # later stop bar
    # as_of before the target → stays open.
    update_open_positions(as_of=date(2026, 1, 7))
    assert _get(s, pid).status == "open"
    # as_of at the target bar → closes target THERE, never peeks to the stop.
    update_open_positions(as_of=date(2026, 1, 8))
    p = _get(s, pid)
    assert p.status == "closed" and p.exit_reason == "target"
    assert p.exit_date == date(2026, 1, 8)


def test_f13_idempotent_immutable_close(pg_legacy_session):
    s = pg_legacy_session
    pid = _mk_open(s, 1, "AAA", date(2026, 1, 5), entry=100, stop=80, target=120)
    _price(s, "AAA", date(2026, 1, 5), 100, 101, 99, 100)
    _price(s, "AAA", date(2026, 1, 6), 100, 121, 99, 120)  # target
    update_open_positions(as_of=date(2026, 1, 6))
    first = _get(s, pid)
    assert first.status == "closed" and first.exit_reason == "target"
    # Re-run + a new stop bar that WOULD change the exit → must not mutate.
    _price(s, "AAA", date(2026, 1, 7), 100, 101, 50, 55)
    update_open_positions(as_of=date(2026, 1, 7))
    again = _get(s, pid)
    assert (again.exit_date, again.exit_price, again.exit_reason, again.pnl) == (
        first.exit_date, first.exit_price, first.exit_reason, first.pnl
    )


def test_f16_split_freeze_closed_immutable(pg_legacy_session):
    s = pg_legacy_session
    pid = _mk_open(s, 1, "AAA", date(2026, 1, 5), entry=100, stop=80, target=120,
                   shares=100)
    _price(s, "AAA", date(2026, 1, 5), 100, 121, 99, 120)  # target same day
    update_open_positions(as_of=date(2026, 1, 5))
    booked = _get(s, pid)
    assert booked.status == "closed"
    booked_tuple = (booked.entry_price, booked.exit_price, booked.pnl, booked.shares)
    # A mid-hold split auto-adjusts stock_prices (halve everything).
    s.execute(text("UPDATE stock_prices SET open=open/2, high=high/2, "
                   "low=low/2, close=close/2 WHERE symbol='AAA'"))
    s.commit()
    update_open_positions(as_of=date(2026, 1, 8))
    after = _get(s, pid)
    assert (after.entry_price, after.exit_price, after.pnl, after.shares) == booked_tuple
