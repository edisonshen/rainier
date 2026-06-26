"""WS D — `rainier recover` detects today's QU100 freshness via `captured_at`,
not per-`capture_session` row counts; the Discord report is gated on freshness
being RESTORED (re-read from the DB after the recovery scrape), not on the scrape
coroutine merely returning.

Under the rebuild-the-day fix a day holds ONE `captured_at` per
`(data_date, ranking_type)` and the rows carry the LATEST scrape's
`capture_session`. The old recover counted rows where
`capture_session == <slot>` to decide a slot was "missed" — after a midday
rebuild every row reads `capture_session="midday"`, so morning would show 0
rows and recover would falsely re-scrape + re-report.

The fix:
  * `_is_qu100_fresh(latest_captured_at, now, schedule, tz)` — per-ranking_type
    primitive: this book is fresh iff a snapshot exists AND its `captured_at` is
    at/after the most-recent scheduled slot already due. Per-DAY, not per-slot.
  * `_qu100_day_is_fresh(db, today, now, schedule, tz)` — the day is fresh only
    when EVERY book (top100 AND bottom100) is fresh. A partial scrape (top100
    lands, bottom100 empty no-op) must read STALE so recover re-fires — a single
    global `max(captured_at)` would let fresh top100 mask stale bottom100.
  * the report is sent only when the day was stale AND `_qu100_day_is_fresh` is
    True again after the scrape — a scrape that returns without raising but lands
    no fresh data (the documented empty-slot slip) must NOT fire a report off a
    frozen snapshot (the exact bug this PR fixes).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from rainier.cli import (
    _is_qu100_fresh,
    _latest_due_session,
    _latest_due_slot,
    _qu100_day_is_fresh,
    _recover_trading_day,
)
from rainier.core.models import MoneyFlowSnapshot

TZ = ZoneInfo("America/New_York")
SCHEDULE = {
    "morning": "11:30",
    "midday": "13:30",
    "afternoon": "15:30",
    "close": "18:00",
}


def _dt(h, m):
    return datetime(2026, 6, 1, h, m, tzinfo=TZ)


# --- "today" anchored to the app-local schedule date -----------------------


def test_recover_today_anchored_to_app_local_date():
    """Codex P1 regression — `recover` resolves "today" via the APP-LOCAL calendar
    date (the schedule's timezone), NOT the ET market date. A late-evening run that
    has crossed ET midnight but is still the same app-local day must resolve to
    that day, so a missed close is recovered against the day that just ended (using
    the ET date would point at the next, not-yet-traded day and — on Friday night —
    fall into the weekend fast-path and skip recovery entirely)."""
    pt = ZoneInfo("America/Los_Angeles")
    # 2026-06-01 22:30 PT == 2026-06-02 01:30 ET. App-local (PT) day is still
    # Monday 2026-06-01 — recover must judge against Monday's close, not Tuesday.
    late_pt = datetime(2026, 6, 1, 22, 30, tzinfo=pt)
    assert _recover_trading_day(late_pt) == date(2026, 6, 1)
    # mid-afternoon PT (during the scraping window): app-local == ET date anyway.
    afternoon_pt = datetime(2026, 6, 1, 13, 0, tzinfo=pt)
    assert _recover_trading_day(afternoon_pt) == date(2026, 6, 1)


# --- per-book freshness primitive ------------------------------------------


def test_fresh_when_captured_at_after_latest_due_slot():
    """At 16:00 the latest due slot is 15:30; a snapshot captured 15:31 is fresh."""
    assert _is_qu100_fresh(_dt(15, 31), _dt(16, 0), SCHEDULE, TZ) is True


def test_fresh_when_captured_at_equals_latest_due_slot():
    """A snapshot landing EXACTLY at the most-recent due slot is fresh (`>=`)."""
    assert _is_qu100_fresh(_dt(15, 30), _dt(16, 0), SCHEDULE, TZ) is True


def test_stale_when_captured_before_latest_due_slot():
    """At 16:00 (15:30 due) a snapshot stuck at the 11:30 morning slot is STALE."""
    assert _is_qu100_fresh(_dt(11, 31), _dt(16, 0), SCHEDULE, TZ) is False


def test_stale_when_no_snapshot_today():
    assert _is_qu100_fresh(None, _dt(16, 0), SCHEDULE, TZ) is False


def test_fresh_before_any_slot_due():
    """Before the first slot (11:30) is due, having no data yet is not 'stale'."""
    assert _is_qu100_fresh(None, _dt(9, 0), SCHEDULE, TZ) is True


def test_fresh_when_only_morning_due_and_morning_ran():
    """At 12:00 only the morning slot (11:30) is due; a morning snapshot is fresh."""
    assert _is_qu100_fresh(_dt(11, 31), _dt(12, 0), SCHEDULE, TZ) is True


# --- latest-due-slot selection ---------------------------------------------


def test_latest_due_slot_returns_most_recent_due():
    name, slot_time = _latest_due_slot(SCHEDULE, _dt(16, 0))
    assert name == "afternoon"  # 15:30, most recent <= 16:00
    assert slot_time == _dt(15, 30)


def test_latest_due_slot_only_morning_due():
    name, slot_time = _latest_due_slot(SCHEDULE, _dt(12, 0))
    assert name == "morning"
    assert slot_time == _dt(11, 30)


def test_latest_due_slot_none_due_yet():
    name, slot_time = _latest_due_slot(SCHEDULE, _dt(9, 0))
    assert name is None and slot_time is None


def test_latest_due_session_falls_back_to_first_when_none_due():
    """Defensive: when no slot is due, label with the first slot."""
    assert _latest_due_session(SCHEDULE, _dt(9, 0)) == "morning"


def test_latest_due_session_picks_most_recent_due():
    assert _latest_due_session(SCHEDULE, _dt(16, 0)) == "afternoon"


# --- per-ranking_type day freshness (DB-backed) ----------------------------

_DDL = """
CREATE TABLE money_flow_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  captured_at TIMESTAMP NOT NULL,
  capture_session VARCHAR(20) NOT NULL,
  data_date DATE NOT NULL,
  view_type VARCHAR(10) NOT NULL DEFAULT 'daily',
  ranking_type VARCHAR(10) NOT NULL,
  symbol VARCHAR(10) NOT NULL,
  rank INTEGER NOT NULL,
  daily_change INTEGER, sector VARCHAR(100), industry VARCHAR(200),
  long_short VARCHAR(50), raw_data JSON
);
"""

DAY = date(2026, 6, 1)


@pytest.fixture
def db_factory():
    engine = create_engine("sqlite://", future=True)
    with engine.begin() as conn:
        conn.execute(text(_DDL.strip()))
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


def _insert(factory, ranking_type: str, captured_at: datetime, symbol: str = "X"):
    with factory() as s:
        s.add(
            MoneyFlowSnapshot(
                captured_at=captured_at,
                capture_session="midday",
                data_date=DAY,
                ranking_type=ranking_type,
                symbol=symbol,
                rank=1,
            )
        )
        s.commit()


def test_day_fresh_when_both_books_fresh(db_factory):
    """Both top100 and bottom100 at/after the latest due slot -> fresh."""
    _insert(db_factory, "top100", datetime(2026, 6, 1, 19, 31, tzinfo=timezone.utc))
    _insert(db_factory, "bottom100", datetime(2026, 6, 1, 19, 31, tzinfo=timezone.utc))
    now = _dt(16, 0)  # 15:30 ET due == 19:30 UTC
    with db_factory() as db:
        assert _qu100_day_is_fresh(db, DAY, now, SCHEDULE, TZ) is True


def test_day_stale_when_bottom100_missing(db_factory):
    """REGRESSION: top100 fresh but bottom100 absent (empty no-op) -> STALE.

    A single global max(captured_at) would read top100's fresh time and report the
    whole day fresh, masking the missing book and skipping recovery."""
    _insert(db_factory, "top100", datetime(2026, 6, 1, 19, 31, tzinfo=timezone.utc))
    now = _dt(16, 0)
    with db_factory() as db:
        assert _qu100_day_is_fresh(db, DAY, now, SCHEDULE, TZ) is False


def test_day_stale_when_one_book_behind(db_factory):
    """REGRESSION: top100 fresh, bottom100 stuck at the morning slot -> STALE."""
    _insert(db_factory, "top100", datetime(2026, 6, 1, 19, 31, tzinfo=timezone.utc))
    _insert(db_factory, "bottom100", datetime(2026, 6, 1, 15, 31, tzinfo=timezone.utc))
    now = _dt(16, 0)  # latest due 15:30 ET == 19:30 UTC; bottom100 is 11:31 ET
    with db_factory() as db:
        assert _qu100_day_is_fresh(db, DAY, now, SCHEDULE, TZ) is False


def test_day_fresh_before_any_slot_due_with_no_data(db_factory):
    """No slot due yet and no data -> fresh (nothing expected)."""
    now = _dt(9, 0)
    with db_factory() as db:
        assert _qu100_day_is_fresh(db, DAY, now, SCHEDULE, TZ) is True


def test_midnight_clamp_keeps_recovered_day_stale(db_factory):
    """REGRESSION (Codex): a recovery that crosses local midnight must still judge
    the recovered day against ITS OWN last slot, not a fresh next-day clock.

    A snapshot stuck at the morning slot, evaluated with the clock clamped to
    end-of-`DAY` (18:00 close is the last due slot), must read STALE. Without the
    clamp the post-midnight clock would find no slot due on the new day and
    falsely report fresh."""
    _insert(db_factory, "top100", datetime(2026, 6, 1, 15, 31, tzinfo=timezone.utc))
    _insert(db_factory, "bottom100", datetime(2026, 6, 1, 15, 31, tzinfo=timezone.utc))
    # Clamp the clock to end-of-DAY (what _recover does when now crosses midnight).
    clamped = datetime(2026, 6, 1, 23, 59, 59, tzinfo=TZ)
    with db_factory() as db:
        # 11:31 ET data vs the 18:00 close slot now due -> STALE.
        assert _qu100_day_is_fresh(db, DAY, clamped, SCHEDULE, TZ) is False


def test_midnight_clamp_fresh_when_close_slot_landed(db_factory):
    """Clamped to end-of-day: a snapshot at/after the 18:00 close slot is fresh."""
    _insert(db_factory, "top100", datetime(2026, 6, 1, 22, 1, tzinfo=timezone.utc))
    _insert(db_factory, "bottom100", datetime(2026, 6, 1, 22, 1, tzinfo=timezone.utc))
    clamped = datetime(2026, 6, 1, 23, 59, 59, tzinfo=TZ)
    with db_factory() as db:
        # 18:01 ET >= 18:00 close slot -> fresh.
        assert _qu100_day_is_fresh(db, DAY, clamped, SCHEDULE, TZ) is True
