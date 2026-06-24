"""WS D — `rainier recover` detects today's QU100 freshness via `captured_at`,
not per-`capture_session` row counts; the Discord report is gated on the
recovery scrape actually succeeding.

Under the rebuild-the-day fix a day holds ONE `captured_at` per
`(data_date, ranking_type)` and the rows carry the LATEST scrape's
`capture_session`. The old recover counted rows where
`capture_session == <slot>` to decide a slot was "missed" — after a midday
rebuild every row reads `capture_session="midday"`, so morning would show 0
rows and recover would falsely re-scrape + re-report.

The fix:
  * `_is_qu100_fresh(latest_captured_at, now, schedule, tz)` — today's data is
    fresh iff a snapshot exists for today AND its `captured_at` is at/after the
    most-recent scheduled slot that is already due. Per-DAY, not per-slot.
  * `_should_send_recovery_report(was_stale, scrape_succeeded)` — re-send the
    daily outlook ONLY when the day was stale AND a recovery scrape this run
    succeeded. A queued-but-failed scrape must NOT fire a report off a stale
    snapshot (the exact frozen-data bug this PR fixes).
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from rainier.cli import _is_qu100_fresh, _should_send_recovery_report

TZ = ZoneInfo("America/New_York")
# Schedule mirrors settings.scraping.schedule (UTC clock strings the recover loop
# reads). Use local-clock slot times for the helper under test.
SCHEDULE = {
    "morning": "11:30",
    "midday": "13:30",
    "afternoon": "15:30",
    "close": "18:00",
}


def _dt(h, m):
    return datetime(2026, 6, 1, h, m, tzinfo=TZ)


# --- freshness detection ---------------------------------------------------


def test_fresh_when_captured_at_after_latest_due_slot():
    """At 16:00 the latest due slot is 15:30; a snapshot captured 15:31 is fresh."""
    now = _dt(16, 0)
    latest = _dt(15, 31)
    assert _is_qu100_fresh(latest, now, SCHEDULE, TZ) is True


def test_stale_when_captured_before_latest_due_slot():
    """At 16:00 (15:30 due) a snapshot stuck at the 11:30 morning slot is STALE —
    later slots never landed."""
    now = _dt(16, 0)
    latest = _dt(11, 31)  # only morning ran
    assert _is_qu100_fresh(latest, now, SCHEDULE, TZ) is False


def test_stale_when_no_snapshot_today():
    now = _dt(16, 0)
    assert _is_qu100_fresh(None, now, SCHEDULE, TZ) is False


def test_fresh_before_any_slot_due():
    """Before the first slot (11:30) is due, having no data yet is not 'stale' —
    nothing was due, so recover should not fire a scrape."""
    now = _dt(9, 0)
    assert _is_qu100_fresh(None, now, SCHEDULE, TZ) is True


def test_fresh_when_only_morning_due_and_morning_ran():
    """At 12:00 only the morning slot (11:30) is due; a morning snapshot is fresh."""
    now = _dt(12, 0)
    latest = _dt(11, 31)
    assert _is_qu100_fresh(latest, now, SCHEDULE, TZ) is True


# --- report gating ---------------------------------------------------------


def test_report_sent_when_stale_and_scrape_succeeds():
    assert _should_send_recovery_report(was_stale=True, scrape_succeeded=True) is True


def test_report_not_sent_when_scrape_fails():
    """A recovery scrape that FAILED must not fire a report off a stale snapshot."""
    assert _should_send_recovery_report(was_stale=True, scrape_succeeded=False) is False


def test_report_not_sent_when_already_fresh():
    """Data already fresh -> treat the report as already sent (no duplicate)."""
    assert _should_send_recovery_report(was_stale=False, scrape_succeeded=False) is False
    assert _should_send_recovery_report(was_stale=False, scrape_succeeded=True) is False
