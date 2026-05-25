"""Tests for the market-tz date helper in ``rainier.scrapers.qu.scraper``.

Regression coverage for the UTC drift bug surfaced by codex iter 2 review
of PR #84: ``_scrape_qu100`` was deriving the API's ``date=`` query param
from ``captured_at.date()`` (UTC). At 19:00 PT (= 02:00 UTC next day) a
manual operator run requested the wrong trading day. The fix is to
derive the market date via ``America/New_York`` conversion.

See ``docs/TASK-PLAN-qu-captured-at-tz-drift-44e3.md`` §4 for acceptance.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from rainier.scrapers.qu.scraper import market_date


def test_market_date_late_pt_returns_same_day():
    """Late evening PT (next-day UTC) maps to the US-market calendar day.

    2026-05-22T02:30:00Z = 22:30 ET on 2026-05-21 = market day 2026-05-21.
    Before the fix, ``timestamp.date()`` returned 2026-05-22, which made the
    scraper request tomorrow's (non-existent) trading data.
    """
    ts = datetime(2026, 5, 22, 2, 30, tzinfo=timezone.utc)
    assert market_date(ts) == date(2026, 5, 21)


def test_market_date_morning_pt_returns_today():
    """Daytime UTC during US market hours stays on the same calendar day.

    2026-05-22T15:00:00Z = 11:00 ET on 2026-05-22 = market day 2026-05-22.
    Sanity check that the helper is not flipping the date for in-market
    timestamps.
    """
    ts = datetime(2026, 5, 22, 15, 0, tzinfo=timezone.utc)
    assert market_date(ts) == date(2026, 5, 22)


def test_market_date_dst_boundary():
    """DST spring-forward boundary still resolves correctly.

    On 2026-03-08, America/New_York jumps from EST (UTC-5) to EDT (UTC-4)
    at 02:00 local time. 2026-03-08T07:00:00Z is 02:00 EST -> 03:00 EDT
    on 2026-03-08 either way — the date should be 2026-03-08 regardless
    of which side of the transition zoneinfo lands on.
    """
    ts = datetime(2026, 3, 8, 7, 0, tzinfo=timezone.utc)
    assert market_date(ts) == date(2026, 3, 8)
