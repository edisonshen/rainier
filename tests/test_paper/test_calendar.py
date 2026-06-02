"""TradingCalendar — injectable weekend/holiday logic (pure, runs everywhere)."""

from __future__ import annotations

from datetime import date

from rainier.paper.calendar import TradingCalendar


def test_weekend_is_not_a_session():
    cal = TradingCalendar()
    assert cal.is_session(date(2026, 1, 9))  # Friday
    assert not cal.is_session(date(2026, 1, 10))  # Saturday
    assert not cal.is_session(date(2026, 1, 11))  # Sunday


def test_next_session_friday_to_monday():
    cal = TradingCalendar()
    assert cal.next_session(date(2026, 1, 9)) == date(2026, 1, 12)


def test_add_sessions_skips_weekend():
    cal = TradingCalendar()
    # T+1 from Friday lands Monday.
    assert cal.add_sessions(date(2026, 1, 9), 1) == date(2026, 1, 12)
    # add_sessions(0) from a weekend rolls forward to the next session.
    assert cal.add_sessions(date(2026, 1, 10), 0) == date(2026, 1, 12)


def test_holiday_injection():
    cal = TradingCalendar(holidays={date(2026, 1, 12)})  # Monday holiday
    assert not cal.is_session(date(2026, 1, 12))
    assert cal.next_session(date(2026, 1, 9)) == date(2026, 1, 13)


def test_sessions_between_inclusive():
    cal = TradingCalendar()
    out = cal.sessions_between(date(2026, 1, 9), date(2026, 1, 13))
    assert out == [date(2026, 1, 9), date(2026, 1, 12), date(2026, 1, 13)]
