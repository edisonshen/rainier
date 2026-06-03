"""Injectable trading calendar.

Determinism (TEST-SPEC §Determinism): position fill/expire counts in *trading
sessions*, not calendar days — so weekend/holiday handling must be explicit and
injectable (never `date.today()` / naive `+1 day`). The default models Mon-Fri
business days (no holidays), matching the existing `eval._trading_days_back`
convention; tests inject a custom calendar to pin holiday/weekend behavior.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

# Monday=0 ... Sunday=6
_WEEKEND = {5, 6}


class TradingCalendar:
    """Mon-Fri business-day calendar with an optional explicit holiday set.

    Inject a custom instance in tests for deterministic weekend/holiday cases.
    """

    def __init__(self, holidays: Iterable[date] | None = None) -> None:
        self._holidays: frozenset[date] = frozenset(holidays or ())

    def is_session(self, d: date) -> bool:
        return d.weekday() not in _WEEKEND and d not in self._holidays

    def next_session(self, d: date) -> date:
        """The first trading session strictly after ``d`` (weekend→Monday)."""
        nxt = d + timedelta(days=1)
        while not self.is_session(nxt):
            nxt += timedelta(days=1)
        return nxt

    def sessions_between(self, start: date, end: date) -> list[date]:
        """Trading sessions in ``[start, end]`` inclusive, ascending."""
        out: list[date] = []
        cur = start
        while cur <= end:
            if self.is_session(cur):
                out.append(cur)
            cur += timedelta(days=1)
        return out

    def add_sessions(self, start: date, n: int) -> date:
        """The session ``n`` trading days after ``start`` (n=0 → start itself
        if it's a session, else the next session)."""
        cur = start
        if not self.is_session(cur):
            cur = self.next_session(cur)
        for _ in range(n):
            cur = self.next_session(cur)
        return cur


DEFAULT_CALENDAR = TradingCalendar()
