"""Pure-helper unit tests for the screened-levels backfill (no Postgres).

The end-to-end backfill tests are Postgres-gated (``requires_postgres``) and
skip when no DB is reachable. But three load-bearing helpers are pure functions
that need no DB — and one of them (``_match_pattern``'s same-type tie-break) is
not exercised by the e2e fixture at all (it yields two patterns of DIFFERENT
types). These ungated tests pin that logic so it has coverage on every run.
"""

from __future__ import annotations

from datetime import date, timezone

import pandas as pd
import pytest

from rainier.core.types import PatternSignal
from rainier.paper.backfill_screened_levels import (
    _as_of_idx,
    _match_pattern,
    backfill_screened_levels,
)


def _sig(
    pattern_type: str,
    *,
    confidence: float,
    entry_price: float,
    stop_loss: float = 0.0,
    target_wave1: float = 0.0,
) -> PatternSignal:
    return PatternSignal(
        symbol="X",
        pattern_type=pattern_type,
        direction="bullish",
        status="confirmed",
        confidence=confidence,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_wave1=target_wave1,
    )


# --- _match_pattern --------------------------------------------------------


def test_match_pattern_no_match_returns_none():
    actionable = [_sig("w_bottom", confidence=0.9, entry_price=100.0)]
    assert _match_pattern(actionable, "false_breakdown") is None


def test_match_pattern_empty_returns_none():
    assert _match_pattern([], "w_bottom") is None


def test_match_pattern_picks_only_matching_type_not_top_ranked():
    # actionable[0] (the live `best_pattern`) is a non-matching type; the
    # matching one ranks lower but must still be chosen because we match by
    # stored type, not by top-ranking.
    actionable = [
        _sig("w_bottom", confidence=0.95, entry_price=100.0),
        _sig("false_breakdown", confidence=0.40, entry_price=89.0),
    ]
    got = _match_pattern(actionable, "false_breakdown")
    assert got is not None
    assert got.pattern_type == "false_breakdown"
    assert got.entry_price == 89.0


def test_match_pattern_same_type_takes_first_in_priority_order():
    # Two SAME-type patterns: the FIRST in _filter_actionable priority order
    # wins — that is the setup the live screener would have written as
    # best_pattern. We must NOT re-sort by confidence: here the first entry has
    # LOWER confidence but higher actionability priority, so it must win.
    actionable = [
        _sig("w_bottom", confidence=0.50, entry_price=80.0),  # higher priority
        _sig("w_bottom", confidence=0.90, entry_price=120.0),  # lower priority
    ]
    got = _match_pattern(actionable, "w_bottom")
    assert got is not None
    # First-in-order wins despite its lower confidence (no confidence re-sort).
    assert got.confidence == 0.50
    assert got.entry_price == 80.0


# --- _as_of_idx ------------------------------------------------------------


def _df(dates: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame({"close": range(len(dates))}, index=pd.DatetimeIndex(dates))


def test_as_of_idx_hits_exact_scan_date_bar():
    dates = [
        pd.Timestamp("2026-06-01", tz="UTC"),
        pd.Timestamp("2026-06-03", tz="UTC"),
        pd.Timestamp("2026-06-05", tz="UTC"),
        pd.Timestamp("2026-06-09", tz="UTC"),  # after scan_date
    ]
    # scan_date 06-05 → the EXACT 06-05 bar (positional index 2) is the as-of bar.
    assert _as_of_idx(_df(dates), date(2026, 6, 5)) == 2


def test_as_of_idx_none_when_scan_date_bar_missing_no_prior_day_fallback():
    """Fidelity gate (codex P1): a missing scan_date bar must NOT silently fall
    back to the prior trading day's bar. Replaying the prior day would write stale
    prior-day levels into a row meant to be reconstructed AS OF scan_date — so the
    row stays still_null instead. Returns None when the exact bar is absent."""
    dates = [
        pd.Timestamp("2026-06-01", tz="UTC"),
        pd.Timestamp("2026-06-03", tz="UTC"),
        pd.Timestamp("2026-06-08", tz="UTC"),
    ]
    # scan_date 06-05 has NO bar → None (no 06-03 fallback). Was index 1 pre-fix.
    assert _as_of_idx(_df(dates), date(2026, 6, 5)) is None


def test_as_of_idx_none_when_all_bars_after_scan_date():
    dates = [
        pd.Timestamp("2026-06-10", tz="UTC"),
        pd.Timestamp("2026-06-11", tz="UTC"),
    ]
    assert _as_of_idx(_df(dates), date(2026, 6, 5)) is None


def test_as_of_idx_handles_tz_naive_index():
    # A tz-naive index (synthetic data) must still compare cleanly.
    dates = [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-05")]
    assert _as_of_idx(_df(dates), date(2026, 6, 5)) == 1


def test_as_of_idx_includes_midnight_utc_scan_date_bar():
    # Production bars are normalized to midnight UTC; the scan_date bar itself
    # must be included (no off-by-one excluding it).
    bar = pd.Timestamp(date(2026, 6, 5), tz=timezone.utc)
    assert _as_of_idx(_df([bar]), date(2026, 6, 5)) == 0


def test_as_of_idx_matches_scan_date_under_non_utc_session_tz():
    """A midnight-UTC bar read back under a BEHIND-UTC session tz must still match
    its scan_date — the bar's UTC calendar date is canonical, not its local date.

    psycopg returns timestamptz in the Postgres session TimeZone GUC, which is NOT
    pinned to UTC on the local TimescaleDB. The 06-05 00:00 UTC bar presents as
    06-04 17:00-07:00 under America/Los_Angeles; taking the LOCAL date would mis-
    date it to 06-04 and the row would never match scan_date 06-05 — silently
    making the whole repair a 0-recovered no-op. _as_of_idx must convert to UTC
    first. FAILS on a local-tz `.normalize().date` implementation."""
    # The same instant the DB stores (midnight UTC on scan_date), but presented
    # in a behind-UTC zone the way a non-UTC psycopg session would return it.
    bar = pd.Timestamp(date(2026, 6, 5), tz=timezone.utc).tz_convert(
        "America/Los_Angeles"
    )  # -> 2026-06-04 17:00-07:00, LOCAL date 06-04
    assert _as_of_idx(_df([bar]), date(2026, 6, 5)) == 0


# --- input validation ------------------------------------------------------


def test_backfill_rejects_from_after_to():
    with pytest.raises(ValueError, match="from_date.*>.*to_date"):
        backfill_screened_levels(
            from_date=date(2026, 6, 12),
            to_date=date(2026, 6, 3),
        )
