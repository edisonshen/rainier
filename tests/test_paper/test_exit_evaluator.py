"""Step 6 — pure evaluate_exit (TEST-SPEC §F). No DB, runs everywhere."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from rainier.paper.exit import PriceBar, evaluate_exit

ENTRY = date(2026, 1, 5)


def _bar(d, o, h, low, c):
    return PriceBar(date=d, open=o, high=h, low=low, close=c)


def _days(n):
    """n consecutive calendar days from ENTRY (the evaluator counts the rows it
    is given as sessions — calendar weekday is irrelevant to the walk)."""
    return [ENTRY + timedelta(days=i) for i in range(n)]


def _eval(rows, *, entry=100.0, stop=90.0, target=120.0, shares=100,
          as_of=None, time_stop_days=None):
    if as_of is None:
        as_of = max(r.date for r in rows)
    return evaluate_exit(
        entry_date=ENTRY, entry_price=entry, stop_loss=stop, target_price=target,
        shares=shares, price_rows=rows, as_of=as_of, time_stop_days=time_stop_days,
    )


def test_f1_target_hit():
    d = _days(2)
    rows = [_bar(d[0], 100, 101, 99, 100), _bar(d[1], 100, 121, 99, 120)]
    r = _eval(rows)
    assert r.exit_reason == "target" and r.exit_price == 120
    assert r.return_pct == pytest.approx(0.20) and r.pnl == pytest.approx(2000)


def test_f2_stop_hit():
    d = _days(2)
    rows = [_bar(d[0], 100, 101, 99, 100), _bar(d[1], 100, 101, 89, 90)]
    r = _eval(rows)
    assert r.exit_reason == "stop_loss" and r.exit_price == 90
    assert r.return_pct == pytest.approx(-0.10) and r.pnl == pytest.approx(-1000)


def test_f3_same_day_sl_tp_to_sl():
    d = _days(2)
    rows = [_bar(d[0], 100, 101, 99, 100), _bar(d[1], 100, 121, 89, 100)]
    r = _eval(rows)
    assert r.exit_reason == "stop_loss" and r.same_bar_ambiguous


def test_f4_gap_through_stop():
    d = _days(2)
    rows = [_bar(d[0], 100, 101, 99, 100), _bar(d[1], 85, 95, 84, 88)]
    r = _eval(rows)
    assert r.exit_reason == "stop_loss" and r.exit_price == 85


def test_f5_gap_through_target():
    d = _days(2)
    rows = [_bar(d[0], 100, 101, 99, 100), _bar(d[1], 125, 130, 124, 128)]
    r = _eval(rows)
    assert r.exit_reason == "target" and r.exit_price == 125


def test_f6_same_day_entry_exit():
    # Entry-day (the first walked bar) low <= stop → exits same day.
    d = _days(1)
    rows = [_bar(d[0], 100, 101, 89, 92)]
    r = _eval(rows)
    assert r.exit_reason == "stop_loss" and r.exit_date == ENTRY


def test_f7_equality_boundaries():
    d = _days(2)
    # low == stop triggers
    assert _eval([_bar(d[0], 100, 101, 99, 100), _bar(d[1], 95, 101, 90, 95)]).exit_reason == "stop_loss"
    # high == target triggers
    assert _eval([_bar(d[0], 100, 101, 99, 100), _bar(d[1], 105, 120, 99, 115)]).exit_reason == "target"
    # open == stop → exit at level (open)
    r = _eval([_bar(d[0], 100, 101, 99, 100), _bar(d[1], 90, 95, 88, 92)])
    assert r.exit_reason == "stop_loss" and r.exit_price == 90


def test_f7b_gap_through_plus_straddle():
    # opens == stop AND high >= target → stop_loss at open (SL-first + gap).
    d = _days(2)
    rows = [_bar(d[0], 100, 101, 99, 100), _bar(d[1], 90, 125, 88, 120)]
    r = _eval(rows)
    assert r.exit_reason == "stop_loss" and r.exit_price == 90


def test_f8_unordered_input():
    d = _days(3)
    ordered = [_bar(d[0], 100, 101, 99, 100), _bar(d[1], 100, 105, 99, 102),
               _bar(d[2], 100, 121, 99, 120)]
    shuffled = [ordered[2], ordered[0], ordered[1]]
    assert _eval(ordered).exit_date == _eval(shuffled).exit_date == d[2]


def test_f9_no_time_exit_when_null():
    rows = [_bar(d, 100, 105, 95, 100) for d in _days(50)]
    assert _eval(rows, time_stop_days=None) is None


def test_f10_honors_time_stop_off_by_one():
    rows = [_bar(d, 100, 105, 95, 100) for d in _days(20)]
    # entry day = session 1; force-close at the 10th session's close.
    r = _eval(rows, time_stop_days=10)
    assert r.exit_reason == "time_stop"
    assert r.exit_date == date(2026, 1, 14)  # 10th business-day index from 1/5
    # Not entry+10, not session 9.


def test_f11_no_lookahead_before_entry():
    pre = _bar(date(2026, 1, 2), 100, 999, 1, 100)  # would trigger if seen
    d = _days(2)
    rows = [pre, _bar(d[0], 100, 101, 99, 100), _bar(d[1], 100, 121, 99, 120)]
    r = _eval(rows, as_of=d[1])
    assert r.exit_reason == "target" and r.exit_date == d[1]


def test_f12_no_lookahead_after_as_of():
    d = _days(6)  # 1/5..1/10
    rows = [_bar(d[0], 100, 101, 99, 100), _bar(d[1], 100, 102, 99, 100),
            _bar(d[2], 100, 103, 99, 100),
            _bar(d[3], 100, 121, 99, 120),   # T+4 = target (1/9 is index 4? compute)
            _bar(d[4], 100, 101, 80, 85)]    # T+5 = stop
    # as_of=T+2 → stays open
    assert _eval(rows, as_of=d[2]) is None
    # as_of=d[3] (the target bar) → closes target at that date, NOT the later stop
    r = _eval(rows, as_of=d[3])
    assert r.exit_reason == "target" and r.exit_date == d[3]


def test_f13_immutable_close_no_double():
    d = _days(3)
    rows = [_bar(d[0], 100, 101, 99, 100), _bar(d[1], 100, 121, 99, 120),
            _bar(d[2], 100, 101, 80, 85)]
    first = _eval(rows, as_of=d[1])
    # A later bar that would produce a different exit must not change the
    # already-determined exit when evaluated up to the original as_of.
    again = _eval(rows, as_of=d[1])
    assert (first.exit_date, first.exit_price, first.exit_reason) == (
        again.exit_date, again.exit_price, again.exit_reason
    )


def test_f14_pnl_precision_fractional_entry():
    d = _days(2)
    rows = [_bar(d[0], 33.0, 34, 32, 33), _bar(d[1], 33, 50, 32, 49)]
    r = _eval(rows, entry=33.0, stop=30.0, target=40.0, shares=303)
    # pnl = shares*(exit-entry) = 303*(40-33) = 2121, NOT return_pct*10000.
    assert r.exit_price == 40 and r.pnl == pytest.approx(303 * (40 - 33))


def test_f15_null_ohlc_skipped():
    d = _days(3)
    rows = [_bar(d[0], 100, 101, 99, 100),
            PriceBar(date=d[1], open=None, high=None, low=None, close=None),
            _bar(d[2], 100, 121, 99, 120)]
    r = _eval(rows)
    assert r.exit_reason == "target" and r.exit_date == d[2]


def test_f15_null_day_does_not_advance_time_stop():
    d = _days(4)
    rows = [_bar(d[0], 100, 105, 95, 100),
            PriceBar(date=d[1], open=None, high=None, low=None, close=None),
            _bar(d[2], 100, 105, 95, 100),
            _bar(d[3], 100, 105, 95, 100)]
    # time_stop_days=3 counts data days only: sessions are d0(1), d2(2), d3(3).
    r = _eval(rows, time_stop_days=3)
    assert r.exit_reason == "time_stop" and r.exit_date == d[3]
