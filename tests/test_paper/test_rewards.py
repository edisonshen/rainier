"""R1 — selection reward ledger: pure reward math + decision scoring (no DB)."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import inspect

from rainier.core.models import SelectionReward
from rainier.paper.calendar import TradingCalendar
from rainier.paper.rewards import (
    REWARD_PNL,
    REWARD_R,
    Decision,
    classify_decision,
    lever_context,
    score_decision,
)
from rainier.research.rewards.selection import (
    REWARD_VERSION,
    SELECTION_REWARDS,
    composite_score,
    expectancy_r,
    max_drawdown_r,
    r_multiple,
    std_r,
    summarize,
    t_stat_r,
    total_r,
    win_rate,
    win_rate_floor_ok,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAL = TradingCalendar()

# ---------------------------------------------------------------------------
# Schema / migration files
# ---------------------------------------------------------------------------


def test_migration_0015_files_exist():
    assert (REPO_ROOT / "migrations" / "0015_selection_reward.sql").exists()
    assert (REPO_ROOT / "migrations" / "0015_selection_reward_downgrade.sql").exists()


def test_selection_reward_orm_shape():
    cols = {c.name for c in inspect(SelectionReward).columns}
    expected = {
        "id", "thesis_id", "symbol", "scan_date", "session_name", "decision",
        "lever_context", "reward_name", "reward_version", "value", "reason",
        "provisional", "counterfactual", "outcome_date", "as_of_date",
        "created_at", "updated_at",
    }
    assert expected <= cols, f"missing: {expected - cols}"
    assert SelectionReward.__table__.c.value.nullable
    uniques = [
        c for c in SelectionReward.__table__.constraints
        if c.__class__.__name__ == "UniqueConstraint"
    ]
    assert any(
        {col.name for col in c.columns} == {"thesis_id", "reward_name"} for c in uniques
    )
    fk = list(SelectionReward.__table__.c.thesis_id.foreign_keys)
    assert fk and fk[0].column.table.name == "analysis_results"


# ---------------------------------------------------------------------------
# R-multiple
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry,stop,exit_px,expected",
    [
        (100.0, 95.0, 110.0, 2.0),  # 2R winner
        (100.0, 95.0, 95.0, -1.0),  # full stop = -1R
        (100.0, 95.0, 100.0, 0.0),  # scratch
        (50.0, 45.0, 47.5, -0.5),  # partial loss
        (100.0, 90.0, 93.0, -0.7),  # gap-through below stop → worse than -1R? no: -0.7
    ],
)
def test_r_multiple_arithmetic(entry, stop, exit_px, expected):
    assert r_multiple(entry, stop, exit_px) == pytest.approx(expected)


def test_r_multiple_gap_through_stop_exceeds_minus_one():
    # exit below the stop (gap-through) is worse than -1R — honest accounting.
    assert r_multiple(100.0, 95.0, 90.0) == pytest.approx(-2.0)


@pytest.mark.parametrize("stop", [100.0, 105.0])
def test_r_multiple_invalid_levels_returns_none(stop):
    assert r_multiple(100.0, stop, 110.0) is None


def test_r_multiple_non_finite_returns_none():
    assert r_multiple(100.0, 95.0, math.nan) is None
    assert r_multiple(math.inf, 95.0, 110.0) is None


# ---------------------------------------------------------------------------
# Cohort axes
# ---------------------------------------------------------------------------


def test_total_and_win_rate():
    rs = [2.0, -1.0, -1.0, 0.5, 0.0]
    assert total_r(rs) == pytest.approx(0.5)
    assert win_rate(rs) == pytest.approx(2 / 5)  # R == 0 is NOT a win
    assert win_rate([]) is None
    assert expectancy_r(rs) == pytest.approx(0.1)
    assert expectancy_r([]) is None


def test_std_and_t_stat():
    rs = [1.0, 3.0]
    assert std_r(rs) == pytest.approx(math.sqrt(2))
    assert t_stat_r(rs) == pytest.approx(2.0 / (math.sqrt(2) / math.sqrt(2)))
    assert std_r([1.0]) is None
    assert t_stat_r([1.0]) is None
    assert t_stat_r([1.0, 1.0, 1.0]) is None  # zero dispersion → undefined


def test_max_drawdown_peak_to_trough():
    # cum: 1, 0, -1, 1  → peak 1 then trough -1 → dd 2
    assert max_drawdown_r([1.0, -1.0, -1.0, 2.0]) == pytest.approx(2.0)
    # losing start counts from the flat 0 start: cum -1, -2, 0 → dd 2
    assert max_drawdown_r([-1.0, -1.0, 2.0]) == pytest.approx(2.0)
    # monotone winners → no drawdown
    assert max_drawdown_r([0.5, 1.0, 2.0]) == 0.0
    assert max_drawdown_r([]) == 0.0


def test_composite_and_win_rate_floor():
    assert composite_score(5.0, 2.0) == pytest.approx(3.0)
    assert composite_score(5.0, 2.0, lam=0.5) == pytest.approx(4.0)
    assert win_rate_floor_ok(0.46, 0.50)  # within 5pp
    assert not win_rate_floor_ok(0.44, 0.50)
    assert win_rate_floor_ok(None, 0.5) and win_rate_floor_ok(0.3, None)


def test_summarize_reports_all_axes_separately():
    s = summarize([2.0, -1.0, 1.0], [400.0, -200.0, 200.0])
    assert s["n"] == 3
    assert s["total_R"] == pytest.approx(2.0)
    assert s["total_pnl_usd"] == pytest.approx(400.0)
    assert s["win_rate"] == pytest.approx(2 / 3)
    assert s["max_drawdown_R"] == pytest.approx(1.0)
    assert s["composite_score"] == pytest.approx(1.0)
    assert s["reward_version"] == REWARD_VERSION
    assert set(SELECTION_REWARDS) == {
        "total_R", "win_rate", "expectancy_R", "std_R", "t_stat_R", "max_drawdown_R",
    }


# ---------------------------------------------------------------------------
# Decision classification + scoring
# ---------------------------------------------------------------------------

SCAN = date(2026, 1, 9)  # Friday → fill Monday 2026-01-12


def _decision(**over) -> Decision:
    base = dict(
        thesis_id=1, symbol="AAA", scan_date=SCAN, session_name="close",
        verdict="watch", llm_confidence=7, prompt_version="v4", model="m",
        signals_used=("b", "a"), pattern_type="w_bottom",
        entry_price=100.0, stop_loss=95.0, target_price=110.0,
    )
    base.update(over)
    return Decision(**base)


def _bar(d, o, h, lo, c):
    return {"date": d, "open": o, "high": h, "low": lo, "close": c}


def _closed_trade(**over):
    t = dict(
        status="closed", entry_date=date(2026, 1, 12), entry_price=100.0,
        stop_loss=95.0, target_price=110.0, shares=100, time_stop_days=None,
        price_basis="adjusted", exit_date=date(2026, 1, 14), exit_price=110.0,
        pnl=1000.0,
    )
    t.update(over)
    return t


@pytest.mark.parametrize(
    "over,expected",
    [
        ({"trade": _closed_trade()}, "setup_long_filled"),
        ({"trade": _closed_trade(status="open")}, "setup_long_filled"),
        ({"trade": _closed_trade(status="pending")}, "setup_long_pending"),
        ({"trade": _closed_trade(status="expired"), "skip_reason": "gap_invalidated"},
         "gap_invalidated"),
        ({"trade": _closed_trade(status="expired")}, "setup_long_expired"),
        ({"verdict": "setup_long", "llm_confidence": 7, "skip_reason": "symbol_already_active"},
         "setup_long_skipped"),
        ({"verdict": "setup_long", "llm_confidence": 5}, "setup_long_gated"),
        ({"verdict": "setup_long", "llm_confidence": 8, "session_name": "morning"},
         "setup_long_gated"),
        ({"verdict": "watch"}, "watch"),
        ({"verdict": "no_setup"}, "no_setup"),
    ],
)
def test_classify_decision(over, expected):
    assert classify_decision(_decision(**over)) == expected


def test_lever_context_snapshot():
    ctx = lever_context(_decision(trade=_closed_trade(time_stop_days=8)))
    assert ctx["prompt_version"] == "v4"
    assert ctx["model"] == "m"
    assert ctx["llm_confidence"] == 7
    assert ctx["confidence_gate"] == 6
    assert ctx["session"] == "close"
    assert ctx["pattern_type"] == "w_bottom"
    assert ctx["time_stop_days"] == 8
    assert ctx["price_basis"] == "adjusted"
    # order-insensitive signal-set hash
    assert ctx["signal_set_hash"] == lever_context(
        _decision(signals_used=("a", "b"))
    )["signal_set_hash"]
    assert lever_context(_decision(signals_used=()))["signal_set_hash"] is None


def test_realized_r_for_closed_trade():
    rows = score_decision(
        _decision(verdict="setup_long", trade=_closed_trade()),
        as_of=date(2026, 1, 20), price_rows=[], calendar=CAL,
    )
    by = {r.reward_name: r for r in rows}
    assert set(by) == {REWARD_R, REWARD_PNL}
    r = by[REWARD_R]
    assert r.decision == "setup_long_filled"
    assert r.value == pytest.approx(2.0)
    assert r.provisional is False and r.counterfactual is False
    assert r.outcome_date == date(2026, 1, 14)
    assert r.reward_version == REWARD_VERSION
    assert by[REWARD_PNL].value == pytest.approx(1000.0)


def test_mtm_r_for_open_trade_respects_as_of():
    bars = [
        _bar(date(2026, 1, 12), 100, 103, 99, 102),
        _bar(date(2026, 1, 13), 102, 104, 101, 104),
        _bar(date(2026, 1, 14), 104, 120, 103, 118),  # after as_of → must be ignored
    ]
    rows = score_decision(
        _decision(verdict="setup_long", trade=_closed_trade(status="open")),
        as_of=date(2026, 1, 13), price_rows=bars, calendar=CAL,
    )
    by = {r.reward_name: r for r in rows}
    assert by[REWARD_R].provisional is True
    assert by[REWARD_R].value == pytest.approx((104 - 100) / 5)
    assert by[REWARD_R].outcome_date is None
    assert by[REWARD_PNL].value == pytest.approx(100 * 4.0)


def test_open_trade_basis_mismatch_unscored():
    rows = score_decision(
        _decision(verdict="setup_long", trade=_closed_trade(status="open", price_basis="raw")),
        as_of=date(2026, 1, 13), price_rows=[_bar(date(2026, 1, 12), 100, 103, 99, 102)],
    )
    assert len(rows) == 1
    assert rows[0].value is None and rows[0].reason == "basis_mismatch"


def test_open_trade_missing_prices_unscored():
    rows = score_decision(
        _decision(verdict="setup_long", trade=_closed_trade(status="open")),
        as_of=date(2026, 1, 13), price_rows=[],
    )
    assert rows[0].value is None and rows[0].reason == "missing_prices"
    assert rows[0].provisional is True


def test_counterfactual_watch_hits_target():
    bars = [
        _bar(date(2026, 1, 12), 100.0, 104, 98, 103),
        _bar(date(2026, 1, 13), 103.0, 111, 102, 110),  # high >= target → exit 110
    ]
    rows = score_decision(_decision(verdict="watch"), as_of=date(2026, 1, 20),
                          price_rows=bars, calendar=CAL)
    by = {r.reward_name: r for r in rows}
    r = by[REWARD_R]
    assert r.decision == "watch"
    assert r.counterfactual is True and r.provisional is False
    assert r.value == pytest.approx(2.0)
    assert r.outcome_date == date(2026, 1, 13)
    assert r.lever_context["cf_entry_date"] == "2026-01-12"
    assert r.lever_context["cf_exit_reason"] == "target"
    assert by[REWARD_PNL].value == pytest.approx(100 * 10.0)  # floor(10000/100) shares


def test_counterfactual_entry_is_next_session_open_not_planned_entry():
    bars = [
        _bar(date(2026, 1, 12), 101.0, 104, 99, 103),  # opens at 101, not the 100 plan
        _bar(date(2026, 1, 13), 103.0, 111, 102, 110),
    ]
    rows = score_decision(_decision(verdict="watch"), as_of=date(2026, 1, 20),
                          price_rows=bars, calendar=CAL)
    r = next(x for x in rows if x.reward_name == REWARD_R)
    assert r.value == pytest.approx((110 - 101) / (101 - 95))


def test_counterfactual_not_enterable_before_fill_session():
    # as_of == scan_date: the next session's open does not exist yet → no rows.
    assert score_decision(_decision(), as_of=SCAN, price_rows=[], calendar=CAL) == []


def test_counterfactual_unresolved_is_provisional_mtm():
    bars = [
        _bar(date(2026, 1, 12), 100.0, 103, 98, 102),
        _bar(date(2026, 1, 13), 102.0, 104, 101, 101),
    ]
    rows = score_decision(_decision(), as_of=date(2026, 1, 13), price_rows=bars, calendar=CAL)
    r = next(x for x in rows if x.reward_name == REWARD_R)
    assert r.provisional is True and r.counterfactual is True
    assert r.value == pytest.approx((101 - 100) / 5)


def test_counterfactual_horizon_cap_time_stops():
    bars = [
        _bar(date(2026, 1, 12), 100.0, 103, 98, 102),
        _bar(date(2026, 1, 13), 102.0, 104, 101, 103),
        _bar(date(2026, 1, 14), 103.0, 105, 101, 104),  # session 3 → time stop at close
        _bar(date(2026, 1, 15), 104.0, 112, 103, 111),  # would have hit target later
    ]
    rows = score_decision(_decision(), as_of=date(2026, 1, 20), price_rows=bars,
                          calendar=CAL, counterfactual_horizon_days=3)
    r = next(x for x in rows if x.reward_name == REWARD_R)
    assert r.provisional is False
    assert r.outcome_date == date(2026, 1, 14)
    assert r.value == pytest.approx((104 - 100) / 5)
    assert r.lever_context["cf_exit_reason"] == "time_stop"


def test_counterfactual_gap_invalidated():
    bars = [_bar(date(2026, 1, 12), 112.0, 115, 110, 114)]  # opens through target
    rows = score_decision(_decision(), as_of=date(2026, 1, 20), price_rows=bars, calendar=CAL)
    assert len(rows) == 1
    assert rows[0].value is None and rows[0].reason == "gap_invalidated"
    assert rows[0].counterfactual is True and rows[0].provisional is False


def test_counterfactual_missing_entry_bar_is_provisional():
    rows = score_decision(_decision(), as_of=date(2026, 1, 20), price_rows=[], calendar=CAL)
    assert rows[0].reason == "missing_prices" and rows[0].provisional is True


def test_no_plan_and_invalid_levels():
    rows = score_decision(
        _decision(verdict="no_setup", entry_price=None, stop_loss=None, target_price=None),
        as_of=date(2026, 1, 20), price_rows=[], calendar=CAL,
    )
    assert rows[0].decision == "no_setup" and rows[0].reason == "no_plan"
    # stop above entry → no positive risk
    rows = score_decision(_decision(stop_loss=101.0), as_of=date(2026, 1, 20),
                          price_rows=[], calendar=CAL)
    assert rows[0].reason == "invalid_levels"
    # bearish pattern is not a long plan
    rows = score_decision(_decision(pattern_type="false_breakout"), as_of=date(2026, 1, 20),
                          price_rows=[], calendar=CAL)
    assert rows[0].reason == "invalid_levels"


def test_pending_and_expired_decisions():
    assert score_decision(
        _decision(verdict="setup_long", trade=_closed_trade(status="pending")),
        as_of=date(2026, 1, 20), price_rows=[],
    ) == []
    rows = score_decision(
        _decision(verdict="setup_long", trade=_closed_trade(status="expired"),
                  skip_reason="gap_invalidated"),
        as_of=date(2026, 1, 20), price_rows=[],
    )
    assert rows[0].decision == "gap_invalidated" and rows[0].reason == "gap_invalidated"
    rows = score_decision(
        _decision(verdict="setup_long", trade=_closed_trade(status="expired")),
        as_of=date(2026, 1, 20), price_rows=[],
    )
    assert rows[0].decision == "setup_long_expired" and rows[0].reason == "missing_prices"


def test_gated_setup_long_is_scored_counterfactually():
    bars = [
        _bar(date(2026, 1, 12), 100.0, 104, 98, 103),
        _bar(date(2026, 1, 13), 103.0, 104, 94, 95),  # low <= stop → -1R
    ]
    rows = score_decision(_decision(verdict="setup_long", llm_confidence=5),
                          as_of=date(2026, 1, 20), price_rows=bars, calendar=CAL)
    r = next(x for x in rows if x.reward_name == REWARD_R)
    assert r.decision == "setup_long_gated" and r.counterfactual is True
    assert r.value == pytest.approx(-1.0)
