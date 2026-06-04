"""Phase 2 D6 — discover_time_stop (learn the force-exit horizon).

Deterministic: fixture price curves + monkeypatched DB loaders (no network, no
Postgres). Asserts the four design invariants:
  * risk-adjusted (Sharpe-like) objective-max k can DIFFER from mean-return-max k;
  * survivor cohort — a trade SL/TP-exiting before day k is excluded from k;
  * as-of ceiling — reconstruction never uses prices past the analysis date;
  * recommend-only (noop) until k stable across >= 2 consecutive runs, then the
    action flips to set_learned_time_stop_days.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

from rainier.llm_thesis.research import (
    TIME_STOP_STABLE_RUNS,
    _day_k_exit_return,
    _return_by_holding_day,
    _sharpe_like,
    _sl_tp_exit_day,
    discover_time_stop,
)
from tests.test_research_helpers import MemSession  # type: ignore


def _bars(entry: date, closes, *, highs=None, lows=None, opens=None):
    """Build (date, open, high, low, close) tuples one per consecutive day."""
    rows = []
    for i, c in enumerate(closes):
        d = entry + timedelta(days=i)
        rows.append(
            (
                d,
                (opens[i] if opens else c),
                (highs[i] if highs else c),
                (lows[i] if lows else c),
                c,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_return_by_holding_day_as_of_guard():
    entry = date(2026, 1, 1)
    rows = _bars(entry, [100, 102, 104, 110])  # day1..day4
    # As-of caps at day 3 (2026-01-03): the 110 (day 4) must be excluded.
    curve = _return_by_holding_day(
        rows, entry, 100.0, as_of=date(2026, 1, 3)
    )
    assert len(curve) == 3
    assert curve[0] == 0.0
    assert abs(curve[2] - 0.04) < 1e-9
    # No hindsight: the day-4 +10% return never appears.
    assert all(abs(x - 0.10) > 1e-9 for x in curve)


def test_return_by_holding_day_skips_null_close():
    entry = date(2026, 1, 1)
    rows = [
        (entry, 100, 100, 100, 100.0),
        (entry + timedelta(days=1), None, None, None, None),  # no-data day
        (entry + timedelta(days=2), 105, 105, 105, 105.0),
    ]
    curve = _return_by_holding_day(rows, entry, 100.0, as_of=entry + timedelta(days=2))
    # NULL day skipped → only two priced sessions.
    assert len(curve) == 2
    assert abs(curve[1] - 0.05) < 1e-9


def test_sl_tp_exit_day_excludes_pre_k_exit():
    entry = date(2026, 1, 1)
    # Day 2 low gaps to the stop (90) → exits on holding-day 2.
    rows = _bars(
        entry,
        closes=[100, 92, 95],
        highs=[101, 95, 96],
        lows=[99, 90, 94],
    )
    assert _sl_tp_exit_day(rows, entry, stop_loss=90.0, target_price=120.0,
                           as_of=date(2026, 1, 3)) == 2
    # Never hits → None.
    rows2 = _bars(entry, closes=[100, 101, 102])
    assert _sl_tp_exit_day(rows2, entry, 80.0, 130.0, as_of=date(2026, 1, 3)) is None


def test_day_k_exit_return_uses_level_on_day_k_touch():
    # codex iter-4 [P2]: evaluate_exit checks SL/TP BEFORE the time-stop on the
    # same session, so a day-k bar that touches the target exits at the target
    # LEVEL — not the (flat) day-k close. The scorer must match that priority.
    entry = date(2026, 1, 1)
    # Day 3 (k=3): high reaches the 120 target intraday but closes flat at 100.
    rows = _bars(
        entry,
        closes=[100, 100, 100],
        highs=[100, 100, 120],   # day-3 high touches target
        lows=[100, 100, 100],
    )
    r = _day_k_exit_return(
        rows, entry, 100.0, stop_loss=80.0, target_price=120.0, k=3,
        as_of=date(2026, 1, 3),
    )
    # +20% target return, NOT the 0% flat close.
    assert abs(r - 0.20) < 1e-9


def test_day_k_exit_return_same_bar_resolves_to_stop():
    # Same-bar SL+TP on day k resolves conservatively to the stop (F3 downward
    # bias, matching evaluate_exit).
    entry = date(2026, 1, 1)
    rows = _bars(
        entry,
        closes=[100, 100, 100],
        highs=[100, 100, 120],   # touches target
        lows=[100, 100, 80],     # AND touches stop on the same day-3 bar
    )
    r = _day_k_exit_return(
        rows, entry, 100.0, stop_loss=80.0, target_price=120.0, k=3,
        as_of=date(2026, 1, 3),
    )
    assert abs(r - (-0.20)) < 1e-9  # -20% stop, not +20% target


def test_day_k_exit_return_survives_past_k_uses_close():
    # No SL/TP touch through day k → score the day-k close return.
    entry = date(2026, 1, 1)
    rows = _bars(entry, closes=[100, 102, 105])
    r = _day_k_exit_return(
        rows, entry, 100.0, stop_loss=80.0, target_price=200.0, k=3,
        as_of=date(2026, 1, 3),
    )
    assert abs(r - 0.05) < 1e-9


def test_day_k_exit_return_excludes_pre_k_exit():
    # Exits before k → None (caller drops from candidate k's cohort).
    entry = date(2026, 1, 1)
    rows = _bars(
        entry, closes=[100, 92, 95],
        highs=[101, 95, 96], lows=[99, 90, 94],  # day-2 low pierces 90 stop
    )
    r = _day_k_exit_return(
        rows, entry, 100.0, stop_loss=90.0, target_price=120.0, k=3,
        as_of=date(2026, 1, 3),
    )
    assert r is None


def test_sharpe_like_prefers_low_variance():
    # Same mean (0.04) but tighter dispersion scores higher.
    tight = [0.03, 0.04, 0.05]
    wide = [-0.06, 0.04, 0.14]
    assert _sharpe_like(tight) > _sharpe_like(wide)
    assert _sharpe_like([0.05]) is None  # n<2 undefined
    assert _sharpe_like([0.05, 0.05]) is None  # zero dispersion undefined


# ---------------------------------------------------------------------------
# discover_time_stop — risk-adjusted choice differs from mean-return-max
# ---------------------------------------------------------------------------


def _trade(symbol, entry, closes, *, stop=1.0, target=1e9):
    """A trade whose curve is `closes` (entry_price=closes[0])."""
    return {
        "id": hash(symbol) & 0xFFFF,
        "symbol": symbol,
        "entry_date": entry,
        "entry_price": float(closes[0]),
        "stop_loss": float(stop),
        "target_price": float(target),
        "price_rows": _bars(entry, [float(c) for c in closes]),
    }


def test_risk_adjusted_max_differs_from_mean_return_max():
    """k=10 has the highest MEAN return but blows up the variance; k=5 wins on
    the risk-adjusted objective. Proves the objective is NOT mean-return-max."""
    entry = date(2026, 1, 1)
    # Build 6 trades. Each has 10 priced days. At day5 the cohort is tight and
    # mildly positive; at day10 it is higher-mean but wildly dispersed.
    day5 = [0.02, 0.03, 0.02, 0.03, 0.02, 0.03]  # mean .025, tight
    day10 = [0.30, -0.20, 0.35, -0.25, 0.40, 0.20]  # mean ~.133, very wide
    trades = []
    for i in range(6):
        closes = [100.0]
        # days 2..5 ramp linearly to the day5 return
        for d in range(1, 5):
            closes.append(100.0 * (1 + day5[i] * d / 4))
        # days 6..10 ramp to the day10 return
        for d in range(5, 10):
            frac = (d - 4) / 5
            closes.append(100.0 * (1 + day5[i] + (day10[i] - day5[i]) * frac))
        trades.append(_trade(f"S{i}", entry, closes))

    as_of = entry + timedelta(days=9)
    with patch(
        "rainier.llm_thesis.research._load_time_stop_trades", return_value=trades
    ), patch(
        "rainier.llm_thesis.research._prior_time_stop_evidence", return_value=None
    ):
        sess = MemSession()
        out = discover_time_stop(
            eval_date=as_of, candidate_ks=(5, 10), min_survivors=3, db_session=sess
        )
    assert len(out) == 1
    ev = out[0].evidence
    # Mean-return-max would pick k=10; the risk-adjusted objective picks k=5.
    per_k = ev["per_k"]
    assert per_k["10"]["mean_return"] > per_k["5"]["mean_return"]
    assert ev["chosen_k"] == 5


def test_survivor_cohort_excludes_pre_k_exit():
    """A trade that stops out at day 2 is excluded from candidate k=5's cohort
    (survivor cohort), so it can't pad the min-survivors count or skew the
    objective."""
    entry = date(2026, 1, 1)
    as_of = entry + timedelta(days=6)
    # Varied day-5 returns so the cohort has non-zero dispersion (Sharpe-like
    # objective is defined).
    survivor_day5_close = [104, 105, 103, 106]
    survivors = [
        _trade(f"W{i}", entry, [100, 101, 102, 103, c, c], stop=80.0)
        for i, c in enumerate(survivor_day5_close)
    ]
    # One early-stopper: day-2 low pierces 90.
    stopper = {
        "id": 999, "symbol": "EARLY", "entry_date": entry,
        "entry_price": 100.0, "stop_loss": 90.0, "target_price": 1e9,
        "price_rows": _bars(
            entry, closes=[100, 88, 88, 88, 88, 88],
            highs=[101, 95, 95, 95, 95, 95], lows=[99, 85, 85, 85, 85, 85],
        ),
    }
    trades = survivors + [stopper]
    with patch(
        "rainier.llm_thesis.research._load_time_stop_trades", return_value=trades
    ), patch(
        "rainier.llm_thesis.research._prior_time_stop_evidence", return_value=None
    ):
        sess = MemSession()
        out = discover_time_stop(
            eval_date=as_of, candidate_ks=(5,), min_survivors=4, db_session=sess
        )
    assert len(out) == 1
    # Only the 4 survivors counted at k=5 (the stopper excluded).
    assert out[0].evidence["per_k"]["5"]["survivors"] == 4


def test_min_survivors_gate_per_candidate_k():
    """min_survivors is enforced PER candidate k: k=10 has too few matured
    survivors and is dropped even though k=5 qualifies."""
    entry = date(2026, 1, 1)
    # 5 trades priced through day 5 only (none mature to day 10). Varied day-5
    # closes give the cohort non-zero dispersion.
    day5 = [104, 105, 103, 106, 102]
    trades = [
        _trade(f"T{i}", entry, [100, 101, 102, 103, c], stop=80.0)
        for i, c in enumerate(day5)
    ]
    as_of = entry + timedelta(days=4)
    with patch(
        "rainier.llm_thesis.research._load_time_stop_trades", return_value=trades
    ), patch(
        "rainier.llm_thesis.research._prior_time_stop_evidence", return_value=None
    ):
        sess = MemSession()
        out = discover_time_stop(
            eval_date=as_of, candidate_ks=(5, 10), min_survivors=5, db_session=sess
        )
    assert len(out) == 1
    per_k = out[0].evidence["per_k"]
    assert "5" in per_k and "10" not in per_k
    assert out[0].evidence["chosen_k"] == 5


# ---------------------------------------------------------------------------
# recommend-only until stable across >= 2 runs
# ---------------------------------------------------------------------------


def _stable_trades():
    entry = date(2026, 1, 1)
    # Varied day-5 closes → non-zero dispersion so the objective is defined.
    day5 = [104, 105, 103, 106, 102]
    return [
        _trade(f"R{i}", entry, [100, 101, 102, 103, c, c], stop=80.0)
        for i, c in enumerate(day5)
    ], entry + timedelta(days=5)


def test_recommend_only_first_run_is_noop():
    trades, as_of = _stable_trades()
    with patch(
        "rainier.llm_thesis.research._load_time_stop_trades", return_value=trades
    ), patch(
        "rainier.llm_thesis.research._prior_time_stop_evidence", return_value=None
    ):
        sess = MemSession()
        out = discover_time_stop(
            eval_date=as_of, candidate_ks=(5,), min_survivors=5, db_session=sess
        )
    ins = out[0]
    assert ins.evidence["stable_run_count"] == 1
    assert ins.evidence["stable"] is False
    assert ins.action["kind"] == "noop"
    assert ins.severity == "info"


def test_flips_to_executable_when_stable():
    trades, as_of = _stable_trades()
    chosen = 5
    prior = {"chosen_k": chosen, "stable_run_count": TIME_STOP_STABLE_RUNS - 1}
    with patch(
        "rainier.llm_thesis.research._load_time_stop_trades", return_value=trades
    ), patch(
        "rainier.llm_thesis.research._prior_time_stop_evidence", return_value=prior
    ):
        sess = MemSession()
        out = discover_time_stop(
            eval_date=as_of, candidate_ks=(5,), min_survivors=5, db_session=sess
        )
    ins = out[0]
    assert ins.evidence["chosen_k"] == chosen
    assert ins.evidence["stable_run_count"] == TIME_STOP_STABLE_RUNS
    assert ins.evidence["stable"] is True
    assert ins.action == {
        "kind": "set_learned_time_stop_days",
        "target": str(chosen),
        "params": {"k": chosen},
    }


def test_stability_resets_when_chosen_k_changes():
    trades, as_of = _stable_trades()  # this run picks k=5
    prior = {"chosen_k": 8, "stable_run_count": 5}  # prior pick was different
    with patch(
        "rainier.llm_thesis.research._load_time_stop_trades", return_value=trades
    ), patch(
        "rainier.llm_thesis.research._prior_time_stop_evidence", return_value=prior
    ):
        sess = MemSession()
        out = discover_time_stop(
            eval_date=as_of, candidate_ks=(5,), min_survivors=5, db_session=sess
        )
    assert out[0].evidence["stable_run_count"] == 1
    assert out[0].action["kind"] == "noop"


def test_same_eval_date_retry_does_not_advance_stability():
    # codex iter-1 [P2]: a retry/replay of the weekly job on the SAME eval_date
    # must not increment the stability counter — only a genuinely new weekly
    # observation (a later eval_date) does. Otherwise re-running once flips the
    # action to executable without a new week of data.
    trades, as_of = _stable_trades()  # picks k=5
    chosen = 5
    # Prior pending row was already recorded for THIS same eval_date at count 1.
    prior = {
        "chosen_k": chosen,
        "stable_run_count": 1,
        "last_run_date": as_of.isoformat(),
    }
    with patch(
        "rainier.llm_thesis.research._load_time_stop_trades", return_value=trades
    ), patch(
        "rainier.llm_thesis.research._prior_time_stop_evidence", return_value=prior
    ):
        sess = MemSession()
        out = discover_time_stop(
            eval_date=as_of, candidate_ks=(5,), min_survivors=5, db_session=sess
        )
    ins = out[0]
    # Held steady at 1 (not advanced to 2), so it stays recommend-only.
    assert ins.evidence["stable_run_count"] == 1
    assert ins.evidence["stable"] is False
    assert ins.action["kind"] == "noop"


def test_distinct_later_eval_date_advances_stability():
    # The complement: same chosen_k but a LATER eval_date is a real new
    # observation — the counter advances and (at the threshold) flips executable.
    trades, as_of = _stable_trades()  # picks k=5
    chosen = 5
    earlier = (as_of - timedelta(days=7)).isoformat()
    prior = {
        "chosen_k": chosen,
        "stable_run_count": TIME_STOP_STABLE_RUNS - 1,
        "last_run_date": earlier,
    }
    with patch(
        "rainier.llm_thesis.research._load_time_stop_trades", return_value=trades
    ), patch(
        "rainier.llm_thesis.research._prior_time_stop_evidence", return_value=prior
    ):
        sess = MemSession()
        out = discover_time_stop(
            eval_date=as_of, candidate_ks=(5,), min_survivors=5, db_session=sess
        )
    ins = out[0]
    assert ins.evidence["stable_run_count"] == TIME_STOP_STABLE_RUNS
    assert ins.evidence["stable"] is True
    assert ins.action["kind"] == "set_learned_time_stop_days"


def test_older_eval_date_replay_does_not_advance_stability():
    # codex iter-2 [P2]: a manual replay with an OLDER eval_date than the prior
    # recorded run must not be treated as a new forward observation — otherwise
    # an out-of-order backfill could flip noop -> executable without two forward
    # consecutive runs. The counter holds steady and the recorded date is kept.
    trades, as_of = _stable_trades()  # picks k=5
    chosen = 5
    newer = (as_of + timedelta(days=7)).isoformat()  # prior run was LATER
    prior = {
        "chosen_k": chosen,
        "stable_run_count": TIME_STOP_STABLE_RUNS - 1,
        "last_run_date": newer,
    }
    with patch(
        "rainier.llm_thesis.research._load_time_stop_trades", return_value=trades
    ), patch(
        "rainier.llm_thesis.research._prior_time_stop_evidence", return_value=prior
    ):
        sess = MemSession()
        out = discover_time_stop(
            eval_date=as_of, candidate_ks=(5,), min_survivors=5, db_session=sess
        )
    ins = out[0]
    # Held steady — not advanced to the threshold — so stays recommend-only.
    assert ins.evidence["stable_run_count"] == TIME_STOP_STABLE_RUNS - 1
    assert ins.evidence["stable"] is False
    assert ins.action["kind"] == "noop"
    # The later prior date is preserved (not clobbered by the older replay).
    assert ins.evidence["last_run_date"] == newer


def test_no_trades_emits_nothing():
    with patch(
        "rainier.llm_thesis.research._load_time_stop_trades", return_value=[]
    ):
        assert discover_time_stop(eval_date=date(2026, 1, 1)) == []
