"""R1 — selection reward ledger against a real DB (Postgres).

Postgres-only: selection_reward uses JSONB + ON CONFLICT ON CONSTRAINT with a
conditional WHERE. Skips cleanly when no Postgres is reachable.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import select, text

from rainier.core.models import PaperTrade, SelectionReward
from rainier.paper.rewards import (
    REWARD_PNL,
    REWARD_R,
    RewardRow,
    compute_rewards,
    load_decisions,
    summarize_rewards,
    upsert_rewards,
)
from rainier.research.rewards.selection import REWARD_VERSION

pytestmark = pytest.mark.requires_postgres

SCAN = date(2026, 1, 9)  # Friday → fill Monday 2026-01-12


def _seed_thesis(session, tid, symbol, *, verdict, conf, levels=(100.0, 95.0, 110.0),
                 pattern="w_bottom", session_name="close"):
    session.execute(
        text(
            "INSERT INTO analysis_results (id, llm_model, prompt_template, recommendation, "
            "confidence, session_name) VALUES (:i, 'm', 'v4', :v, :c, :s) "
            "ON CONFLICT DO NOTHING"
        ),
        {"i": tid, "v": verdict, "c": conf, "s": session_name},
    )
    e, s, t = levels if levels else (None, None, None)
    session.execute(
        text(
            "INSERT INTO screened_stocks (scan_date, session_name, symbol, rule_rank, "
            "composite_score, pattern_type, entry_price, stop_loss, target_price, thesis_id) "
            "VALUES (:d, :s, :sym, 1, 1.0, :p, :e, :st, :t, :tid)"
        ),
        {"d": SCAN, "s": session_name, "sym": symbol, "p": pattern, "e": e, "st": s,
         "t": t, "tid": tid},
    )
    session.commit()


def _seed_bars(session, symbol, bars):
    for d, o, h, lo, c in bars:
        session.execute(
            text(
                "INSERT INTO stock_prices (symbol, date, open, high, low, close) "
                "VALUES (:s, :d, :o, :h, :l, :c) ON CONFLICT DO NOTHING"
            ),
            {"s": symbol, "d": datetime(d.year, d.month, d.day, tzinfo=timezone.utc),
             "o": o, "h": h, "l": lo, "c": c},
        )
    session.commit()


def _row(tid, name, value, *, provisional=False, version=REWARD_VERSION, as_of=SCAN):
    return RewardRow(
        thesis_id=tid, symbol="AAA", scan_date=SCAN, session_name="close",
        decision="watch", lever_context={"prompt_version": "v4"}, reward_name=name,
        value=value, reason=None, provisional=provisional, counterfactual=True,
        outcome_date=None if provisional else as_of, as_of_date=as_of,
        reward_version=version,
    )


def _ledger(session):
    rows = session.execute(
        select(SelectionReward).order_by(SelectionReward.thesis_id, SelectionReward.reward_name)
    ).scalars().all()
    return {(r.thesis_id, r.reward_name): r for r in rows}


def test_upsert_is_idempotent_and_matured_rows_are_immutable(pg_legacy_session):
    s = pg_legacy_session
    _seed_thesis(s, 1, "AAA", verdict="watch", conf=7)

    assert upsert_rewards([_row(1, REWARD_R, 0.4, provisional=True)]) == 1
    # Provisional → refreshed in place (one row, new value).
    assert upsert_rewards([_row(1, REWARD_R, 0.8, provisional=True)]) == 1
    s.expire_all()
    led = _ledger(s)
    assert len(led) == 1 and led[(1, REWARD_R)].value == pytest.approx(0.8)

    # Provisional → matured.
    assert upsert_rewards([_row(1, REWARD_R, 2.0)]) == 1
    s.expire_all()
    assert _ledger(s)[(1, REWARD_R)].provisional is False

    # Matured is immutable: a later provisional write and a same-version
    # re-write are both no-ops.
    assert upsert_rewards([_row(1, REWARD_R, 0.1, provisional=True)]) == 0
    assert upsert_rewards([_row(1, REWARD_R, 2.0)]) == 0
    s.expire_all()
    led = _ledger(s)
    assert len(led) == 1
    assert led[(1, REWARD_R)].value == pytest.approx(2.0)
    assert led[(1, REWARD_R)].provisional is False

    # A reward_version bump DOES rewrite a matured row.
    assert upsert_rewards([_row(1, REWARD_R, 2.5, version=REWARD_VERSION + 1)]) == 1
    s.expire_all()
    assert _ledger(s)[(1, REWARD_R)].reward_version == REWARD_VERSION + 1


def test_check_constraint_requires_value_or_reason(pg_legacy_session):
    from sqlalchemy.exc import IntegrityError

    s = pg_legacy_session
    _seed_thesis(s, 1, "AAA", verdict="watch", conf=7)
    s.add(SelectionReward(
        thesis_id=1, symbol="AAA", scan_date=SCAN, session_name="close", decision="watch",
        lever_context={}, reward_name=REWARD_R, reward_version=1, value=None, reason=None,
        as_of_date=SCAN,
    ))
    with pytest.raises(IntegrityError):
        s.commit()
    s.rollback()


def test_compute_rewards_end_to_end(pg_legacy_session):
    s = pg_legacy_session
    # 1: setup_long that the book took and closed at target (realized 2R).
    _seed_thesis(s, 1, "AAA", verdict="setup_long", conf=8)
    s.add(PaperTrade(
        thesis_id=1, symbol="AAA", scan_date=SCAN, session_name="close", status="closed",
        planned_entry_price=100.0, stop_loss=95.0, target_price=110.0, verdict="setup_long",
        llm_confidence=8, entry_date=date(2026, 1, 12), entry_price=100.0, shares=100,
        allocated_amount=10000.0, residual_cash=0.0, price_basis="adjusted",
        exit_date=date(2026, 1, 13), exit_price=110.0, exit_reason="target",
        return_pct=0.10, pnl=1000.0,
    ))
    s.commit()
    # 2: WATCH declined — counterfactual walks to the stop (-1R).
    _seed_thesis(s, 2, "BBB", verdict="watch", conf=7)
    _seed_bars(s, "BBB", [
        (date(2026, 1, 12), 100.0, 103, 98, 102),
        (date(2026, 1, 13), 102.0, 104, 94, 95),
        (date(2026, 1, 14), 95.0, 97, 93, 96),
    ])
    # 3: no_setup without a plan — counted, not scored.
    _seed_thesis(s, 3, "CCC", verdict="no_setup", conf=3, levels=None, pattern=None)
    # 4: WATCH still running at as_of — provisional MTM.
    _seed_thesis(s, 4, "DDD", verdict="watch", conf=6)
    _seed_bars(s, "DDD", [
        (date(2026, 1, 12), 100.0, 103, 98, 102),
        (date(2026, 1, 13), 102.0, 104, 101, 103),
    ])
    # 5: scanned ON as_of — not a decision with an outcome yet.
    s.execute(text(
        "INSERT INTO analysis_results (id, llm_model, prompt_template, recommendation, "
        "confidence) VALUES (5, 'm', 'v4', 'watch', 7)"
    ))
    s.execute(text(
        "INSERT INTO screened_stocks (scan_date, session_name, symbol, rule_rank, "
        "composite_score, thesis_id) VALUES (:d, 'close', 'EEE', 1, 1.0, 5)"
    ), {"d": date(2026, 1, 13)})
    s.commit()

    as_of = date(2026, 1, 13)
    decisions = load_decisions(as_of=as_of)
    assert sorted(d.thesis_id for d in decisions) == [1, 2, 3, 4]
    assert decisions[0].trade is not None and decisions[0].trade["status"] == "closed"

    res = compute_rewards(as_of=as_of)
    assert res["decisions"] == 4
    assert res["scored"] == 3 and res["unscored"] == 1
    assert res["provisional"] == 1 and res["counterfactual"] == 3
    assert res["decision_setup_long_filled"] == 1
    assert res["decision_watch"] == 2
    assert res["decision_no_setup"] == 1

    s.expire_all()
    led = _ledger(s)
    assert led[(1, REWARD_R)].value == pytest.approx(2.0)
    assert led[(1, REWARD_R)].provisional is False and led[(1, REWARD_R)].counterfactual is False
    assert led[(1, REWARD_PNL)].value == pytest.approx(1000.0)
    assert led[(1, REWARD_R)].lever_context["prompt_version"] == "v4"
    assert led[(1, REWARD_R)].lever_context["llm_confidence"] == 8
    assert led[(2, REWARD_R)].value == pytest.approx(-1.0)
    assert led[(2, REWARD_R)].counterfactual is True
    assert led[(2, REWARD_R)].outcome_date == date(2026, 1, 13)
    assert led[(3, REWARD_R)].value is None and led[(3, REWARD_R)].reason == "no_plan"
    assert (3, REWARD_PNL) not in led
    assert led[(4, REWARD_R)].provisional is True
    assert led[(4, REWARD_R)].value == pytest.approx((103 - 100) / 5)

    # Idempotent re-run: matured rows untouched, provisional refreshed.
    res2 = compute_rewards(as_of=as_of)
    s.expire_all()
    assert len(_ledger(s)) == len(led)
    assert res2["rows_written"] == 2  # thesis 4's two provisional rows only

    # Summary: matured only (1 live + 1 counterfactual), provisional excluded.
    summ = summarize_rewards(as_of=as_of)
    assert summ["all"]["n"] == 2
    assert summ["all"]["total_R"] == pytest.approx(1.0)
    assert summ["all"]["win_rate"] == pytest.approx(0.5)
    assert summ["all"]["max_drawdown_R"] == pytest.approx(1.0)
    assert summ["all"]["composite_score"] == pytest.approx(0.0)
    assert summ["live"]["n"] == 1 and summ["live"]["total_R"] == pytest.approx(2.0)
    assert summ["counterfactual"]["n"] == 1


def test_provisional_to_realized_transition_and_as_of_discipline(pg_legacy_session):
    s = pg_legacy_session
    _seed_thesis(s, 1, "AAA", verdict="setup_long", conf=8)
    s.add(PaperTrade(
        thesis_id=1, symbol="AAA", scan_date=SCAN, session_name="close", status="open",
        planned_entry_price=100.0, stop_loss=95.0, target_price=110.0, verdict="setup_long",
        llm_confidence=8, entry_date=date(2026, 1, 12), entry_price=100.0, shares=100,
        allocated_amount=10000.0, residual_cash=0.0, price_basis="adjusted",
    ))
    s.commit()
    _seed_bars(s, "AAA", [
        (date(2026, 1, 12), 100.0, 103, 98, 102),
        (date(2026, 1, 13), 102.0, 111, 101, 110),
    ])

    compute_rewards(as_of=date(2026, 1, 12))
    s.expire_all()
    r = _ledger(s)[(1, REWARD_R)]
    assert r.provisional is True and r.value == pytest.approx(0.4)

    # The book closes the trade at target; the ledger matures in place.
    s.execute(text(
        "UPDATE paper_trade SET status='closed', exit_date=:d, exit_price=110.0, "
        "exit_reason='target', return_pct=0.1, pnl=1000.0 WHERE thesis_id=1"
    ), {"d": date(2026, 1, 13)})
    s.commit()
    compute_rewards(as_of=date(2026, 1, 13))
    s.expire_all()
    led = _ledger(s)
    assert len(led) == 2
    assert led[(1, REWARD_R)].provisional is False
    assert led[(1, REWARD_R)].value == pytest.approx(2.0)
    assert led[(1, REWARD_R)].outcome_date == date(2026, 1, 13)

    # A historical re-run with an EARLIER as_of must not regress the matured
    # row — the closed trade stays realized (immutability) and the summary as
    # of the earlier date excludes an outcome that had not happened yet.
    compute_rewards(as_of=date(2026, 1, 12))
    s.expire_all()
    assert _ledger(s)[(1, REWARD_R)].value == pytest.approx(2.0)
    assert summarize_rewards(as_of=date(2026, 1, 12))["all"]["n"] == 0
    assert summarize_rewards(as_of=date(2026, 1, 13))["all"]["n"] == 1
