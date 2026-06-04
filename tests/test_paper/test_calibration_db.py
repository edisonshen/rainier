"""Phase 2 D7a — calibration compute/persist against a real DB (Postgres).

Postgres-only: paper_calibration uses JSONB + ON CONFLICT ON CONSTRAINT, and
the headline reads the thesis_evaluations ARRAY-bearing table. Skips cleanly
when no Postgres is reachable (CI provides one).
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select, text

from rainier.core.models import PaperCalibration, PaperTrade
from rainier.paper.calibration import (
    compute_calibration_payload,
    load_latest_calibration,
    persist_calibration,
)

pytestmark = pytest.mark.requires_postgres

AS_OF = date(2026, 2, 10)


def _seed_thesis(session, tid):
    session.execute(
        text("INSERT INTO analysis_results (id, llm_model, prompt_template) "
             "VALUES (:i,'m','t') ON CONFLICT DO NOTHING"),
        {"i": tid},
    )


def _add_eval(session, tid, *, horizon, verdict, conf, ret, scan_date):
    _seed_thesis(session, tid)
    session.execute(
        text(
            "INSERT INTO thesis_evaluations "
            "(thesis_id, horizon, scan_date, symbol, verdict, llm_confidence, "
            " entry_price, exit_price, return_pct, hit) "
            "VALUES (:t,:h,:d,'SYM',:v,:c,100,100,:r,:hit)"
        ),
        {"t": tid, "h": horizon, "d": scan_date, "v": verdict, "c": conf,
         "r": ret, "hit": ret > 0},
    )
    session.commit()


def _add_closed_trade(session, tid, symbol, *, exit_date, return_pct, pnl,
                      exit_reason="target"):
    _seed_thesis(session, tid)
    session.add(PaperTrade(
        thesis_id=tid, symbol=symbol, scan_date=date(2026, 2, 1),
        session_name="close", status="closed", planned_entry_price=100.0,
        stop_loss=90.0, target_price=120.0, entry_date=date(2026, 2, 2),
        entry_price=100.0, shares=100, residual_cash=0.0,
        exit_date=exit_date, exit_price=100 * (1 + return_pct),
        exit_reason=exit_reason, return_pct=return_pct, pnl=pnl,
    ))
    session.commit()


def test_headline_is_unbiased_fixed_horizon(pg_legacy_session):
    s = pg_legacy_session
    # Three 5d theses: 2 setup_long wins + 1 loss.
    _add_eval(s, 1, horizon="5d", verdict="setup_long", conf=9, ret=0.05,
              scan_date=date(2026, 2, 5))
    _add_eval(s, 2, horizon="5d", verdict="setup_long", conf=8, ret=0.03,
              scan_date=date(2026, 2, 5))
    _add_eval(s, 3, horizon="5d", verdict="setup_long", conf=4, ret=-0.02,
              scan_date=date(2026, 2, 5))

    payload = compute_calibration_payload(AS_OF)
    h5 = payload["headline_fixed_horizon"]["5d"]
    assert h5["overall"]["n"] == 3
    # 2 of 3 gains.
    assert abs(h5["overall"]["win_rate"] - 2 / 3) < 1e-9
    # The supplementary realized record exists and is labeled survivorship-prone.
    assert "survivorship-prone" in payload["supplementary_realized"]["label"]


def test_supplementary_last_k_closed_and_mtm(pg_legacy_session):
    s = pg_legacy_session
    # 12 closed trades — only the last K=10 by exit_date appear in the sample.
    for i in range(12):
        _add_closed_trade(
            s, 100 + i, f"C{i:02d}",
            exit_date=date(2026, 2, 2 + (i % 7)),
            return_pct=0.01 * (i + 1), pnl=10.0 * (i + 1),
        )
    payload = compute_calibration_payload(AS_OF, k=10)
    supp = payload["supplementary_realized"]
    assert supp["n_closed_sample"] == 10
    # MTM-including-open figure is carried (no open positions → equals realized).
    assert "mtm_including_open_pnl_usd" in supp
    # Headline empty (no thesis_evaluations) but supplementary still computed.
    assert payload["headline_fixed_horizon"] == {}


def test_persist_and_load_round_trip(pg_legacy_session):
    s = pg_legacy_session
    _add_eval(s, 1, horizon="1d", verdict="setup_long", conf=7, ret=0.02,
              scan_date=date(2026, 2, 9))
    payload = compute_calibration_payload(AS_OF)
    persist_calibration(AS_OF, payload)

    loaded = load_latest_calibration(AS_OF)
    assert loaded is not None
    assert loaded["as_of_date"] == AS_OF.isoformat()
    # Idempotent upsert: re-persist same as_of → still one row.
    persist_calibration(AS_OF, payload)
    n = s.execute(select(PaperCalibration)).scalars().all()
    assert len(n) == 1


def test_load_latest_picks_most_recent_on_or_before(pg_legacy_session):
    s = pg_legacy_session  # noqa: F841
    persist_calibration(date(2026, 2, 5), {"as_of_date": "2026-02-05", "tag": "old"})
    persist_calibration(date(2026, 2, 8), {"as_of_date": "2026-02-08", "tag": "new"})
    # Future row must not be picked when asking as-of 2026-02-06.
    persist_calibration(date(2026, 2, 12), {"as_of_date": "2026-02-12", "tag": "fut"})
    got = load_latest_calibration(date(2026, 2, 6))
    assert got["tag"] == "old"
    got2 = load_latest_calibration(date(2026, 2, 9))
    assert got2["tag"] == "new"
