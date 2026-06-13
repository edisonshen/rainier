"""Step 7 — daily wiring order (TEST-SPEC §G) + R-E feature-step isolation.

Patches every pipeline step to assert call order and failure isolation.
"""

from __future__ import annotations

import asyncio
from datetime import date


class _FakeDiscord:
    enabled = False
    webhook_url = ""


class _FakeAlerts:
    discord = _FakeDiscord()


class _FakeLLMThesis:
    learned_time_stop_days = None
    enabled = False  # R-A reflections gated off so no LLM call is attempted
    model = "test-model"
    chart_lookback_days = 120  # R-D close-chart window


class _FakeSettings:
    alerts = _FakeAlerts()
    llm_thesis = _FakeLLMThesis()


def _patch_pipeline(
    monkeypatch,
    calls,
    captured,
    *,
    cohort=None,
    cohort_exc=None,
    ingest_result=None,
    cohort_ingest_result=None,
    cohort_ingest_exc=None,
    feature_exc=None,
):
    """Install fakes for every daily-eval step, recording into `calls` /
    `captured`. Keyword knobs inject failures or canned returns.

    Ingest runs as TWO calls (trading-critical active ∪ screened, then the
    failure-isolated cohort-only extras); the fake routes on the symbol set:
    a call containing AAA/BBB is the trading ingest, anything else is the
    extras ingest. ``captured["ingest_symbols"]`` accumulates the union.
    """
    from rainier.scheduler import service

    cohort = cohort if cohort is not None else [
        {"symbol": "CCC", "rank": 1, "data_date": date(2026, 1, 9),
         "captured_at": None},
    ]
    ingest_result = ingest_result or {"upserted": 0, "changed": []}
    cohort_ingest_result = cohort_ingest_result or {"upserted": 0, "changed": []}

    def fake_cohort(as_of):
        calls.append("cohort")
        if cohort_exc:
            raise cohort_exc
        return cohort

    def fake_ingest(symbols, **kw):
        symbols = list(symbols)
        captured.setdefault("ingest_symbols", []).extend(symbols)
        if {"AAA", "BBB"} & set(symbols):
            calls.append("ingest")
            return ingest_result
        calls.append("ingest_extra")
        if cohort_ingest_exc:
            raise cohort_ingest_exc
        return cohort_ingest_result

    def fake_feature_step(as_of, changed=(), cohort=None):
        calls.append("features")
        captured["feature_changed"] = list(changed)
        captured["feature_cohort"] = cohort
        if feature_exc:
            raise feature_exc
        return {"computed": 0, "recomputed": 0, "failed": 0}

    monkeypatch.setattr(
        "rainier.paper.ingest.get_current_qu100_cohort", fake_cohort
    )
    monkeypatch.setattr("rainier.paper.ingest.ingest_prices", fake_ingest)
    monkeypatch.setattr("rainier.paper.ingest.active_symbols", lambda: ["AAA"])
    monkeypatch.setattr(
        "rainier.paper.ingest.screened_symbols", lambda as_of: ["BBB"]
    )
    monkeypatch.setattr(
        "rainier.paper.features.run_daily_feature_step", fake_feature_step
    )

    def fake_fill(**kw):
        calls.append("fill")
        return {"filled": 0, "expired": 0}

    def fake_update(**kw):
        calls.append("update")
        return {"closed": 0, "same_bar_ambiguous_exits": 0}

    monkeypatch.setattr(
        "rainier.paper.positions.fill_pending_positions", fake_fill
    )
    monkeypatch.setattr(
        "rainier.paper.positions.update_open_positions", fake_update
    )

    # Existing horizon eval (step iv).
    def fake_evaluate_horizon(eval_date, horizon):
        calls.append("horizon")
        return 0

    monkeypatch.setattr(
        "rainier.llm_thesis.eval.evaluate_horizon", fake_evaluate_horizon
    )
    monkeypatch.setattr(
        "rainier.llm_thesis.eval.compute_verdict_hit_rate", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        "rainier.llm_thesis.eval.compute_signal_contribution",
        lambda *a, **k: [],
    )
    monkeypatch.setattr("rainier.alerts.discord.send_eval_report", lambda **k: None)

    # R-D close-side chart capture (step v addendum) — stub so it never hits
    # the chart DB; not part of the asserted order, just kept non-fatal.
    monkeypatch.setattr(
        "rainier.paper.chart_archive.capture_trade_close_charts",
        lambda **k: 0,
    )

    # Paper report (step v).
    def fake_compute(as_of):
        calls.append("report")
        return {}

    monkeypatch.setattr("rainier.paper.report.compute_daily_payload", fake_compute)
    monkeypatch.setattr(
        "rainier.paper.report.persist_daily_snapshot", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "rainier.paper.report.send_daily_paper_report", lambda *a, **k: True
    )

    # D7a calibration (step vi) — runs AFTER the report.
    def fake_calib_compute(as_of):
        calls.append("calibration")
        return {}

    monkeypatch.setattr(
        "rainier.paper.calibration.compute_calibration_payload",
        fake_calib_compute,
    )
    monkeypatch.setattr(
        "rainier.paper.calibration.persist_calibration", lambda *a, **k: None
    )

    # Avoid the yesterday-rows DB fetch.
    monkeypatch.setattr(service, "load_settings_fresh", lambda: _FakeSettings())


def _run(monkeypatch, **knobs):
    from rainier.scheduler import service

    calls: list[str] = []
    captured: dict = {}
    _patch_pipeline(monkeypatch, calls, captured, **knobs)
    asyncio.run(service.run_daily_eval("2026-01-09"))
    return calls, captured


def test_g1_daily_eval_runs_steps_in_order(monkeypatch):
    calls, _ = _run(monkeypatch)
    # ingest precedes the R-E feature step (which recomputes off the ingest's
    # changed set); features precede fill (G2 extension); fill precedes update;
    # update precedes horizon eval; horizon precedes the paper report; report
    # precedes calibration (G1 authoritative order).
    assert calls.index("ingest") < calls.index("features")
    assert calls.index("features") < calls.index("fill")
    assert calls.index("fill") < calls.index("update")
    assert calls.index("update") < calls.index("horizon")
    assert calls.index("horizon") < calls.index("report")
    assert calls.index("report") < calls.index("calibration")


def test_g3_horizon_eval_still_runs(monkeypatch):
    calls, _ = _run(monkeypatch)
    assert "horizon" in calls


def test_g4_feature_step_failure_never_blocks_trading_steps(monkeypatch):
    """R-E isolation contract — a feature-step exception can never block
    ingest/fill/exit/eval/report (design §8 success metric)."""
    calls, _ = _run(monkeypatch, feature_exc=RuntimeError("boom"))
    for step in ("ingest", "fill", "update", "horizon", "report", "calibration"):
        assert step in calls, f"{step} was blocked by the feature step"


def test_g5_feature_step_receives_changed_set_and_cohort(monkeypatch):
    """Same-run recompute contract — the changed (symbol, date) sets returned
    by BOTH ingest calls (trading + cohort extras) flow into the feature step
    of the SAME run; the cohort is fetched once and shared."""
    changed = [("AAA", date(2026, 1, 8)), ("AAA", date(2026, 1, 9))]
    extra_changed = [("CCC", date(2026, 1, 9))]
    cohort = [
        {"symbol": "CCC", "rank": 3, "data_date": date(2026, 1, 9),
         "captured_at": None},
    ]
    _, captured = _run(
        monkeypatch,
        cohort=cohort,
        ingest_result={"upserted": 2, "changed": changed},
        cohort_ingest_result={"upserted": 1, "changed": extra_changed},
    )
    assert captured["feature_changed"] == changed + extra_changed
    assert captured["feature_cohort"] == cohort


def test_g6_ingest_covers_full_cohort(monkeypatch):
    """D9 scope change — daily ingest = SPY ∪ active ∪ screened ∪ FULL current
    cohort (~100 symbols), not just active ∪ top-50. SPY is always in the
    trading window (R-C regime tag)."""
    cohort = [
        {"symbol": "CCC", "rank": 1, "data_date": date(2026, 1, 9),
         "captured_at": None},
        {"symbol": "DDD", "rank": 2, "data_date": date(2026, 1, 9),
         "captured_at": None},
    ]
    _, captured = _run(monkeypatch, cohort=cohort)
    assert set(captured["ingest_symbols"]) == {"SPY", "AAA", "BBB", "CCC", "DDD"}


def test_g7_cohort_selector_failure_never_blocks_ingest(monkeypatch):
    """A cohort-selector exception must not block ingest of SPY ∪ active ∪
    screened (nor any later step); the feature step still runs recompute-only."""
    calls, captured = _run(monkeypatch, cohort_exc=RuntimeError("db down"))
    assert "ingest" in calls
    assert set(captured["ingest_symbols"]) == {"SPY", "AAA", "BBB"}
    assert "ingest_extra" not in calls  # empty cohort → no extras call
    assert "features" in calls
    assert captured["feature_cohort"] == []  # degraded: recompute-only
    for step in ("fill", "update", "horizon", "report", "calibration"):
        assert step in calls


def test_g8_cohort_ingest_failure_never_blocks_trading_steps(monkeypatch):
    """Review iter-1 regression — the cohort-ONLY extras ingest is a SEPARATE,
    failure-isolated call: an exception there (bad cohort symbol, failed extra
    yfinance batch) can never block fill/exit/eval/report, and the feature
    step still runs with the trading ingest's changed set."""
    changed = [("AAA", date(2026, 1, 9))]
    calls, captured = _run(
        monkeypatch,
        cohort_ingest_exc=RuntimeError("yfinance down"),
        ingest_result={"upserted": 1, "changed": changed},
    )
    assert "ingest" in calls and "ingest_extra" in calls
    # Trading ingest ran FIRST — its blast radius is unchanged by the extras.
    assert calls.index("ingest") < calls.index("ingest_extra")
    assert captured["feature_changed"] == changed  # extras' changed degraded to []
    for step in ("features", "fill", "update", "horizon", "report", "calibration"):
        assert step in calls, f"{step} was blocked by the cohort ingest"
