"""Step 7 — daily wiring order (TEST-SPEC §G). Patches steps to assert order."""

from __future__ import annotations

import asyncio

from rainier.scheduler import service


def test_g1_daily_eval_runs_steps_in_order(monkeypatch):
    calls: list[str] = []

    # Paper steps (i)-(iii).
    def fake_ingest(symbols, **kw):
        calls.append("ingest")
        return {"upserted": 0}

    def fake_active():
        return ["AAA"]

    def fake_screened(as_of):
        return ["BBB"]

    def fake_fill(**kw):
        calls.append("fill")
        return {"filled": 0, "expired": 0}

    def fake_update(**kw):
        calls.append("update")
        return {"closed": 0, "same_bar_ambiguous_exits": 0}

    monkeypatch.setattr("rainier.paper.ingest.ingest_prices", fake_ingest)
    monkeypatch.setattr("rainier.paper.ingest.active_symbols", fake_active)
    monkeypatch.setattr("rainier.paper.ingest.screened_symbols", fake_screened)
    monkeypatch.setattr("rainier.paper.positions.fill_pending_positions", fake_fill)
    monkeypatch.setattr("rainier.paper.positions.update_open_positions", fake_update)

    # Existing horizon eval (step iv).
    def fake_evaluate_horizon(eval_date, horizon):
        calls.append("horizon")
        return 0

    monkeypatch.setattr(
        "rainier.llm_thesis.eval.evaluate_horizon", fake_evaluate_horizon
    )
    monkeypatch.setattr(
        "rainier.llm_thesis.eval.compute_verdict_hit_rate",
        lambda *a, **k: {},
    )
    monkeypatch.setattr(
        "rainier.llm_thesis.eval.compute_signal_contribution",
        lambda *a, **k: [],
    )
    monkeypatch.setattr("rainier.alerts.discord.send_eval_report", lambda **k: None)

    # R-D close-side chart capture (step v addendum, unconditional). The
    # scheduler must thread the FRESH settings' chart_lookback_days through
    # (codex P2: never the process-cached singleton mid-daemon).
    seen_windows: list = []

    def fake_close_charts(*, as_of, window=None):
        calls.append("close_charts")
        seen_windows.append(window)
        return 0

    monkeypatch.setattr(
        "rainier.paper.chart_archive.capture_trade_close_charts", fake_close_charts
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
        "rainier.paper.calibration.compute_calibration_payload", fake_calib_compute
    )
    monkeypatch.setattr(
        "rainier.paper.calibration.persist_calibration", lambda *a, **k: None
    )

    # R-A reflections — run AFTER step (v) (the report/chart-capture step), so
    # the close-side chart exists by reflection time once chart-archive lands.
    def fake_reflections(as_of, **kw):
        calls.append("reflections")
        return {"written": 0, "failed": 0}

    monkeypatch.setattr(
        "rainier.paper.reflection.generate_reflections", fake_reflections
    )

    # Avoid the yesterday-rows DB fetch.
    monkeypatch.setattr(service, "load_settings_fresh", lambda: _FakeSettings())

    asyncio.run(service.run_daily_eval("2026-01-09"))

    # ingest precedes fill (G2); fill precedes update; update precedes horizon
    # eval; horizon precedes the paper report; report precedes reflections
    # (R-A runs after step (v)); report precedes calibration (step vi reuses
    # the report's MTM figure — G1 authoritative order).
    assert calls.index("ingest") < calls.index("fill")
    assert calls.index("fill") < calls.index("update")
    assert calls.index("update") < calls.index("horizon")
    assert calls.index("horizon") < calls.index("report")
    assert calls.index("report") < calls.index("reflections")
    assert calls.index("reflections") < calls.index("calibration")
    # R-D: close-side chart capture runs in the step (v) block — after the
    # exits are booked (update) and before the daily report goes out.
    assert calls.index("update") < calls.index("close_charts")
    assert calls.index("close_charts") < calls.index("report")
    # ... and the window is the fresh-settings value (117 is deliberately NOT
    # the 120 config default — proves it came from load_settings_fresh()).
    assert seen_windows == [117]


class _FakeDiscord:
    enabled = False
    webhook_url = ""


class _FakeAlerts:
    discord = _FakeDiscord()


class _FakeLLMThesis:
    learned_time_stop_days = None
    model = "test-model"
    # The operator's LLM kill switch — reflections (R-A) gate on it too.
    enabled = True
    # Deliberately NOT the 120 default: G1 asserts this exact value reaches
    # capture_trade_close_charts, proving the fresh-settings threading.
    chart_lookback_days = 117


class _FakeSettings:
    alerts = _FakeAlerts()
    llm_thesis = _FakeLLMThesis()


class _FakeLLMThesisDisabled(_FakeLLMThesis):
    enabled = False


class _FakeSettingsLLMOff(_FakeSettings):
    llm_thesis = _FakeLLMThesisDisabled()


def _patch_daily_eval_steps(monkeypatch, *, settings=None) -> None:
    """Stub every run_daily_eval step except reflections/calibration tracking."""
    monkeypatch.setattr("rainier.paper.ingest.ingest_prices", lambda *a, **k: {})
    monkeypatch.setattr("rainier.paper.ingest.active_symbols", lambda: [])
    monkeypatch.setattr("rainier.paper.ingest.screened_symbols", lambda d: [])
    monkeypatch.setattr(
        "rainier.paper.positions.fill_pending_positions", lambda **k: {}
    )
    monkeypatch.setattr(
        "rainier.paper.positions.update_open_positions", lambda **k: {}
    )
    monkeypatch.setattr(
        "rainier.llm_thesis.eval.evaluate_horizon", lambda *a, **k: 0
    )
    monkeypatch.setattr(
        "rainier.llm_thesis.eval.compute_verdict_hit_rate", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        "rainier.llm_thesis.eval.compute_signal_contribution", lambda *a, **k: []
    )
    monkeypatch.setattr("rainier.alerts.discord.send_eval_report", lambda **k: None)
    monkeypatch.setattr("rainier.paper.report.compute_daily_payload", lambda d: {})
    monkeypatch.setattr(
        "rainier.paper.report.persist_daily_snapshot", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "rainier.paper.report.send_daily_paper_report", lambda *a, **k: True
    )
    monkeypatch.setattr(
        "rainier.paper.calibration.compute_calibration_payload", lambda d: {}
    )
    monkeypatch.setattr(
        "rainier.paper.calibration.persist_calibration", lambda *a, **k: None
    )
    monkeypatch.setattr(
        service, "load_settings_fresh", lambda: settings or _FakeSettings()
    )


def test_g3_horizon_eval_still_runs(monkeypatch):
    ran = {"horizon": False}

    _patch_daily_eval_steps(monkeypatch)

    def fake_evaluate_horizon(eval_date, horizon):
        ran["horizon"] = True
        return 0

    monkeypatch.setattr(
        "rainier.llm_thesis.eval.evaluate_horizon", fake_evaluate_horizon
    )
    monkeypatch.setattr(
        "rainier.paper.reflection.generate_reflections", lambda *a, **k: {}
    )

    asyncio.run(service.run_daily_eval("2026-01-09"))
    assert ran["horizon"] is True


def test_reflections_skipped_when_llm_thesis_disabled(monkeypatch):
    """R-A spend gate: `llm_thesis.enabled = false` (the operator's LLM kill
    switch) must stop reflection LLM calls, not just thesis generation."""
    _patch_daily_eval_steps(monkeypatch, settings=_FakeSettingsLLMOff())

    called = {"reflections": False}

    def fake_reflections(*a, **kw):
        called["reflections"] = True
        return {}

    monkeypatch.setattr(
        "rainier.paper.reflection.generate_reflections", fake_reflections
    )

    asyncio.run(service.run_daily_eval("2026-01-09"))
    assert called["reflections"] is False


def test_reflections_failure_does_not_block_calibration(monkeypatch):
    """A raising reflections step is non-fatal: step (vi) calibration still
    runs (the daily eval's per-step isolation contract)."""
    _patch_daily_eval_steps(monkeypatch)

    ran = {"calibration": False}

    def fake_calib_compute(as_of):
        ran["calibration"] = True
        return {}

    monkeypatch.setattr(
        "rainier.paper.calibration.compute_calibration_payload", fake_calib_compute
    )

    def broken_reflections(*a, **kw):
        raise RuntimeError("reflections blew up")

    monkeypatch.setattr(
        "rainier.paper.reflection.generate_reflections", broken_reflections
    )

    asyncio.run(service.run_daily_eval("2026-01-09"))
    assert ran["calibration"] is True
