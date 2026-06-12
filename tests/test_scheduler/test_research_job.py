"""Tests for `run_research_job` and the Friday 09:00 PT cron registration (PR3)."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from rainier.core.config import (
    AlertsConfig,
    AppConfig,
    DiscordConfig,
    LLMThesisConfig,
    ScrapingConfig,
    ScrapingSchedule,
    Settings,
)


def _settings():
    return Settings(
        database_url="postgresql://test:test@localhost/test",
        app=AppConfig(timezone="America/Los_Angeles"),
        scraping=ScrapingConfig(
            schedule=ScrapingSchedule(
                morning="08:35", midday="10:35",
                afternoon="12:35", close="14:35",
            )
        ),
        alerts=AlertsConfig(
            discord=DiscordConfig(enabled=True, webhook_url="https://example/test")
        ),
        llm_thesis=LLMThesisConfig(),
    )


# ---------------------------------------------------------------------------
# run_research_job orchestration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_research_job_calls_mark_stale_then_run_research():
    """The job must run mark_stale FIRST (so old insights flip to stale
    before the new run_research adds fresh ones).
    """
    call_order: list[str] = []

    def _mark_stale(days):
        call_order.append("mark_stale")
        assert days == 30
        return 0

    def _run_research(*, eval_date, days):  # noqa: ARG001
        call_order.append("run_research")
        return []

    with (
        patch(
            "rainier.scheduler.service.load_settings_fresh",
            return_value=_settings(),
        ),
        patch(
            "rainier.llm_thesis.research.mark_stale", side_effect=_mark_stale,
        ),
        patch(
            "rainier.llm_thesis.research.run_research", side_effect=_run_research,
        ),
        patch(
            "rainier.paper.sweep.sweep_missed_winners",
            return_value={"missed_winners": [], "dodged_losers": {"count": 0}},
        ),
        patch(
            "rainier.alerts.discord.send_research_report"
        ) as mock_discord,
    ):
        from rainier.scheduler.service import run_research_job
        await run_research_job(eval_date_iso="2026-05-08")

    assert call_order == ["mark_stale", "run_research"]
    mock_discord.assert_called_once()


@pytest.mark.asyncio
async def test_run_research_job_continues_when_mark_stale_fails():
    """A mark_stale exception must not skip the run_research + Discord post."""
    def _mark_stale(_days):
        raise RuntimeError("boom")

    def _run_research(*, eval_date, days):
        _ = eval_date, days
        return []

    with (
        patch(
            "rainier.scheduler.service.load_settings_fresh",
            return_value=_settings(),
        ),
        patch(
            "rainier.llm_thesis.research.mark_stale", side_effect=_mark_stale,
        ),
        patch(
            "rainier.llm_thesis.research.run_research", side_effect=_run_research,
        ),
        patch(
            "rainier.paper.sweep.sweep_missed_winners",
            return_value={"missed_winners": [], "dodged_losers": {"count": 0}},
        ),
        patch(
            "rainier.alerts.discord.send_research_report"
        ) as mock_discord,
    ):
        from rainier.scheduler.service import run_research_job
        await run_research_job(eval_date_iso="2026-05-08")

    mock_discord.assert_called_once()


@pytest.mark.asyncio
async def test_run_research_job_continues_when_run_research_fails():
    """A run_research exception must still post the (empty) Discord report
    so the operator sees a heartbeat that the job ran."""
    def _run_research(*, eval_date, days):
        _ = eval_date, days
        raise RuntimeError("checks all failed")

    with (
        patch(
            "rainier.scheduler.service.load_settings_fresh",
            return_value=_settings(),
        ),
        patch(
            "rainier.llm_thesis.research.mark_stale", return_value=0,
        ),
        patch(
            "rainier.llm_thesis.research.run_research", side_effect=_run_research,
        ),
        patch(
            "rainier.paper.sweep.sweep_missed_winners",
            return_value={"missed_winners": [], "dodged_losers": {"count": 0}},
        ),
        patch(
            "rainier.alerts.discord.send_research_report"
        ) as mock_discord,
    ):
        from rainier.scheduler.service import run_research_job
        await run_research_job(eval_date_iso="2026-05-08")

    mock_discord.assert_called_once()
    args, kwargs = mock_discord.call_args
    # When checks fail wholesale the insights list is empty.
    assert kwargs["insights"] == []


@pytest.mark.asyncio
async def test_run_research_job_defaults_to_today():
    captured = {}

    def _run_research(*, eval_date, days):  # noqa: ARG001
        captured["eval_date"] = eval_date
        return []

    with (
        patch(
            "rainier.scheduler.service.load_settings_fresh",
            return_value=_settings(),
        ),
        patch(
            "rainier.llm_thesis.research.mark_stale", return_value=0,
        ),
        patch(
            "rainier.llm_thesis.research.run_research", side_effect=_run_research,
        ),
        patch(
            "rainier.paper.sweep.sweep_missed_winners",
            return_value={"missed_winners": [], "dodged_losers": {"count": 0}},
        ),
        patch("rainier.alerts.discord.send_research_report"),
    ):
        from rainier.scheduler.service import run_research_job
        await run_research_job()

    assert captured["eval_date"] == date.today()


# ---------------------------------------------------------------------------
# Phase 3 — weekly missed-winner sweep wiring (TASK-PLAN qu100-miss-sweep-6b63)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_research_job_runs_miss_sweep_with_eval_date():
    """The Fri 09:00 PT job runs sweep_missed_winners with as_of=eval_date,
    the production yfinance fetch_fn, and the Discord config threaded."""
    captured = {}

    def _sweep(*, as_of, fetch_fn, calendar=None, discord_config=None):
        _ = calendar
        captured["as_of"] = as_of
        captured["fetch_fn"] = fetch_fn
        captured["discord"] = discord_config
        return {"missed_winners": [], "dodged_losers": {"count": 0}}

    settings = _settings()
    with (
        patch(
            "rainier.scheduler.service.load_settings_fresh",
            return_value=settings,
        ),
        patch("rainier.llm_thesis.research.mark_stale", return_value=0),
        patch("rainier.llm_thesis.research.run_research", return_value=[]),
        patch(
            "rainier.paper.sweep.sweep_missed_winners", side_effect=_sweep,
        ),
        patch("rainier.alerts.discord.send_research_report"),
    ):
        from rainier.scheduler.service import run_research_job
        await run_research_job(eval_date_iso="2026-06-12")

    from rainier.paper.ingest import _yfinance_fetch_fn

    assert captured["as_of"] == date(2026, 6, 12)
    assert captured["fetch_fn"] is _yfinance_fetch_fn
    assert captured["discord"] is settings.alerts.discord


@pytest.mark.asyncio
async def test_run_research_job_continues_when_sweep_fails():
    """A sweep failure must not kill the research Discord report."""
    with (
        patch(
            "rainier.scheduler.service.load_settings_fresh",
            return_value=_settings(),
        ),
        patch("rainier.llm_thesis.research.mark_stale", return_value=0),
        patch("rainier.llm_thesis.research.run_research", return_value=[]),
        patch(
            "rainier.paper.sweep.sweep_missed_winners",
            side_effect=RuntimeError("yfinance down"),
        ),
        patch("rainier.alerts.discord.send_research_report") as mock_discord,
    ):
        from rainier.scheduler.service import run_research_job
        await run_research_job(eval_date_iso="2026-06-12")

    mock_discord.assert_called_once()


@pytest.mark.asyncio
async def test_run_research_job_sweep_failure_log_never_leaks_token():
    """The miss-sweep failure handler must log status + exception class ONLY —
    never str(exc), whose httpx variant embeds the token-bearing webhook URL
    (regression for codex [P1] 2026-06-09, commit 96fbd13; review iter-1)."""
    import httpx

    secret = "SWEEPtokenSECRET999"
    url = f"https://discord.com/api/webhooks/55555/{secret}"
    req = httpx.Request("POST", url)
    resp = httpx.Response(401, request=req)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        err = e
    assert secret in str(err)  # the leak vector exists in the exception

    with (
        patch(
            "rainier.scheduler.service.load_settings_fresh",
            return_value=_settings(),
        ),
        patch("rainier.llm_thesis.research.mark_stale", return_value=0),
        patch("rainier.llm_thesis.research.run_research", return_value=[]),
        patch("rainier.paper.sweep.sweep_missed_winners", side_effect=err),
        patch("rainier.alerts.discord.send_research_report"),
        patch("rainier.scheduler.service.log") as mock_log,
    ):
        from rainier.scheduler.service import run_research_job
        await run_research_job(eval_date_iso="2026-06-12")

    blob = repr(mock_log.mock_calls)
    assert secret not in blob, "webhook token leaked into the failure log"
    failed = [
        c for c in mock_log.error.call_args_list
        if c.args and c.args[0] == "research_miss_sweep_failed"
    ]
    assert len(failed) == 1
    assert failed[0].kwargs == {"status": 401, "error_type": "HTTPStatusError"}


# ---------------------------------------------------------------------------
# Cron registration — Friday 09:00 PT
# ---------------------------------------------------------------------------


def test_build_scheduler_registers_weekly_research_friday_0900():
    with patch("rainier.scheduler.service.get_settings", return_value=_settings()):
        from rainier.scheduler.service import build_scheduler
        sched = build_scheduler()

    job = sched.get_job("weekly_research")
    assert job is not None
    trigger = job.trigger
    # The CronTrigger stores fields keyed by name on `.fields`.
    field_lookup = {f.name: str(f) for f in trigger.fields}
    assert field_lookup["day_of_week"] == "fri"
    assert field_lookup["hour"] == "9"
    assert field_lookup["minute"] == "0"
