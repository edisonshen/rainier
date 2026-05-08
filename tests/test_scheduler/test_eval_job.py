"""Tests for the run_daily_eval scheduler entry (PR2)."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

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
            discord=DiscordConfig(
                enabled=True, webhook_url="https://example/test",
            )
        ),
        llm_thesis=LLMThesisConfig(),
    )


@pytest.mark.asyncio
async def test_run_daily_eval_calls_evaluate_horizon_for_each_horizon():
    with (
        patch(
            "rainier.scheduler.service.load_settings_fresh",
            return_value=_settings(),
        ),
        patch(
            "rainier.llm_thesis.eval.evaluate_horizon", return_value=0
        ) as mock_eval,
        patch(
            "rainier.llm_thesis.eval.compute_signal_contribution", return_value=[]
        ),
        patch(
            "rainier.llm_thesis.eval.compute_verdict_hit_rate", return_value={}
        ),
        patch(
            "rainier.alerts.discord.send_eval_report"
        ) as mock_discord,
        patch(
            "rainier.core.database.get_session"
        ) as mock_gs,
    ):
        # The yesterday-rows query path uses get_session — return an empty list.
        cm = MagicMock()
        chain = (
            cm.__enter__.return_value.query.return_value
            .join.return_value
            .filter.return_value
            .order_by.return_value
        )
        chain.all.return_value = []
        cm.__exit__.return_value = False
        mock_gs.return_value = cm

        from rainier.scheduler.service import run_daily_eval
        await run_daily_eval(eval_date_iso="2026-05-07")

    # Three horizons → three calls to evaluate_horizon.
    horizons = {call.args[1] for call in mock_eval.call_args_list}
    assert horizons == {"1d", "5d", "10d"}
    mock_discord.assert_called_once()


@pytest.mark.asyncio
async def test_run_daily_eval_continues_when_one_horizon_fails():
    """A failure inside evaluate_horizon for one horizon must not crash the
    scheduler entry — log + continue, then post the Discord report."""
    call_count = {"n": 0}

    def _eval(eval_date, horizon):
        call_count["n"] += 1
        if horizon == "5d":
            raise RuntimeError("boom")
        return 0

    with (
        patch(
            "rainier.scheduler.service.load_settings_fresh",
            return_value=_settings(),
        ),
        patch(
            "rainier.llm_thesis.eval.evaluate_horizon", side_effect=_eval
        ),
        patch(
            "rainier.llm_thesis.eval.compute_signal_contribution", return_value=[]
        ),
        patch(
            "rainier.llm_thesis.eval.compute_verdict_hit_rate", return_value={}
        ),
        patch(
            "rainier.alerts.discord.send_eval_report"
        ) as mock_discord,
        patch(
            "rainier.core.database.get_session"
        ) as mock_gs,
    ):
        cm = MagicMock()
        chain = (
            cm.__enter__.return_value.query.return_value
            .join.return_value
            .filter.return_value
            .order_by.return_value
        )
        chain.all.return_value = []
        cm.__exit__.return_value = False
        mock_gs.return_value = cm

        from rainier.scheduler.service import run_daily_eval
        await run_daily_eval(eval_date_iso="2026-05-07")

    # All three horizons attempted despite the 5d failure.
    assert call_count["n"] == 3
    mock_discord.assert_called_once()


@pytest.mark.asyncio
async def test_run_daily_eval_defaults_to_today_when_no_arg():
    today = date.today()
    captured = {}

    def _eval(eval_date, horizon):
        captured["eval_date"] = eval_date
        return 0

    with (
        patch(
            "rainier.scheduler.service.load_settings_fresh",
            return_value=_settings(),
        ),
        patch(
            "rainier.llm_thesis.eval.evaluate_horizon", side_effect=_eval
        ),
        patch(
            "rainier.llm_thesis.eval.compute_signal_contribution", return_value=[]
        ),
        patch(
            "rainier.llm_thesis.eval.compute_verdict_hit_rate", return_value={}
        ),
        patch("rainier.alerts.discord.send_eval_report"),
        patch(
            "rainier.core.database.get_session"
        ) as mock_gs,
    ):
        cm = MagicMock()
        chain = (
            cm.__enter__.return_value.query.return_value
            .join.return_value
            .filter.return_value
            .order_by.return_value
        )
        chain.all.return_value = []
        cm.__exit__.return_value = False
        mock_gs.return_value = cm

        from rainier.scheduler.service import run_daily_eval
        await run_daily_eval()

    assert captured["eval_date"] == today
