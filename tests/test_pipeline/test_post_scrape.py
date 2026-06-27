"""Tests for the shared post-scrape pipeline used by both CLI and scheduler.

This module is the single source of truth for the steps that happen AFTER
a successful QU scrape:

    screen_stocks -> persist top-50 -> (LLM thesis on top-5 if session is
    in enabled_sessions) -> Discord send_stock_candidates(top-20)

Both ``rainier scrape qu`` (CLI / cron path) and the long-running
``rainier run`` scheduler delegate here so the LLM thesis embeds reach
DISCORD_STOCK_WEBHOOK_URL regardless of how the scrape was triggered.
"""

from __future__ import annotations

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
from rainier.core.types import StockCandidate


def _make_settings(
    *,
    llm_enabled: bool = True,
    enabled_sessions: tuple[str, ...] = ("afternoon", "close"),
    dashboard_base_url: str | None = "https://dash.example.com",
) -> Settings:
    """Build a Settings object exercising the LLM thesis fields the pipeline
    inspects.
    """
    return Settings(
        database_url="postgresql://test:test@localhost/test",
        app=AppConfig(timezone="America/Los_Angeles"),
        scraping=ScrapingConfig(
            schedule=ScrapingSchedule(
                morning="08:35",
                midday="10:35",
                afternoon="12:35",
                close="14:35",
            )
        ),
        alerts=AlertsConfig(
            discord=DiscordConfig(webhook_url="https://x/y", enabled=True),
        ),
        llm_thesis=LLMThesisConfig(
            enabled=llm_enabled,
            model="test",
            max_usd_per_scan=1.0,
            enabled_sessions=list(enabled_sessions),
            dashboard_base_url=dashboard_base_url,
        ),
    )


def _candidates(n: int) -> list[StockCandidate]:
    return [
        StockCandidate(
            symbol=f"S{i:03d}",
            rank=i + 1,
            rank_change=0,
            long_short="Long in",
            capital_flow_direction="+",
            sector="Technology",
            signal_strength=1.0 - i * 0.001,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Module / function existence — guards against the file being deleted or the
# public entry point being renamed.
# ---------------------------------------------------------------------------


def test_module_exposes_run_post_scrape_pipeline():
    from rainier.pipeline.post_scrape import run_post_scrape_pipeline

    assert callable(run_post_scrape_pipeline)


# ---------------------------------------------------------------------------
# Session gating (the core regression this PR fixes).
# ---------------------------------------------------------------------------


class TestSessionGating:
    """LLM thesis fires only on sessions in ``llm_thesis.enabled_sessions``."""

    def test_morning_session_skips_llm_but_still_sends_screener(self):
        """Morning is NOT in enabled_sessions by default — LLM must not run,
        but the regular screener post still goes out so the QU100 channel
        gets the top-20 summary.
        """
        cands = _candidates(25)
        settings = _make_settings()

        with (
            patch(
                "rainier.pipeline.post_scrape.screen_stocks",
                return_value=(cands, {}),
            ),
            patch(
                "rainier.pipeline.post_scrape.persist_screened_stocks"
            ) as mock_persist,
            patch(
                "rainier.pipeline.post_scrape.compute_theses_and_persist"
            ) as mock_llm,
            patch(
                "rainier.pipeline.post_scrape.send_stock_candidates"
            ) as mock_discord,
        ):
            from rainier.pipeline.post_scrape import run_post_scrape_pipeline

            run_post_scrape_pipeline(settings, "morning")

        mock_llm.assert_not_called()
        mock_discord.assert_called_once()
        # Discord display set is top-20 regardless of session.
        sent = mock_discord.call_args.args[0]
        assert len(sent) == 20
        # theses kwarg is None when LLM didn't run.
        assert mock_discord.call_args.kwargs.get("theses") is None
        # iter-2 regression: session label must be forwarded so the summary
        # embed title carries the "— Morning" suffix (pre-extraction CLI did
        # this via _build_payloads(candidates, session=session); the codex
        # iter-2 P2 finding caught the regression when this dropped).
        assert mock_discord.call_args.kwargs.get("session") == "morning"
        # Persist still got top-50 (or all if fewer).
        mock_persist.assert_called_once()
        persisted = mock_persist.call_args.args[0]
        assert len(persisted) == min(50, len(cands))

    def test_session_label_forwarded_for_all_sessions(self):
        """Regression for codex [P2] iter-2: every session (morning, midday,
        afternoon, close) must propagate session= to send_stock_candidates so
        the summary embed title gets the session-label suffix. Without this,
        four same-day scans become indistinguishable in the Discord channel.
        """
        cands = _candidates(5)
        settings = _make_settings()

        for session_name in ("morning", "midday", "afternoon", "close"):
            with (
                patch(
                    "rainier.pipeline.post_scrape.screen_stocks",
                    return_value=(cands, {}),
                ),
                patch("rainier.pipeline.post_scrape.persist_screened_stocks"),
                patch(
                    "rainier.pipeline.post_scrape.compute_theses_and_persist",
                    return_value={},
                ),
                patch(
                    "rainier.pipeline.post_scrape.send_stock_candidates"
                ) as mock_discord,
            ):
                from rainier.pipeline.post_scrape import run_post_scrape_pipeline

                run_post_scrape_pipeline(settings, session_name)

            assert (
                mock_discord.call_args.kwargs.get("session") == session_name
            ), (
                f"session={session_name!r} not forwarded to send_stock_candidates"
            )

    def test_afternoon_session_fires_llm_on_top_5(self):
        cands = _candidates(25)
        ohlcv = {c.symbol: MagicMock() for c in cands}
        settings = _make_settings()

        with (
            patch(
                "rainier.pipeline.post_scrape.screen_stocks",
                return_value=(cands, ohlcv),
            ),
            patch("rainier.pipeline.post_scrape.persist_screened_stocks"),
            patch(
                "rainier.pipeline.post_scrape.compute_theses_and_persist",
                return_value={"S000": {"verdict": "setup_long"}},
            ) as mock_llm,
            patch(
                "rainier.pipeline.post_scrape.send_stock_candidates"
            ) as mock_discord,
        ):
            from rainier.pipeline.post_scrape import run_post_scrape_pipeline

            run_post_scrape_pipeline(settings, "afternoon")

        mock_llm.assert_called_once()
        # compute_theses_and_persist receives top-5 candidates only.
        llm_args = mock_llm.call_args.args
        assert len(llm_args[0]) == 5
        # And the OHLCV map is forwarded so it can build charts.
        assert llm_args[1] is ohlcv
        # Discord call carries theses dict + dashboard URL.
        assert mock_discord.call_args.kwargs.get("theses") == {
            "S000": {"verdict": "setup_long"}
        }
        assert (
            mock_discord.call_args.kwargs.get("dashboard_base_url")
            == "https://dash.example.com"
        )

    def test_close_session_fires_llm(self):
        cands = _candidates(10)
        settings = _make_settings()

        with (
            patch(
                "rainier.pipeline.post_scrape.screen_stocks",
                return_value=(cands, {}),
            ),
            patch("rainier.pipeline.post_scrape.persist_screened_stocks"),
            patch(
                "rainier.pipeline.post_scrape.compute_theses_and_persist",
                return_value={},
            ) as mock_llm,
            patch("rainier.pipeline.post_scrape.send_stock_candidates"),
        ):
            from rainier.pipeline.post_scrape import run_post_scrape_pipeline

            run_post_scrape_pipeline(settings, "close")

        mock_llm.assert_called_once()

    def test_llm_globally_disabled_skips_even_on_afternoon(self):
        """If ``settings.llm_thesis.enabled`` is False the LLM never runs
        regardless of session — the kill switch must be respected."""
        cands = _candidates(10)
        settings = _make_settings(llm_enabled=False)

        with (
            patch(
                "rainier.pipeline.post_scrape.screen_stocks",
                return_value=(cands, {}),
            ),
            patch("rainier.pipeline.post_scrape.persist_screened_stocks"),
            patch(
                "rainier.pipeline.post_scrape.compute_theses_and_persist"
            ) as mock_llm,
            patch("rainier.pipeline.post_scrape.send_stock_candidates"),
        ):
            from rainier.pipeline.post_scrape import run_post_scrape_pipeline

            run_post_scrape_pipeline(settings, "afternoon")

        mock_llm.assert_not_called()

    def test_empty_candidates_skips_llm_block(self):
        """Screener returned nothing — no LLM call, no Discord call."""
        settings = _make_settings()

        with (
            patch(
                "rainier.pipeline.post_scrape.screen_stocks",
                return_value=([], {}),
            ),
            patch(
                "rainier.pipeline.post_scrape.persist_screened_stocks"
            ) as mock_persist,
            patch(
                "rainier.pipeline.post_scrape.compute_theses_and_persist"
            ) as mock_llm,
            patch(
                "rainier.pipeline.post_scrape.send_stock_candidates"
            ) as mock_discord,
        ):
            from rainier.pipeline.post_scrape import run_post_scrape_pipeline

            run_post_scrape_pipeline(settings, "afternoon")

        mock_llm.assert_not_called()
        # send_stock_candidates short-circuits on empty input — see
        # alerts/discord.py:send_stock_candidates. Either it isn't called,
        # or it's called with an empty list. Both are valid.
        if mock_discord.called:
            assert mock_discord.call_args.args[0] == []
        # Persist still called for empty list (with [] — no rows to write).
        mock_persist.assert_called_once()
        assert mock_persist.call_args.args[0] == []


# ---------------------------------------------------------------------------
# Slicing — persistence/Discord/LLM see different windows.
# ---------------------------------------------------------------------------


class TestSlicing:
    def test_persist_top_50_discord_top_20_llm_top_5(self):
        """Mirrors the service.py contract: persist 50, Discord 20, LLM 5."""
        cands = _candidates(80)
        settings = _make_settings()

        with (
            patch(
                "rainier.pipeline.post_scrape.screen_stocks",
                return_value=(cands, {}),
            ),
            patch(
                "rainier.pipeline.post_scrape.persist_screened_stocks"
            ) as mock_persist,
            patch(
                "rainier.pipeline.post_scrape.compute_theses_and_persist",
                return_value={},
            ) as mock_llm,
            patch(
                "rainier.pipeline.post_scrape.send_stock_candidates"
            ) as mock_discord,
        ):
            from rainier.pipeline.post_scrape import run_post_scrape_pipeline

            run_post_scrape_pipeline(settings, "afternoon")

        assert len(mock_persist.call_args.args[0]) == 50
        assert len(mock_discord.call_args.args[0]) == 20
        assert len(mock_llm.call_args.args[0]) == 5

    def test_explicit_scan_date_stamps_persisted_artifacts(self):
        """Codex P1 regression — an explicit ``scan_date`` (recover passes the
        RECOVERED trading day) is threaded to persist_screened_stocks AND
        compute_theses_and_persist, so a post-midnight / cross-tz replay stamps
        derived rows with the day the data is from, not date.today()."""
        from datetime import date

        recovered_day = date(2026, 5, 22)
        cands = _candidates(10)
        settings = _make_settings()

        with (
            patch(
                "rainier.pipeline.post_scrape.screen_stocks",
                return_value=(cands, {}),
            ),
            patch(
                "rainier.pipeline.post_scrape.persist_screened_stocks"
            ) as mock_persist,
            patch(
                "rainier.pipeline.post_scrape.compute_theses_and_persist",
                return_value={},
            ) as mock_llm,
            patch("rainier.pipeline.post_scrape.send_stock_candidates"),
        ):
            from rainier.pipeline.post_scrape import run_post_scrape_pipeline

            run_post_scrape_pipeline(settings, "afternoon", recovered_day)

        assert mock_persist.call_args.kwargs["scan_date"] == recovered_day
        assert mock_llm.call_args.kwargs["scan_date"] == recovered_day

    def test_fewer_than_5_candidates_llm_sees_all(self):
        cands = _candidates(3)
        settings = _make_settings()

        with (
            patch(
                "rainier.pipeline.post_scrape.screen_stocks",
                return_value=(cands, {}),
            ),
            patch("rainier.pipeline.post_scrape.persist_screened_stocks"),
            patch(
                "rainier.pipeline.post_scrape.compute_theses_and_persist",
                return_value={},
            ) as mock_llm,
            patch("rainier.pipeline.post_scrape.send_stock_candidates"),
        ):
            from rainier.pipeline.post_scrape import run_post_scrape_pipeline

            run_post_scrape_pipeline(settings, "afternoon")

        assert len(mock_llm.call_args.args[0]) == 3


# ---------------------------------------------------------------------------
# Error tolerance — persist + LLM failures must NOT prevent the Discord post.
# ---------------------------------------------------------------------------


class TestErrorTolerance:
    def test_persist_failure_does_not_block_discord(self):
        cands = _candidates(5)
        settings = _make_settings()

        with (
            patch(
                "rainier.pipeline.post_scrape.screen_stocks",
                return_value=(cands, {}),
            ),
            patch(
                "rainier.pipeline.post_scrape.persist_screened_stocks",
                side_effect=RuntimeError("DB down"),
            ),
            patch(
                "rainier.pipeline.post_scrape.compute_theses_and_persist",
                return_value={},
            ),
            patch(
                "rainier.pipeline.post_scrape.send_stock_candidates"
            ) as mock_discord,
        ):
            from rainier.pipeline.post_scrape import run_post_scrape_pipeline

            # Must not raise.
            run_post_scrape_pipeline(settings, "afternoon")

        mock_discord.assert_called_once()

    def test_llm_failure_falls_back_to_empty_theses(self):
        cands = _candidates(5)
        settings = _make_settings()

        with (
            patch(
                "rainier.pipeline.post_scrape.screen_stocks",
                return_value=(cands, {}),
            ),
            patch("rainier.pipeline.post_scrape.persist_screened_stocks"),
            patch(
                "rainier.pipeline.post_scrape.compute_theses_and_persist",
                side_effect=RuntimeError("LLM 500"),
            ),
            patch(
                "rainier.pipeline.post_scrape.send_stock_candidates"
            ) as mock_discord,
        ):
            from rainier.pipeline.post_scrape import run_post_scrape_pipeline

            run_post_scrape_pipeline(settings, "afternoon")

        mock_discord.assert_called_once()
        # When LLM crashes, the call falls back to ``theses=None`` so the
        # Discord renderer takes the regular top-20-only path. ``theses or
        # None`` covers both the empty-dict and the missing-dict case.
        assert mock_discord.call_args.kwargs.get("theses") is None


# ---------------------------------------------------------------------------
# Delegation — service.py & cli.py both route through the shared module.
# ---------------------------------------------------------------------------


class TestSchedulerServiceDelegation:
    @pytest.mark.asyncio
    async def test_run_qu_scrape_delegates_to_shared_pipeline(self):
        """service.py:run_qu_scrape must call run_post_scrape_pipeline rather
        than re-implementing the screener→LLM→Discord block inline.
        """
        from unittest.mock import AsyncMock

        mock_result = MagicMock(records_created=20, errors=[], duration_seconds=1.0)
        mock_scraper = AsyncMock()
        mock_scraper.execute = AsyncMock(return_value=mock_result)
        settings = _make_settings()

        with (
            patch("rainier.scrapers.browser.BrowserManager") as MockBM,
            patch("rainier.scrapers.get_scraper", return_value=mock_scraper),
            patch(
                "rainier.scheduler.service.load_settings_fresh",
                return_value=settings,
            ),
            patch("rainier.notifications.notifier.notify_scrape_result"),
            patch(
                "rainier.scheduler.service.run_post_scrape_pipeline"
            ) as mock_pipeline,
        ):
            MockBM.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            MockBM.return_value.__aexit__ = AsyncMock(return_value=False)

            from rainier.scheduler.service import run_qu_scrape

            await run_qu_scrape("afternoon")

        mock_pipeline.assert_called_once_with(settings, "afternoon")


class TestCliDelegation:
    """The cron CLI path (``rainier scrape qu``) must also route through the
    shared module — that's the whole point of the task."""

    @pytest.mark.asyncio
    async def test_cli_run_qu_scrape_calls_shared_pipeline(self):
        from unittest.mock import AsyncMock

        mock_result = MagicMock(records_created=20, errors=[], duration_seconds=1.0)
        mock_scraper = AsyncMock()
        mock_scraper.execute = AsyncMock(return_value=mock_result)
        settings = _make_settings()

        with (
            patch("rainier.scrapers.browser.BrowserManager") as MockBM,
            patch("rainier.scrapers.get_scraper", return_value=mock_scraper),
            patch("rainier.core.config.get_settings", return_value=settings),
            patch("rainier.cli.run_post_scrape_pipeline") as mock_pipeline,
        ):
            MockBM.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            MockBM.return_value.__aexit__ = AsyncMock(return_value=False)

            from rainier.cli import _run_qu_scrape

            await _run_qu_scrape(
                session="afternoon",
                detail_top=None,
                dates=None,
                days_back=0,
                start_date=None,
                delay=None,
                headed=False,
                cdp=None,
            )

        mock_pipeline.assert_called_once_with(settings, "afternoon")

    @pytest.mark.asyncio
    async def test_cli_skips_pipeline_when_no_records_created(self):
        """Existing guard: if the scrape produced zero rows OR a backfill
        date_list was provided, skip the post-scrape pipeline."""
        from unittest.mock import AsyncMock

        mock_result = MagicMock(records_created=0, errors=[], duration_seconds=1.0)
        mock_scraper = AsyncMock()
        mock_scraper.execute = AsyncMock(return_value=mock_result)
        settings = _make_settings()

        with (
            patch("rainier.scrapers.browser.BrowserManager") as MockBM,
            patch("rainier.scrapers.get_scraper", return_value=mock_scraper),
            patch("rainier.core.config.get_settings", return_value=settings),
            patch("rainier.cli.run_post_scrape_pipeline") as mock_pipeline,
        ):
            MockBM.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            MockBM.return_value.__aexit__ = AsyncMock(return_value=False)

            from rainier.cli import _run_qu_scrape

            await _run_qu_scrape(
                session="afternoon",
                detail_top=None,
                dates=None,
                days_back=0,
                start_date=None,
                delay=None,
                headed=False,
                cdp=None,
            )

        mock_pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_cli_skips_pipeline_during_backfill(self):
        from unittest.mock import AsyncMock

        mock_result = MagicMock(records_created=42, errors=[], duration_seconds=1.0)
        mock_scraper = AsyncMock()
        mock_scraper.execute = AsyncMock(return_value=mock_result)
        settings = _make_settings()

        with (
            patch("rainier.scrapers.browser.BrowserManager") as MockBM,
            patch("rainier.scrapers.get_scraper", return_value=mock_scraper),
            patch("rainier.core.config.get_settings", return_value=settings),
            patch("rainier.cli.run_post_scrape_pipeline") as mock_pipeline,
        ):
            MockBM.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            MockBM.return_value.__aexit__ = AsyncMock(return_value=False)

            from rainier.cli import _run_qu_scrape

            await _run_qu_scrape(
                session="afternoon",
                detail_top=None,
                dates="2026-05-01,2026-05-02",
                days_back=0,
                start_date=None,
                delay=None,
                headed=False,
                cdp=None,
            )

        # date_list is non-empty -> skip post-scrape pipeline.
        mock_pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_cli_pinned_date_keeps_inline_pipeline_off(self):
        """``recover`` pins the scrape to the stale day (dates=<today>), so
        _run_qu_scrape's INLINE pipeline stays OFF — recover runs the pipeline
        itself, gated on restored two-book freshness, so a stale half-book can't
        fire screener/LLM/candidate output inline (Codex P1)."""
        from unittest.mock import AsyncMock

        mock_result = MagicMock(records_created=200, errors=[], duration_seconds=1.0)
        mock_scraper = AsyncMock()
        mock_scraper.execute = AsyncMock(return_value=mock_result)
        settings = _make_settings()

        with (
            patch("rainier.scrapers.browser.BrowserManager") as MockBM,
            patch("rainier.scrapers.get_scraper", return_value=mock_scraper),
            patch("rainier.core.config.get_settings", return_value=settings),
            patch("rainier.cli.run_post_scrape_pipeline") as mock_pipeline,
        ):
            MockBM.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            MockBM.return_value.__aexit__ = AsyncMock(return_value=False)

            from rainier.cli import _run_qu_scrape

            await _run_qu_scrape(
                session="afternoon",
                detail_top=0,
                dates="2026-05-22",  # pinned single day (recover targets `today`)
                days_back=0,
                start_date=None,
                delay=None,
                headed=False,
                cdp=None,
            )

        # Date pinned -> inline pipeline must NOT run (recover gates it separately).
        mock_pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_cli_pipeline_runs_in_thread_so_inner_asyncio_run_is_safe(
        self,
    ):
        """Regression for codex [P1] iter-1: _run_qu_scrape is already running
        inside an event loop, so the pipeline (which internally does
        asyncio.run via compute_theses_and_persist) must be handed off via
        asyncio.to_thread. Verify by patching the pipeline with a callable
        that itself calls asyncio.run — if we were calling the pipeline
        directly from the running loop, that would raise RuntimeError. Run
        in a thread, it works.
        """
        import asyncio
        from unittest.mock import AsyncMock

        mock_result = MagicMock(records_created=20, errors=[], duration_seconds=1.0)
        mock_scraper = AsyncMock()
        mock_scraper.execute = AsyncMock(return_value=mock_result)
        settings = _make_settings()

        async def _noop_coro():
            return None

        def _pipeline_that_uses_asyncio_run(_settings, _session):
            # Mirrors compute_theses_and_persist's pattern of wrapping async
            # work in asyncio.run inside a sync function. If the call site in
            # _run_qu_scrape ever drops the asyncio.to_thread hand-off, this
            # call raises RuntimeError("asyncio.run() cannot be called from a
            # running event loop") and the test fails.
            asyncio.run(_noop_coro())

        with (
            patch("rainier.scrapers.browser.BrowserManager") as MockBM,
            patch("rainier.scrapers.get_scraper", return_value=mock_scraper),
            patch("rainier.core.config.get_settings", return_value=settings),
            patch(
                "rainier.cli.run_post_scrape_pipeline",
                side_effect=_pipeline_that_uses_asyncio_run,
            ) as mock_pipeline,
        ):
            MockBM.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            MockBM.return_value.__aexit__ = AsyncMock(return_value=False)

            from rainier.cli import _run_qu_scrape

            await _run_qu_scrape(
                session="afternoon",
                detail_top=None,
                dates=None,
                days_back=0,
                start_date=None,
                delay=None,
                headed=False,
                cdp=None,
            )

        mock_pipeline.assert_called_once_with(settings, "afternoon")

    @pytest.mark.asyncio
    async def test_cli_pipeline_failure_does_not_morph_into_scrape_failure(self):
        """Regression for codex [P2] iter-1: pre-extraction CLI guarded the
        screener block in its own try/except and reported a partial-success
        warning. After extraction, a screener / pipeline failure was bubbling
        all the way up to _run_qu_scrape's outer except, turning a successful
        scrape into a red "Scrape FAILED" alert and a non-zero exit.

        Expected behavior: pipeline exception is caught locally, an orange
        partial-success Discord alert is emitted, and the function returns
        normally (no re-raise).
        """
        from unittest.mock import AsyncMock

        mock_result = MagicMock(records_created=20, errors=[], duration_seconds=1.0)
        mock_scraper = AsyncMock()
        mock_scraper.execute = AsyncMock(return_value=mock_result)
        settings = _make_settings()

        with (
            patch("rainier.scrapers.browser.BrowserManager") as MockBM,
            patch("rainier.scrapers.get_scraper", return_value=mock_scraper),
            patch("rainier.core.config.get_settings", return_value=settings),
            patch(
                "rainier.cli.run_post_scrape_pipeline",
                side_effect=RuntimeError("screener db blip"),
            ),
            patch("rainier.cli._notify_scrape_discord") as mock_notify,
        ):
            MockBM.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            MockBM.return_value.__aexit__ = AsyncMock(return_value=False)

            from rainier.cli import _run_qu_scrape

            # Must NOT raise — pipeline failures stay local.
            await _run_qu_scrape(
                session="afternoon",
                detail_top=None,
                dates=None,
                days_back=0,
                start_date=None,
                delay=None,
                headed=False,
                cdp=None,
            )

        # The partial-success notification fired (orange, not red).
        # _notify_scrape_discord may also be called for other reasons in the
        # path (e.g. scrape warnings) so we scan calls for the post-scrape
        # alert specifically rather than asserting call count.
        titles = [
            call.kwargs.get("title") or (call.args[2] if len(call.args) >= 3 else "")
            for call in mock_notify.call_args_list
        ]
        assert any("Post-Scrape FAILED" in t for t in titles), (
            f"expected a Post-Scrape FAILED partial-success alert; got: {titles}"
        )
        # The red "Scrape FAILED" alert (outer except) must NOT have fired.
        assert not any("Scrape FAILED" in t and "Post-Scrape" not in t for t in titles), (
            f"unexpected red Scrape FAILED alert fired: {titles}"
        )
