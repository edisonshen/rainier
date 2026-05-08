"""Rainier scheduler — runs scrape jobs on a daily cron schedule."""

from __future__ import annotations

import asyncio
import signal
from datetime import date

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from rainier.core.config import get_settings, load_settings_fresh

log = structlog.get_logger()


async def run_qu_scrape(session_name: str) -> None:
    """Run a single QU100 scrape for the given session. Called by APScheduler.

    Pipeline (post-scrape):
        1. load_settings_fresh — pick up YAML toggles (eng review D2)
        2. screen_stocks       — 3-layer screener; returns (candidates, ohlcv)
        3. persist_screened_stocks — DB row for every candidate, every session
        4. compute_theses_and_persist — LLM thesis on top-5, only on
           sessions in `settings.llm_thesis.enabled_sessions`
        5. send_stock_candidates — Discord; theses dict empty for non-LLM sessions
    """
    from rainier.scrapers import get_scraper
    from rainier.scrapers.browser import BrowserManager

    log.info("scheduled_scrape_starting", session=session_name)

    try:
        async with BrowserManager(headless=True) as browser:
            scraper = get_scraper("qu", browser)
            result = await scraper.execute(session=session_name)

        log.info(
            "scheduled_scrape_finished",
            session=session_name,
            records=result.records_created,
            errors=len(result.errors),
            duration=result.duration_seconds,
        )
        if result.errors:
            for err in result.errors:
                log.warning("scrape_error", session=session_name, error=err)

        # Notify on success (or partial success with errors)
        from rainier.notifications.notifier import notify_scrape_result
        notify_scrape_result(session_name, result)

        # 1. Reload settings every scan so YAML toggles take effect (D2).
        settings = await asyncio.to_thread(load_settings_fresh)

        # 2. Screener — refactored to return (candidates, ohlcv_dict).
        from rainier.alerts.discord import send_stock_candidates
        from rainier.analysis.stock_screener import screen_stocks

        all_candidates, ohlcv_by_symbol = await asyncio.to_thread(
            screen_stocks, settings
        )
        # PR2 carry-over P2 #3: persist top-50 (or all if fewer) to give the
        # 30-day shadow validation an unbiased dataset. Top-20 alone biases
        # the per-rank correlation toward the upper tail. Discord still gets
        # top-20 (display rule), and the LLM still only runs on top-5.
        scan_candidates = all_candidates[:50]
        candidates = all_candidates[:20]  # Discord display set

        scan_date = date.today()

        # 3. Always persist screener output for every scan — full top-50, not
        #    only the Discord set.
        try:
            from rainier.llm_thesis.persistence import persist_screened_stocks
            await asyncio.to_thread(
                persist_screened_stocks,
                scan_candidates,
                scan_date=scan_date,
                session_name=session_name,
            )
        except Exception as exc:
            log.error(
                "persist_screened_stocks_failed",
                session=session_name,
                error=str(exc),
            )

        # 4. LLM thesis ONLY on configured sessions (afternoon + close).
        theses: dict[str, dict] = {}
        if (
            settings.llm_thesis.enabled
            and session_name in settings.llm_thesis.enabled_sessions
            and candidates
        ):
            try:
                from rainier.llm_thesis.service import compute_theses_and_persist
                theses = await asyncio.to_thread(
                    compute_theses_and_persist,
                    candidates[:5],
                    ohlcv_by_symbol,
                    scan_date=scan_date,
                    session_name=session_name,
                    settings=settings,
                )
            except Exception as exc:
                log.error("compute_theses_unexpected_failure", error=str(exc))
                theses = {}

        # 5. Discord — empty theses dict means existing top-20-only behavior.
        await asyncio.to_thread(
            send_stock_candidates,
            candidates,
            settings.alerts.discord,
            theses=theses or None,
        )

    except Exception as exc:
        log.error("scheduled_scrape_failed", session=session_name, error=str(exc))

        # Notify on failure
        from rainier.notifications.notifier import notify_scrape_failure
        notify_scrape_failure(session_name, str(exc))


def build_scheduler() -> AsyncIOScheduler:
    """
    Build an AsyncIOScheduler with cron jobs for each QU100 session.

    Schedule from settings.yaml:
        morning:   08:35 PST  (Mon-Fri)
        midday:    10:35 PST  (Mon-Fri)
        afternoon: 12:35 PST  (Mon-Fri)
        close:     14:35 PST  (Mon-Fri)
    """
    settings = get_settings()
    tz = settings.app.timezone

    scheduler = AsyncIOScheduler(timezone=tz)
    schedule = settings.scraping.schedule

    sessions = {
        "morning": schedule.morning,
        "midday": schedule.midday,
        "afternoon": schedule.afternoon,
        "close": schedule.close,
    }

    for session_name, time_str in sessions.items():
        hour, minute = time_str.split(":")
        trigger = CronTrigger(
            day_of_week="mon-fri",
            hour=int(hour),
            minute=int(minute),
            timezone=tz,
        )
        scheduler.add_job(
            run_qu_scrape,
            trigger=trigger,
            args=[session_name],
            id=f"qu_scrape_{session_name}",
            name=f"QU100 scrape ({session_name})",
            misfire_grace_time=300,  # 5 min grace if system was asleep
        )
        log.info("job_registered", session=session_name, time=time_str, days="Mon-Fri")

    return scheduler


async def start_scheduler() -> None:
    """Start the scheduler and run until interrupted."""
    scheduler = build_scheduler()
    scheduler.start()

    settings = get_settings()
    schedule = settings.scraping.schedule

    log.info(
        "scheduler_running",
        jobs=len(scheduler.get_jobs()),
        schedule={
            "morning": schedule.morning,
            "midday": schedule.midday,
            "afternoon": schedule.afternoon,
            "close": schedule.close,
        },
    )

    # Wait until signalled to stop
    stop_event = asyncio.Event()

    def _handle_signal(sig: int, _frame) -> None:
        sig_name = signal.Signals(sig).name
        log.info("shutdown_signal", signal=sig_name)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        await stop_event.wait()
    finally:
        scheduler.shutdown(wait=False)
        log.info("scheduler_stopped")
