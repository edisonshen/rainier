"""Rainier scheduler — runs scrape jobs on a daily cron schedule."""

from __future__ import annotations

import asyncio
import signal

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from rainier.core.config import get_settings

log = structlog.get_logger()


async def run_qu_scrape(session_name: str) -> None:
    """Run a single QU100 scrape for the given session. Called by APScheduler."""
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

        # Send stock candidates to Discord after scrape
        from rainier.alerts.discord import send_stock_candidates
        from rainier.analysis.stock_screener import screen_stocks
        settings = get_settings()
        candidates = screen_stocks(settings)[:20]
        send_stock_candidates(candidates, settings.alerts.discord)

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
