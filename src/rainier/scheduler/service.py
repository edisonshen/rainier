"""Rainier scheduler — runs scrape jobs on a daily cron schedule."""

from __future__ import annotations

import asyncio
import signal

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from rainier.core.config import get_settings, load_settings_fresh
from rainier.pipeline.post_scrape import run_post_scrape_pipeline

log = structlog.get_logger()


async def run_qu_scrape(session_name: str) -> None:
    """Run a single QU100 scrape for the given session. Called by APScheduler.

    After the scrape itself succeeds, the post-scrape pipeline (screen ->
    persist -> LLM thesis -> Discord) is delegated to
    :func:`rainier.pipeline.post_scrape.run_post_scrape_pipeline` — the same
    function the cron CLI path uses, so both entry points emit identical
    Discord output.
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

        # Reload settings every scan so YAML toggles take effect (D2). The
        # shared pipeline is sync — we hand off in a single to_thread call
        # rather than threading each sub-step.
        settings = await asyncio.to_thread(load_settings_fresh)
        await asyncio.to_thread(run_post_scrape_pipeline, settings, session_name)

    except Exception as exc:
        log.error("scheduled_scrape_failed", session=session_name, error=str(exc))

        # Notify on failure
        from rainier.notifications.notifier import notify_scrape_failure
        notify_scrape_failure(session_name, str(exc))


async def run_daily_eval(eval_date_iso: str | None = None) -> None:
    """Backfill ThesisEvaluation rows + post the Discord eval report.

    Triggered nightly at 17:00 PT (Mon-Fri). Idempotent: re-runs against the
    same eval_date insert nothing new and re-render the same Discord message.

    eval_date_iso lets callers (CLI, manual replay) override "today".
    """
    from datetime import date as _date

    from rainier.alerts.discord import send_eval_report
    from rainier.core.database import get_session
    from rainier.core.models import ScreenedStockRecord, ThesisEvaluation
    from rainier.llm_thesis.eval import (
        HORIZONS,
        compute_signal_contribution,
        compute_verdict_hit_rate,
        evaluate_horizon,
    )

    eval_date = (
        _date.fromisoformat(eval_date_iso) if eval_date_iso else _date.today()
    )

    log.info("daily_eval_starting", eval_date=eval_date.isoformat())

    # Paper-tracker (design §5 authoritative order): ingest (active ∪ screened)
    # → fill → update. These run BEFORE the existing horizon eval; the paper
    # daily report is sent AFTER the horizon eval (step v). Non-fatal: a failure
    # here must not block the horizon eval / report.
    try:
        await asyncio.to_thread(run_paper_daily_steps, eval_date)
    except Exception as exc:
        log.error("daily_paper_steps_failed", error=str(exc))

    inserted_total = 0
    for horizon in HORIZONS:
        try:
            n = await asyncio.to_thread(evaluate_horizon, eval_date, horizon)
            inserted_total += n
        except Exception as exc:
            log.error(
                "daily_eval_horizon_failed",
                horizon=horizon,
                error=str(exc),
            )

    log.info("daily_eval_inserts_done", inserts=inserted_total)

    settings = await asyncio.to_thread(load_settings_fresh)

    # Pull "yesterday's scan" rows for the report block. We use the
    # 1d horizon since that captures the freshest readout the operator cares
    # about each morning.
    #
    # PR3 carry-over [P2]: roll back via TRADING days, not calendar days. On
    # a Monday eval, calendar-rollback lands on Sunday and quietly returns
    # zero theses; the operator never sees Friday's afternoon/close picks
    # graded. `_trading_days_back` from llm_thesis.eval already handles the
    # weekend skip — reuse it here.
    def _fetch_yesterday():
        from rainier.llm_thesis.eval import _trading_days_back

        prior = _trading_days_back(eval_date, 1)
        with get_session() as session:
            rows = (
                session.query(
                    ThesisEvaluation.scan_date,
                    ThesisEvaluation.symbol,
                    ThesisEvaluation.verdict,
                    ThesisEvaluation.llm_confidence,
                    ThesisEvaluation.return_pct,
                    ThesisEvaluation.hit,
                    ScreenedStockRecord.session_name,
                )
                .join(
                    ScreenedStockRecord,
                    ScreenedStockRecord.id
                    == ThesisEvaluation.screened_record_id,
                )
                .filter(
                    ThesisEvaluation.scan_date == prior,
                    ThesisEvaluation.horizon == "1d",
                )
                .order_by(ThesisEvaluation.symbol)
                .all()
            )
        return [
            {
                "scan_date": r.scan_date,
                "session_name": r.session_name,
                "symbol": r.symbol,
                "verdict": r.verdict,
                "llm_confidence": r.llm_confidence,
                "return_pct": r.return_pct,
                "hit": r.hit,
            }
            for r in rows
        ]

    try:
        yesterday_rows = await asyncio.to_thread(_fetch_yesterday)
    except Exception as exc:
        log.error("daily_eval_fetch_yesterday_failed", error=str(exc))
        yesterday_rows = []

    try:
        # PR3 carry-over [P3]: pass eval_date so the rolling window anchors on
        # the run's logical "today" rather than `date.today()` at function-call
        # time (matters for historical replay / manual eval).
        base_rates = await asyncio.to_thread(
            lambda: compute_verdict_hit_rate(30, eval_date=eval_date)
        )
    except Exception as exc:
        log.error("daily_eval_base_rates_failed", error=str(exc))
        base_rates = {}

    try:
        contribs = await asyncio.to_thread(
            lambda: compute_signal_contribution(30, "5d", eval_date=eval_date)
        )
    except Exception as exc:
        log.error("daily_eval_contribs_failed", error=str(exc))
        contribs = []

    try:
        await asyncio.to_thread(
            send_eval_report,
            eval_date=eval_date,
            yesterday_rows=yesterday_rows,
            base_rates=base_rates,
            signal_contribs=contribs,
            config=settings.alerts.discord,
        )
    except Exception as exc:
        log.error("daily_eval_discord_failed", error=str(exc))

    # Step (v): paper-book daily report — compute, persist the snapshot, push to
    # Discord. Runs after the horizon eval per the authoritative order. Non-fatal.
    try:
        await asyncio.to_thread(
            run_paper_daily_report, eval_date, settings.alerts.discord
        )
    except Exception as exc:
        log.error("daily_paper_report_failed", error=str(exc))

    # R-A: post-exit reflections — one LLM post-mortem per trade closed within
    # the trailing 30 days that has none yet. Runs AFTER step (v) (the report/
    # chart-capture step) so the close-side chart exists by reflection time once
    # the chart-archive lands; pre-archive schemas take the text-only path.
    # Non-fatal: a failure leaves reflections NULL, retried tomorrow.
    try:
        await asyncio.to_thread(
            run_paper_reflections, eval_date, settings.llm_thesis.model
        )
    except Exception as exc:
        log.error("daily_paper_reflections_failed", error=str(exc))

    # Step (vi): D7a calibration block — compute the unbiased fixed-horizon
    # headline + labeled realized supplementary and persist it for tomorrow's
    # thesis prompt. Runs AFTER the daily report (it reuses the report's
    # MTM-including-open figure). Non-fatal.
    try:
        await asyncio.to_thread(run_paper_calibration, eval_date)
    except Exception as exc:
        log.error("daily_paper_calibration_failed", error=str(exc))


def run_paper_daily_steps(eval_date) -> None:
    """Paper steps (i)-(iii): ingest (active ∪ screened) → fill → update.

    Ingest MUST precede fill/update or a pending's T+1 open would be absent
    (G2). Each step is its own DB-driven, idempotent operation.
    """
    from rainier.paper.ingest import (
        _yfinance_fetch_fn,
        active_symbols,
        ingest_prices,
        screened_symbols,
    )
    from rainier.paper.positions import fill_pending_positions, update_open_positions

    settings = load_settings_fresh()
    learned_ts = settings.llm_thesis.learned_time_stop_days

    symbols = sorted(set(active_symbols()) | set(screened_symbols(eval_date)))
    if symbols:
        ingest_prices(symbols, as_of=eval_date, fetch_fn=_yfinance_fetch_fn)
    fill_pending_positions(as_of=eval_date, learned_time_stop_days=learned_ts)
    update_open_positions(as_of=eval_date)


def run_paper_daily_report(eval_date, discord_config) -> None:
    """Paper step (v): compute → persist snapshot → push to Discord."""
    from rainier.paper.report import (
        compute_daily_payload,
        persist_daily_snapshot,
        send_daily_paper_report,
    )

    payload = compute_daily_payload(eval_date)
    persist_daily_snapshot(eval_date, payload)
    send_daily_paper_report(payload, discord_config)


def run_paper_reflections(eval_date, model: str) -> None:
    """R-A: write post-exit reflections for newly closed trades.

    Selection is DB-driven (`reflection IS NULL`, trailing 30 days), so a
    failed day is naturally retried on the next run. Idempotent.
    """
    from rainier.paper.reflection import generate_reflections

    generate_reflections(eval_date, model=model)


def run_paper_calibration(eval_date) -> None:
    """Paper step (vi): compute the D7a calibration payload → persist it.

    The persisted row feeds ``build_user_message`` on the next scan. Headline =
    unbiased fixed-horizon thesis stats; realized paper record is labeled
    supplementary. Idempotent per as-of date.
    """
    from rainier.paper.calibration import (
        compute_calibration_payload,
        persist_calibration,
    )

    payload = compute_calibration_payload(eval_date)
    persist_calibration(eval_date, payload)


async def run_research_job(eval_date_iso: str | None = None) -> None:
    """Weekly auto-research entry — emits ResearchInsight rows + posts the
    Discord research-report.

    Triggered Friday 09:00 PT. The job:
      1. mark_stale() — flips pending insights >30 days old to status=stale.
      2. run_research(eval_date=today, days=30) — runs all 6 check classes.
      3. send_research_report() — posts a Discord summary of pending
         warn/critical insights.

    eval_date_iso lets CLI / replay callers override "today".
    """
    from datetime import date as _date

    from rainier.alerts.discord import send_research_report
    from rainier.llm_thesis.research import mark_stale, run_research

    eval_date = (
        _date.fromisoformat(eval_date_iso) if eval_date_iso else _date.today()
    )

    log.info("research_job_starting", eval_date=eval_date.isoformat())

    try:
        stale_count = await asyncio.to_thread(mark_stale, 30)
        log.info("research_marked_stale", count=stale_count)
    except Exception as exc:
        log.error("research_mark_stale_failed", error=str(exc))

    try:
        insights = await asyncio.to_thread(
            lambda: run_research(eval_date=eval_date, days=30)
        )
    except Exception as exc:
        log.error("research_run_failed", error=str(exc))
        insights = []

    log.info("research_job_emitted", count=len(insights))

    # Phase 3 — weekly missed-winner sweep (design §5(4), coverage diagnostic
    # only). Own try/except so a sweep failure never kills the research report
    # (and vice versa). Discord delivery inside the sweep is already non-fatal.
    try:
        from rainier.paper.ingest import _yfinance_fetch_fn
        from rainier.paper.sweep import sweep_missed_winners

        settings = await asyncio.to_thread(load_settings_fresh)
        payload = await asyncio.to_thread(
            lambda: sweep_missed_winners(
                as_of=eval_date,
                fetch_fn=_yfinance_fetch_fn,
                discord_config=settings.alerts.discord,
            )
        )
        log.info(
            "research_miss_sweep_done",
            missed=len(payload["missed_winners"]),
            dodged=payload["dodged_losers"]["count"],
        )
    except Exception as exc:
        # Never log str(exc) here: this try block loads the Discord config —
        # a pydantic ValidationError from load_settings_fresh() embeds
        # input_value=... which can carry the token-bearing webhook URL, and
        # any exception escaping the sweep would be stringified one frame up
        # from the send (codex [P1] 2026-06-09, commit 96fbd13). Status +
        # class only.
        from rainier.alerts.discord import _http_status

        log.error(
            "research_miss_sweep_failed",
            status=_http_status(exc),
            error_type=type(exc).__name__,
        )

    try:
        settings = await asyncio.to_thread(load_settings_fresh)
        await asyncio.to_thread(
            send_research_report,
            eval_date=eval_date,
            insights=insights,
            config=settings.alerts.discord,
        )
    except Exception as exc:
        log.error("research_discord_failed", error=str(exc))


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

    # PR2: daily eval at 17:00 PT Mon-Fri (~2h after market close). Backfills
    # ThesisEvaluation rows for 1d/5d/10d horizons and posts the eval Discord
    # report. Idempotent — only fills missing rows on each fire.
    eval_trigger = CronTrigger(
        day_of_week="mon-fri",
        hour=17,
        minute=0,
        timezone=tz,
    )
    scheduler.add_job(
        run_daily_eval,
        trigger=eval_trigger,
        id="daily_eval",
        name="Daily thesis evaluation + Discord report",
        misfire_grace_time=900,  # 15 min grace
    )
    log.info("job_registered", session="daily_eval", time="17:00", days="Mon-Fri")

    # PR3: weekly auto-research at Friday 09:00 PT (eng review D5). Analyzes
    # the rolling 30d window and emits ResearchInsight rows. The schedule is
    # Friday so the eval data has had a full work-week to accumulate.
    research_trigger = CronTrigger(
        day_of_week="fri",
        hour=9,
        minute=0,
        timezone=tz,
    )
    scheduler.add_job(
        run_research_job,
        trigger=research_trigger,
        id="weekly_research",
        name="Weekly auto-research + Discord report",
        misfire_grace_time=1800,  # 30 min grace — research is non-urgent
    )
    log.info("job_registered", session="weekly_research", time="09:00", days="Fri")

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
