"""Post-restart recovery: service checks and missed-scrape re-runs.

Extracted verbatim from ``rainier.cli`` (the ``recover`` command). The CLI
remains the composition root: the scrape/pipeline/Discord callables are
injected by the thin ``rainier recover`` wrapper rather than imported here.
"""

from __future__ import annotations

from datetime import date, datetime

import click

RECOVER_WEBHOOK = (
    "https://discord.com/api/webhooks/1486760877867794503/"
    "A7-DOUrsQMmJfzxaZ2GRlqIPSJHrA3KXRxTUXLFOc4K4_cNRZVnwPdwdnBLPvAIkVSAk"
)


def _notify_recover(title: str, description: str, color: int = 0x3498DB):
    """Send a recovery event notification to Discord."""
    import httpx

    payload = {
        "embeds": [{
            "title": f"🔧 Recovery: {title}",
            "description": description,
            "color": color,
            "timestamp": datetime.now().astimezone().isoformat(),
        }],
    }
    try:
        httpx.post(RECOVER_WEBHOOK, json=payload, timeout=10)
    except Exception:
        pass  # Don't let notification failures block recovery


# Ranking types the QU scraper persists every slot. The day is only "fresh" when
# BOTH books are at/after the latest due slot — a partial scrape (e.g. top100 lands
# but bottom100 returns an empty no-op) must read as STALE so recover re-fires.
_QU100_RANKING_TYPES = ("top100", "bottom100")

# Minimum rank coverage for a book to count as "fresh". QU100 is exactly 100 ranks
# and a full scrape overrides every slot (count == 100); a glitched non-empty
# PARTIAL scrape (a handful of ranks, nothing earlier that day to carry forward)
# lands far below this floor, so it reads STALE and recover re-fires rather than
# treating a near-empty book as a fresh full scrape. 90 = a strong-majority cohort,
# above any realistic dedup-unfill (a moved symbol leaves at most a few slots
# unfilled) yet well clear of a partial response.
_QU100_MIN_FRESH_RANKS = 90


def _recover_trading_day(now: datetime) -> date:
    """Anchor recover's "today" to the APP-LOCAL calendar date of ``now`` — the
    same timezone the schedule slots fire in.

    Recover must compare today's snapshots against the schedule, and the schedule
    slots are app-tz (the scheduler runs ``AsyncIOScheduler(timezone=app.tz)``).
    The day key MUST therefore be the app-local date so the latest-due-slot scan
    (``_latest_due_slot``) and the snapshot day-filter agree.

    This also matches the STORED ``data_date = market_date(captured_at)`` (ET) for
    every legitimately-scraped slot: scrapes fire only during US market hours,
    when the ET calendar date equals the app-local date (market hours are daytime
    in any reasonable app tz). So a close scrape at 1pm PT stores ``data_date`` =
    that Monday (ET) and an evening recover at 9pm PT resolves ``today`` = the same
    Monday (app-local) — they match.

    Using the ET date here (``market_date(now)``) instead would BREAK late-evening
    recovery: after ~9pm PT the ET clock has rolled to the next calendar day, so
    ``today`` would point at a not-yet-traded date while the latest due slot still
    belongs to the day that just ended — a missed Monday close would be checked
    against Tuesday, and on Friday night it would fall into the weekend fast-path
    and skip recovery entirely.
    """
    return now.date()


def _latest_due_slot(schedule: dict, now):
    """Return ``(name, slot_time)`` of the most-recent scheduled slot already due
    at ``now`` (``slot_time <= now``), or ``(None, None)`` when none is due yet.

    ``schedule`` maps slot name -> ``"HH:MM"`` in the app timezone; ``now`` is a
    tz-aware datetime in that zone. Single source of truth for "which slot should
    have run by now" — both the freshness check and the recovery-scrape label read
    from it so they can never drift.
    """
    latest_name = None
    latest_time = None
    for name, time_str in schedule.items():
        hour, minute = map(int, time_str.split(":"))
        slot_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if slot_time <= now and (latest_time is None or slot_time > latest_time):
            latest_time = slot_time
            latest_name = name
    return latest_name, latest_time


def _is_qu100_fresh(latest_captured_at, now, schedule: dict, tz) -> bool:
    """Is ONE book's ``captured_at`` present and fresh as of the latest due slot?

    Under the rebuild-the-day fix a day holds ONE ``captured_at`` per
    ``(data_date, ranking_type)`` and rows carry the LATEST scrape's
    ``capture_session`` — so the old per-``capture_session`` row count can no
    longer tell which slot ran. Freshness is detected PER DAY via ``captured_at``:

      * if no slot is due yet, nothing was expected -> fresh (no scrape needed);
      * otherwise this book is fresh iff a snapshot exists AND its ``captured_at``
        is at/after the most-recent-due slot.

    This is the per-ranking_type primitive; ``_qu100_day_is_fresh`` requires it to
    hold for EVERY book so a partial scrape never reads as fresh. ``now`` and
    ``latest_captured_at`` are tz-aware datetimes.
    """
    _, latest_due = _latest_due_slot(schedule, now)
    if latest_due is None:
        return True  # nothing due yet today
    if latest_captured_at is None:
        return False  # a slot is due but no data landed
    # captured_at is a timestamptz on Postgres (aware); a naive value (e.g. a
    # SQLite-backed read) is assumed UTC so `.astimezone` never reinterprets it as
    # local wall-clock. Compare in the app timezone.
    if latest_captured_at.tzinfo is None:
        from datetime import timezone as _timezone

        latest_captured_at = latest_captured_at.replace(tzinfo=_timezone.utc)
    return latest_captured_at.astimezone(tz) >= latest_due


def _qu100_day_is_fresh(db, today, now, schedule: dict, tz) -> bool:
    """Is today's QU100 snapshot fresh for EVERY ranking_type?

    Reads ``max(captured_at)`` AND the row count per ranking_type for
    ``data_date == today`` and requires each expected book (``top100`` AND
    ``bottom100``) to be both timely AND adequately covered:

      * TIMELY — its ``captured_at`` is at/after the latest due slot (a single
        global ``max(captured_at)`` would let a fresh top100 mask a stale/empty
        bottom100 — a partial-failure scrape — and report the day fresh when half
        the book is frozen);
      * COVERED — once a slot is due, the book holds at least
        ``_QU100_MIN_FRESH_RANKS`` rows. Under the rebuild fix a day's whole book
        shares one ``captured_at``, so a non-empty PARTIAL first scrape (a handful
        of ranks, nothing earlier to carry forward) would otherwise advance
        ``captured_at`` and read as fully fresh even though the book is mostly
        missing (Codex P1). A full scrape is exactly 100 ranks; a glitched partial
        is well below the floor, so recover re-fires instead of running the
        screener off a near-empty book. (Coverage is NOT required before any slot
        is due — there may legitimately be no data yet.)

    Used both at detection AND post-scrape to gate the Discord report.
    """
    from sqlalchemy import func

    from rainier.core.models import MoneyFlowSnapshot

    by_type = {
        rt: (max_cap, count)
        for rt, max_cap, count in db.query(
            MoneyFlowSnapshot.ranking_type,
            func.max(MoneyFlowSnapshot.captured_at),
            func.count(),
        )
        .filter(MoneyFlowSnapshot.data_date == today)
        .group_by(MoneyFlowSnapshot.ranking_type)
        .all()
    }
    _, latest_due = _latest_due_slot(schedule, now)

    def _book_fresh(rt: str) -> bool:
        max_cap, count = by_type.get(rt, (None, 0))
        if not _is_qu100_fresh(max_cap, now, schedule, tz):
            return False
        # Timely. Demand rank coverage too — but only once a slot is due (before
        # that, "nothing expected" is fresh regardless of coverage).
        if latest_due is None:
            return True
        return count >= _QU100_MIN_FRESH_RANKS

    return all(_book_fresh(rt) for rt in _QU100_RANKING_TYPES)


def _latest_due_session(schedule: dict, now) -> str:
    """The session name of the most-recent scheduled slot already due at ``now``.

    Used to label the single recovery scrape when today's data is stale. Falls
    back to the first slot if (defensively) none is due — the caller only invokes
    this when a slot IS due, so the fallback is just a safety net.
    """
    name, _ = _latest_due_slot(schedule, now)
    return name if name is not None else next(iter(schedule))


async def run_recovery(
    settings,
    dry_run: bool,
    *,
    run_qu_scrape,
    run_post_scrape_pipeline,
    get_discord_webhook,
    send_discord_embeds,
):
    """Check Chrome CDP, scheduler, and missed scrape sessions."""
    import subprocess
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import httpx

    tz = ZoneInfo(settings.app.timezone)
    now = datetime.now(tz)
    issues = []
    actions = []

    # --- 1. Check Chrome CDP ---
    click.echo("Checking Chrome CDP...")
    cdp_ok = False
    try:
        resp = httpx.get("http://127.0.0.1:9222/json/version", timeout=3)
        cdp_ok = resp.status_code == 200
    except Exception:
        pass

    if cdp_ok:
        click.echo("  Chrome CDP: running")
    else:
        issues.append("Chrome CDP not running")
        actions.append("start_cdp")
        click.echo("  Chrome CDP: DOWN")

    # --- 2. Check scheduler ---
    click.echo("Checking scheduler...")
    result = subprocess.run(
        ["pgrep", "-f", "rainier run"], capture_output=True, text=True,
    )
    scheduler_ok = result.returncode == 0

    if scheduler_ok:
        click.echo("  Scheduler: running")
    else:
        issues.append("Scheduler not running")
        actions.append("start_scheduler")
        click.echo("  Scheduler: DOWN")

    # --- 3. Check missed scrape sessions ---
    click.echo("Checking missed scrape sessions...")
    schedule = settings.scraping.schedule
    sessions_config = {
        "morning": schedule.morning,
        "midday": schedule.midday,
        "afternoon": schedule.afternoon,
        "close": schedule.close,
    }

    # Detect today's QU100 freshness via the latest captured_at vs the schedule
    # (per-DAY, not per-capture_session — under the rebuild-the-day fix a day
    # holds one captured_at and the rows carry the LATEST scrape's session, so
    # per-session counts can no longer tell which slot ran). ``qu100_stale`` drives
    # both the recovery scrape AND whether the daily outlook is re-sent.
    qu100_stale = False
    # Anchor the day key to the APP-LOCAL date (the schedule's timezone) so the
    # latest-due-slot scan and the snapshot day-filter agree, and so late-evening
    # recovery still targets the day that just ended — see _recover_trading_day.
    today = _recover_trading_day(now)
    if today.weekday() >= 5:
        click.echo("  Weekend — no scrape sessions to check")
    else:
        from rainier.core.database import get_session

        with get_session() as db:
            day_fresh = _qu100_day_is_fresh(db, today, now, sessions_config, tz)

        if day_fresh:
            click.echo("  QU100 today: fresh (every book at/after the latest due slot)")
        else:
            qu100_stale = True
            issues.append("QU100 data stale (a book is missing today's latest slot)")
            # One recovery scrape targeting the most-recent due slot's session.
            recover_session = _latest_due_session(sessions_config, now)
            actions.append(f"scrape_{recover_session}")
            click.echo("  QU100 today: STALE — re-scraping the latest due slot")

    # --- 4. QU100 Discord report ---
    # Decoupled from scrape-action queueing: the daily outlook is re-sent ONLY
    # when freshness is RESTORED (re-read post-scrape, per ranking_type — see the
    # execution stage). A scrape that returns without raising but lands no fresh
    # data must not fire a report off a stale snapshot — that would reintroduce the
    # frozen-data bug. So we do NOT queue a "discord_report" action here; the
    # decision is made post-scrape from the DB, not from "the coroutine returned".
    click.echo("Checking QU100 Discord report...")
    if qu100_stale:
        click.echo("  Discord report: deferred — gated on restored freshness")
    else:
        click.echo("  Discord report: likely OK (data fresh)")

    # --- Summary ---
    click.echo()
    if not issues:
        click.echo("All systems healthy — nothing to recover.")
        _notify_recover(
            "Health Check — All Clear",
            "All services running, no missed jobs.",
            color=0x2ECC71,
        )
        return

    click.echo(f"Found {len(issues)} issue(s):")
    for issue in issues:
        click.echo(f"  - {issue}")
    click.echo()

    if dry_run:
        click.echo("Dry run — would take these actions:")
        for action in actions:
            click.echo(f"  - {action}")
        return

    # --- Execute recovery ---
    click.echo("Recovering...")
    _notify_recover(
        "Recovery Started",
        "Issues detected:\n" + "\n".join(f"• {i}" for i in issues),
        color=0xE67E22,
    )

    uid = subprocess.getoutput("id -u")

    if "start_cdp" in actions:
        click.echo("  Starting Chrome CDP via launchd...")
        subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{uid}/com.rainier.chrome-cdp"],
            capture_output=True,
        )
        # Wait for CDP to be ready
        import asyncio
        for _ in range(15):
            await asyncio.sleep(2)
            try:
                resp = httpx.get("http://127.0.0.1:9222/json/version", timeout=2)
                if resp.status_code == 200:
                    click.echo("  Chrome CDP: started")
                    _notify_recover("Chrome CDP", "Started successfully", color=0x2ECC71)
                    break
            except Exception:
                pass
        else:
            click.echo("  Chrome CDP: FAILED to start — skipping scrapes")
            _notify_recover("Chrome CDP", "FAILED to start — aborting recovery", color=0xE74C3C)
            return

    if "start_scheduler" in actions:
        click.echo("  Starting scheduler via launchd...")
        subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{uid}/com.rainier.scheduler"],
            capture_output=True,
        )
        click.echo("  Scheduler: restarted")
        _notify_recover("Scheduler", "Restarted via launchd", color=0x2ECC71)

    # Re-run missed scrapes via CDP (Chrome is already running)
    recovered_scrapes = []
    for action in actions:
        if action.startswith("scrape_"):
            session_name = action.replace("scrape_", "")
            click.echo(f"  Running missed {session_name} scrape...")
            try:
                # Pin the scrape to the STALE trading day (`today`), not the
                # scraper's own clock-derived date. With dates=None the QU scraper
                # derives the API/persist date from market_date(captured_at), which
                # in the late-evening window (after ET midnight) resolves to the
                # NEXT, not-yet-traded day — so the rerun would fetch the wrong day
                # and never refresh `today`, leaving _qu100_day_is_fresh(db, today)
                # false (Codex P1). Passing the explicit date makes the scraper
                # query and persist `data_date = today`. The pinned date also keeps
                # _run_qu_scrape's INLINE post-scrape pipeline OFF — recover runs
                # the pipeline itself below, gated on RESTORED two-book freshness,
                # so a stale half-book never fires screener/LLM/candidate output.
                await run_qu_scrape(
                    session=session_name,
                    detail_top=0,
                    dates=today.isoformat(),
                    days_back=0,
                    start_date=None,
                    delay=None,
                    headed=False,
                    cdp="http://127.0.0.1:9222",
                )
                click.echo(f"  {session_name} scrape: done")
                recovered_scrapes.append(session_name)
                _notify_recover(
                    f"Scrape: {session_name}",
                    f"Re-ran missed {session_name} scrape successfully",
                    color=0x2ECC71,
                )
            except Exception as exc:
                click.echo(f"  {session_name} scrape: FAILED ({exc})")
                _notify_recover(
                    f"Scrape: {session_name}",
                    f"FAILED: {exc}",
                    color=0xE74C3C,
                )

    # Re-send the daily outlook ONLY if the day was stale AND freshness was
    # actually RESTORED this run. We re-read the DB rather than trust "the scrape
    # coroutine returned": an empty/partial scrape (the documented empty-slot slip
    # -> _persist_qu100 no-op) returns without raising yet leaves the snapshot
    # stale. Gating on a post-scrape freshness re-read (per ranking_type) is the
    # only signal that won't fire a report off a frozen snapshot — the exact bug
    # this PR fixes. Data already fresh -> treated as already sent.
    report_sent = False
    freshness_restored = False
    if qu100_stale and recovered_scrapes:
        from datetime import time

        from rainier.core.database import get_session

        # Re-evaluate against the CURRENT clock, but CLAMPED to the day we set out
        # to recover (`today`). A recover that crossed a slot boundary (15:29 ->
        # 15:31) must judge against the now-overdue 15:30 slot, so the clock must
        # advance. But if the rerun crosses local MIDNIGHT, `now_after` lands on a
        # NEW day where no slot is due yet -> `_latest_due_slot` returns None ->
        # `_qu100_day_is_fresh` returns True and we'd resend the outlook off the
        # still-stale prior day (Codex). Clamp the clock to end-of-`today` so the
        # recovered day is always judged against its OWN last due slot, never a
        # fresh next-day clock. The day is fixed; the clock advances within it only.
        now_after = datetime.now(tz)
        if _recover_trading_day(now_after) > today:
            now_after = datetime.combine(today, time.max, tzinfo=tz)
        with get_session() as db:
            freshness_restored = _qu100_day_is_fresh(
                db, today, now_after, sessions_config, tz
            )
        if not freshness_restored:
            click.echo(
                "  Discord report: skipped — recovery scrape landed no fresh data"
            )
    # Restore the DERIVED outputs a scheduled scrape would have produced
    # (screened_stocks, LLM theses, the top-20 candidate alert) — but ONLY now
    # that BOTH books are fresh. Running this gated on freshness_restored (not
    # inline in the recovery scrape) is what prevents a stale half-book from
    # firing screener/LLM/candidate output (Codex P1). run_post_scrape_pipeline is
    # sync and internally does asyncio.run, so hand it off via asyncio.to_thread
    # (same reason as _run_qu_scrape) to avoid a nested-event-loop RuntimeError.
    if qu100_stale and freshness_restored:
        import asyncio

        click.echo("  Restoring post-scrape pipeline (screener/LLM/candidates)...")
        try:
            # Stamp the derived artifacts with the RECOVERED trading day (`today`),
            # not date.today(): a post-midnight / cross-timezone replay must label
            # ScreenedStockRecord / thesis rows with the day the data is from
            # (the pinned scrape data_date), not the wall-clock run date (Codex P1).
            await asyncio.to_thread(
                run_post_scrape_pipeline, settings, recover_session, today
            )
            click.echo("  Post-scrape pipeline: done")
        except Exception as pipeline_exc:
            click.echo(
                f"  Post-scrape pipeline: FAILED ({pipeline_exc})", err=True
            )
            _notify_recover(
                "Post-Scrape Pipeline",
                f"FAILED during recovery: {pipeline_exc}",
                color=0xE74C3C,
            )

    if qu100_stale and freshness_restored:
        click.echo("  Sending QU100 Discord report...")
        try:
            from rainier.backtest.qu100_backtest import (
                format_discord_report,
                run_qu100_backtest,
            )
            result = run_qu100_backtest()
            webhook = get_discord_webhook(settings)
            if webhook:
                embeds = format_discord_report(result)
                send_discord_embeds(webhook, embeds)
                click.echo("  Discord report: sent")
                report_sent = True
                _notify_recover("QU100 Report", "Sent to Discord", color=0x2ECC71)
            else:
                click.echo("  Discord report: no webhook configured")
        except Exception as exc:
            click.echo(f"  Discord report: FAILED ({exc})")
            _notify_recover("QU100 Report", f"FAILED: {exc}", color=0xE74C3C)

    # --- Final summary ---
    summary_parts = []
    if "start_cdp" in actions:
        summary_parts.append("Chrome CDP restarted")
    if "start_scheduler" in actions:
        summary_parts.append("Scheduler restarted")
    if recovered_scrapes:
        summary_parts.append(f"Scrapes recovered: {', '.join(recovered_scrapes)}")
    if report_sent:
        summary_parts.append("QU100 report re-sent")

    _notify_recover(
        "Recovery Complete",
        "\n".join(f"✓ {p}" for p in summary_parts),
        color=0x2ECC71,
    )

    click.echo()
    click.echo("Recovery complete.")

