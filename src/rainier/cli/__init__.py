"""CLI interface: rainier — trading analysis platform."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import click

from rainier.core.config import load_settings
from rainier.pipeline.post_scrape import run_post_scrape_pipeline


@click.group()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
@click.pass_context
def cli(ctx, config_path):
    """Rainier — trading analysis platform."""
    ctx.ensure_object(dict)
    path = Path(config_path) if config_path else Path("config/settings.yaml")
    ctx.obj["settings"] = load_settings(path)
    # Codex P1: thesis subcommands need to honor `--config staging.yaml` so
    # they don't silently fall back to config/settings.yaml in non-default
    # environments. We stash the resolved path here; load_settings_fresh()
    # call sites pull it from ctx.obj via _settings_path(ctx).
    ctx.obj["settings_path"] = str(path)


def _settings_path(ctx) -> str:
    """Return the YAML path the user selected at the CLI root.

    Defaults to `config/settings.yaml` when ctx.obj is missing or empty
    (callsites that build a click.Context just for unit tests).
    """
    obj = getattr(ctx, "obj", None) or {}
    return obj.get("settings_path") or "config/settings.yaml"


@contextmanager
def _legacy_db_for_config(ctx):
    """Point the legacy ``get_session()``/``get_engine()`` at the DB named by the
    root ``--config`` for the duration of the block.

    Commands that write via the legacy ``core.database`` global session factory
    otherwise ignore ``--config`` and hit whatever ``config/settings.yaml``
    resolves to (codex P1 — same failure the backfill-screened-levels command
    fixes). Seed the process settings singleton from the operator-selected YAML
    and reset the cached engine/session factory so they rebind, then RESTORE all
    three process globals on exit so an in-process caller (CliRunner /
    programmatic reuse) does not silently inherit this command's DB.
    """
    from rainier.core import config as _config_mod
    from rainier.core import database as _database_mod
    from rainier.core.config import load_settings_fresh

    _prev_settings = _config_mod._settings
    _prev_engine = _database_mod._engine
    _prev_factory = _database_mod._session_factory

    _config_mod._settings = load_settings_fresh(_settings_path(ctx))
    _database_mod._engine = None
    _database_mod._session_factory = None
    try:
        yield
    finally:
        # Dispose the engine created INSIDE the block (if any) before restoring
        # the prior one — otherwise its pooled connections leak on every
        # invocation in the in-process/CliRunner reuse path this helper exists
        # to protect. `_engine` is still None here if no session was opened.
        _new_engine = _database_mod._engine
        _config_mod._settings = _prev_settings
        _database_mod._engine = _prev_engine
        _database_mod._session_factory = _prev_factory
        if _new_engine is not None and _new_engine is not _prev_engine:
            _new_engine.dispose()



def _get_discord_webhook(settings) -> str | None:
    """Get Discord webhook URL from settings (stock/scrape alerts)."""
    return settings.discord_stock_webhook_url or settings.discord_webhook_url or None


def _get_discord_backtest_webhook(settings) -> str | None:
    """Get Discord webhook URL for backtest notifications."""
    return settings.discord_backtest_webhook_url or _get_discord_webhook(settings)


def _send_discord_embeds(webhook: str, embeds: list[dict]) -> None:
    """Send embeds to Discord, splitting into batches of 10."""
    import httpx

    for i in range(0, len(embeds), 10):
        batch = embeds[i : i + 10]
        resp = httpx.post(webhook, json={"embeds": batch}, timeout=10)
        resp.raise_for_status()




async def _run_qu_scrape(session, detail_top, dates, days_back, start_date, delay, headed, cdp):
    import asyncio
    from datetime import timedelta

    from rainier.core.config import get_settings
    from rainier.scrapers import get_scraper
    from rainier.scrapers.browser import BrowserManager

    settings = get_settings()

    # Override backfill delay if --delay is specified
    if delay is not None:
        settings.scraping.quantunicorn.backfill_delay_seconds = delay

    date_list = None
    if dates:
        date_list = [d.strip() for d in dates.split(",")]
    elif start_date:
        import exchange_calendars as xcals
        nyse = xcals.get_calendar("XNYS")
        start = date.fromisoformat(start_date)
        end = date.today() - timedelta(days=1)
        sessions = nyse.sessions_in_range(start.isoformat(), end.isoformat())
        date_list = [s.date().isoformat() for s in sessions]
    elif days_back > 0:
        import exchange_calendars as xcals
        nyse = xcals.get_calendar("XNYS")
        end = date.today() - timedelta(days=1)
        # Go back far enough to find N trading days
        start = end - timedelta(days=int(days_back * 1.6))
        sessions = nyse.sessions_in_range(
            start.isoformat(), end.isoformat(),
        )
        date_list = [
            s.date().isoformat() for s in sessions[-days_back:]
        ]

    if date_list:
        effective_delay = settings.scraping.quantunicorn.backfill_delay_seconds
        est_minutes = len(date_list) * effective_delay / 60
        click.echo(
            f"Scraping {len(date_list)} dates "
            f"(~{est_minutes:.0f} min at {effective_delay}s/date)"
        )
        click.echo(f"  First: {date_list[0]}, Last: {date_list[-1]}")

    try:
        async with BrowserManager(headless=not headed, cdp_url=cdp) as browser:
            scraper = get_scraper("qu", browser)
            result = await scraper.execute(
                session=session, top_n=detail_top, dates=date_list,
            )

        click.echo(f"Scrape complete: {result.records_created} records created")
        if result.errors:
            for err in result.errors:
                click.echo(f"  - {err}")
        if result.duration_seconds is not None:
            click.echo(f"  Duration: {result.duration_seconds:.1f}s")

        # Notify Discord on success with errors
        if result.errors:
            _notify_scrape_discord(
                settings, session,
                title=f"QU100 Scrape Warning ({session})",
                message=(
                    f"Scraped {result.records_created} records "
                    f"with {len(result.errors)} error(s):\n"
                    + "\n".join(f"- {e}" for e in result.errors[:5])
                ),
                color=0xFFA500,  # orange
            )

        # Post-scrape: delegate to the shared pipeline so the cron CLI path
        # emits the same Discord output as the long-running ``rainier run``
        # scheduler — including the LLM thesis embeds for sessions in
        # settings.llm_thesis.enabled_sessions. Skip during backfill runs
        # (date_list is set) since those replay historical scans and would
        # spam the channel. ``recover`` also pins a date (single-day replay) but
        # drives the pipeline ITSELF, gated on restored two-book freshness — see
        # _recover — so a stale half-book can't fire derived output here.
        #
        # Hand off via asyncio.to_thread because compute_theses_and_persist
        # wraps its async work in asyncio.run(); calling it directly from this
        # already-running event loop raises RuntimeError. The shared pipeline
        # is sync (mirrors the scheduler hand-off in scheduler/service.py).
        #
        # Wrap in try/except so a screener / pipeline failure POST scrape
        # surfaces as a partial-success warning rather than morphing the
        # already-successful scrape into a red "Scrape FAILED" alert. The
        # outer except is for actual scrape errors only.
        if result.records_created > 0 and not date_list:
            try:
                await asyncio.to_thread(
                    run_post_scrape_pipeline, settings, session,
                )
            except Exception as pipeline_exc:
                err_msg = str(pipeline_exc)
                if len(err_msg) > 400:
                    err_msg = err_msg[:400] + "..."
                click.echo(
                    f"  Post-scrape pipeline failed: {err_msg}", err=True,
                )
                _notify_scrape_discord(
                    settings, session,
                    title=f"QU100 Post-Scrape FAILED ({session})",
                    message=(
                        f"Scrape succeeded but post-scrape pipeline failed:\n"
                        f"{err_msg}"
                    ),
                    color=0xFFA500,  # orange — partial success
                )

    except Exception as exc:
        error_msg = str(exc)
        # Truncate long playwright tracebacks
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."

        click.echo(f"Scrape FAILED: {error_msg}", err=True)

        # Send failure alert to Discord
        _notify_scrape_discord(
            settings, session,
            title=f"QU100 Scrape FAILED ({session})",
            message=(
                f"**Session:** {session}\n"
                f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M PT')}\n"
                f"**Error:** {error_msg}"
            ),
            color=0xFF1744,  # red
        )
        raise


def _notify_scrape_discord(
    settings, session: str, title: str, message: str, color: int,
) -> None:
    """Send a scrape status notification to Discord."""
    import httpx

    webhook = _get_discord_webhook(settings)
    if not webhook:
        click.echo("  (no Discord webhook configured, skipping notification)")
        return

    embed = {
        "title": title,
        "description": message,
        "color": color,
    }
    try:
        resp = httpx.post(webhook, json={"embeds": [embed]}, timeout=10)
        resp.raise_for_status()
        click.echo(f"  Discord notification sent: {title}")
    except Exception as notify_exc:
        click.echo(f"  Failed to send Discord notification: {notify_exc}")



# Import command-group submodules so their commands register on the root
# ``cli`` group (they attach via decorators at import time). Keep these at the
# bottom: the submodules import ``cli`` and the shared helpers defined above.
from rainier.cli import (  # noqa: E402, F401
    dashboard,
    db,
    futures,
    jobs,
    ml,
    paper,
    qu100,
    thematic,
    thesis,
)

# Backwards-compatible re-exports: tests and external callers historically
# reached these through ``rainier.cli``.
from rainier.cli.db import _resolve_alembic_config  # noqa: E402, F401
from rainier.cli.thematic import (  # noqa: E402, F401
    _check_ohlcv_freshness,
    _dual_write_benchmark_pg,
    _dual_write_breadth_pg,
    _dual_write_features_pg,
    _dual_write_labels_pg,
    _frame_to_pg_rows,
)
from rainier.cli.thesis import _FAKE_THESIS_VERDICTS, _mask_webhook_url  # noqa: E402, F401
from rainier.scheduler.recovery import (  # noqa: E402, F401
    _is_qu100_fresh,
    _latest_due_session,
    _latest_due_slot,
    _qu100_day_is_fresh,
    _recover_trading_day,
)

__all__ = ["cli"]
