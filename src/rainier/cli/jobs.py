"""Scheduler/ops commands: jobs, alert, run, recover."""

from __future__ import annotations

from pathlib import Path

import click

from rainier.cli import (
    _get_discord_webhook,
    cli,
)


@cli.group()
def jobs():
    """Manage scheduled jobs (config/cron.yaml → system crontab)."""
    pass


@jobs.command(name="list")
def jobs_list():
    """Show all jobs from cron.yaml and their crontab status."""
    from rainier.scheduler.jobs import list_active, load_config

    config_jobs = load_config()
    active = {j["name"] for j in list_active()}

    for job in config_jobs:
        name = job["name"]
        enabled = job.get("enabled", True)
        status = "ACTIVE" if name in active else ("DISABLED" if not enabled else "NOT SYNCED")
        click.echo(f"  {name:20s} {job['schedule']:20s} {status}")
        click.echo(f"    {job.get('description', '')}")
        click.echo(f"    cmd: {job['command']}")
        click.echo()


@jobs.command(name="sync")
def jobs_sync():
    """Sync cron.yaml jobs to system crontab."""
    from rainier.scheduler.jobs import sync

    actions = sync(project_dir=Path.cwd())
    for name, action in actions.items():
        click.echo(f"  {name}: {action}")
    click.echo("Done.")


@jobs.command(name="stop")
@click.option("--name", required=True, help="Job name to remove from crontab")
def jobs_stop(name):
    """Remove a job from system crontab (keeps it in cron.yaml)."""
    from rainier.scheduler.jobs import _remove_job, list_active

    active = {j["name"] for j in list_active()}
    if name not in active:
        click.echo(f"Job '{name}' not found in crontab.")
        return
    _remove_job(name)
    click.echo(f"Removed '{name}' from crontab.")


# ---------------------------------------------------------------------------
# Alert commands
# ---------------------------------------------------------------------------


@cli.group()
def alert():
    """Send alerts manually (Discord, etc.)."""


@alert.command(name="discord")
@click.option("--dry-run", is_flag=True, default=False, help="Format and print without sending")
@click.option("--top-n", default=20, help="Max candidates to include")
@click.pass_context
def alert_discord(ctx, dry_run, top_n):
    """Send latest QU100 screening results to Discord."""
    from rainier.alerts.discord import (
        _build_payloads,
        format_stock_candidates_json,
    )
    from rainier.analysis.stock_screener import screen_stocks

    settings = ctx.obj["settings"]

    click.echo("Running QU100 stock screener (3-layer pipeline)...")
    candidates, _ohlcv = screen_stocks(settings)
    candidates = candidates[:top_n]

    if not candidates:
        click.echo("No candidates found from screener.")
        return

    # Show summary
    with_pattern = sum(1 for c in candidates if c.pattern_type)
    click.echo(
        f"Screener returned {len(candidates)} candidates "
        f"({with_pattern} with pattern match)"
    )

    if dry_run:
        click.echo(format_stock_candidates_json(candidates))
        click.echo(f"\n({len(candidates)} candidates formatted, not sent)")
        return

    webhook = _get_discord_webhook(settings)
    if not webhook:
        click.echo("No Discord webhook configured")
        return

    import httpx
    payloads = _build_payloads(candidates)
    for payload in payloads:
        resp = httpx.post(webhook, json=payload, timeout=10)
        resp.raise_for_status()
    click.echo(f"Sent {len(candidates)} candidates to Discord.")




@cli.command(name="run")
@click.option(
    "--once", default=None,
    type=click.Choice(["morning", "midday", "afternoon", "close"]),
    help="Run a single scrape immediately instead of starting the scheduler",
)
@click.pass_context
def run_scheduler(ctx, once):
    """Start the scraper scheduler (long-running service)."""
    import asyncio

    if once:
        click.echo(f"Running one-off scrape: {once}")
        asyncio.run(_run_once(once))
    else:
        click.echo("Starting Rainier scheduler (Ctrl+C to stop)...")
        asyncio.run(_start_scheduler())


async def _start_scheduler():
    from rainier.scheduler.service import start_scheduler
    await start_scheduler()


async def _run_once(session):
    from rainier.scheduler.service import run_qu_scrape
    await run_qu_scrape(session)




@cli.command()
@click.option("--dry-run", is_flag=True, default=False, help="Show what would be done without doing it")
@click.pass_context
def recover(ctx, dry_run):
    """Recover after a restart: check services and re-run missed jobs."""
    import asyncio

    import rainier.cli as cli_root
    from rainier.scheduler.recovery import run_recovery

    asyncio.run(
        run_recovery(
            ctx.obj["settings"],
            dry_run,
            run_qu_scrape=cli_root._run_qu_scrape,
            run_post_scrape_pipeline=cli_root.run_post_scrape_pipeline,
            get_discord_webhook=cli_root._get_discord_webhook,
            send_discord_embeds=cli_root._send_discord_embeds,
        )
    )

