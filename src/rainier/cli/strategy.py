"""FNG-TQQQ strategy signal and backtest commands."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import click
import exchange_calendars as xcals
import pandas as pd

from rainier.cli import _legacy_db_for_config, cli
from rainier.data.prices import load_daily_close, warmup_start
from rainier.pipeline.fng_tqqq_signal import (
    StaleInputError,
    preview_daily_signal,
    run_daily_signal,
    run_stop_check,
)


@cli.group()
def strategy() -> None:
    """Trading strategies without broker execution."""


@strategy.group(name="fng-tqqq")
def fng_tqqq() -> None:
    """CNN Fear & Greed / QQQ signal strategy for TQQQ."""


@fng_tqqq.command(name="signal")
@click.option("--dry-run", is_flag=True, help="Read inputs and print without writes or Discord")
@click.option("--no-discord", is_flag=True, help="Persist the signal without posting to Discord")
@click.pass_context
def signal(ctx, dry_run: bool, no_discord: bool) -> None:
    """Evaluate and persist today's FNG-TQQQ signal."""
    from rainier.alerts.discord import send_strategy_signal
    from rainier.core.database import get_session

    settings = ctx.obj["settings"]
    cfg = settings.strategies.fng_tqqq
    calendar = xcals.get_calendar("XNYS")
    today = date.today()

    def poster(day, strategy_cfg, next_session):
        if not no_discord:
            posted = send_strategy_signal(
                day,
                strategy_cfg,
                next_session,
                settings.alerts.discord,
            )
            if not posted:
                raise click.ClickException(
                    "Discord strategy webhook not configured/enabled; "
                    f"signal {day.action} for {day.date.isoformat()} was persisted "
                    "but not posted"
                )

    try:
        with _legacy_db_for_config(ctx):
            session_factory = get_session
            if dry_run:
                prepared = preview_daily_signal(
                    today=today,
                    cfg=cfg,
                    session_factory=session_factory,
                    price_loader=load_daily_close,
                    calendar=calendar,
                )
                if prepared is None:
                    click.echo(f"No XNYS session on {today.isoformat()}")
                    return
                day, next_session = prepared
                click.echo(
                    f"{day.date.isoformat()} action={day.action} "
                    f"fg={day.fg:.2f} qqq={day.qqq_close:.2f} "
                    f"sma{cfg.sma_window}={day.qqq_sma:.2f} "
                    f"holding={day.holding} armed={day.armed}; "
                    f"next_session={next_session.isoformat()} "
                    "(dry-run: no write or Discord post)"
                )
                return
            day = run_daily_signal(
                today=today,
                cfg=cfg,
                session_factory=session_factory,
                price_loader=load_daily_close,
                poster=poster,
                calendar=calendar,
            )
    except StaleInputError as exc:
        raise click.ClickException(str(exc)) from exc

    if day is None:
        click.echo(f"No XNYS session on {today.isoformat()}")
    else:
        click.echo(f"{day.date.isoformat()} action={day.action}")


@fng_tqqq.command(name="backtest")
@click.option("--start", default=None, help="Replay start date (YYYY-MM-DD)")
@click.option("--buy-below", default=None, type=float, help="F&G BUY threshold")
@click.option("--sma-window", default=None, type=int, help="QQQ SMA window")
@click.option(
    "--regime-sma",
    default=None,
    type=int,
    help="QQQ bear-market SMA window (0 disables)",
)
@click.option(
    "--stop-loss",
    default=None,
    type=float,
    help="Same-day stop percentage (0 disables)",
)
@click.pass_context
def backtest(
    ctx,
    start: str | None,
    buy_below: float | None,
    sma_window: int | None,
    regime_sma: int | None,
    stop_loss: float | None,
) -> None:
    """Replay FNG-TQQQ strategy against adjusted QQQ and TQQQ closes."""
    from rainier.core.database import get_session
    from rainier.data.fear_greed import load_daily_scores
    from rainier.signals.fng_tqqq import evaluate, replay

    cfg = ctx.obj["settings"].strategies.fng_tqqq
    updates = {}
    if start is not None:
        try:
            updates["start"] = date.fromisoformat(start)
        except ValueError as exc:
            raise click.BadParameter("must be YYYY-MM-DD", param_hint="--start") from exc
    if buy_below is not None:
        updates["buy_below"] = buy_below
    if sma_window is not None:
        updates["sma_window"] = sma_window
    if regime_sma is not None:
        updates["regime_sma_window"] = regime_sma or None
    if stop_loss is not None:
        updates["stop_loss_pct"] = stop_loss or None
    cfg = cfg.model_copy(update=updates)

    try:
        with _legacy_db_for_config(ctx):
            fg = load_daily_scores(session_factory=get_session)
            sessions = max(cfg.sma_window, cfg.regime_sma_window or 0) + 20
            price_start = warmup_start(cfg.start, sessions)
            qqq_close = load_daily_close("QQQ", price_start)
            tqqq_close = load_daily_close("TQQQ", price_start)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    days = evaluate(fg, qqq_close, tqqq_close, cfg)
    if not days:
        raise click.ClickException("No F&G/QQQ rows available for the selected start date")
    trades, final_equity = replay(days, tqqq_close)
    first_date, last_date = days[0].date, days[-1].date
    elapsed_days = max(1, (last_date - first_date).days)
    cagr = (final_equity / 10_000.0) ** (365.25 / elapsed_days) - 1

    first_px = float(tqqq_close.loc[pd.Timestamp(first_date)])
    last_px = float(tqqq_close.loc[pd.Timestamp(last_date)])
    buy_hold_final = 10_000.0 * (1 - 0.001) * last_px / first_px

    stop_dates = {day.date for day in days if day.action == "STOP"}
    click.echo("Entry date  Exit date   Exit type  Entry px  Exit px   Return")
    for trade in trades:
        exit_date = trade.exit_date.isoformat() if trade.exit_date else "OPEN"
        exit_type = (
            "STOP"
            if trade.exit_date in stop_dates
            else "SELL" if trade.exit_date else "OPEN"
        )
        exit_px = f"{trade.exit_px:.2f}" if trade.exit_px is not None else "—"
        click.echo(
            f"{trade.entry_date.isoformat()}  {exit_date:10}  {exit_type:9} "
            f"{trade.entry_px:8.2f}  {exit_px:7}  {trade.ret:+.2%}"
        )
    click.echo(f"Final value: ${final_equity:,.2f}")
    click.echo(f"CAGR: {cagr:.2%}")
    click.echo(f"TQQQ buy-and-hold final: ${buy_hold_final:,.2f}")


@fng_tqqq.command(name="stop-check")
@click.pass_context
def stop_check(ctx) -> None:
    """Check the live TQQQ price for a stop in the final five minutes."""
    from rainier.alerts.discord import send_strategy_stop_alert
    from rainier.core.database import get_session
    from rainier.data.fear_greed import load_daily_scores
    from rainier.data.prices import load_last_price

    settings = ctx.obj["settings"]
    cfg = settings.strategies.fng_tqqq
    calendar = xcals.get_calendar("XNYS")

    def poster(day, price, price_time):
        posted = send_strategy_stop_alert(
            day,
            price,
            price_time,
            settings.alerts.discord,
        )
        if not posted:
            raise click.ClickException(
                "Discord strategy webhook not configured/enabled; "
                f"stop alert for {day.date.isoformat()} was not posted"
            )

    try:
        with _legacy_db_for_config(ctx):
            triggered = run_stop_check(
                cfg=cfg,
                session_factory=get_session,
                fg_loader=load_daily_scores,
                price_loader=load_daily_close,
                live_price_fn=load_last_price,
                poster=poster,
                calendar=calendar,
                clock=lambda: datetime.now(ZoneInfo("America/Los_Angeles")),
            )
    except StaleInputError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("STOP alert posted" if triggered else "No stop alert needed")
