"""Paper-trading commands."""

from __future__ import annotations

import click

from rainier.cli import (
    cli,
)


@cli.group(name="paper")
def paper_group() -> None:
    """QU100 paper-trade tracker — open positions, update exits, report."""


@paper_group.command(name="shadow-replay")
@click.option(
    "--start", "start_iso", required=True, help="Window start (YYYY-MM-DD)"
)
@click.option("--end", "end_iso", required=True, help="Window end (YYYY-MM-DD)")
@click.option(
    "--thresholds",
    default="4,5,6",
    show_default=True,
    help="Comma-separated WATCH confidence thresholds to sweep.",
)
@click.option(
    "--benchmark", default="SPY", show_default=True, help="Benchmark symbol."
)
def paper_shadow_replay(start_iso, end_iso, thresholds, benchmark):
    """WS A — replay the shadow WATCH-buy book over a window per threshold T.

    Drives the REAL fill/exit engine day by day over historical theses, opening
    SHADOW positions (excluded from the live book) at each T, and prints the
    shadow book's return vs cash and vs the benchmark. Use the two review
    windows (2026-05-29→06-12 and 2026-05-22→06-12) to compare arms.

    NOTE: shadow rows persist in the DB. Run against a throwaway/replay schema
    or `rainier db gc-test-schemas` afterward — this command does not isolate
    arms from each other; it is a measurement tool, not a live mutation.
    """
    from datetime import date as _date

    from rainier.paper.calendar import DEFAULT_CALENDAR
    from rainier.paper.replay import replay_threshold

    start = _date.fromisoformat(start_iso)
    end = _date.fromisoformat(end_iso)
    ts = [int(t) for t in thresholds.split(",") if t.strip()]
    days = DEFAULT_CALENDAR.sessions_between(start, end)

    click.echo(f"Shadow replay {start}..{end} ({len(days)} sessions), bench={benchmark}")
    click.echo(f"{'T':>3} {'fired':>6} {'book%':>8} {'cash%':>7} {'bench%':>8}")
    for t in ts:
        arm = replay_threshold(
            threshold=t, trading_days=days, benchmark_symbol=benchmark
        )
        click.echo(
            f"{arm.threshold:>3} {arm.fired:>6} "
            f"{arm.book_return_pct * 100:>7.2f}% "
            f"{arm.cash_return_pct * 100:>6.2f}% "
            f"{arm.benchmark_return_pct * 100:>7.2f}%"
        )


@paper_group.command(name="open")
@click.option("--date", "as_of_iso", default=None, help="Fill as-of date (YYYY-MM-DD)")
@click.pass_context
def paper_open(ctx, as_of_iso):
    """Fill pending positions at their T+1 trading-session open."""
    from datetime import date as _date

    from rainier.paper.positions import fill_pending_positions

    as_of = _date.fromisoformat(as_of_iso) if as_of_iso else _date.today()
    # Snapshot the learned time-stop at fill, same as the scheduled daily path
    # (codex iter-2 P3). Read from the Click-loaded settings (honors --config,
    # codex iter-7) — NOT get_settings(), which would ignore a non-default root.
    settings = ctx.obj["settings"]
    learned_ts = settings.llm_thesis.learned_time_stop_days
    res = fill_pending_positions(as_of=as_of, learned_time_stop_days=learned_ts)
    click.echo(f"Filled {res['filled']}, expired {res['expired']}.")


@paper_group.command(name="update")
@click.option("--date", "as_of_iso", default=None, help="As-of date (YYYY-MM-DD)")
def paper_update(as_of_iso):
    """Walk open positions' OHLC and close any that triggered an exit."""
    from datetime import date as _date

    from rainier.paper.positions import update_open_positions

    as_of = _date.fromisoformat(as_of_iso) if as_of_iso else _date.today()
    res = update_open_positions(as_of=as_of)
    click.echo(
        f"Closed {res['closed']} "
        f"(same-bar ambiguous: {res['same_bar_ambiguous_exits']})."
    )


@paper_group.command(name="report")
@click.option("--date", "as_of_iso", default=None, help="Report as-of date (YYYY-MM-DD)")
@click.option("--week", is_flag=True, help="Weekly missed-winner report (Phase 3)")
@click.option(
    "--regenerate",
    is_flag=True,
    help="Recompute from raw inputs + upsert (else plain re-render from snapshot)",
)
def paper_report(as_of_iso, week, regenerate):
    """Render a paper-book report.

    Plain (default) re-renders from the persisted ``paper_report_snapshot`` only.
    ``--regenerate`` recomputes from raw inputs and upserts — for ``--week``
    that's money_flow_snapshots/stock_prices/screened_stocks/analysis_results/
    paper_trade (never the mutable ResearchInsight queue).
    """
    from datetime import date as _date

    from rainier.paper.report import (
        REPORT_TYPE_DAILY,
        REPORT_TYPE_WEEKLY,
        compute_daily_payload,
        load_snapshot,
        persist_daily_snapshot,
        render_payload,
    )

    as_of = _date.fromisoformat(as_of_iso) if as_of_iso else _date.today()

    if week:
        from rainier.paper.sweep import (
            compute_weekly_payload,
            persist_weekly_snapshot,
            render_weekly_payload,
        )

        if regenerate:
            # Recompute from the raw period inputs (historical as_of selects
            # the historical cohort: max data_date <= as_of). No price ingest,
            # no insight emission, no Discord — just compute + upsert + render.
            payload = compute_weekly_payload(as_of)
            persist_weekly_snapshot(as_of, payload)
            click.echo(render_weekly_payload(payload))
            return
        payload = load_snapshot(REPORT_TYPE_WEEKLY, as_of)
        if payload is None:
            raise click.ClickException(
                f"No weekly snapshot for {as_of}. "
                "Run with --regenerate to compute one."
            )
        click.echo(render_weekly_payload(payload))
        return

    if regenerate:
        payload = compute_daily_payload(as_of)
        persist_daily_snapshot(as_of, payload)
        click.echo(render_payload(payload))
        return

    # Plain: snapshot-only.
    payload = load_snapshot(REPORT_TYPE_DAILY, as_of)
    if payload is None:
        raise click.ClickException(
            f"No daily snapshot for {as_of}. Run with --regenerate to compute one."
        )
    click.echo(render_payload(payload))


@paper_group.command(name="appearances")
@click.argument("symbol")
@click.option(
    "--as-of", "as_of_iso", default=None, help="As-of date (YYYY-MM-DD, default today)"
)
@click.option(
    "--window",
    default=None,
    type=int,
    help="Trailing trading-session window (default: chart_lookback_days config)",
)
@click.option(
    "--all-sessions",
    is_flag=True,
    help="Every intraday top100 capture (default: one row/day, latest capture)",
)
@click.pass_context
def paper_appearances(ctx, symbol, as_of_iso, window, all_sessions):
    """Days SYMBOL was in the QU100 top-100 (and its rank each day)."""
    from datetime import date as _date

    from rainier.paper.ingest import get_qu100_appearances

    symbol = symbol.strip().upper()
    as_of = _date.fromisoformat(as_of_iso) if as_of_iso else _date.today()
    if window is None:
        window = ctx.obj["settings"].llm_thesis.chart_lookback_days

    apps = get_qu100_appearances(
        symbol, as_of=as_of, window=window, all_sessions=all_sessions
    )
    if not apps:
        click.echo(
            f"No QU100 top-100 appearances for {symbol} in the last "
            f"{window} sessions ending {as_of}."
        )
        return
    for a in apps:
        line = f"{a.data_date}  #{a.rank}"
        if all_sessions:
            line += f"  {a.capture_session}  {a.captured_at.isoformat()}"
        click.echo(line)


@paper_group.command(name="chart")
@click.argument("symbol")
@click.option(
    "--as-of", "as_of_iso", default=None, help="As-of date (YYYY-MM-DD, default today)"
)
@click.option(
    "--window",
    default=None,
    type=int,
    help="Daily-bar window (default: chart_lookback_days config)",
)
@click.option(
    "--out",
    "out_path",
    default=None,
    help="Output PNG path (default ./<SYMBOL>_<as-of>.png)",
)
@click.pass_context
def paper_chart(ctx, symbol, as_of_iso, window, out_path):
    """Render-on-demand archive chart (zero stored pixels, design App. C).

    Rebuilds the annotated chart for SYMBOL at --as-of from stored inputs
    (stock_prices + money_flow_snapshots + paper_trade). On unchanged data
    this reproduces a stored trade-close chart byte-identically.
    """
    from datetime import date as _date
    from pathlib import Path as _Path

    from rainier.paper.chart_archive import regenerate_chart

    symbol = symbol.strip().upper()
    as_of = _date.fromisoformat(as_of_iso) if as_of_iso else _date.today()
    if window is None:
        window = ctx.obj["settings"].llm_thesis.chart_lookback_days

    try:
        png, sha = regenerate_chart(symbol, as_of=as_of, window=window)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    out = _Path(out_path) if out_path else _Path(f"{symbol}_{as_of}.png")
    out.write_bytes(png)
    click.echo(f"Wrote {out} ({len(png):,} bytes, sha256 {sha[:12]})")


# ---------------------------------------------------------------------------
# Feature store commands
# ---------------------------------------------------------------------------

