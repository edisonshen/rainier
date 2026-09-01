"""LLM thesis and research commands."""

from __future__ import annotations

from pathlib import Path

import click

from rainier.cli import (
    _settings_path,
    cli,
)


@cli.group()
def thesis():
    """LLM thesis layer (signals, daily run, single-ticker debug, log, signals registry)."""


@thesis.command("daily")
@click.option(
    "--session",
    "session_name",
    default="afternoon",
    type=click.Choice(["morning", "midday", "afternoon", "close"]),
    help="Which scan to attribute this run to.",
)
@click.option("--top-n", default=5, type=int, help="How many top candidates to LLM-thesis.")
@click.option("--discord/--no-discord", default=False, help="Post results to Discord.")
@click.option("--dry-run", is_flag=True, help="Skip Discord, print theses to stdout.")
@click.option(
    "--max-usd",
    default=None,
    type=float,
    help="Hard kill switch on cumulative spend. Defaults to config "
    "(llm_thesis.max_usd_per_scan) when omitted.",
)
@click.pass_context
def thesis_daily(ctx, session_name, top_n, discord, dry_run, max_usd):
    """Manual scheduler-hook trigger — runs the same pipeline as `rainier run`."""
    from datetime import date as _date

    from rainier.alerts.discord import send_stock_candidates
    from rainier.analysis.stock_screener import screen_stocks
    from rainier.core.config import load_settings_fresh
    from rainier.llm_thesis.persistence import persist_screened_stocks
    from rainier.llm_thesis.service import compute_theses_and_persist

    settings = load_settings_fresh(_settings_path(ctx))
    # Override the kill switch only when explicitly passed; otherwise inherit the
    # config value (raised to 2.5 for xhigh thinking) instead of a stale default.
    if max_usd is not None:
        settings.llm_thesis.max_usd_per_scan = float(max_usd)
    effective_max_usd = settings.llm_thesis.max_usd_per_scan

    click.echo(f"Running screener for session={session_name}...")
    all_candidates, ohlcv_by_symbol = screen_stocks(settings)
    # PR2 carry-over P2 #3: persist top-50 for unbiased shadow validation;
    # display set capped at top-20; LLM thesis on top-N (default 5).
    scan_candidates = all_candidates[:50]
    candidates = all_candidates[:20]
    if not candidates:
        click.echo("No candidates from screener.")
        return

    scan_date = _date.today()
    persist_screened_stocks(
        scan_candidates, scan_date=scan_date, session_name=session_name
    )

    click.echo(f"Running LLM thesis on top {top_n} (max_usd={effective_max_usd:.2f})...")
    theses = compute_theses_and_persist(
        candidates[:top_n],
        ohlcv_by_symbol,
        scan_date=scan_date,
        session_name=session_name,
        settings=settings,
    )

    if dry_run or not discord:
        import json as _json
        click.echo(_json.dumps(theses, indent=2, default=str))
        click.echo(f"\n({len(theses)} theses generated, Discord skipped)")
        return

    send_stock_candidates(
        candidates,
        settings.alerts.discord,
        theses=theses or None,
        # PR5: pipe the dashboard base URL through so embed titles get a
        # clickable deep-link. None disables the link.
        dashboard_base_url=settings.llm_thesis.dashboard_base_url,
        # iter-2: forward session so the summary embed title carries the
        # `— Morning/Midday/Afternoon/Close` suffix when multiple scans
        # land in the same channel on the same day.
        session=session_name,
    )
    click.echo(f"Sent {len(theses)} theses + summary to Discord.")


@thesis.command("ticker")
@click.argument("symbol")
@click.option(
    "--session", "session_name", default="afternoon",
    type=click.Choice(["morning", "midday", "afternoon", "close"]),
)
@click.option(
    "--max-usd",
    default=None,
    type=float,
    help="Hard kill switch on cumulative spend. Defaults to config "
    "(llm_thesis.max_usd_per_scan) when omitted.",
)
@click.pass_context
def thesis_ticker(ctx, symbol, session_name, max_usd):
    """Single-ticker debug pipeline against the latest QU100 snapshot."""
    from datetime import date as _date

    from rainier.analysis.stock_screener import screen_stocks
    from rainier.core.config import load_settings_fresh
    from rainier.llm_thesis.service import compute_theses_and_persist

    settings = load_settings_fresh(_settings_path(ctx))
    # Inherit the config cap unless the operator explicitly overrides it.
    if max_usd is not None:
        settings.llm_thesis.max_usd_per_scan = float(max_usd)

    click.echo(f"Running screener (will filter to {symbol})...")
    candidates, ohlcv = screen_stocks(settings)
    target = next((c for c in candidates if c.symbol.upper() == symbol.upper()), None)
    if target is None:
        click.echo(f"{symbol} not in screener output.")
        return

    theses = compute_theses_and_persist(
        [target], ohlcv,
        scan_date=_date.today(),
        session_name=session_name,
        settings=settings,
    )
    import json as _json
    click.echo(_json.dumps(theses, indent=2, default=str))


@thesis.command("log")
@click.option("--ticker", required=True)
@click.option("--date", "scan_date_s", required=True, help="Scan date YYYY-MM-DD.")
@click.option(
    "--session",
    "session_name",
    required=True,
    type=click.Choice(["morning", "midday", "afternoon", "close"]),
    help="Which scan session row to log against (afternoon, close, etc.).",
)
@click.option(
    "--action", required=True, type=click.Choice(["took", "skipped", "watched"])
)
@click.option("--outcome", "outcome", default=None, help="e.g. +2.3% or -1.1%.")
@click.option("--notes", default=None)
def thesis_log(ticker, scan_date_s, session_name, action, outcome, notes):
    """Manually record action_taken + outcome on a ScreenedStockRecord.

    Codex P1: scoped to a single (scan_date, session_name, symbol) row.
    Without the session filter, the same ticker showing up in morning and
    afternoon scans would have its outcome stamped onto every row, which
    corrupts the per-scan history PR2's outcome backfill consumes.
    """
    from datetime import date as _date
    from datetime import datetime as _datetime
    from datetime import timezone as _tz

    from sqlalchemy import update

    from rainier.core.database import get_session
    from rainier.core.models import ScreenedStockRecord

    scan_date = _date.fromisoformat(scan_date_s)

    pct: float | None = None
    if outcome is not None:
        try:
            pct = float(outcome.strip().rstrip("%"))
        except ValueError as exc:
            raise click.BadParameter(f"Bad --outcome: {exc}") from exc

    with get_session() as session:
        result = session.execute(
            update(ScreenedStockRecord)
            .where(
                ScreenedStockRecord.scan_date == scan_date,
                ScreenedStockRecord.session_name == session_name,
                ScreenedStockRecord.symbol == ticker.upper(),
            )
            .values(
                action_taken=action,
                outcome_pct=pct,
                outcome_recorded_at=_datetime.now(_tz.utc),
                notes=notes,
            )
        )
    affected = int(result.rowcount or 0)
    if affected == 0:
        raise click.ClickException(
            "No ScreenedStockRecord row found for "
            f"symbol={ticker} scan_date={scan_date_s} session={session_name}"
        )
    click.echo(
        f"Updated {affected} row(s) for {ticker} on {scan_date_s} ({session_name})."
    )


@thesis.group("signals")
def thesis_signals():
    """Inspect / toggle / test individual signals in the LLM thesis registry."""


@thesis_signals.command("list")
@click.pass_context
def thesis_signals_list(ctx):
    """Print every signal in the registry + its enabled flag from settings.yaml."""
    from rainier.core.config import load_settings_fresh
    from rainier.llm_thesis.signals import REGISTRY

    settings = load_settings_fresh(_settings_path(ctx))
    cfg_map = settings.llm_thesis.signals
    click.echo(f"{'Name':<24} {'Enabled':<8} {'Version':<8} {'Cost(ms)':<10} Weight")
    click.echo("-" * 64)
    for name, sig_cls in REGISTRY.items():
        instance = sig_cls()
        cfg = cfg_map.get(name)
        enabled = "yes" if (cfg and cfg.enabled) else "no"
        weight = cfg.weight if cfg else 1.0
        click.echo(
            f"{name:<24} {enabled:<8} {instance.version:<8} "
            f"{instance.cost_estimate_ms:<10} {weight:.2f}"
        )


def _set_signal_enabled_yaml(name: str, enabled: bool, config_path: str) -> None:
    """Toggle a signal in `config_path` via plain PyYAML.

    PR1 uses PyYAML which does not preserve comments/order — that's intentional
    for PR1 minimality. PR3 will swap in ruamel.yaml when the auto-research
    accept flow needs to mutate config without nuking layout.

    Codex P1: targets the same YAML the caller's `--config` selected so that
    `rainier --config staging.yaml thesis signals enable X` doesn't surprise-
    edit the production `config/settings.yaml`.
    """
    import yaml as _yaml
    path = Path(config_path)
    if not path.exists():
        raise click.ClickException(f"Missing {path}")
    with path.open("r") as f:
        data = _yaml.safe_load(f) or {}
    section = data.setdefault("llm_thesis", {}).setdefault("signals", {})
    if name not in section:
        section[name] = {"enabled": enabled, "params": {}, "weight": 1.0}
    else:
        section[name]["enabled"] = enabled
    with path.open("w") as f:
        _yaml.safe_dump(data, f, sort_keys=False)


@thesis_signals.command("enable")
@click.argument("name")
@click.pass_context
def thesis_signals_enable(ctx, name):
    """Flip the signal to enabled=true in settings.yaml."""
    from rainier.llm_thesis.signals import REGISTRY
    if name not in REGISTRY:
        raise click.ClickException(
            f"Unknown signal {name!r}. Known: {', '.join(REGISTRY)}"
        )
    _set_signal_enabled_yaml(name, True, _settings_path(ctx))
    click.echo(f"Enabled signal: {name}")


@thesis_signals.command("disable")
@click.argument("name")
@click.pass_context
def thesis_signals_disable(ctx, name):
    """Flip the signal to enabled=false in settings.yaml."""
    from rainier.llm_thesis.signals import REGISTRY
    if name not in REGISTRY:
        raise click.ClickException(
            f"Unknown signal {name!r}. Known: {', '.join(REGISTRY)}"
        )
    _set_signal_enabled_yaml(name, False, _settings_path(ctx))
    click.echo(f"Disabled signal: {name}")


@thesis_signals.command("test")
@click.argument("name")
@click.option("--symbol", required=True, help="Symbol to dry-run the signal against.")
@click.pass_context
def thesis_signals_test(ctx, name, symbol):
    """Dry-run a single signal's compute() against the latest QU100 snapshot."""
    from datetime import date as _date

    from rainier.analysis.stock_screener import screen_stocks
    from rainier.core.config import load_settings_fresh
    from rainier.core.types import StockCandidate
    from rainier.llm_thesis.signals import REGISTRY
    from rainier.llm_thesis.signals.base import SignalContext

    if name not in REGISTRY:
        raise click.ClickException(
            f"Unknown signal {name!r}. Known: {', '.join(REGISTRY)}"
        )

    settings = load_settings_fresh(_settings_path(ctx))
    candidates, ohlcv = screen_stocks(settings)
    target: StockCandidate | None = next(
        (c for c in candidates if c.symbol.upper() == symbol.upper()), None
    )
    if target is None:
        click.echo(f"{symbol} not in screener output — building stub candidate.")
        target = StockCandidate(
            symbol=symbol.upper(), rank=0, rank_change=0, long_short="Long in",
            capital_flow_direction="N", sector="Unknown", signal_strength=0.0,
        )

    cfg = settings.llm_thesis.signals.get(name)
    params = dict(cfg.params) if cfg else {}
    ctx = SignalContext(
        symbol=target.symbol, scan_date=_date.today(),
        session_name="manual", candidate=target,
        ohlcv_df=ohlcv.get(target.symbol), params=params,
    )
    sig = REGISTRY[name]()
    value = sig.compute(ctx)
    click.echo(f"value: {value}")
    if value is not None:
        click.echo(f"render_for_prompt: {sig.render_for_prompt(value)}")


@thesis.command("eval")
@click.option(
    "--date", "eval_date_s", default=None,
    help="Evaluation date YYYY-MM-DD; defaults to today.",
)
@click.option(
    "--horizon", "horizon", default=None,
    type=click.Choice(["1d", "5d", "10d"]),
    help="Single horizon to evaluate; default runs all three.",
)
@click.pass_context
def thesis_eval(ctx, eval_date_s, horizon):
    """Run the daily-eval job manually.

    Backfills ThesisEvaluation rows + posts the Discord eval report.
    Idempotent: only inserts missing rows.
    """
    import asyncio as _asyncio
    from datetime import date as _date

    from rainier.scheduler.service import run_daily_eval

    if horizon is None:
        # Full pass — invoke the scheduler entry which runs all three plus
        # composes the Discord report.
        _asyncio.run(run_daily_eval(eval_date_iso=eval_date_s))
        click.echo("Daily eval finished.")
        return

    # Single-horizon: skip the Discord report, just run the backfill so the
    # operator can see the insert count for one horizon at a time.
    from rainier.llm_thesis.eval import evaluate_horizon

    eval_date = (
        _date.fromisoformat(eval_date_s) if eval_date_s else _date.today()
    )
    n = evaluate_horizon(eval_date, horizon)  # type: ignore[arg-type]
    click.echo(
        f"Evaluated horizon={horizon} on {eval_date.isoformat()}: {n} rows inserted."
    )
    _ = ctx  # ctx unused for single-horizon path; kept for API symmetry


# ---------------------------------------------------------------------------
# `rainier thesis research` — auto-research loop (PR3)
# ---------------------------------------------------------------------------


@thesis.group("research")
def thesis_research():
    """Weekly auto-research loop — produce, browse, accept, reject insights."""


@thesis_research.command("run")
@click.option(
    "--eval-date", "eval_date_s", default=None,
    help="Logical 'today' for the rolling window, YYYY-MM-DD; defaults to today.",
)
@click.option("--days", "days", default=30, type=int)
def thesis_research_run(eval_date_s, days):
    """Manually trigger the weekly research job.

    Same code path the Friday 09:00 PT scheduler entry uses — runs all 6
    check classes, posts the Discord report, idempotent re-runs UPSERT
    pending insights instead of duplicating.
    """
    import asyncio as _asyncio

    from rainier.scheduler.service import run_research_job

    _ = days  # currently fixed at 30 in run_research_job; param kept for symmetry
    _asyncio.run(run_research_job(eval_date_iso=eval_date_s))
    click.echo("Research job finished.")


@thesis_research.group("insights")
def thesis_research_insights():
    """Browse / accept / reject ResearchInsight rows."""


@thesis_research_insights.command("list")
@click.option(
    "--status",
    "status_filter",
    default="pending",
    type=click.Choice(["pending", "accepted", "rejected", "stale", "auto_applied", "all"]),
    help="Filter rows by status; default 'pending'.",
)
def thesis_research_insights_list(status_filter):
    """Print the ResearchInsight queue."""
    from sqlalchemy import select as _select

    from rainier.core.database import get_session
    from rainier.core.models import ResearchInsight

    with get_session() as session:
        stmt = _select(ResearchInsight).order_by(ResearchInsight.id.desc())
        if status_filter != "all":
            stmt = stmt.where(ResearchInsight.status == status_filter)
        rows = session.execute(stmt).scalars().all()

    if not rows:
        click.echo(f"No {status_filter} insights.")
        return

    click.echo(
        f"{'ID':<6} {'Severity':<10} {'Kind':<24} {'Subject':<24} "
        f"{'Recur':<6} {'Action':<22} Status"
    )
    click.echo("-" * 110)
    for r in rows:
        action_kind = "?"
        if isinstance(r.action, dict):
            action_kind = str(r.action.get("kind", "?"))
        click.echo(
            f"{r.id:<6} {r.severity:<10} {r.kind:<24} {(r.subject or '')[:23]:<24} "
            f"{r.recurrence_count:<6} {action_kind:<22} {r.status}"
        )
        rationale = (r.rationale or "").strip().replace("\n", " ")
        if rationale:
            if len(rationale) > 100:
                rationale = rationale[:97] + "..."
            click.echo(f"       -> {rationale}")


@thesis_research_insights.command("accept")
@click.argument("insight_id", type=int)
@click.pass_context
def thesis_research_insights_accept(ctx, insight_id):
    """Apply the suggested action and mark the insight accepted.

    Looks up the row, dispatches `insight.action.kind` through
    `research.ACTION_EXECUTORS`, mutates `config/settings.yaml` via ruamel.yaml,
    then UPDATEs the DB row to `status='accepted'` with the diff stored in
    `applied_change`. Errors out cleanly if the row is non-pending or the
    action kind is unknown.
    """
    from datetime import datetime as _datetime
    from datetime import timezone as _tz

    from rainier.core.database import get_session
    from rainier.core.models import ResearchInsight
    from rainier.llm_thesis.research import apply_action

    settings_path = Path(_settings_path(ctx))

    # Review iter-1 [P2]: order matters. Validate the action kind FIRST (no
    # side-effects), then take the DB row + apply YAML inside one
    # transaction so a YAML-write failure rolls back the DB row update via
    # the contextmanager. Previously YAML was written before the DB commit
    # — a transient DB error left settings.yaml mutated while the row stayed
    # `pending`, allowing a second accept to re-apply the action.
    with get_session() as session:
        row = session.get(ResearchInsight, insight_id)
        if row is None:
            raise click.ClickException(f"No ResearchInsight with id={insight_id}")
        if row.status != "pending":
            raise click.ClickException(
                f"Insight {insight_id} has status={row.status!r}, not pending. "
                "Only pending insights can be accepted."
            )
        action = row.action or {}
        try:
            diff = apply_action(action, settings_path)
        except ValueError as exc:
            # Bad action shape — never wrote anything. Surface clearly.
            raise click.ClickException(f"Could not apply action: {exc}") from exc

        row.status = "accepted"
        row.decided_at = _datetime.now(_tz.utc)
        row.applied_change = diff
        session.flush()
        # contextmanager will COMMIT on clean exit; if anything below this
        # point raised inside the `with`, it would rollback. The YAML write
        # already happened atomically (temp-file rename) — operator can
        # always reconcile by editing YAML by hand and accepting again.

    click.echo(
        f"Accepted insight #{insight_id}: action={action.get('kind')} "
        f"target={action.get('target')!r}"
    )
    click.echo(f"Applied change: {diff}")


@thesis_research_insights.command("reject")
@click.argument("insight_id", type=int)
@click.option("--reason", required=True, help="Free-text reason stored on the row.")
def thesis_research_insights_reject(insight_id, reason):
    """Dismiss the insight without applying its action.

    Sets status='rejected' and stores the reason in `decided_by`. The
    settings.yaml is not touched.
    """
    from datetime import datetime as _datetime
    from datetime import timezone as _tz

    from rainier.core.database import get_session
    from rainier.core.models import ResearchInsight

    with get_session() as session:
        row = session.get(ResearchInsight, insight_id)
        if row is None:
            raise click.ClickException(f"No ResearchInsight with id={insight_id}")
        if row.status != "pending":
            raise click.ClickException(
                f"Insight {insight_id} has status={row.status!r}, not pending."
            )
        row.status = "rejected"
        row.decided_at = _datetime.now(_tz.utc)
        row.decided_by = reason[:200]
        session.flush()

    click.echo(f"Rejected insight #{insight_id} with reason: {reason}")


@thesis_research.command("signals")
@click.option("--signal", "signal_name", default=None,
              help="Filter to a single signal name; default lists all.")
@click.option("--days", "days", default=30, type=int)
@click.option("--horizon", "horizon", default="5d",
              type=click.Choice(["1d", "5d", "10d"]))
def thesis_research_signals(signal_name, days, horizon):
    """Ad-hoc per-signal contribution dump (Mann-Whitney U on used vs absent)."""
    from rainier.llm_thesis.eval import compute_signal_contribution

    contribs = compute_signal_contribution(days=days, horizon=horizon)
    if signal_name:
        contribs = [c for c in contribs if c.name == signal_name]
    if not contribs:
        click.echo("No signal contribution rows found.")
        return

    click.echo(f"{'Signal':<24} {'lift':<10} {'p-value':<10} {'n_used':<8} {'n_absent':<10}")
    click.echo("-" * 72)
    for c in contribs:
        p = f"{c.p_value:.3f}" if c.p_value is not None else "n/a"
        click.echo(
            f"{c.name:<24} {c.lift:+.4f}   {p:<10} {c.n_used:<8} {c.n_absent:<10}"
        )


@thesis_research.command("verdicts")
@click.option("--days", "days", default=30, type=int)
def thesis_research_verdicts(days):
    """Ad-hoc per-verdict hit-rate dump."""
    from rainier.llm_thesis.eval import compute_verdict_hit_rate

    rates = compute_verdict_hit_rate(days=days)
    click.echo(f"{'Verdict':<12} {'Horizon':<8} {'n':<6} {'win-rate':<10} avg-return")
    click.echo("-" * 56)
    for verdict, hits in rates.items():
        for hr in hits:
            click.echo(
                f"{verdict:<12} {hr.horizon:<8} {hr.n:<6} {hr.win_rate:<10.2%} "
                f"{hr.avg_return_pct:+.4f}"
            )
