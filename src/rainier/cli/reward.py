"""Selection reward ledger commands (R1)."""

from __future__ import annotations

import click

from rainier.cli import cli


@cli.group(name="reward")
def reward_group() -> None:
    """QU100-LLM selection reward ledger — compute R-multiples, summarize axes."""


@reward_group.command(name="compute")
@click.option("--as-of", "as_of_iso", default=None, help="As-of date (YYYY-MM-DD)")
@click.option(
    "--horizon",
    "horizon_days",
    type=int,
    default=None,
    help="Counterfactual horizon cap in trading sessions (default 20)",
)
def reward_compute(as_of_iso, horizon_days):
    """Score every decision as of a date and upsert the selection_reward ledger.

    Manual / backfill entry for daily step (vii). Idempotent: provisional
    (mark-to-market) rows are refreshed in place, matured rows are untouched.
    """
    from datetime import date as _date

    from rainier.paper.rewards import compute_rewards
    from rainier.research.rewards.selection import DEFAULT_COUNTERFACTUAL_HORIZON_DAYS

    as_of = _date.fromisoformat(as_of_iso) if as_of_iso else _date.today()
    horizon = horizon_days if horizon_days is not None else DEFAULT_COUNTERFACTUAL_HORIZON_DAYS
    res = compute_rewards(as_of=as_of, counterfactual_horizon_days=horizon)
    click.echo(
        f"Rewards as of {as_of}: {res['decisions']} decisions, "
        f"{res['scored']} scored ({res['provisional']} provisional, "
        f"{res['counterfactual']} counterfactual), {res['unscored']} unscored, "
        f"{res['skipped']} not yet scoreable; {res['rows_written']} rows written."
    )
    for k in sorted(res):
        if k.startswith("decision_"):
            click.echo(f"  {k[len('decision_'):]:<22} {res[k]}")


def _fmt(v, *, pct: bool = False) -> str:
    if v is None:
        return "-"
    if pct:
        return f"{v * 100:.1f}%"
    return f"{v:,.2f}"


@reward_group.command(name="summary")
@click.option(
    "--as-of", "as_of_iso", default=None, help="Only outcomes on/before this date"
)
@click.option(
    "--drawdown-lambda",
    "lam",
    type=float,
    default=None,
    help="Drawdown penalty λ in composite = total_R − λ·max_drawdown_R (default 1.0)",
)
def reward_summary(as_of_iso, lam):
    """Three-axis summary (total return / risk / win rate) of the matured ledger."""
    from datetime import date as _date

    from rainier.paper.rewards import summarize_rewards
    from rainier.research.rewards.selection import DEFAULT_DRAWDOWN_LAMBDA

    as_of = _date.fromisoformat(as_of_iso) if as_of_iso else None
    res = summarize_rewards(as_of=as_of, lam=lam if lam is not None else DEFAULT_DRAWDOWN_LAMBDA)
    click.echo(
        f"{'cohort':<16}{'n':>5}{'total_R':>10}{'pnl_$':>12}{'win':>8}"
        f"{'maxDD_R':>10}{'score':>10}{'E[R]':>8}{'t':>7}"
    )
    for name in ("all", "live", "counterfactual"):
        s = res[name]
        click.echo(
            f"{name:<16}{s['n']:>5}{_fmt(s['total_R']):>10}{_fmt(s['total_pnl_usd']):>12}"
            f"{_fmt(s['win_rate'], pct=True):>8}{_fmt(s['max_drawdown_R']):>10}"
            f"{_fmt(s['composite_score']):>10}{_fmt(s['expectancy_R']):>8}"
            f"{_fmt(s['t_stat_R']):>7}"
        )
    click.echo(f"(composite = total_R − {res['lambda']}·max_drawdown_R; matured rows only)")
