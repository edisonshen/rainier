"""``uv run rainier llm-research <cmd>`` subcommand group.

Slice 0 ships four subcommands:
  - ``providers test`` — auth + endpoint sanity check
  - ``call --dry-run`` — deterministic prompt assembly
  - ``cost-pilot`` — 500-call envelope measurement (live or fixture)
  - ``survivorship-check`` — D-016 PASS gate against qu100_history

Stdout discipline (D-034): the verbose paths write to a per-run log under
``~/.rainier/research/runs/<run_id>/run.log``; stdout emits only the
[D-031] summary block or the JSON-typed record the subcommand documents.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import click

from rainier.research import (
    cost_pilot as cost_pilot_mod,
)
from rainier.research import (
    output_schema as output_schema_mod,
)
from rainier.research import (
    providers as providers_mod,
)
from rainier.research import (
    survivorship as survivorship_mod,
)
from rainier.research.evaluator.builder import assemble_dry_run_pack


def _resolve_ledger_sha() -> str:
    """Return the ledger SHA used to stamp archive rows (D-014 amendment).

    Best-effort — falls back to ``"unknown"`` when git isn't available
    (e.g., when running from a wheel install). Live invocations re-stamp
    this when the parquet is persisted.
    """
    import subprocess

    ledger_path = (
        Path(__file__).parent / "data_availability.yaml"
    ).resolve()
    try:
        out = subprocess.run(
            ["git", "rev-parse", f"HEAD:{ledger_path.relative_to(Path.cwd())}"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, ValueError):
        pass
    return "unknown"


@click.group(name="llm-research")
def llm_research() -> None:
    """LLM research engine — Slice 0 CLI surface."""


# --------------------------------------------------------------------------- #
# providers test
# --------------------------------------------------------------------------- #


@llm_research.group(name="providers")
def providers_group() -> None:
    """Manage LLM provider adapters."""


@providers_group.command(name="test")
@click.option("--provider", "provider_name", default=None,
              type=click.Choice(["anthropic", "deepseek", "openai_compatible"]),
              help="Test a single provider; default tests all three.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Emit JSON to stdout (CI / subagent path).")
def providers_test(provider_name: str | None, as_json: bool) -> None:
    """Sanity-check API keys + endpoints for configured providers."""
    names = [provider_name] if provider_name else providers_mod.list_providers()
    results = [providers_mod.get_provider(n).test_auth() for n in names]
    if as_json:
        click.echo(json.dumps([r.as_dict() for r in results]))
        return
    for r in results:
        line = f"  {r.provider:20s} auth={r.auth.value}"
        if r.next_step:
            line += f"  next-step: {r.next_step}"
        click.echo(line)


# --------------------------------------------------------------------------- #
# call --dry-run
# --------------------------------------------------------------------------- #


@llm_research.command(name="call")
@click.option("--skill", required=True, help="Skill id (e.g. baseline_opus_multimodal_v1).")
@click.option("--ticker", required=True, help="Stock ticker.")
@click.option("--day", required=True, help="Evaluation day YYYY-MM-DD.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Assemble + print the prompt; do not call the LLM.")
@click.option("--provider", default="anthropic",
              type=click.Choice(["anthropic", "deepseek", "openai_compatible"]))
@click.option("--model", default=None, help="Override provider default model.")
def call_command(skill: str, ticker: str, day: str, dry_run: bool,
                 provider: str, model: str | None) -> None:
    """Single LLM call (or dry-run prompt assembly)."""
    eval_day = date.fromisoformat(day)
    if not dry_run:
        # Slice 1 ships the real run path; until then we surface the
        # operator-actionable next-step rather than silently failing.
        raise click.ClickException(
            "Real LLM dispatch lands in Slice 1. Re-run with --dry-run to "
            "exercise the prompt-assembly + determinism contract."
        )
    pack = assemble_dry_run_pack(skill=skill, ticker=ticker, day=eval_day)
    payload = {
        "skill": skill,
        "ticker": ticker,
        "day": day,
        "provider": provider,
        "model": model,
        "prompt": pack.as_prompt(),
        "dry_run": True,
    }
    click.echo(json.dumps(payload, sort_keys=True))


# --------------------------------------------------------------------------- #
# cost-pilot
# --------------------------------------------------------------------------- #


@llm_research.command(name="cost-pilot")
@click.option("--skill", "skill_id", default="cost_pilot_v0",
              help="Skill id stamped on per-call rows (default cost_pilot_v0).")
@click.option("--n", default=500, type=int, help="Number of calls to run.")
@click.option("--providers", "provider_list", default="anthropic,deepseek,openai_compatible",
              help="Comma-separated providers to dial against.")
@click.option("--out", "out_parquet", default="data/cache/slice0_cost_pilot.parquet",
              type=click.Path(), help="Per-call parquet output path.")
@click.option("--report-html", default="docs/slice0-cost-report.html",
              type=click.Path(), help="HTML report output path.")
@click.option("--hard-cap-usd", default=cost_pilot_mod.DEFAULT_HARD_CAP_USD,
              type=float, help="Abort if cumulative spend exceeds this cap.")
@click.option("--envelope-mean-usd", default=cost_pilot_mod.DEFAULT_ENVELOPE_MEAN_USD,
              type=float, help="Expected mean cost per the rationale doc.")
@click.option("--fixture", "fixture_path", default=None, type=click.Path(exists=True),
              help="Replay a recorded fixture instead of dialing the live providers.")
def cost_pilot_command(skill_id: str, n: int, provider_list: str,
                       out_parquet: str, report_html: str,
                       hard_cap_usd: float, envelope_mean_usd: float,
                       fixture_path: str | None) -> None:
    """Run the cost-pilot — measures the spend envelope per D-015."""
    if fixture_path is None:
        # Slice 1 implements the live dispatcher; surface the next-step.
        raise click.ClickException(
            "Live cost-pilot dispatcher lands in Slice 1. Re-run with "
            "--fixture <path-to-recorded-calls.json> to exercise the "
            "summary / report / verdict math here in Slice 0."
        )
    raw = json.loads(Path(fixture_path).read_text())
    calls = raw.get("calls", raw if isinstance(raw, list) else [])
    ledger_sha = raw.get("ledger_sha") or _resolve_ledger_sha()
    skill_id = raw.get("skill_id", skill_id)

    materialized = cost_pilot_mod.run_calls(
        calls,
        hard_cap_usd=hard_cap_usd,
        out_parquet=out_parquet,
        skill_id=skill_id,
        ledger_sha=ledger_sha,
    )
    # Drop the budget_aborted bookkeeping row before computing per-call stats.
    real_calls = [c for c in materialized if c.get("outcome") != "budget_aborted"]
    summary = cost_pilot_mod.summarize(real_calls)
    per_provider = cost_pilot_mod.per_provider_stats(real_calls)
    verdict = cost_pilot_mod.verdict_for(
        summary,
        hard_cap_usd=hard_cap_usd,
        envelope_mean_usd=envelope_mean_usd,
    )
    cost_pilot_mod.render_report(
        summary,
        per_provider=per_provider,
        verdict=verdict,
        out_path=report_html,
        skill_id=skill_id,
        ledger_sha=ledger_sha,
    )

    block = cost_pilot_mod.summary_block(
        summary,
        verdict=verdict,
        skill_id=skill_id,
        report_html_path=report_html,
        ledger_sha=ledger_sha,
        calls_total=n,
    )
    click.echo(output_schema_mod.format_block("cost_pilot", block))


# --------------------------------------------------------------------------- #
# survivorship-check
# --------------------------------------------------------------------------- #


@llm_research.command(name="survivorship-check")
@click.option("--from", "from_date", default="2025-01-01",
              help="Start of survivorship window (YYYY-MM-DD).")
@click.option("--to", "to_date", default="2026-05-01",
              help="End of survivorship window (YYYY-MM-DD).")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Emit JSON to stdout (CI / subagent path).")
def survivorship_check_command(from_date: str, to_date: str, as_json: bool) -> None:
    """Validate qu100_history coverage + insert-only proof (D-016)."""
    from_d = date.fromisoformat(from_date)
    to_d = date.fromisoformat(to_date)

    # Test override: an env-var sqlite path means we run against an isolated
    # in-memory DB rather than the project DB. Lets CLI tests run without a
    # live Postgres + lets the operator dry-run against a known-good fixture.
    sqlite_path = os.environ.get("RAINIER_RESEARCH_SURVIVORSHIP_SQLITE")
    if sqlite_path:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        engine = create_engine(f"sqlite:///{sqlite_path}", future=True)
        survivorship_mod.bootstrap_sqlite(engine)
        session = Session(engine, future=True)
    else:
        from rainier.core.database import get_session

        session = get_session().__enter__()  # type: ignore[attr-defined]

    try:
        result = survivorship_mod.check(
            session=session, from_date=from_d, to_date=to_d
        )
    finally:
        session.close()

    payload = result.as_dict()
    if as_json:
        click.echo(json.dumps(payload, sort_keys=True))
        return
    # Non-JSON path: print the verdict line + a YAML block (D-034 single-block).
    click.echo(output_schema_mod.format_block(
        "survivorship",
        {
            "from_date": payload["from_date"],
            "to_date": payload["to_date"],
            "verdict": payload["verdict"],
            "delisted_tickers": payload["delisted_tickers"],
            "missing_days": payload["missing_days"],
            "next_step": payload["next_step"],
        },
    ))


def register(root: click.Group) -> None:
    """Register the `llm-research` subgroup on the root CLI.

    Called from ``rainier.cli`` (the composition root). Keeps the
    research package free of any imports from production CLI plumbing.
    """
    root.add_command(llm_research)
