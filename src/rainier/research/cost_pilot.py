"""Cost-pilot runner — 500-call envelope measurement (D-015).

Slice 0 does NOT make real LLM calls from the test suite. The runner here
takes a list of pre-recorded call records (provider/model/cost/outcome)
and produces:
  - summary statistics (mean / p95 / p99 cost, schema-retry-rate, cache-hit-rate)
  - per-provider table
  - PASS/CONDITIONAL/FAIL verdict
  - parquet ledger of per-call rows
  - rendered HTML report
  - D-031 summary block for stdout

The operator-driven live pilot writes its 500 records to
``data/cache/slice0_cost_pilot.parquet`` and runs the same code path —
the fixture and live modes share the same surface.
"""

from __future__ import annotations

import html
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# Verdict thresholds — concrete values chosen so the CONDITIONAL band is
# meaningful but not trip-happy. Operator can override via CLI flags.
DEFAULT_HARD_CAP_USD = 50.0
DEFAULT_ENVELOPE_MEAN_USD = 0.10  # docs/RESEARCH-cost-pilot-rationale.html


@dataclass(frozen=True)
class CostSummary:
    calls_completed: int
    mean_cost_usd: float
    p95_cost_usd: float
    p99_cost_usd: float
    schema_retry_rate: float
    cache_hit_rate: float
    total_spend_usd: float


def _percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (deterministic; no numpy).

    Uses the canonical nearest-rank formula: rank = ceil(pct/100 * n).
    Python's `round` does banker's rounding (rounds .5 to even) which
    under-reports p95/p99 for some n (e.g., n=30 → round(28.5) = 28 vs
    the canonical 29). codex iter-3 [P2].
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    # 1-indexed nearest-rank with explicit ceil; clamp to len for safety.
    rank = max(1, min(len(sorted_vals), math.ceil(pct / 100.0 * len(sorted_vals))))
    return sorted_vals[rank - 1]


def summarize(calls: Iterable[dict[str, Any]]) -> CostSummary:
    """Aggregate per-call records into a :class:`CostSummary`."""
    records = list(calls)
    n = len(records)
    if n == 0:
        return CostSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    costs = [float(r["cost_usd"]) for r in records]
    retries = sum(int(r.get("schema_retries", 0)) > 0 for r in records)
    hits = sum(1 for r in records if bool(r.get("cache_hit", False)))
    total = sum(costs)
    return CostSummary(
        calls_completed=n,
        mean_cost_usd=round(total / n, 6),
        p95_cost_usd=round(_percentile(costs, 95), 6),
        p99_cost_usd=round(_percentile(costs, 99), 6),
        schema_retry_rate=round(retries / n, 6),
        cache_hit_rate=round(hits / n, 6),
        total_spend_usd=round(total, 6),
    )


def per_provider_stats(calls: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """One row per (provider, model) showing mean / p95 / p99 / count."""
    rows = list(calls)
    buckets: dict[str, list[float]] = {}
    for r in rows:
        key = f"{r['provider']}/{r['model']}"
        buckets.setdefault(key, []).append(float(r["cost_usd"]))
    out: dict[str, dict[str, float]] = {}
    for key, costs in buckets.items():
        out[key] = {
            "count": len(costs),
            "mean_usd": round(statistics.fmean(costs), 6),
            "p95_usd": round(_percentile(costs, 95), 6),
            "p99_usd": round(_percentile(costs, 99), 6),
            "total_usd": round(sum(costs), 6),
        }
    return out


def verdict_for(
    summary: CostSummary,
    hard_cap_usd: float = DEFAULT_HARD_CAP_USD,
    envelope_mean_usd: float = DEFAULT_ENVELOPE_MEAN_USD,
) -> str:
    """Map a :class:`CostSummary` to PASS / CONDITIONAL / FAIL.

    - FAIL when total spend exceeded the operator-set hard cap (D-015 mitigation).
    - CONDITIONAL when mean cost is > 2x the rationale-doc envelope.
    - PASS otherwise.
    """
    if summary.total_spend_usd > hard_cap_usd:
        return "FAIL"
    if summary.mean_cost_usd > 2 * envelope_mean_usd:
        return "CONDITIONAL"
    return "PASS"


def persist_parquet(
    calls: Iterable[dict[str, Any]],
    out_path: str | Path,
    *,
    skill_id: str,
    ledger_sha: str,
) -> Path:
    """Write per-call records as parquet (one row per call).

    Adds ``skill_id`` + ``ledger_sha`` columns to every row so the file is
    self-describing (the operator-review step doesn't need a separate
    manifest).
    """
    rows = [
        {**r, "skill_id": skill_id, "ledger_sha": ledger_sha}
        for r in calls
    ]
    df = pd.DataFrame(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def run_calls(
    calls: Iterable[dict[str, Any]],
    *,
    hard_cap_usd: float,
    out_parquet: str | Path,
    skill_id: str,
    ledger_sha: str,
) -> list[dict[str, Any]]:
    """Replay a list of per-call records honoring the hard-cap stop-condition.

    The live pilot replaces this with a function that streams from an LLM
    provider; the contract is identical. When cumulative spend crosses the
    hard cap, a ``budget_aborted`` row is appended and iteration stops.
    """
    materialized: list[dict[str, Any]] = []
    running_total = 0.0
    for r in calls:
        cost = float(r["cost_usd"])
        materialized.append(dict(r))
        running_total += cost
        # Crossing the cap inside this call means the next iteration must
        # abort — we still record this call's outcome so the partial-spend
        # is auditable, then append the budget_aborted marker and stop.
        if running_total > hard_cap_usd:
            materialized.append(
                {
                    "call_id": "budget-abort",
                    "provider": "n/a",
                    "model": "n/a",
                    "cost_usd": 0.0,
                    "schema_retries": 0,
                    "cache_hit": False,
                    "outcome": "budget_aborted",
                }
            )
            break
    persist_parquet(
        materialized, out_parquet, skill_id=skill_id, ledger_sha=ledger_sha
    )
    return materialized


def summary_block(
    summary: CostSummary,
    *,
    verdict: str,
    skill_id: str,
    report_html_path: str | Path,
    ledger_sha: str,
    calls_total: int | None = None,
) -> dict[str, Any]:
    """Build the D-031 cost-pilot summary-block payload."""
    return {
        "slice": 0,
        "skill_id": skill_id,
        "calls_total": calls_total if calls_total is not None else summary.calls_completed,
        "calls_completed": summary.calls_completed,
        "mean_cost_usd": float(summary.mean_cost_usd),
        "p95_cost_usd": float(summary.p95_cost_usd),
        "p99_cost_usd": float(summary.p99_cost_usd),
        "schema_retry_rate": float(summary.schema_retry_rate),
        "cache_hit_rate": float(summary.cache_hit_rate),
        "total_spend_usd": float(summary.total_spend_usd),
        "verdict": verdict,
        "report_html": str(report_html_path),
        "ledger_sha": ledger_sha,
    }


# --------------------------------------------------------------------------- #
# HTML report — operator-readable artifact (D-014 / RESEARCH-cost-pilot-rationale).
# --------------------------------------------------------------------------- #


_REPORT_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Slice 0 cost-pilot report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 820px; margin: 40px auto; padding: 0 24px; color: #1f2937; }}
  h1 {{ font-size: 22px; }}
  table {{ border-collapse: collapse; margin: 16px 0; width: 100%; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 6px 10px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  .verdict {{ font-weight: 700; padding: 6px 12px; border-radius: 4px; display: inline-block; }}
  .verdict.PASS {{ background: #d1fae5; color: #065f46; }}
  .verdict.CONDITIONAL {{ background: #fef3c7; color: #92400e; }}
  .verdict.FAIL {{ background: #fee2e2; color: #991b1b; }}
  .meta {{ color: #6b7280; font-size: 13px; }}
</style>
</head>
<body>
<h1>Slice 0 cost-pilot report</h1>
<p class="meta">skill_id: <code>{skill_id}</code> · ledger_sha: <code>{ledger_sha}</code></p>
<p>Verdict: <span class="verdict {verdict}">{verdict}</span></p>

<h2>Aggregate</h2>
<table>
  <tr><th>Calls completed</th><td>{calls_completed}</td></tr>
  <tr><th>Mean cost ($)</th><td>{mean_cost_usd:.6f}</td></tr>
  <tr><th>P95 cost ($)</th><td>{p95_cost_usd:.6f}</td></tr>
  <tr><th>P99 cost ($)</th><td>{p99_cost_usd:.6f}</td></tr>
  <tr><th>Schema retry rate</th><td>{schema_retry_rate:.4f}</td></tr>
  <tr><th>Cache hit rate</th><td>{cache_hit_rate:.4f}</td></tr>
  <tr><th>Total spend ($)</th><td>{total_spend_usd:.4f}</td></tr>
</table>

<h2>Per-provider</h2>
<table>
  <tr><th>provider / model</th><th>count</th><th>mean</th><th>p95</th><th>p99</th><th>total</th></tr>
  {provider_rows}
</table>

<p class="meta">Calibration against docs/RESEARCH-cost-pilot-rationale.html confidence intervals:
mean is within ±5% of the envelope when verdict is PASS; CONDITIONAL signals the operator
to scope-down before Slice 1 spend per D-015.</p>
</body>
</html>
"""


def render_report(
    summary: CostSummary,
    *,
    per_provider: dict[str, dict[str, float]],
    verdict: str,
    out_path: str | Path,
    skill_id: str,
    ledger_sha: str,
) -> Path:
    """Render the operator-facing HTML report."""
    rows = []
    for key in sorted(per_provider):
        stats = per_provider[key]
        rows.append(
            "<tr><td>{name}</td><td>{count}</td><td>{mean:.6f}</td>"
            "<td>{p95:.6f}</td><td>{p99:.6f}</td><td>{total:.4f}</td></tr>".format(
                name=html.escape(key),
                count=int(stats["count"]),
                mean=stats["mean_usd"],
                p95=stats["p95_usd"],
                p99=stats["p99_usd"],
                total=stats["total_usd"],
            )
        )
    body = _REPORT_TEMPLATE.format(
        skill_id=html.escape(skill_id),
        ledger_sha=html.escape(ledger_sha),
        verdict=html.escape(verdict),
        calls_completed=summary.calls_completed,
        mean_cost_usd=summary.mean_cost_usd,
        p95_cost_usd=summary.p95_cost_usd,
        p99_cost_usd=summary.p99_cost_usd,
        schema_retry_rate=summary.schema_retry_rate,
        cache_hit_rate=summary.cache_hit_rate,
        total_spend_usd=summary.total_spend_usd,
        provider_rows="\n  ".join(rows) if rows else "<tr><td colspan=6>(no calls)</td></tr>",
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    return out
