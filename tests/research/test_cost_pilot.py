"""Cost-pilot runner tests — fixture mode, no real LLM calls.

The 500-call live pilot is the operator-driven artifact; here we lock in
the math (mean / p95 / p99 / schema-retry-rate / cache-hit-rate), the
parquet writer, the report renderer, and the hard-cap abort path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rainier.research import cost_pilot, output_schema


def _load_fixture(path: Path) -> dict:
    return json.loads(path.read_text())


def test_summarize_math_matches_fixture(cost_pilot_10call_fixture):
    fixture = _load_fixture(cost_pilot_10call_fixture)
    summary = cost_pilot.summarize(fixture["calls"])

    # 10 calls; one schema retry; two cache hits.
    assert summary.calls_completed == 10
    assert summary.schema_retry_rate == 0.1  # 1/10
    assert summary.cache_hit_rate == 0.2  # 2/10
    # Costs were chosen so the mean is computable by inspection.
    expected_total = sum(c["cost_usd"] for c in fixture["calls"])
    assert summary.total_spend_usd == round(expected_total, 6)
    assert summary.mean_cost_usd == round(expected_total / 10, 6)
    # p95 / p99 are non-decreasing and bounded by the max observed cost.
    max_cost = max(c["cost_usd"] for c in fixture["calls"])
    assert summary.p95_cost_usd <= max_cost
    assert summary.p99_cost_usd <= max_cost
    assert summary.p99_cost_usd >= summary.p95_cost_usd >= summary.mean_cost_usd


def test_verdict_pass_when_within_envelope(cost_pilot_10call_fixture):
    fixture = _load_fixture(cost_pilot_10call_fixture)
    verdict = cost_pilot.verdict_for(
        cost_pilot.summarize(fixture["calls"]),
        hard_cap_usd=50.0,
        envelope_mean_usd=0.10,
    )
    assert verdict == "PASS"


def test_verdict_conditional_when_mean_double_envelope(cost_pilot_10call_fixture):
    fixture = _load_fixture(cost_pilot_10call_fixture)
    summary = cost_pilot.summarize(fixture["calls"])
    # Force CONDITIONAL by setting a very tight envelope.
    verdict = cost_pilot.verdict_for(
        summary,
        hard_cap_usd=50.0,
        envelope_mean_usd=summary.mean_cost_usd / 3,
    )
    assert verdict == "CONDITIONAL"


def test_verdict_fail_when_hard_cap_exceeded(cost_pilot_10call_fixture):
    fixture = _load_fixture(cost_pilot_10call_fixture)
    summary = cost_pilot.summarize(fixture["calls"])
    verdict = cost_pilot.verdict_for(
        summary,
        hard_cap_usd=summary.total_spend_usd - 0.001,
        envelope_mean_usd=1.0,
    )
    assert verdict == "FAIL"


def test_persist_parquet_writes_row_per_call(tmp_path, cost_pilot_10call_fixture):
    fixture = _load_fixture(cost_pilot_10call_fixture)
    out = tmp_path / "slice0_cost_pilot.parquet"
    cost_pilot.persist_parquet(fixture["calls"], out, skill_id=fixture["skill_id"],
                               ledger_sha=fixture["ledger_sha"])
    df = pd.read_parquet(out)
    assert len(df) == 10
    assert set(["call_id", "provider", "model", "cost_usd",
                "schema_retries", "cache_hit", "outcome",
                "skill_id", "ledger_sha"]).issubset(df.columns)
    assert (df["skill_id"] == "cost_pilot_v0").all()


def test_render_report_html(tmp_path, cost_pilot_10call_fixture):
    fixture = _load_fixture(cost_pilot_10call_fixture)
    summary = cost_pilot.summarize(fixture["calls"])
    out = tmp_path / "report.html"
    cost_pilot.render_report(
        summary,
        per_provider=cost_pilot.per_provider_stats(fixture["calls"]),
        verdict="PASS",
        out_path=out,
        skill_id=fixture["skill_id"],
        ledger_sha=fixture["ledger_sha"],
    )
    assert out.exists()
    text = out.read_text()
    # Verdict + skill + per-provider table headers are present.
    assert "PASS" in text
    assert "cost_pilot_v0" in text
    assert "anthropic" in text
    assert "deepseek" in text
    assert "mean" in text.lower() and "p95" in text.lower()


def test_summary_block_parses_against_output_schema(cost_pilot_10call_fixture, tmp_path):
    fixture = _load_fixture(cost_pilot_10call_fixture)
    summary = cost_pilot.summarize(fixture["calls"])
    block = cost_pilot.summary_block(
        summary,
        verdict="PASS",
        skill_id=fixture["skill_id"],
        report_html_path=tmp_path / "report.html",
        ledger_sha=fixture["ledger_sha"],
    )
    text = output_schema.format_block("cost_pilot", block)
    parsed = output_schema.parse_block(text)
    assert parsed["verdict"] == "PASS"
    assert parsed["skill_id"] == "cost_pilot_v0"
    output_schema.validate("cost_pilot", parsed)


def test_percentile_uses_ceil_not_bankers_round():
    """Regression: codex iter-3 [P2] — nearest-rank percentile uses
    ceil(pct/100 * n), not Python's banker's-rounding `round`. For
    n=30, p95 = sorted_vals[ceil(0.95*30)-1] = sorted_vals[28]; the
    pre-fix code returned sorted_vals[27] (banker's round of 28.5 → 28).
    Underreports p95/p99 on common fixture / per-provider bucket sizes.
    """
    # Constructed values where rank-28 vs rank-29 disagree: 0.0..29.0
    # ascending so the percentile equals (rank - 1).
    vals = [float(i) for i in range(30)]
    # nearest-rank: ceil(0.95 * 30) = 29 → sorted[28] = 28.0
    assert cost_pilot._percentile(vals, 95) == 28.0
    # ceil(0.99 * 30) = 30 → sorted[29] = 29.0
    assert cost_pilot._percentile(vals, 99) == 29.0
    # Edge cases stay correct.
    assert cost_pilot._percentile(vals, 100) == 29.0
    assert cost_pilot._percentile(vals, 0) == 0.0
    assert cost_pilot._percentile([], 95) == 0.0


def test_budget_aborted_row_written_on_hard_cap(tmp_path):
    """`run_fixture` with a hard-cap that trips part-way writes the budget_aborted row."""
    calls = [
        {"call_id": f"c{i}", "provider": "deepseek", "model": "v4-pro",
         "cost_usd": 1.0, "schema_retries": 0, "cache_hit": False,
         "outcome": "filled"}
        for i in range(5)
    ]
    out_parquet = tmp_path / "cap.parquet"
    cost_pilot.run_calls(
        calls,
        hard_cap_usd=2.5,  # trips after the third call ($3 cumulative).
        out_parquet=out_parquet,
        skill_id="cost_pilot_v0",
        ledger_sha="0" * 40,
    )
    df = pd.read_parquet(out_parquet)
    # Three "filled" calls then a budget_aborted row.
    assert (df["outcome"] == "budget_aborted").sum() >= 1
    assert (df["outcome"] == "filled").sum() == 3
