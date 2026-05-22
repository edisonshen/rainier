"""End-to-end CLI tests for `uv run rainier llm-research <cmd>`.

Uses click's testing harness; no shell process spawned. The composition
root (`rainier.cli`) registers the `llm-research` subgroup, so these
tests double as the integration check that wiring landed.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from rainier.cli import cli as root_cli


def _run(args, env=None):
    runner = CliRunner()
    return runner.invoke(root_cli, args, env=env or {})


def test_root_help_includes_llm_research_group():
    result = _run(["--help"])
    assert result.exit_code == 0, result.output
    assert "llm-research" in result.output


def test_llm_research_help_lists_all_subcommands():
    result = _run(["llm-research", "--help"])
    assert result.exit_code == 0, result.output
    for cmd in ("providers", "call", "cost-pilot", "survivorship-check"):
        assert cmd in result.output


def test_providers_test_subcommand_runs(monkeypatch):
    # Force-clear via empty string so the project's .env-loader (which respects
    # existing env vars and only fills missing ones) doesn't re-populate keys
    # under us. The provider adapters treat "" the same as missing.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "")
    result = _run(["llm-research", "providers", "test", "--json"])
    assert result.exit_code == 0, result.output
    # JSON output is a list of provider records.
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert {row["provider"] for row in data} == {
        "anthropic", "deepseek", "openai_compatible",
    }
    # All three should show auth=missing in our cleared env.
    assert all(row["auth"] == "missing" for row in data)


def test_call_dry_run_is_deterministic_for_same_inputs():
    """D-031 + determinism rule: same (skill, ticker, day) → byte-identical prompt."""
    args = [
        "llm-research", "call",
        "--skill", "baseline_opus_multimodal_v1",
        "--ticker", "AAPL",
        "--day", "2025-06-10",
        "--dry-run",
    ]
    a = _run(args)
    b = _run(args)
    assert a.exit_code == 0, a.output
    assert b.exit_code == 0, b.output
    # The dry-run path emits the assembled prompt as JSON to stdout.
    payload_a = json.loads(a.output)
    payload_b = json.loads(b.output)
    assert payload_a == payload_b
    # Sanity: the prompt is keyed by inputs.
    assert payload_a["skill"] == "baseline_opus_multimodal_v1"
    assert payload_a["ticker"] == "AAPL"
    assert payload_a["day"] == "2025-06-10"


def test_survivorship_check_json_shape(tmp_path, monkeypatch):
    """`survivorship-check --json` returns the canonical record shape."""
    # Steer the check at an in-memory session via env var (CLI honors it).
    monkeypatch.setenv("RAINIER_RESEARCH_SURVIVORSHIP_SQLITE", str(tmp_path / "s.db"))
    # The check should not blow up even when the table doesn't exist —
    # it returns a CONDITIONAL verdict with a concrete next-step.
    result = _run([
        "llm-research", "survivorship-check",
        "--from", "2025-01-01", "--to", "2026-05-01", "--json",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "verdict" in data
    assert data["verdict"] in {"PASS", "CONDITIONAL", "FAIL"}
    assert "from_date" in data and data["from_date"] == "2025-01-01"
    assert "to_date" in data and data["to_date"] == "2026-05-01"
    assert "delisted_tickers" in data
    assert "next_step" in data


def test_cost_pilot_fixture_accepts_bare_list_shape(tmp_path):
    """Regression: codex iter-1 [P2] — `--fixture` must accept either
    `{calls: [...], ledger_sha, skill_id}` OR a bare top-level JSON list
    of call records. The pre-fix code called `raw.get("calls", ...)`
    unconditionally, which AttributeErrors when `raw` is a list.
    """
    calls = [
        {"call_id": "c0", "provider": "anthropic", "model": "opus-4.7",
         "cost_usd": 0.01, "schema_retries": 0, "cache_hit": False,
         "outcome": "filled"},
        {"call_id": "c1", "provider": "deepseek", "model": "v4-pro",
         "cost_usd": 0.005, "schema_retries": 0, "cache_hit": True,
         "outcome": "filled"},
    ]
    fixture = tmp_path / "bare_list.json"
    fixture.write_text(json.dumps(calls))
    result = _run([
        "llm-research", "cost-pilot",
        "--fixture", str(fixture),
        "--n", "2",  # must match fixture size; see iter-4 [P2].
        "--out", str(tmp_path / "out.parquet"),
        "--report-html", str(tmp_path / "out.html"),
        "--skill", "test_skill",
    ])
    assert result.exit_code == 0, result.output
    # Summary block should round-trip — verdict is one of the three legal values.
    from rainier.research import output_schema
    parsed = output_schema.parse_block(result.output)
    assert parsed["verdict"] in {"PASS", "CONDITIONAL", "FAIL"}
    assert parsed["calls_completed"] == 2


def test_cost_pilot_fixture_rejects_invalid_shape(tmp_path):
    """Non-dict, non-list JSON should raise a ClickException with a
    helpful next-step rather than crashing with a cryptic AttributeError.
    """
    fixture = tmp_path / "scalar.json"
    fixture.write_text(json.dumps("not-a-list-or-dict"))
    result = _run([
        "llm-research", "cost-pilot",
        "--fixture", str(fixture),
        "--out", str(tmp_path / "out.parquet"),
        "--report-html", str(tmp_path / "out.html"),
    ])
    assert result.exit_code != 0
    # The click error message names the supported shapes.
    assert "list" in result.output.lower() or "object" in result.output.lower()


def test_cost_pilot_rejects_undersized_fixture(tmp_path):
    """Regression: codex iter-4 [P2] — fixture shorter than --n must
    fail-fast. Pre-fix the CLI silently sliced the fixture and emitted
    a calls_total=500 summary backed by 10 actual calls — letting an
    incomplete replay pose as a successful 500-call pilot.
    """
    calls = [
        {"call_id": f"c{i}", "provider": "anthropic", "model": "opus-4.7",
         "cost_usd": 0.01, "schema_retries": 0, "cache_hit": False,
         "outcome": "filled"}
        for i in range(3)
    ]
    fixture = tmp_path / "three_calls.json"
    fixture.write_text(json.dumps(calls))
    result = _run([
        "llm-research", "cost-pilot",
        "--fixture", str(fixture),
        "--n", "10",  # bigger than fixture; should error.
        "--out", str(tmp_path / "out.parquet"),
        "--report-html", str(tmp_path / "out.html"),
    ])
    assert result.exit_code != 0
    # The click message names both numbers so the operator knows what to do.
    assert "3" in result.output and "10" in result.output


def test_cost_pilot_honors_n_below_fixture_size(tmp_path):
    """Regression: codex iter-2 [P2] — `--n 2` against a 5-call fixture
    must replay only the first 2 calls (not all 5). Pre-fix the CLI
    passed the whole fixture into run_calls and emitted a calls_total=2
    summary block while persisting 5 rows + summarizing 5 calls.
    """
    calls = [
        {"call_id": f"c{i}", "provider": "anthropic", "model": "opus-4.7",
         "cost_usd": 0.01, "schema_retries": 0, "cache_hit": False,
         "outcome": "filled"}
        for i in range(5)
    ]
    fixture = tmp_path / "five_calls.json"
    fixture.write_text(json.dumps(calls))
    out_parquet = tmp_path / "out.parquet"
    result = _run([
        "llm-research", "cost-pilot",
        "--fixture", str(fixture),
        "--n", "2",
        "--out", str(out_parquet),
        "--report-html", str(tmp_path / "out.html"),
    ])
    assert result.exit_code == 0, result.output
    from rainier.research import output_schema
    parsed = output_schema.parse_block(result.output)
    assert parsed["calls_total"] == 2
    assert parsed["calls_completed"] == 2
    # The parquet should also reflect the bounded replay (2 rows).
    import pandas as pd
    df = pd.read_parquet(out_parquet)
    # No budget_aborted row in this case — cap is not exceeded.
    assert len(df) == 2
