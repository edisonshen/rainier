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
