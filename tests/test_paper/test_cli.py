"""Step 8 — `rainier paper` CLI (TEST-SPEC §H5). No DB for the Phase-3 guard."""

from __future__ import annotations

from click.testing import CliRunner

from rainier.cli import cli


def test_h5_weekly_regenerate_is_phase3_error():
    runner = CliRunner()
    result = runner.invoke(
        cli, ["paper", "report", "--date", "2026-01-09", "--week", "--regenerate"]
    )
    assert result.exit_code != 0
    assert "Phase 3" in result.output or "not implemented" in result.output


def test_paper_group_registered():
    runner = CliRunner()
    result = runner.invoke(cli, ["paper", "--help"])
    assert result.exit_code == 0
    for sub in ("open", "update", "report"):
        assert sub in result.output


def test_db_ingest_prices_registered():
    runner = CliRunner()
    result = runner.invoke(cli, ["db", "ingest-prices", "--help"])
    assert result.exit_code == 0
    assert "active" in result.output
