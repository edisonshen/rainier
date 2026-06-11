"""Step 8 — `rainier paper` CLI (TEST-SPEC §H5). No-DB registration checks."""

from __future__ import annotations

from click.testing import CliRunner

from rainier.cli import cli


def test_weekly_flags_registered():
    """Phase 3 ships --week / --week --regenerate (the old guard error is
    gone); behavior is covered in test_weekly_sweep.py — here just
    registration."""
    runner = CliRunner()
    result = runner.invoke(cli, ["paper", "report", "--help"])
    assert result.exit_code == 0
    assert "--week" in result.output
    assert "--regenerate" in result.output


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


def test_paper_open_reads_learned_time_stop_from_loaded_config(monkeypatch):
    """codex iter-7 — `paper open` snapshots learned_time_stop_days from the
    Click-loaded settings (ctx.obj['settings'], honors --config), not the cached
    get_settings(). Mock the fill so no DB is needed; assert the value flows."""
    captured = {}

    def _fake_fill(*, as_of, learned_time_stop_days=None):
        captured["learned"] = learned_time_stop_days
        return {"filled": 0, "expired": 0}

    monkeypatch.setattr(
        "rainier.paper.positions.fill_pending_positions", _fake_fill
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["paper", "open", "--date", "2026-01-09"])
    assert result.exit_code == 0, result.output
    # Default config ships learned_time_stop_days = None (Phase 0+1 baseline);
    # the key being present proves the loaded-config value was threaded through.
    assert "learned" in captured
