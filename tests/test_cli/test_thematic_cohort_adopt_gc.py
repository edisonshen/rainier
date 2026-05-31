"""CLI wiring tests for the sanctioned cohort->canonical bridge.

Covers:
  * `thematic backfill --adopt` exposes the flag and passes adopt=True (with
    force=True) through to the script's backfill().
  * `thematic gc-cohorts` reaps orphan sibling cohorts, keeps canonical + N
    latest, dry-runs by default, --apply deletes, never touches canonical.

The adopt fetch/replace semantics are unit-tested in
tests/research/test_backfill_thematic_universe.py; here we verify the CLI
surface + the gc command wiring.
"""

from __future__ import annotations

import os
from pathlib import Path

from click.testing import CliRunner

# ---------------------------------------------------------------------------
# backfill --adopt
# ---------------------------------------------------------------------------


def test_adopt_flag_in_help():
    from rainier.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["thematic", "backfill", "--help"])
    assert result.exit_code == 0
    assert "--adopt" in result.output


def test_adopt_passes_through_to_script(monkeypatch, tmp_path):
    """`thematic backfill --force --adopt` reaches script backfill() with
    adopt=True and force=True."""
    import importlib.util as _ilu

    import rainier.cli as cli_mod

    yaml_path = tmp_path / "thematic_universe.yaml"
    yaml_path.write_text(
        "version: 1\n"
        "schema: thematic_universe.v1\n"
        "asof_seeded: 2024-10-01\n"
        "universe:\n"
        "  test_sector:\n"
        "    - XLK\n"
        "    - SMH\n"
    )
    out_path = tmp_path / "thematic_universe.parquet"

    captured: dict[str, object] = {}

    def _fake_backfill(**kwargs):
        captured.update(kwargs)
        return Path(kwargs["out_path"])

    orig_module_from_spec = _ilu.module_from_spec

    def _patched_module_from_spec(spec):
        mod = orig_module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        mod.backfill = _fake_backfill  # type: ignore[attr-defined]
        spec.loader.exec_module = lambda _m: None  # type: ignore[assignment]
        return mod

    monkeypatch.setattr(_ilu, "module_from_spec", _patched_module_from_spec)

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "thematic",
            "backfill",
            "--force",
            "--adopt",
            "--yaml",
            str(yaml_path),
            "--out",
            str(out_path),
            "--ticker-registry",
            str(tmp_path / "ticker_registry.parquet"),
            "--sector-registry",
            str(tmp_path / "sector_registry.parquet"),
        ],
    )
    assert result.exit_code == 0, f"exit={result.exit_code} output={result.output}"
    assert captured.get("adopt") is True, f"captured={captured}"
    assert captured.get("force") is True


# ---------------------------------------------------------------------------
# gc-cohorts
# ---------------------------------------------------------------------------


def _touch(p: Path, mtime: int) -> Path:
    p.write_bytes(b"x")
    os.utime(p, (mtime, mtime))
    return p


def test_gc_cohorts_in_help():
    from rainier.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["thematic", "gc-cohorts", "--help"])
    assert result.exit_code == 0
    assert "--apply" in result.output
    assert "--keep" in result.output


def test_gc_cohorts_dry_run_default_lists_no_delete(tmp_path):
    from rainier.cli import cli

    canonical = tmp_path / "thematic_universe.parquet"
    _touch(canonical, 1000)
    c1 = _touch(tmp_path / "thematic_universe_20260501_010101_000001.parquet", 100)
    c2 = _touch(tmp_path / "thematic_universe_20260502_010101_000001.parquet", 200)

    runner = CliRunner()
    result = runner.invoke(
        cli, ["thematic", "gc-cohorts", "--out", str(canonical), "--keep", "1"]
    )
    assert result.exit_code == 0, result.output
    # Default is dry-run: nothing deleted.
    assert c1.exists() and c2.exists() and canonical.exists()
    assert "DRY-RUN" in result.output or "dry-run" in result.output


def test_gc_cohorts_apply_deletes_keeps_canonical(tmp_path):
    from rainier.cli import cli

    canonical = tmp_path / "thematic_universe.parquet"
    _touch(canonical, 1000)
    c1 = _touch(tmp_path / "thematic_universe_20260501_010101_000001.parquet", 100)
    c2 = _touch(tmp_path / "thematic_universe_20260502_010101_000001.parquet", 200)
    c3 = _touch(tmp_path / "thematic_universe_20260503_010101_000001.parquet", 300)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["thematic", "gc-cohorts", "--out", str(canonical), "--keep", "2", "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert canonical.exists()  # NEVER deleted
    assert c2.exists() and c3.exists()  # 2 latest kept
    assert not c1.exists()  # oldest reaped
