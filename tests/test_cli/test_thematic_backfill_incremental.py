"""CLI wiring tests for `rainier thematic backfill --incremental`.

The CLI command loads `scripts/backfill_thematic_universe.py` and calls its
`backfill()`. These tests confirm:

  * `--incremental` is exposed on the command surface.
  * Passing `--incremental` reaches the script's `backfill()` with
    `incremental=True` (and does NOT require `--force`, mirroring the breadth
    `market-breadth backfill-ohlcv --incremental` cron contract).

The actual incremental fetch/upsert semantics are unit-tested in
tests/research/test_backfill_thematic_universe.py; here we only verify the
CLI passes the flag through, so the cron command
`uv run rainier thematic backfill --incremental && uv run rainier thematic run-daily`
is wired correctly.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner


def test_incremental_flag_in_help():
    from rainier.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["thematic", "backfill", "--help"])
    assert result.exit_code == 0
    assert "--incremental" in result.output


def test_incremental_passes_through_to_script(monkeypatch, tmp_path):
    """`thematic backfill --incremental` calls the script with incremental=True
    and writes in place (no --force needed)."""
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

    # The CLI dynamically loads scripts/backfill_thematic_universe.py and calls
    # its backfill(). Patch importlib.util.module_from_spec so the loaded module
    # exposes our fake backfill — and execute the real module body first so the
    # CLI can still read its DEFAULT_* constants and loader helpers.
    import importlib.util as _ilu

    orig_module_from_spec = _ilu.module_from_spec

    def _patched_module_from_spec(spec):
        mod = orig_module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        mod.backfill = _fake_backfill  # type: ignore[attr-defined]
        # The CLI calls spec.loader.exec_module(mod) again after building the
        # module; make that a no-op so our fake backfill survives.
        spec.loader.exec_module = lambda _m: None  # type: ignore[assignment]
        return mod

    monkeypatch.setattr(_ilu, "module_from_spec", _patched_module_from_spec)

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "thematic",
            "backfill",
            "--incremental",
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
    assert captured.get("incremental") is True, (
        f"--incremental must reach script backfill(); captured={captured}"
    )
    # incremental does NOT require force.
    assert captured.get("force") in (False, None)
