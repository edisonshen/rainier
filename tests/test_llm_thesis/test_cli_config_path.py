"""Tests for codex P1: thesis subcommands honor `--config <path>`.

Without this, `rainier --config staging.yaml thesis ...` would silently fall
back to `config/settings.yaml` because every thesis subcommand reloads via
`load_settings_fresh()` with the default path. In a non-default environment
that means the new workflow can screen, persist, and notify against the
WRONG database/webhook.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from rainier.cli import _settings_path, cli


class _FakeCtx:
    def __init__(self, obj):
        self.obj = obj


def test_settings_path_returns_default_without_obj():
    assert _settings_path(_FakeCtx({})) == "config/settings.yaml"


def test_settings_path_returns_obj_value_when_set():
    assert (
        _settings_path(_FakeCtx({"settings_path": "config/staging.yaml"}))
        == "config/staging.yaml"
    )


def test_thesis_signals_list_passes_config_path_through(tmp_path):
    """The list subcommand must call load_settings_fresh with the --config
    path, NOT the default."""
    cfg = tmp_path / "staging.yaml"
    cfg.write_text("alerts:\n  discord:\n    webhook_url: \"\"\n")

    captured: dict = {}

    def _fake_load(config_path: str = "config/settings.yaml"):
        captured["path"] = config_path

        # Return a stub Settings shape with the bits the command touches.
        class _Sig:
            enabled = True
            weight = 1.0

        class _Thesis:
            signals = {"rank_trajectory": _Sig()}

        class _S:
            llm_thesis = _Thesis()

        return _S()

    runner = CliRunner()
    with patch("rainier.core.config.load_settings_fresh", side_effect=_fake_load), \
         patch("rainier.cli.load_settings"):
        # NOTE: `--config` lives on the cli root, not the subcommand.
        result = runner.invoke(
            cli,
            ["--config", str(cfg), "thesis", "signals", "list"],
        )

    assert result.exit_code == 0, result.output
    assert captured.get("path") == str(cfg)


def test_thesis_signals_enable_writes_to_selected_config(tmp_path):
    """`thesis signals enable X` with `--config staging.yaml` must mutate the
    staging YAML, not config/settings.yaml."""
    cfg = tmp_path / "staging.yaml"
    cfg.write_text(
        "llm_thesis:\n"
        "  signals:\n"
        "    rank_trajectory:\n"
        "      enabled: false\n"
        "      params: {}\n"
        "      weight: 1.0\n",
    )

    runner = CliRunner()
    # Patch the cli root's load_settings so it doesn't fail on the stub YAML.
    with patch("rainier.cli.load_settings") as mock_load_root:
        from rainier.core.config import Settings
        mock_load_root.return_value = Settings(
            database_url="postgresql://test:test@localhost/test"
        )
        result = runner.invoke(
            cli,
            ["--config", str(cfg), "thesis", "signals", "enable", "rank_trajectory"],
        )

    assert result.exit_code == 0, result.output
    contents = cfg.read_text()
    assert "enabled: true" in contents
    # And the production-default path must NOT have been touched.
    # (We can't really assert "production untouched" cleanly here without
    # touching the real fs, but the helper now takes config_path explicitly,
    # so a mismatched path would have raised "Missing ...".)
