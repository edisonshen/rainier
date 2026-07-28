"""The thesis CLI must inherit the config kill-switch cap unless --max-usd is
explicitly passed.

Regression: the CLI hardcoded --max-usd default 1.0 and unconditionally wrote
it onto settings.llm_thesis.max_usd_per_scan, so manual `rainier thesis daily`
runs ignored the config cap (raised to 2.5 for xhigh thinking) and could trip
the old $1.00 kill switch mid-scan.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from rainier.cli import cli


def _fake_settings(cap: float = 2.5):
    return SimpleNamespace(llm_thesis=SimpleNamespace(max_usd_per_scan=cap))


def _run(args):
    """Run `thesis daily` with an empty screener so it early-returns after the
    cap is (or isn't) overridden, and capture the settings it built."""
    captured: dict = {}

    def _fake_load(config_path: str = "config/settings.yaml"):
        s = _fake_settings()
        captured["settings"] = s
        return s

    def _fake_screen(settings):
        return [], {}  # no candidates → command returns before the pipeline

    with patch("rainier.core.config.load_settings_fresh", side_effect=_fake_load), \
            patch("rainier.analysis.stock_screener.screen_stocks", side_effect=_fake_screen):
        result = CliRunner().invoke(cli, args)
    return result, captured["settings"]


def test_daily_inherits_config_cap_when_max_usd_omitted():
    result, settings = _run(["thesis", "daily", "--session", "afternoon"])
    assert result.exit_code == 0, result.output
    # Config cap (2.5) preserved — NOT clobbered by a stale 1.0 default.
    assert settings.llm_thesis.max_usd_per_scan == 2.5


def test_daily_explicit_max_usd_overrides_config():
    result, settings = _run(
        ["thesis", "daily", "--session", "afternoon", "--max-usd", "0.5"]
    )
    assert result.exit_code == 0, result.output
    assert settings.llm_thesis.max_usd_per_scan == 0.5
