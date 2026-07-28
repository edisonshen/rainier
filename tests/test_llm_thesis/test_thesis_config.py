"""Config tests for the xhigh extended-thinking knobs on LLMThesisConfig.

Guards: the thinking budget default + positive-int validation, the raised
per-scan kill-switch cap, and that the committed settings.yaml block actually
reaches the typed config (YAML-reload path the scheduler uses).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rainier.core.config import LLMThesisConfig, load_settings


def test_thinking_budget_and_cap_defaults():
    cfg = LLMThesisConfig()
    # "xhigh" tier.
    assert cfg.thinking_budget_tokens == 24000
    # Cap raised 1.0 -> 2.5 so a 5-ticker xhigh scan doesn't trip the kill switch.
    assert cfg.max_usd_per_scan == 2.5


# 0/-1 are non-positive; 500/1023 are positive but below Anthropic's 1024
# minimum thinking budget — both must fail fast at config load rather than
# reach the provider and 400 every thesis call.
@pytest.mark.parametrize("bad", [0, -1, -24000, 500, 1023])
def test_thinking_budget_rejects_below_minimum(bad: int):
    with pytest.raises(ValidationError):
        LLMThesisConfig(thinking_budget_tokens=bad)


def test_thinking_budget_accepts_at_and_above_minimum():
    assert LLMThesisConfig(thinking_budget_tokens=1024).thinking_budget_tokens == 1024
    assert LLMThesisConfig(thinking_budget_tokens=8000).thinking_budget_tokens == 8000


def test_yaml_block_reaches_thesis_config(monkeypatch, tmp_path):
    yaml_path = tmp_path / "settings.yaml"
    yaml_path.write_text(
        "llm_thesis:\n"
        "  model: \"claude-sonnet-4-6\"\n"
        "  max_usd_per_scan: 2.5\n"
        "  thinking_budget_tokens: 30000\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    s = load_settings(config_path=yaml_path)
    assert s.llm_thesis.thinking_budget_tokens == 30000
    assert s.llm_thesis.max_usd_per_scan == 2.5


def test_committed_settings_yaml_uses_xhigh_budget_and_raised_cap():
    """The real config must ship the xhigh budget + raised cap (yaml/model
    lockstep) so the live cron actually runs the intended spend profile."""
    repo_root = Path(__file__).resolve().parents[2]
    s = load_settings(config_path=repo_root / "config" / "settings.yaml")
    assert s.llm_thesis.thinking_budget_tokens == 24000
    assert s.llm_thesis.max_usd_per_scan == 2.5
