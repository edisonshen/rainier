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
    # Cap raised 1.0 -> 2.5 (xhigh thinking) -> 3.5 (Opus 5.5 pricing).
    assert cfg.max_usd_per_scan == 3.5


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


def test_committed_settings_yaml_ships_the_afternoon_batch_spend_profile():
    """The real config must ship the intended spend profile so the live cron runs
    it: one batched afternoon scan that theses the whole pool and posts the
    ranked top 10, with a cap that fits a full-price fallback per thesis."""
    repo_root = Path(__file__).resolve().parents[2]
    s = load_settings(config_path=repo_root / "config" / "settings.yaml")
    cfg = s.llm_thesis
    assert cfg.thinking_budget_tokens == 24000
    assert cfg.enabled_sessions == ["afternoon"]
    assert "afternoon" in cfg.batch_sessions
    assert cfg.selection_stage == "after_thesis"
    assert (cfg.top_n, cfg.selection_pool_size) == (10, 100)
    assert cfg.max_usd_per_scan / cfg.selection_pool_size > 0.051  # max sync thesis seen
