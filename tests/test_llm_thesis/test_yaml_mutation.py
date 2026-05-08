"""Round-trip YAML preservation tests — ensures the ACTION_EXECUTOR helpers
mutate `config/settings.yaml` without nuking comments, key order, or
unrelated sections.

ruamel.yaml is the only YAML lib that round-trips comments + indent
preferences. PyYAML drops both. The tests here use a fixture YAML with
comments + multiple sections to verify only the targeted line changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rainier.llm_thesis.research import (
    _bump_prompt_version,
    _disable_signal,
    _increment_prompt_version,
    _lower_signal_weight,
    _raise_signal_weight,
    apply_action,
)

SAMPLE_YAML = """\
# Top-level settings comment.
app:
  name: rainier
  # data_dir is overridable via .env
  data_dir: ./data
  log_level: INFO

# --- LLM thesis layer (PR1) ---
#
# Per-scan settings reload (eng review D2).
llm_thesis:
  enabled: true
  model: "claude-sonnet-4-6"
  max_usd_per_scan: 1.0
  prompt_version: "v1"
  enabled_sessions:
    - afternoon
    - close
  signals:
    rank_trajectory:
      enabled: true        # leading signal
      params:
        days: 10
      weight: 1.0
    fundamentals:
      enabled: true
      params: {}
      weight: 1.0

# Trailing comment that must survive.
"""


@pytest.fixture
def settings_path(tmp_path: Path) -> Path:
    p = tmp_path / "settings.yaml"
    p.write_text(SAMPLE_YAML)
    return p


# ---------------------------------------------------------------------------
# disable_signal — flips enabled, preserves comments/order
# ---------------------------------------------------------------------------


def test_disable_signal_flips_enabled(settings_path: Path):
    diff = _disable_signal("rank_trajectory", {}, settings_path)
    assert diff == {
        "signal": "rank_trajectory",
        "field": "enabled",
        "new_value": False,
    }
    text = settings_path.read_text()
    # The targeted line moved from `enabled: true` to `enabled: false`.
    assert "rank_trajectory:" in text
    rt_section = text.split("rank_trajectory:")[1].split("fundamentals:")[0]
    assert "enabled: false" in rt_section
    # The other signal untouched.
    fund_section = text.split("fundamentals:")[1]
    assert "enabled: true" in fund_section


def test_disable_signal_preserves_comments(settings_path: Path):
    _disable_signal("rank_trajectory", {}, settings_path)
    text = settings_path.read_text()
    # Top-level + section comments + inline + trailing comments all survive.
    assert "# Top-level settings comment." in text
    assert "# --- LLM thesis layer (PR1) ---" in text
    assert "# Per-scan settings reload (eng review D2)." in text
    assert "leading signal" in text  # inline comment on rank_trajectory.enabled
    assert "# Trailing comment that must survive." in text


def test_disable_signal_preserves_key_order(settings_path: Path):
    _disable_signal("rank_trajectory", {}, settings_path)
    text = settings_path.read_text()
    # Order: app -> llm_thesis. Within llm_thesis: enabled, model,
    # max_usd_per_scan, prompt_version, enabled_sessions, signals.
    app_idx = text.index("app:")
    thesis_idx = text.index("llm_thesis:")
    assert app_idx < thesis_idx
    enabled_idx = text.index("enabled: true", thesis_idx)
    model_idx = text.index('model: "claude-sonnet-4-6"', thesis_idx)
    max_idx = text.index("max_usd_per_scan:", thesis_idx)
    prompt_idx = text.index("prompt_version:", thesis_idx)
    sessions_idx = text.index("enabled_sessions:", thesis_idx)
    signals_idx = text.index("signals:", thesis_idx)
    assert enabled_idx < model_idx < max_idx < prompt_idx < sessions_idx < signals_idx


def test_disable_signal_creates_missing_entry(settings_path: Path):
    """When the target signal is not in YAML yet, create a sensible default
    rather than raise. Keeps the executor idempotent.
    """
    _disable_signal("brand_new_signal", {}, settings_path)
    text = settings_path.read_text()
    assert "brand_new_signal:" in text
    new_section = text.split("brand_new_signal:")[1]
    assert "enabled: false" in new_section


def test_disable_signal_only_targeted_line_changes(settings_path: Path):
    original = settings_path.read_text()
    _disable_signal("rank_trajectory", {}, settings_path)
    after = settings_path.read_text()
    # Compute line-level diff: the only change should be the rank_trajectory
    # `enabled` line; every other line in the file must be unchanged.
    orig_lines = original.splitlines()
    new_lines = after.splitlines()
    diffs = [
        (i, o, n)
        for i, (o, n) in enumerate(zip(orig_lines, new_lines))
        if o != n
    ]
    assert len(diffs) == 1, f"expected 1 line change, got {diffs}"
    _, old, new = diffs[0]
    assert "rank_trajectory" not in old  # the change is on the enabled line
    assert "enabled: true" in old
    assert "enabled: false" in new


# ---------------------------------------------------------------------------
# bump_prompt_version
# ---------------------------------------------------------------------------


def test_bump_prompt_version_increments(settings_path: Path):
    diff = _bump_prompt_version("v1", {}, settings_path)
    assert diff["new_value"] == "v2"
    assert diff["old_value"] == "v1"
    text = settings_path.read_text()
    assert 'prompt_version: "v2"' in text or "prompt_version: v2" in text


def test_increment_prompt_version_handles_double_digits():
    assert _increment_prompt_version("v9") == "v10"
    assert _increment_prompt_version("v99") == "v100"


def test_increment_prompt_version_falls_back():
    assert _increment_prompt_version("custom") == "v2"
    assert _increment_prompt_version("") == "v2"


# ---------------------------------------------------------------------------
# raise / lower signal weight
# ---------------------------------------------------------------------------


def test_raise_signal_weight_default_factor(settings_path: Path):
    diff = _raise_signal_weight("rank_trajectory", {}, settings_path)
    assert abs(diff["new_value"] - 1.2) < 1e-9
    assert diff["old_value"] == 1.0
    assert diff["factor"] == 1.2


def test_raise_signal_weight_custom_factor(settings_path: Path):
    diff = _raise_signal_weight(
        "rank_trajectory", {"factor": 1.5}, settings_path
    )
    assert abs(diff["new_value"] - 1.5) < 1e-9


def test_lower_signal_weight_default_factor(settings_path: Path):
    diff = _lower_signal_weight("rank_trajectory", {}, settings_path)
    assert abs(diff["new_value"] - 0.8) < 1e-9


def test_signal_weight_clamps_at_zero(settings_path: Path):
    """Negative factor results in clamped 0.0 (no negative weights)."""
    diff = _raise_signal_weight(
        "rank_trajectory", {"factor": -1.0}, settings_path
    )
    assert diff["new_value"] == 0.0


def test_signal_weight_clamps_at_five(settings_path: Path):
    """A runaway factor caps at 5.0 to prevent dominance."""
    diff = _raise_signal_weight(
        "rank_trajectory", {"factor": 100.0}, settings_path
    )
    assert diff["new_value"] == 5.0


# ---------------------------------------------------------------------------
# apply_action — dispatcher
# ---------------------------------------------------------------------------


def test_apply_action_disable_signal_dispatches(settings_path: Path):
    diff = apply_action(
        {"kind": "disable_signal", "target": "fundamentals", "params": {}},
        settings_path,
    )
    assert diff["signal"] == "fundamentals"
    assert diff["new_value"] is False


def test_apply_action_noop_returns_marker(settings_path: Path):
    diff = apply_action(
        {"kind": "noop", "target": "anything", "params": {}}, settings_path
    )
    assert diff["noop"] is True
    # File contents unchanged (binary equality).
    assert settings_path.read_text() == SAMPLE_YAML


def test_apply_action_unknown_kind_raises(settings_path: Path):
    with pytest.raises(ValueError, match="Unknown action kind"):
        apply_action(
            {"kind": "explode_universe", "target": "x", "params": {}},
            settings_path,
        )


def test_apply_action_non_dict_raises(settings_path: Path):
    with pytest.raises(ValueError, match="must be a dict"):
        apply_action("not_a_dict", settings_path)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Atomic write — temp file is gone after success
# ---------------------------------------------------------------------------


def test_atomic_write_no_temp_files_left(settings_path: Path):
    _disable_signal("rank_trajectory", {}, settings_path)
    siblings = list(settings_path.parent.iterdir())
    # Only the target file should remain — no .tmp_ leftovers.
    tmp_files = [f for f in siblings if f.name.startswith(".tmp_")]
    assert tmp_files == []
