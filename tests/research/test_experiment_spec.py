"""Experiment-spec interpreter tests (A/B substrate, design §10.4).

The spec is a YAML file (config/experiments/*.yaml): champion + challengers
+ the one contended `layer` + reward keys + window. The interpreter — not
the merge — validates every override key, because
`merge_stock_screener_config` validates nothing and
`StockScreenerConfig(**merged)` silently drops unknown kwargs: an
unvalidated typo would yield a challenger identical to the champion (a
silent no-op A/B).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from rainier.core.config import StockScreenerConfig
from rainier.research import experiment


def _spec_dict(**overrides: Any) -> dict[str, Any]:
    """The design §10.4 example spec, with real StockScreenerConfig fields."""
    base: dict[str, Any] = {
        "id": "layer-weights-rebalance",
        "status": "active",
        "champion": "champion.yaml",
        "layer": "layer_weights",
        "challengers": [
            {
                "id": "mf35",
                "override": {
                    "layer_weight_money_flow": 0.35,
                    "layer_weight_pattern": 0.55,
                },
            },
            {
                "id": "mf40",
                "override": {
                    "layer_weight_money_flow": 0.40,
                    "layer_weight_pattern": 0.50,
                },
            },
        ],
        "primary": "sharpe",
        "guardrails": ["max_drawdown", "turnover"],
        "window": {
            "train": "2025-05-27..2026-03-31",
            "holdout": "2026-04-01..2026-06-25",
            "embargo_days": 20,
        },
    }
    base.update(overrides)
    return base


def _write_spec(directory: Path, filename: str, spec: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(yaml.safe_dump(spec))
    return path


# ---------------------------------------------------------------------------
# Parsing the valid §10.4 shape
# ---------------------------------------------------------------------------


def test_parse_valid_spec_shape():
    spec = experiment.parse_spec(_spec_dict())
    assert spec.id == "layer-weights-rebalance"
    assert spec.status == "active"
    assert spec.champion == "champion.yaml"
    assert spec.layer == "layer_weights"
    assert [c.id for c in spec.challengers] == ["mf35", "mf40"]
    assert spec.primary == "sharpe"
    assert spec.guardrails == ("max_drawdown", "turnover")
    assert spec.window.train == "2025-05-27..2026-03-31"
    assert spec.window.holdout == "2026-04-01..2026-06-25"


def test_load_spec_from_yaml_file(tmp_path):
    path = _write_spec(tmp_path, "layer-weights.yaml", _spec_dict())
    spec = experiment.load_spec(path)
    assert spec.id == "layer-weights-rebalance"
    assert len(spec.challengers) == 2


def test_embargo_days_defaults_to_20():
    raw = _spec_dict()
    del raw["window"]["embargo_days"]
    spec = experiment.parse_spec(raw)
    assert spec.window.embargo_days == 20


def test_embargo_days_explicit_value_parsed():
    raw = _spec_dict()
    raw["window"]["embargo_days"] = 5
    spec = experiment.parse_spec(raw)
    assert spec.window.embargo_days == 5


def test_missing_window_rejected():
    raw = _spec_dict()
    del raw["window"]
    with pytest.raises(experiment.ExperimentSpecError, match="window"):
        experiment.parse_spec(raw)


def test_duplicate_challenger_ids_rejected():
    raw = _spec_dict()
    raw["challengers"][1]["id"] = "mf35"
    with pytest.raises(experiment.ExperimentSpecError, match="mf35"):
        experiment.parse_spec(raw)


def test_empty_override_rejected():
    # An empty override yields a challenger identical to the champion —
    # exactly the silent no-op A/B the interpreter exists to prevent.
    raw = _spec_dict()
    raw["challengers"][0]["override"] = {}
    with pytest.raises(experiment.ExperimentSpecError, match="non-empty mapping"):
        experiment.parse_spec(raw)


# ---------------------------------------------------------------------------
# Override-key validation (the load-bearing check)
# ---------------------------------------------------------------------------


def test_unknown_flat_override_key_rejected_with_key_named():
    raw = _spec_dict()
    raw["challengers"][0]["override"] = {"layer_weight_money_floww": 0.35}
    with pytest.raises(experiment.ExperimentSpecError, match="layer_weight_money_floww"):
        experiment.parse_spec(raw)


def test_unknown_pattern_weights_dotted_key_rejected_with_key_named():
    raw = _spec_dict()
    raw["challengers"][0]["override"] = {"pattern_weights.bulll_flag": 0.9}
    with pytest.raises(experiment.ExperimentSpecError, match="bulll_flag"):
        experiment.parse_spec(raw)


def test_unknown_nested_pattern_weights_key_rejected():
    raw = _spec_dict()
    raw["challengers"][0]["override"] = {"pattern_weights": {"bulll_flag": 0.9}}
    with pytest.raises(experiment.ExperimentSpecError, match="bulll_flag"):
        experiment.parse_spec(raw)


def test_arbitrary_dotted_path_rejected():
    # The interpreter never invents config paths — a knob must exist as a
    # StockScreenerConfig field before it can be experimented on.
    raw = _spec_dict()
    raw["challengers"][0]["override"] = {"signal.momentum.method": "ema"}
    with pytest.raises(experiment.ExperimentSpecError, match="signal.momentum.method"):
        experiment.parse_spec(raw)


def test_pattern_weights_dotted_path_translated_to_nested_dict():
    raw = _spec_dict()
    raw["challengers"][0]["override"] = {"pattern_weights.bull_flag": 0.9}
    spec = experiment.parse_spec(raw)
    assert spec.challengers[0].override == {"pattern_weights": {"bull_flag": 0.9}}


# ---------------------------------------------------------------------------
# Challenger materialization (deep-merge via merge_stock_screener_config)
# ---------------------------------------------------------------------------


def test_challengers_materialize_differing_only_in_overridden_keys():
    spec = experiment.parse_spec(_spec_dict())
    base_overrides = {"buy_threshold": 0.70}
    champion_cfg, challenger_cfgs = experiment.materialize_challengers(spec, base_overrides)

    assert set(challenger_cfgs) == {"mf35", "mf40"}
    mf35 = challenger_cfgs["mf35"]
    assert mf35.layer_weight_money_flow == 0.35
    assert mf35.layer_weight_pattern == 0.55
    # Non-overridden fields fall through to the champion layer / defaults.
    assert mf35.buy_threshold == 0.70
    champion_dump = champion_cfg.model_dump()
    mf35_dump = mf35.model_dump()
    differing = {k for k in champion_dump if champion_dump[k] != mf35_dump[k]}
    assert differing == {"layer_weight_money_flow", "layer_weight_pattern"}


def test_pattern_weight_merge_preserves_other_pattern_weights():
    raw = _spec_dict()
    raw["challengers"] = [
        {"id": "bf90", "override": {"pattern_weights.bull_flag": 0.9}},
    ]
    spec = experiment.parse_spec(raw)
    _, challenger_cfgs = experiment.materialize_challengers(spec, None)
    weights = challenger_cfgs["bf90"].pattern_weights
    defaults = StockScreenerConfig().pattern_weights
    assert weights["bull_flag"] == 0.9
    for name, default in defaults.items():
        if name != "bull_flag":
            assert weights[name] == default


# ---------------------------------------------------------------------------
# Spec discovery + mutual exclusion (config/experiments/*.yaml)
# ---------------------------------------------------------------------------


def test_load_active_specs_returns_only_active(tmp_path):
    _write_spec(tmp_path, "a.yaml", _spec_dict())
    retired = _spec_dict(id="old-experiment", layer="thresholds", status="retired")
    _write_spec(tmp_path, "b.yaml", retired)
    specs = experiment.load_active_specs(tmp_path)
    assert [s.id for s in specs] == ["layer-weights-rebalance"]


def test_malformed_status_is_a_loud_error(tmp_path):
    _write_spec(tmp_path, "a.yaml", _spec_dict(status="paused"))
    with pytest.raises(experiment.ExperimentSpecError, match="paused"):
        experiment.load_active_specs(tmp_path)


def test_two_active_specs_same_layer_mutually_exclusive(tmp_path):
    _write_spec(tmp_path, "a.yaml", _spec_dict())
    _write_spec(tmp_path, "b.yaml", _spec_dict(id="second-experiment"))
    with pytest.raises(experiment.ExperimentSpecError, match="layer_weights"):
        experiment.load_active_specs(tmp_path)


def test_two_active_specs_different_layers_both_load(tmp_path):
    _write_spec(tmp_path, "a.yaml", _spec_dict())
    other = _spec_dict(
        id="threshold-experiment",
        layer="thresholds",
        challengers=[{"id": "bt70", "override": {"buy_threshold": 0.70}}],
    )
    _write_spec(tmp_path, "b.yaml", other)
    specs = experiment.load_active_specs(tmp_path)
    assert sorted(s.id for s in specs) == [
        "layer-weights-rebalance",
        "threshold-experiment",
    ]


def test_retired_spec_never_conflicts_on_layer(tmp_path):
    _write_spec(tmp_path, "a.yaml", _spec_dict())
    retired = _spec_dict(id="old-layer-weights", status="retired")
    _write_spec(tmp_path, "b.yaml", retired)
    specs = experiment.load_active_specs(tmp_path)
    assert [s.id for s in specs] == ["layer-weights-rebalance"]


def test_duplicate_active_experiment_ids_rejected(tmp_path):
    _write_spec(tmp_path, "a.yaml", _spec_dict())
    dup = _spec_dict(layer="thresholds")
    _write_spec(tmp_path, "b.yaml", dup)
    with pytest.raises(experiment.ExperimentSpecError, match="layer-weights-rebalance"):
        experiment.load_active_specs(tmp_path)


def test_load_active_specs_empty_dir_returns_empty(tmp_path):
    assert experiment.load_active_specs(tmp_path) == []


# ---------------------------------------------------------------------------
# Strict spec shape (review iter-1): unknown keys at every level are loud —
# a typo'd `guardrail:` must not yield a guardrail-free experiment.
# ---------------------------------------------------------------------------


def test_unknown_top_level_spec_key_rejected():
    raw = _spec_dict()
    raw["guardrail"] = ["max_drawdown"]  # singular typo of `guardrails`
    with pytest.raises(experiment.ExperimentSpecError, match="guardrail"):
        experiment.parse_spec(raw)


def test_unknown_window_key_rejected():
    raw = _spec_dict()
    raw["window"]["embargo_dayz"] = 5  # typo would silently keep default 20
    with pytest.raises(experiment.ExperimentSpecError, match="embargo_dayz"):
        experiment.parse_spec(raw)


def test_unknown_challenger_key_rejected():
    raw = _spec_dict()
    raw["challengers"][0]["overrride"] = {"buy_threshold": 0.7}
    with pytest.raises(experiment.ExperimentSpecError, match="overrride"):
        experiment.parse_spec(raw)


def test_empty_challenger_id_rejected():
    raw = _spec_dict()
    raw["challengers"][0]["id"] = ""
    with pytest.raises(experiment.ExperimentSpecError, match="non-empty"):
        experiment.parse_spec(raw)


@pytest.mark.parametrize("bad", [-1, True, "20", 2.5])
def test_bad_embargo_days_rejected(bad):
    raw = _spec_dict()
    raw["window"]["embargo_days"] = bad
    with pytest.raises(experiment.ExperimentSpecError, match="embargo_days"):
        experiment.parse_spec(raw)


# ---------------------------------------------------------------------------
# Override ambiguity + value validation (review iter-1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"pattern_weights.bull_flag": 0.9, "pattern_weights": {"bull_flag": 0.5}},
        {"pattern_weights": {"bull_flag": 0.5}, "pattern_weights.bull_flag": 0.9},
    ],
    ids=["dotted-then-nested", "nested-then-dotted"],
)
def test_dotted_and_nested_same_pattern_conflict_rejected(override):
    # Silent last-write-wins for the same pattern is exactly the ambiguity
    # class the interpreter rejects everywhere else.
    raw = _spec_dict()
    raw["challengers"][0]["override"] = override
    with pytest.raises(experiment.ExperimentSpecError, match="bull_flag"):
        experiment.parse_spec(raw)


def test_override_value_type_rejected_at_parse_time():
    # Bad VALUES fail at spec-load time as ExperimentSpecError — not at cron
    # runtime inside materialize_challengers as a raw pydantic error.
    raw = _spec_dict()
    raw["challengers"][0]["override"] = {"layer_weight_money_flow": "not-a-number"}
    with pytest.raises(experiment.ExperimentSpecError, match="layer_weight_money_flow"):
        experiment.parse_spec(raw)


# ---------------------------------------------------------------------------
# Discovery hardening (review iter-1)
# ---------------------------------------------------------------------------


def test_invalid_yaml_file_raises_spec_error(tmp_path):
    (tmp_path / "broken.yaml").write_text("id: [unclosed\n")
    with pytest.raises(experiment.ExperimentSpecError, match="invalid YAML"):
        experiment.load_active_specs(tmp_path)


def test_overlapping_override_keys_across_layer_labels_rejected(tmp_path):
    # Layer labels are just names — two active specs contending the SAME
    # config knob under different labels must be rejected, or the A/B
    # results entangle silently.
    _write_spec(tmp_path, "a.yaml", _spec_dict())
    other = _spec_dict(
        id="relabeled-weights",
        layer="weights",  # different label, same contended knob
        challengers=[{"id": "mf45", "override": {"layer_weight_money_flow": 0.45}}],
    )
    _write_spec(tmp_path, "b.yaml", other)
    with pytest.raises(experiment.ExperimentSpecError, match="layer_weight_money_flow"):
        experiment.load_active_specs(tmp_path)


def test_disjoint_pattern_weight_keys_across_specs_allowed(tmp_path):
    # pattern_weights is expanded per-pattern: touching DIFFERENT patterns
    # in two active experiments is not a conflict.
    a = _spec_dict(
        id="bull-weight",
        layer="pw-bull",
        challengers=[{"id": "b9", "override": {"pattern_weights.bull_flag": 0.9}}],
    )
    b = _spec_dict(
        id="bear-weight",
        layer="pw-bear",
        challengers=[{"id": "k9", "override": {"pattern_weights.bear_flag": 0.9}}],
    )
    _write_spec(tmp_path, "a.yaml", a)
    _write_spec(tmp_path, "b.yaml", b)
    specs = experiment.load_active_specs(tmp_path)
    assert sorted(s.id for s in specs) == ["bear-weight", "bull-weight"]


# ---------------------------------------------------------------------------
# Review iter-2: empty nested pattern_weights + non-str keys + falsy guardrails
# ---------------------------------------------------------------------------


def test_empty_nested_pattern_weights_rejected():
    # {"pattern_weights": {}} is a non-empty override MAPPING that translates
    # to an EMPTY override — a silent no-op challenger that would also escape
    # cross-spec mutual exclusion (override_keys == empty set).
    raw = _spec_dict()
    raw["challengers"][0]["override"] = {"pattern_weights": {}}
    with pytest.raises(experiment.ExperimentSpecError, match="non-empty"):
        experiment.parse_spec(raw)


def test_non_str_top_level_key_rejected():
    # YAML 1.1 parses a bare `on:` as boolean True.
    raw = _spec_dict()
    raw[True] = "x"
    with pytest.raises(experiment.ExperimentSpecError, match="must be str"):
        experiment.parse_spec(raw)


def test_non_str_override_key_rejected():
    raw = _spec_dict()
    raw["challengers"][0]["override"] = {1: 0.5}
    with pytest.raises(experiment.ExperimentSpecError, match="must be str"):
        experiment.parse_spec(raw)


def test_non_str_pattern_weights_key_rejected():
    raw = _spec_dict()
    raw["challengers"][0]["override"] = {"pattern_weights": {1: 0.5}}
    with pytest.raises(experiment.ExperimentSpecError, match="must be str"):
        experiment.parse_spec(raw)


def test_non_str_window_key_rejected():
    raw = _spec_dict()
    raw["window"][True] = "x"
    with pytest.raises(experiment.ExperimentSpecError, match="must be str"):
        experiment.parse_spec(raw)


@pytest.mark.parametrize("bad", [False, 0, ""], ids=["false", "zero", "empty-str"])
def test_falsy_non_list_guardrails_rejected(bad):
    # `guardrails: false` must be a type error, not silently "no guardrails".
    raw = _spec_dict()
    raw["guardrails"] = bad
    with pytest.raises(experiment.ExperimentSpecError, match="guardrails"):
        experiment.parse_spec(raw)


def test_unreadable_spec_path_raises_spec_error(tmp_path):
    # A directory named *.yaml (or a permission failure) must surface as the
    # contract exception, not a raw OSError.
    (tmp_path / "adir.yaml").mkdir()
    with pytest.raises(experiment.ExperimentSpecError, match="unreadable"):
        experiment.load_active_specs(tmp_path)


def test_empty_string_guardrail_rejected():
    raw = _spec_dict()
    raw["guardrails"] = ["max_drawdown", ""]
    with pytest.raises(experiment.ExperimentSpecError, match="guardrails"):
        experiment.parse_spec(raw)
