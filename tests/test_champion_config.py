"""WS C — champion.yaml model-config system.

Acceptance (task plan):
- behavior-preservation: seeded champion.yaml -> byte-identical ranking on a
  fixture (here: byte-identical StockScreenerConfig, since the ranking is a pure
  function of the config + fixed prices).
- precedence/deep-merge: a champion.yaml setting ONE field proves unspecified
  fields fall through to settings.yaml (NOT to code defaults).
- hot-reload: a champion.yaml change is picked up via load_settings_fresh
  without restart.
"""

from __future__ import annotations

import json

import pandas as pd
import yaml

from rainier.core.champion import (
    METADATA_KEYS,
    append_registry_entry,
    load_champion_overrides,
    merge_stock_screener_config,
    read_registry,
)
from rainier.core.config import StockScreenerConfig, load_settings, load_settings_fresh

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_settings_yaml(path, screener_block: dict | None):
    body: dict = {}
    if screener_block is not None:
        body["stock_screener"] = screener_block
    path.write_text(yaml.safe_dump(body))


def _write_champion(model_dir, fields: dict):
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "champion.yaml").write_text(yaml.safe_dump(fields))


# ---------------------------------------------------------------------------
# load_champion_overrides — strips metadata, returns flat field dict
# ---------------------------------------------------------------------------


def test_load_champion_strips_metadata_header(tmp_path):
    _write_champion(
        tmp_path,
        {
            "version": 3,
            "parent": 2,
            "created": "2026-06-20",
            "note": "x",
            "score": {"hit_rate": 0.5},
            "layer_weight_pattern": 0.55,
            "pattern_weights": {"bull_flag": 0.70},
        },
    )
    out = load_champion_overrides(tmp_path)
    assert not (set(out) & METADATA_KEYS)
    assert out["layer_weight_pattern"] == 0.55
    assert out["pattern_weights"] == {"bull_flag": 0.70}


def test_load_champion_missing_file_returns_empty(tmp_path):
    assert load_champion_overrides(tmp_path) == {}


# ---------------------------------------------------------------------------
# deep-merge precedence: champion wins; unspecified fields fall through to
# settings.yaml (NOT to code defaults); pattern_weights deep-merges.
# ---------------------------------------------------------------------------


def test_merge_champion_wins_per_key():
    yaml_overrides = {"layer_weight_pattern": 0.65, "buy_threshold": 0.65}
    champion = {"layer_weight_pattern": 0.55}
    merged = merge_stock_screener_config(yaml_overrides, champion)
    # champion wins on the field it sets
    assert merged["layer_weight_pattern"] == 0.55
    # the field it does NOT set falls through to settings.yaml, not the default
    assert merged["buy_threshold"] == 0.65


def test_merge_pattern_weights_deep_merge():
    yaml_overrides = {"pattern_weights": {"bull_flag": 0.75, "m_top": 0.85}}
    champion = {"pattern_weights": {"bull_flag": 0.70}}
    merged = merge_stock_screener_config(yaml_overrides, champion)
    # champion overrides bull_flag; m_top survives from settings.yaml
    assert merged["pattern_weights"] == {"bull_flag": 0.70, "m_top": 0.85}


def test_precedence_unspecified_field_falls_through_to_settings_yaml(
    tmp_path, monkeypatch
):
    """A champion.yaml that sets ONE field: every other field must come from
    settings.yaml, proving a dict-level deep-merge (not whole-object replace)."""
    monkeypatch.chdir(tmp_path)
    # settings.yaml sets a NON-default buy_threshold so we can tell it apart
    # from the code default (0.65).
    settings_path = tmp_path / "settings.yaml"
    _write_settings_yaml(
        settings_path,
        {"buy_threshold": 0.61, "watch_threshold": 0.49},
    )
    # champion sets ONLY layer_weight_pattern.
    _write_champion(tmp_path / "config" / "model", {"layer_weight_pattern": 0.55})

    s = load_settings(config_path=settings_path)
    sc = s.stock_screener
    assert sc.layer_weight_pattern == 0.55  # from champion
    assert sc.buy_threshold == 0.61  # from settings.yaml (NOT code default 0.65)
    assert sc.watch_threshold == 0.49  # from settings.yaml


# ---------------------------------------------------------------------------
# behavior-preservation: the SEEDED champion.yaml + settings.yaml in the repo
# yields the SAME StockScreenerConfig as the code defaults (no value changed).
# ---------------------------------------------------------------------------


def test_seeded_champion_is_behavior_preserving():
    """Loading the real repo config (settings.yaml + seeded champion.yaml) must
    produce a StockScreenerConfig byte-identical to the code defaults — the WS1
    invariant that wiring the loader changes no live behavior."""
    s = load_settings()  # real config/settings.yaml + config/model/champion.yaml
    assert s.stock_screener.model_dump() == StockScreenerConfig().model_dump()


# ---------------------------------------------------------------------------
# hot-reload: a champion.yaml edit takes effect via load_settings_fresh.
# ---------------------------------------------------------------------------


def test_hot_reload_picks_up_champion_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings_path = tmp_path / "settings.yaml"
    _write_settings_yaml(settings_path, {"layer_weight_pattern": 0.65})
    model_dir = tmp_path / "config" / "model"

    _write_champion(model_dir, {"layer_weight_pattern": 0.65})
    s1 = load_settings_fresh(config_path=str(settings_path))
    assert s1.stock_screener.layer_weight_pattern == 0.65

    # Edit champion.yaml; a fresh load (no restart) must pick it up.
    _write_champion(model_dir, {"layer_weight_pattern": 0.55})
    s2 = load_settings_fresh(config_path=str(settings_path))
    assert s2.stock_screener.layer_weight_pattern == 0.55


# ---------------------------------------------------------------------------
# results registry — Parquet, append-only, off Neon.
# ---------------------------------------------------------------------------


def test_registry_append_and_read(tmp_path):
    model_dir = tmp_path / "config" / "model"
    append_registry_entry(
        version=1,
        window="2025-06..2026-06",
        metrics={"hit_rate": 0.58, "risk_adj_return": 0.42},
        recorded_at="2026-06-16T00:00:00Z",
        model_dir=model_dir,
    )
    append_registry_entry(
        version=2,
        window="2025-06..2026-06",
        metrics={"hit_rate": 0.40},
        recorded_at="2026-06-17T00:00:00Z",
        model_dir=model_dir,
    )
    df = read_registry(model_dir)
    assert len(df) == 2
    assert list(df["version"]) == ["1", "2"]
    # losers are retained too (version 2 underperforms) — it's a research dataset
    first = json.loads(df.iloc[0]["metrics_json"])
    assert first["hit_rate"] == 0.58


def test_registry_empty_when_absent(tmp_path):
    df = read_registry(tmp_path / "config" / "model")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
