"""champion.yaml model-config system (WS C of the pattern-tuning design).

A single versioned YAML *is* the live screener model. It holds the flat
``StockScreenerConfig`` field names (matching ``settings.yaml:stock_screener``)
plus a metadata header (version / parent / created / note / score). The loader
deep-merges, at the dict level, BEFORE constructing ``StockScreenerConfig``:

    code defaults  ◄  settings.yaml:stock_screener  ◄  champion.yaml

so champion.yaml wins, settings.yaml fills the gaps, and any field neither
file mentions falls through to the pydantic model default. The metadata header
is stripped before model construction.

This is the unit a later workstream A/B-tunes: a promotion is a new
champion.yaml (version++, parent=old, score recorded); history is retained in
``config/model/history/`` and a Parquet results registry. NONE of that lives in
a Neon ``market.*`` table — the registry is a Parquet file (feature-store
convention; avoids the two-DATABASE_URL footgun).

WS1 seeds champion.yaml byte-identical to today's effective config, so wiring
the loader in is behavior-preserving (proven by a parity test).

    settings.yaml:stock_screener (dict)
              │   deep-merge (champion wins)
              ▼
    champion.yaml fields (dict, metadata stripped)
              │
              ▼
    merged dict ──► StockScreenerConfig(**merged)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Metadata header keys that describe the model version but are NOT
# StockScreenerConfig fields — stripped before model construction.
METADATA_KEYS: frozenset[str] = frozenset(
    {"version", "parent", "created", "note", "score"}
)

# Default location of the champion model file and its history dir / registry.
DEFAULT_MODEL_DIR = Path("config/model")
CHAMPION_FILENAME = "champion.yaml"
HISTORY_DIRNAME = "history"
REGISTRY_FILENAME = "registry.parquet"


def champion_path(model_dir: Path | None = None) -> Path:
    """Path to the champion.yaml model file."""
    base = model_dir if model_dir is not None else DEFAULT_MODEL_DIR
    return base / CHAMPION_FILENAME


def load_champion_overrides(model_dir: Path | None = None) -> dict[str, Any]:
    """Read champion.yaml and return ONLY its StockScreenerConfig field dict.

    The metadata header (version/parent/created/note/score) is stripped. A
    missing file (or empty file) returns an empty dict — the loader then falls
    back to settings.yaml + code defaults, so champion.yaml is strictly
    optional. ``pattern_weights`` is preserved as a nested dict (the one nested
    config field).

    A PRESENT-but-malformed file (a list, scalar, or other non-mapping YAML)
    raises ``ValueError`` rather than silently falling back: this file is the
    live screener model, so a botched promotion must fail loudly, not boot the
    process on stale settings.yaml thresholds as a hard-to-detect no-op.
    """
    path = champion_path(model_dir)
    if not path.exists():
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw is None:  # empty file → optional, fall back to settings.yaml
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"champion.yaml at {path} must be a YAML mapping of "
            f"StockScreenerConfig fields, got {type(raw).__name__}"
        )
    return {k: v for k, v in raw.items() if k not in METADATA_KEYS}


def merge_stock_screener_config(
    yaml_overrides: dict[str, Any] | None,
    champion_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Deep-merge ``settings.yaml`` then ``champion.yaml`` over code defaults.

    Precedence (lowest → highest): code defaults are supplied by pydantic at
    construction, so this returns ONLY the override layer:
    ``settings.yaml:stock_screener`` first, then ``champion.yaml`` on top
    (champion wins per-key).

    The merge is dict-level, not whole-object-replace: a champion.yaml that
    sets ONE field leaves every unspecified field falling through to
    settings.yaml (and, for fields neither file names, to the pydantic
    default). ``pattern_weights`` (the one nested dict) is itself deep-merged so
    a champion that tweaks one pattern weight keeps the rest from settings.yaml.
    """
    merged: dict[str, Any] = {}
    for layer in (yaml_overrides or {}, champion_overrides or {}):
        for key, value in layer.items():
            if (
                key == "pattern_weights"
                and isinstance(value, dict)
                and isinstance(merged.get(key), dict)
            ):
                # Deep-merge the nested pattern_weights dict so a partial
                # champion override does not drop the settings.yaml entries.
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Results registry — a Parquet file recording (version, window, metric) per
# config run (winners AND losers). NOT a Neon market.* table: the registry is a
# regenerable feature-store artifact, kept off the canonical DB to avoid the
# two-DATABASE_URL footgun.
# ---------------------------------------------------------------------------

# Columns the registry always carries; the metric payload is a free JSON-ish
# dict serialized to a string so the schema stays stable across metric sets.
REGISTRY_COLUMNS: tuple[str, ...] = ("version", "window", "metrics_json", "recorded_at")


def registry_path(model_dir: Path | None = None) -> Path:
    """Path to the Parquet results registry."""
    base = model_dir if model_dir is not None else DEFAULT_MODEL_DIR
    return base / REGISTRY_FILENAME


def history_dir(model_dir: Path | None = None) -> Path:
    """Path to the promoted-config history directory."""
    base = model_dir if model_dir is not None else DEFAULT_MODEL_DIR
    return base / HISTORY_DIRNAME


def append_registry_entry(
    *,
    version: int | str,
    window: str,
    metrics: dict[str, Any],
    recorded_at: str,
    model_dir: Path | None = None,
) -> Path:
    """Append one (version, window, metrics) row to the Parquet registry.

    Reads the existing registry (if any), appends the row, rewrites the file.
    The registry is append-only by convention — both winners and losers are
    retained so it doubles as a research dataset.
    """
    import json

    import pandas as pd

    path = registry_path(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "version": str(version),
        "window": window,
        "metrics_json": json.dumps(metrics, sort_keys=True),
        "recorded_at": recorded_at,
    }
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=list(REGISTRY_COLUMNS))
    df.to_parquet(path, index=False)
    return path


def read_registry(model_dir: Path | None = None) -> "Any":
    """Return the registry as a DataFrame (empty with the canonical columns if absent)."""
    import pandas as pd

    path = registry_path(model_dir)
    if not path.exists():
        return pd.DataFrame(columns=list(REGISTRY_COLUMNS))
    return pd.read_parquet(path)
