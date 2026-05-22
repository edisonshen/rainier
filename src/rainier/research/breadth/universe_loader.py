"""Loader for `config/thematic_universe.yaml` + universe-state log writer.

Two responsibilities:

  1. ``load_universe(path)`` — parse the YAML into a deterministic
     ``UniverseSpec`` dataclass carrying ``sectors``, ``all_tickers`` (flat),
     and ``yaml_sha`` (SHA-1 of the raw file bytes). Pure function;
     byte-identical inputs → byte-identical output.

  2. ``snapshot_universe(yaml_path, log_path, effective_from, note)`` —
     append-only Layer C audit. First call seeds a row capturing the
     current YAML SHA + ticker list. Subsequent calls are no-ops if the
     SHA is unchanged; appended only when the YAML mutates.

The SHA is computed over the raw bytes (not the parsed dict) so reordering
the YAML — which doesn't change the semantic universe — still appends a new
log row. That's intentional: it matches the operator's mental model
("I edited the file → I want to see the edit recorded").

Design ref:
    docs/DESIGN-thematic-ranks-dashboard.md §4 §5.3 ([D-001] [D-011])
    docs/TASK-PLAN-thematic-backfill-fd88.md §3
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniverseSpec:
    """Parsed universe contents.

    Attributes
    ----------
    sectors:
        Mapping of sector_name -> ordered list of ticker symbols. Order is
        preserved from the YAML so per-sector display layouts can rely on it.
    all_tickers:
        Flat list of all tickers in YAML appearance order. Useful for the
        backfill script which doesn't care about sector grouping.
    yaml_sha:
        Hex SHA-1 of the raw YAML file bytes. The log table keys off this.
    version:
        The integer ``version:`` field from the YAML (currently always 1).
    """

    sectors: dict[str, list[str]]
    all_tickers: list[str]
    yaml_sha: str
    version: int


def load_universe(yaml_path: Path) -> UniverseSpec:
    """Parse ``yaml_path`` into a ``UniverseSpec``.

    The SHA is computed over the file bytes as read from disk — this
    captures whitespace edits + reorderings the operator may have made,
    matching the "I edited this file" mental model.
    """
    path = Path(yaml_path)
    raw = path.read_bytes()
    sha = hashlib.sha1(raw).hexdigest()
    parsed = yaml.safe_load(raw)

    sectors_raw = parsed.get("universe", {}) or {}
    sectors: dict[str, list[str]] = {}
    all_tickers: list[str] = []
    for sector, tickers in sectors_raw.items():
        if not isinstance(tickers, list):
            raise ValueError(
                f"sector {sector!r} must map to a list of tickers, got "
                f"{type(tickers).__name__}"
            )
        sectors[sector] = list(tickers)
        all_tickers.extend(tickers)

    version = int(parsed.get("version", 1))
    return UniverseSpec(
        sectors=sectors,
        all_tickers=all_tickers,
        yaml_sha=sha,
        version=version,
    )


# ---------------------------------------------------------------------------
# Universe-state log (Layer C)
# ---------------------------------------------------------------------------


_LOG_SCHEMA = pa.schema(
    [
        ("effective_from", pa.date32()),
        ("yaml_sha", pa.string()),
        ("universe_size", pa.int16()),
        ("tickers", pa.list_(pa.string())),
        ("note", pa.string()),
    ]
)


def _read_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(
            columns=["effective_from", "yaml_sha", "universe_size", "tickers", "note"]
        )
    return pd.read_parquet(log_path)


def _write_log_atomic(df: pd.DataFrame, log_path: Path) -> None:
    """pyarrow write + os.replace — never leaves a half-written parquet visible."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = log_path.with_suffix(log_path.suffix + ".tmp")
    table = pa.Table.from_pandas(df, schema=_LOG_SCHEMA, preserve_index=False)
    try:
        pq.write_table(table, tmp, compression="snappy")
        os.replace(tmp, log_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def snapshot_universe(
    yaml_path: Path,
    log_path: Path,
    effective_from: date,
    note: str = "",
) -> bool:
    """Append a row to the universe-state log if the YAML SHA changed.

    Returns ``True`` if a new row was appended, ``False`` if the SHA matched
    the most recent log entry (no-op).

    The log is append-only — never updates or deletes existing rows. Historical
    centile ranks can be reproduced by looking up the YAML state in force at
    any given ``asof_date``.

    Parameters
    ----------
    yaml_path:
        Path to ``config/thematic_universe.yaml`` (or an operator-supplied
        fork).
    log_path:
        Parquet to append to (typically ``data/cache/thematic_universe_log.parquet``).
    effective_from:
        First trading day this universe applies to. Operator-supplied; the
        function does not infer "today" because the cron job may be running
        late or running a backfill against a historical date.
    note:
        Free-form human-readable note ("added SOXX", "removed dead ETF").
    """
    spec = load_universe(yaml_path)
    existing = _read_log(log_path)

    if not existing.empty and (existing["yaml_sha"] == spec.yaml_sha).any():
        # SHA already on record → no-op (universe content unchanged).
        return False

    new_row = pd.DataFrame(
        [
            {
                "effective_from": effective_from,
                "yaml_sha": spec.yaml_sha,
                "universe_size": len(spec.all_tickers),
                "tickers": list(spec.all_tickers),
                "note": note,
            }
        ]
    )
    out = pd.concat([existing, new_row], ignore_index=True)
    _write_log_atomic(out, log_path)
    return True
