"""Append-only audit of `thematic_universe.yaml` SHA changes (Layer C).

Per DESIGN §5.3 + [D-011], the universe-state log is required so historical
centile ranks remain reproducible after operator YAML edits.

Behavior contract:
  - First call: seeds the parquet with one row capturing today's YAML SHA.
  - Second call (YAML unchanged): NO new row appended.
  - Third call (YAML mutated → new SHA): new row appended; old row preserved.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from rainier.research.breadth import universe_loader as ul


SAMPLE_YAML_V1 = """\
version: 1
schema: thematic_universe.v1
asof_seeded: 2026-05-22
universe:
  technology:
    - XLK
    - SMH
  energy:
    - XLE
    - XOP
"""

SAMPLE_YAML_V2 = """\
version: 1
schema: thematic_universe.v1
asof_seeded: 2026-05-22
universe:
  technology:
    - XLK
    - SMH
    - SOXX
  energy:
    - XLE
    - XOP
"""


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content)


def test_snapshot_universe_first_call_seeds_row(tmp_path: Path):
    yaml_path = tmp_path / "thematic_universe.yaml"
    log_path = tmp_path / "thematic_universe_log.parquet"
    _write_yaml(yaml_path, SAMPLE_YAML_V1)

    appended = ul.snapshot_universe(
        yaml_path=yaml_path,
        log_path=log_path,
        effective_from=date(2026, 5, 22),
        note="seed",
    )
    assert appended is True
    assert log_path.exists()

    df = pd.read_parquet(log_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["effective_from"] == date(2026, 5, 22)
    assert isinstance(row["yaml_sha"], str)
    assert len(row["yaml_sha"]) == 40  # SHA-1 hex
    assert row["universe_size"] == 4
    assert sorted(list(row["tickers"])) == ["SMH", "XLE", "XLK", "XOP"]


def test_snapshot_universe_no_change_is_noop(tmp_path: Path):
    """Same YAML → same SHA → no new row appended."""
    yaml_path = tmp_path / "thematic_universe.yaml"
    log_path = tmp_path / "thematic_universe_log.parquet"
    _write_yaml(yaml_path, SAMPLE_YAML_V1)

    ul.snapshot_universe(
        yaml_path=yaml_path,
        log_path=log_path,
        effective_from=date(2026, 5, 22),
    )
    appended_second = ul.snapshot_universe(
        yaml_path=yaml_path,
        log_path=log_path,
        effective_from=date(2026, 5, 23),  # later date, same content
    )
    assert appended_second is False
    df = pd.read_parquet(log_path)
    assert len(df) == 1  # still just the seed row


def test_snapshot_universe_yaml_edit_appends_new_row(tmp_path: Path):
    """YAML content changes → SHA changes → new row appended.

    Old row preserved (append-only).
    """
    yaml_path = tmp_path / "thematic_universe.yaml"
    log_path = tmp_path / "thematic_universe_log.parquet"
    _write_yaml(yaml_path, SAMPLE_YAML_V1)
    ul.snapshot_universe(
        yaml_path=yaml_path,
        log_path=log_path,
        effective_from=date(2026, 5, 22),
        note="seed",
    )

    # Operator edits the YAML (adds SOXX).
    _write_yaml(yaml_path, SAMPLE_YAML_V2)
    appended = ul.snapshot_universe(
        yaml_path=yaml_path,
        log_path=log_path,
        effective_from=date(2026, 6, 1),
        note="added SOXX",
    )
    assert appended is True

    df = pd.read_parquet(log_path).sort_values("effective_from").reset_index(drop=True)
    assert len(df) == 2
    # Old row untouched.
    assert df.iloc[0]["universe_size"] == 4
    assert df.iloc[0]["note"] == "seed"
    # New row reflects mutation.
    assert df.iloc[1]["universe_size"] == 5
    assert df.iloc[1]["note"] == "added SOXX"
    assert df.iloc[0]["yaml_sha"] != df.iloc[1]["yaml_sha"]


def test_load_universe_returns_sectors_and_tickers_and_sha(tmp_path: Path):
    yaml_path = tmp_path / "thematic_universe.yaml"
    _write_yaml(yaml_path, SAMPLE_YAML_V1)

    spec = ul.load_universe(yaml_path)
    assert spec.sectors == {
        "technology": ["XLK", "SMH"],
        "energy": ["XLE", "XOP"],
    }
    assert sorted(spec.all_tickers) == ["SMH", "XLE", "XLK", "XOP"]
    assert isinstance(spec.yaml_sha, str)
    assert len(spec.yaml_sha) == 40


def test_load_universe_sha_deterministic(tmp_path: Path):
    yaml_path = tmp_path / "thematic_universe.yaml"
    _write_yaml(yaml_path, SAMPLE_YAML_V1)
    sha1 = ul.load_universe(yaml_path).yaml_sha
    sha2 = ul.load_universe(yaml_path).yaml_sha
    assert sha1 == sha2


def test_load_universe_sha_changes_with_content(tmp_path: Path):
    yaml_path = tmp_path / "thematic_universe.yaml"
    _write_yaml(yaml_path, SAMPLE_YAML_V1)
    sha_v1 = ul.load_universe(yaml_path).yaml_sha

    _write_yaml(yaml_path, SAMPLE_YAML_V2)
    sha_v2 = ul.load_universe(yaml_path).yaml_sha

    assert sha_v1 != sha_v2
