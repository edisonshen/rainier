"""Migration 0014 is a PLAIN table, not a hypertable.

Deterministic (no DB): parses the committed SQL + asserts the ORM never lists
`fear_greed_index` in HYPERTABLES. A hypertable would need the partition column
in every unique index (composite PK) and would defeat the append-on-change
`MIN(observed_at)` PIT scan the design relies on — so it must stay plain.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FORWARD = ROOT / "migrations" / "0014_fear_greed_index.sql"
DOWNGRADE = ROOT / "migrations" / "0014_fear_greed_index_downgrade.sql"

SCORE_COLUMNS = [
    "momentum_sp500_score",
    "momentum_sp125_score",
    "price_strength_score",
    "price_breadth_score",
    "put_call_score",
    "volatility_vix_score",
    "volatility_vix_50_score",
    "junk_bond_demand_score",
    "safe_haven_demand_score",
]


@pytest.fixture(scope="module")
def forward_sql() -> str:
    return FORWARD.read_text()


def test_forward_and_downgrade_exist():
    assert FORWARD.exists(), f"missing {FORWARD}"
    assert DOWNGRADE.exists(), f"missing paired downgrade {DOWNGRADE}"


def test_creates_fear_greed_table(forward_sql):
    lowered = forward_sql.lower()
    assert "create table if not exists fear_greed_index" in lowered


def test_is_not_a_hypertable(forward_sql):
    assert "create_hypertable" not in forward_sql.lower()


def test_all_nine_score_columns_declared(forward_sql):
    for col in SCORE_COLUMNS:
        assert col in forward_sql, f"score column {col} missing from migration"


def test_has_raw_jsonb_and_provenance(forward_sql):
    lowered = forward_sql.lower()
    assert "raw" in lowered and "jsonb" in lowered
    assert "observed_at" in lowered
    assert "source_version" in lowered


def test_indexes_present(forward_sql):
    lowered = forward_sql.lower()
    assert "(date)" in lowered.replace(" ", " ")  # ix_fng_date
    assert "observed_at desc" in lowered  # ix_fng_date_obs (latest/earliest scan)


def test_downgrade_drops_table():
    lowered = DOWNGRADE.read_text().lower()
    assert "drop table if exists fear_greed_index" in lowered


def test_not_registered_as_hypertable():
    from rainier.core.models import HYPERTABLES

    assert "fear_greed_index" not in HYPERTABLES
