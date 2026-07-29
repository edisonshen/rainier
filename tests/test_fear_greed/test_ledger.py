"""Data-availability ledger records F&G's observability rule.

The ledger is the look-ahead control surface. F&G is a post-close, revised
series — so the entry declares `end_of_trading_day` observability and
`revision_immutability: false`. It does NOT invent a `first_observed_date`
field (the loader whitelists keys and would drop it); the first-live-capture
boundary is derived from the table via
`MIN(observed_at) WHERE source_version='daily'`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rainier.research.data_availability_loader import load

ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = ROOT / "src" / "rainier" / "research" / "data_availability.yaml"

DATASET = "fear_greed_index"


@pytest.fixture(scope="module")
def entries():
    return load(LEDGER_PATH).by_dataset(DATASET)


def test_ledger_has_fear_greed_entry(entries):
    assert entries, f"no ledger entry for dataset {DATASET!r}"


def test_observability_and_revision_rule(entries):
    entry = entries[0]
    assert entry.observability_timestamp_rule == "end_of_trading_day"
    # CNN restates the series intraday and revises history → not immutable.
    assert entry.revision_immutability is False


def test_no_first_observed_date_field(entries):
    """The boundary is derived from the table, never stored as a ledger field."""
    entry = entries[0]
    assert "first_observed_date" not in entry.fields
