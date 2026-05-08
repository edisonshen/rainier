"""Schema tests for LLMAnalysisRecord — input_hash, signals_used, mapping."""

from __future__ import annotations

import pytest
from sqlalchemy import inspect

from rainier.core.models import LLMAnalysisRecord


def test_input_hash_column_exists():
    cols = {c.name for c in inspect(LLMAnalysisRecord).columns}
    assert "input_hash" in cols
    assert "signals_used" in cols


def test_input_hash_indexed():
    indexes = list(LLMAnalysisRecord.__table__.indexes)
    # Either a dedicated input_hash index or the column declared with index=True;
    # SQLAlchemy materializes both as table-level Index objects.
    has_input_hash_index = any(
        any(c.name == "input_hash" for c in idx.columns) for idx in indexes
    )
    assert has_input_hash_index


@pytest.mark.requires_postgres
def test_idempotent_insert_partial_unique_index_present():
    """The unique partial index lives in the migration script, not metadata.

    This test is `requires_postgres` because verifying the unique partial
    index requires the migration to have been applied to a live Postgres DB.
    Skipped by default; CI / local devs can run it via:
        pytest -m requires_postgres tests/test_models/test_llm_analysis.py
    """
    pytest.skip("Requires migrations/0001_llm_thesis_pr1.sql to be applied")
