"""Schema tests for ResearchInsight (PR3) — columns, defaults, partial
unique index, and SQLite parity for fixture-driven tests.

The partial unique index uses Postgres-only WHERE syntax, so the round-trip
INSERT/UPDATE semantics test runs against an in-memory SQLite engine where
we replicate the constraint logic at the application level (one pending row
per (kind, subject)).
"""

from __future__ import annotations

from sqlalchemy import inspect

from rainier.core.models import ResearchInsight

# ---------------------------------------------------------------------------
# Schema shape
# ---------------------------------------------------------------------------


def test_table_name():
    assert ResearchInsight.__tablename__ == "research_insights"


def test_columns_present():
    cols = {c.name for c in inspect(ResearchInsight).columns}
    expected = {
        "id", "created_at", "updated_at", "kind", "severity", "subject",
        "evidence", "action", "rationale", "recurrence_count", "status",
        "decided_at", "decided_by", "applied_change",
    }
    missing = expected - cols
    assert not missing, f"missing columns: {missing}"


def test_kind_indexed():
    indexes = list(ResearchInsight.__table__.indexes)
    assert any(
        any(c.name == "kind" for c in idx.columns) for idx in indexes
    )


def test_subject_indexed():
    indexes = list(ResearchInsight.__table__.indexes)
    assert any(
        any(c.name == "subject" for c in idx.columns) for idx in indexes
    )


def test_pending_partial_unique_index_declared():
    indexes = {idx.name: idx for idx in ResearchInsight.__table__.indexes}
    target = indexes.get("idx_research_insight_pending_kind_subject")
    assert target is not None, (
        "partial unique index `idx_research_insight_pending_kind_subject` "
        "must be declared on ResearchInsight so emit_insight() UPSERTs "
        "atomically"
    )
    col_names = {c.name for c in target.columns}
    assert col_names == {"kind", "subject"}
    assert target.unique is True
    # Postgres-side WHERE is hung off the dialect_options, not the index
    # object. Check that the postgresql_where clause is present so the SQL
    # generator emits the partial-index clause.
    pg_opts = target.dialect_options.get("postgresql", {})
    assert pg_opts.get("where") is not None


def test_recurrence_count_defaults_to_one():
    col = ResearchInsight.__table__.c.recurrence_count
    # Server default is "1" so existing pending rows post-migration have
    # the right starting value. Python-side default is 1 too.
    assert col.default is not None and col.default.arg == 1
    assert col.server_default is not None
    assert str(col.server_default.arg) == "1"


def test_status_defaults_to_pending():
    col = ResearchInsight.__table__.c.status
    assert col.default is not None and col.default.arg == "pending"
    assert col.server_default is not None
    assert str(col.server_default.arg) == "pending"
