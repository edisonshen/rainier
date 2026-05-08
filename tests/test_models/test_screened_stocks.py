"""Schema tests for ScreenedStockRecord — UNIQUE constraint, FK, columns."""

from __future__ import annotations

from datetime import date

from sqlalchemy import inspect

from rainier.core.models import ScreenedStockRecord


def test_table_name():
    assert ScreenedStockRecord.__tablename__ == "screened_stocks"


def test_columns_present():
    cols = {c.name for c in inspect(ScreenedStockRecord).columns}
    expected = {
        "id", "captured_at", "scan_date", "session_name", "symbol",
        "rule_rank", "composite_score", "money_flow_score", "sector",
        "pattern_type", "pattern_confidence",
        "llm_confidence", "shadow_combined_score", "would_be_combined_rank",
        "thesis_id", "patterns_in_chart_not_in_indicators_count",
        "action_taken", "outcome_pct", "outcome_recorded_at", "notes",
        "forward_return_5d", "forward_return_10d", "outcome_backfilled_at",
    }
    missing = expected - cols
    assert not missing, f"missing columns: {missing}"


def test_unique_constraint_on_scan_session_symbol():
    constraints = ScreenedStockRecord.__table__.constraints
    uniques = [c for c in constraints if c.__class__.__name__ == "UniqueConstraint"]
    assert any(
        c.name == "uq_screened_stocks_scan_session_symbol" for c in uniques
    )


def test_thesis_id_fk_to_analysis_results():
    thesis_col = ScreenedStockRecord.__table__.c.thesis_id
    fks = list(thesis_col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].column.table.name == "analysis_results"


def test_in_memory_sqlite_can_create_table():
    """Smoke: the table can be created without TimescaleDB."""
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///:memory:")
    ScreenedStockRecord.__table__.create(bind=engine)
    inspector = inspect(engine)
    assert "screened_stocks" in inspector.get_table_names()


def test_basic_insert_round_trip_via_sqlite_in_memory():
    """In-memory SQLite — basic ORM insert + read back to verify mapping works."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    ScreenedStockRecord.__table__.create(bind=engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        session.add(
            ScreenedStockRecord(
                scan_date=date(2026, 5, 7), session_name="afternoon",
                symbol="NVDA", rule_rank=1, composite_score=0.85,
                pattern_type="w_bottom", pattern_confidence=0.8,
            )
        )
        session.commit()
        row = session.query(ScreenedStockRecord).one()
        assert row.symbol == "NVDA"
        assert row.rule_rank == 1
        assert row.session_name == "afternoon"
