"""Step 1 — migrations + ORM models (TEST-SPEC §A)."""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from rainier.core.models import (
    PaperReportSnapshot,
    PaperSkip,
    PaperTrade,
    ScreenedStockRecord,
)

# --------------------------------------------------------------------------
# ORM-metadata assertions (run always — no Postgres needed)
# --------------------------------------------------------------------------


def test_paper_trade_orm_shape():
    cols = {c.name for c in inspect(PaperTrade).columns}
    expected = {
        "id", "thesis_id", "screened_record_id", "symbol", "scan_date",
        "session_name", "created_at", "updated_at", "status",
        "planned_entry_price", "stop_loss", "target_price", "rr_ratio",
        "pattern_type", "llm_confidence", "verdict",
        "entry_date", "entry_price", "shares", "allocated_amount",
        "residual_cash", "time_stop_days", "price_basis",
        "exit_date", "exit_price", "exit_reason", "return_pct", "pnl",
    }
    assert expected <= cols, f"missing: {expected - cols}"
    # price_basis + time_stop_days are nullable (NULL while pending).
    assert PaperTrade.__table__.c.price_basis.nullable
    assert PaperTrade.__table__.c.time_stop_days.nullable
    # fill fields nullable
    for f in ("entry_date", "entry_price", "shares", "allocated_amount",
              "residual_cash"):
        assert PaperTrade.__table__.c[f].nullable


def test_paper_trade_unique_thesis_id():
    uniques = [
        c for c in PaperTrade.__table__.constraints
        if c.__class__.__name__ == "UniqueConstraint"
    ]
    assert any({col.name for col in c.columns} == {"thesis_id"} for c in uniques)


def test_paper_trade_fks():
    thesis_fk = list(PaperTrade.__table__.c.thesis_id.foreign_keys)
    assert thesis_fk and thesis_fk[0].column.table.name == "analysis_results"
    scr_fk = list(PaperTrade.__table__.c.screened_record_id.foreign_keys)
    assert scr_fk and scr_fk[0].column.table.name == "screened_stocks"


def test_paper_trade_partial_unique_predicate():
    idx = next(
        i for i in PaperTrade.__table__.indexes
        if i.name == "idx_paper_trade_active_symbol"
    )
    assert idx.unique
    where = idx.dialect_options["postgresql"]["where"]
    assert "status IN ('pending', 'open')" in str(where)


def test_paper_skip_orm_shape_and_reason_set():
    cols = {c.name for c in inspect(PaperSkip).columns}
    assert {"id", "thesis_id", "symbol", "scan_date", "reason", "created_at"} <= cols
    uniques = [
        c for c in PaperSkip.__table__.constraints
        if c.__class__.__name__ == "UniqueConstraint"
    ]
    assert any({col.name for col in c.columns} == {"thesis_id"} for c in uniques)
    checks = [
        c for c in PaperSkip.__table__.constraints
        if c.__class__.__name__ == "CheckConstraint"
    ]
    sqltext = " ".join(str(c.sqltext) for c in checks)
    for reason in (
        "symbol_already_active", "missing_levels", "missing_screened_record",
        "gap_invalidated", "same_symbol_lower_conviction",
    ):
        assert reason in sqltext, f"reason {reason} missing from CHECK"


def test_paper_report_snapshot_orm_shape():
    cols = {c.name for c in inspect(PaperReportSnapshot).columns}
    assert {"id", "report_type", "as_of_date", "payload", "created_at"} <= cols
    uniques = [
        c for c in PaperReportSnapshot.__table__.constraints
        if c.__class__.__name__ == "UniqueConstraint"
    ]
    assert any(
        {col.name for col in c.columns} == {"report_type", "as_of_date"}
        for c in uniques
    )


def test_screened_stocks_additive_columns():
    """A2 — entry_price/stop_loss/target_price/rr_ratio added (nullable);
    pattern_type unchanged/not duplicated."""
    cols = inspect(ScreenedStockRecord).columns
    names = [c.name for c in cols]
    for c in ("entry_price", "stop_loss", "target_price", "rr_ratio"):
        assert c in names
        assert ScreenedStockRecord.__table__.c[c].nullable
    # pattern_type present exactly once (not re-added).
    assert names.count("pattern_type") == 1


def test_paper_trade_not_a_hypertable_marker():
    """A1 — paper_trade is NOT registered for hypertable conversion."""
    from rainier.core.models import HYPERTABLES

    assert "paper_trade" not in HYPERTABLES


# --------------------------------------------------------------------------
# Postgres-only schema assertions (sqlite can't model these)
# --------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_a1_paper_trade_pg_shape(pg_legacy_engine):
    insp = inspect(pg_legacy_engine)
    cols = {c["name"] for c in insp.get_columns("paper_trade")}
    assert {"price_basis", "time_stop_days", "planned_entry_price"} <= cols
    # NOT a hypertable — the timescaledb catalog has no entry (or the ext is
    # absent, which also proves not-a-hypertable).
    with pg_legacy_engine.connect() as conn:
        is_ht = conn.execute(
            text(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema='_timescaledb_catalog' AND table_name='hypertable'"
            )
        ).scalar()
        if is_ht:
            n = conn.execute(
                text(
                    "SELECT count(*) FROM _timescaledb_catalog.hypertable "
                    "WHERE table_name='paper_trade'"
                )
            ).scalar()
            assert n == 0


@pytest.mark.requires_postgres
def test_a3_a4_pg_objects_exist(pg_legacy_engine):
    insp = inspect(pg_legacy_engine)
    tables = set(insp.get_table_names())
    assert {"paper_trade", "paper_skip", "paper_report_snapshot"} <= tables
    payload_col = next(
        c for c in insp.get_columns("paper_report_snapshot") if c["name"] == "payload"
    )
    assert "JSON" in str(payload_col["type"]).upper()


@pytest.mark.requires_postgres
def test_a4_paper_skip_reason_check_rejects_unknown(pg_legacy_engine):
    with pg_legacy_engine.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(
                text(
                    "INSERT INTO paper_skip (thesis_id, symbol, scan_date, reason) "
                    "VALUES (1, 'AAA', '2026-01-01', 'not_a_real_reason')"
                )
            )


@pytest.mark.requires_postgres
@pytest.mark.parametrize(
    "first,second,should_conflict",
    [
        ("pending", "pending", True),
        ("pending", "open", True),
        ("open", "open", True),
        ("closed", "pending", False),
        ("expired", "open", False),
    ],
)
def test_a6_partial_unique_pending_open(pg_legacy_engine, first, second, should_conflict):
    """A6 — partial-unique on symbol WHERE status IN ('pending','open')."""
    with pg_legacy_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO analysis_results (id, llm_model, prompt_template) "
                "VALUES (101, 'm', 't'), (102, 'm', 't') ON CONFLICT DO NOTHING"
            )
        )

    def _insert(conn, thesis_id, status):
        conn.execute(
            text(
                "INSERT INTO paper_trade "
                "(thesis_id, symbol, scan_date, session_name, status, "
                " planned_entry_price, stop_loss, target_price) "
                "VALUES (:t, 'ZZZ', '2026-01-01', 'close', :s, 100, 90, 120)"
            ),
            {"t": thesis_id, "s": status},
        )

    with pg_legacy_engine.begin() as conn:
        _insert(conn, 101, first)
    if should_conflict:
        with pytest.raises(IntegrityError):
            with pg_legacy_engine.begin() as conn:
                _insert(conn, 102, second)
    else:
        with pg_legacy_engine.begin() as conn:
            _insert(conn, 102, second)


@pytest.mark.requires_postgres
def test_a5_downgrade_reverses(pg_legacy_engine):
    """A5 — downgrade drops exactly the 0005 objects; FK-target tables remain."""
    from pathlib import Path

    down = Path(__file__).resolve().parents[2] / "migrations" / "0005_paper_tracker_downgrade.sql"
    with pg_legacy_engine.begin() as conn:
        conn.execute(text(down.read_text()))

    insp = inspect(pg_legacy_engine)
    tables = set(insp.get_table_names())
    assert "paper_trade" not in tables
    assert "paper_skip" not in tables
    assert "paper_report_snapshot" not in tables
    # FK targets survive.
    assert {"analysis_results", "screened_stocks"} <= tables
    # The four additive columns are gone; pattern_type survives.
    scr_cols = {c["name"] for c in insp.get_columns("screened_stocks")}
    assert not ({"entry_price", "stop_loss", "target_price", "rr_ratio"} & scr_cols)
    assert "pattern_type" in scr_cols
