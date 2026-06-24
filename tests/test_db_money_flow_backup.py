"""Tests for the nightly money_flow_snapshots -> Neon backup.

Design: docs/DESIGN-money-flow-neon-backup.md §2/§3 (9-item test plan).

The copy/verify core (``rainier.db.money_flow_backup``) is engine-agnostic: it
takes a ``src`` engine (local TimescaleDB in prod) and a ``dst`` engine (Neon in
prod). Here both sides are SQLite in-memory so the logic runs with no Postgres
dependency (the SCHEMA/JSONB-on-Postgres specifics are covered by the
``requires_postgres`` migration tests in ``test_db_migrations.py``).

Cross-engine HWM-by-id, insert-only, idempotent copy + ``--verify`` integrity
(max-id match, missing-row reconciliation on the full composite key, full-window
canonicalized checksum, id-uniqueness guard). The CLI surface (DATABASE_URL
fail-loud, ``--skip-if-unconfigured``) is exercised via CliRunner with the local
engine monkeypatched onto a temp SQLite DB.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from click.testing import CliRunner
from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
    text,
)
from sqlalchemy.types import JSON

# ---------------------------------------------------------------------------
# Synthetic source table — mirrors core/models.py:MoneyFlowSnapshot columns but
# without the stocks FK / TimescaleDB hypertable so it runs on SQLite. The PK is
# the same composite (id, captured_at) as the real source.
# ---------------------------------------------------------------------------

_SRC_META = MetaData()
_SRC = Table(
    "money_flow_snapshots",
    _SRC_META,
    Column("id", BigInteger, nullable=False),
    Column("captured_at", DateTime(timezone=True), nullable=False),
    Column("capture_session", String(20), nullable=False),
    Column("data_date", Date, nullable=False),
    Column("view_type", String(10), nullable=False),
    Column("ranking_type", String(10), nullable=False),
    Column("symbol", String(10), nullable=False),
    Column("rank", Integer, nullable=False),
    Column("daily_change", Integer),
    Column("sector", String(100)),
    Column("industry", String(200)),
    Column("long_short", String(50)),
    Column("raw_data", JSON),
)

_CAP = datetime(2026, 6, 1, 22, 0, tzinfo=timezone.utc)


def _row(i: int, **over) -> dict:
    base = {
        "id": i,
        "captured_at": _CAP,
        "capture_session": "close",
        "data_date": date(2026, 6, 1),
        "view_type": "daily",
        "ranking_type": "long",
        "symbol": f"S{i:03d}",
        "rank": i,
        "daily_change": i - 50,
        "sector": "Tech",
        "industry": "Software",
        "long_short": "long",
        "raw_data": {"rank": i, "nested": {"z": 1, "a": 2}, "tags": ["x", "y"]},
    }
    base.update(over)
    return base


@pytest.fixture
def src_engine():
    eng = create_engine("sqlite://")
    _SRC_META.create_all(eng)
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture
def dst_engine():
    eng = create_engine("sqlite://")
    try:
        yield eng
    finally:
        eng.dispose()


def _seed_src(eng, rows):
    with eng.begin() as conn:
        conn.execute(insert(_SRC), rows)


def _count(eng, table_name: str) -> int:
    with eng.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()


def _dst_ids(eng, backup_table) -> list[int]:
    with eng.connect() as conn:
        return sorted(
            conn.execute(select(backup_table.c.id)).scalars().all()
        )


# ===========================================================================
# Module import (TDD-red: module does not exist yet)
# ===========================================================================


def _import():
    from rainier.db import money_flow_backup as mfb

    return mfb


# ===========================================================================
# Test 1 — bootstrap: empty backup -> copies all rows <= run_max
# ===========================================================================


def test_bootstrap_copies_all_rows(src_engine, dst_engine):
    mfb = _import()
    _seed_src(src_engine, [_row(i) for i in range(1, 6)])

    result = mfb.backup_money_flow(src_engine, dst_engine)

    assert result.copied == 5
    assert result.hwm_before == 0
    assert result.run_max == 5
    assert _count(dst_engine, "backup_money_flow_snapshots") == 5
    assert _dst_ids(dst_engine, mfb.backup_table()) == [1, 2, 3, 4, 5]


# ===========================================================================
# Test 2 — incremental: a brand-new DAY appends; the unchanged old day is skipped
# ===========================================================================

_CAP_DAY2 = datetime(2026, 6, 2, 22, 0, tzinfo=timezone.utc)


def test_incremental_new_day_appends(src_engine, dst_engine):
    """Under the data_date-aware reconcile, genuinely-new rows are a NEW trading
    day; the unchanged prior day is not recopied."""
    mfb = _import()
    _seed_src(src_engine, [_row(i) for i in range(1, 4)])  # day 2026-06-01
    mfb.backup_money_flow(src_engine, dst_engine)

    # A new trading day (2026-06-02) — distinct data_date + captured_at.
    _seed_src(
        src_engine,
        [_row(i, data_date=date(2026, 6, 2), captured_at=_CAP_DAY2) for i in range(4, 7)],
    )
    result = mfb.backup_money_flow(src_engine, dst_engine)

    assert result.copied == 3  # only the new day's 3 rows; old day skipped
    assert result.hwm_before == 3
    assert result.run_max == 6
    assert _dst_ids(dst_engine, mfb.backup_table()) == [1, 2, 3, 4, 5, 6]
    assert mfb.verify_backup(src_engine, dst_engine, run_max=6).ok


# ===========================================================================
# Test 3 — idempotent re-run: zero copied, no duplicates/errors
# ===========================================================================


def test_idempotent_rerun(src_engine, dst_engine):
    mfb = _import()
    _seed_src(src_engine, [_row(i) for i in range(1, 4)])
    mfb.backup_money_flow(src_engine, dst_engine)

    result = mfb.backup_money_flow(src_engine, dst_engine)

    assert result.copied == 0
    assert _count(dst_engine, "backup_money_flow_snapshots") == 3


# ===========================================================================
# Test 4 — JSONB fidelity: raw_data round-trips intact (typed Core insert)
# ===========================================================================


def test_jsonb_fidelity(src_engine, dst_engine):
    mfb = _import()
    payload = {"a": 1, "b": [1, 2, {"c": 3}], "d": None, "e": "txt"}
    _seed_src(src_engine, [_row(1, raw_data=payload)])

    mfb.backup_money_flow(src_engine, dst_engine)

    bt = mfb.backup_table()
    with dst_engine.connect() as conn:
        got = conn.execute(select(bt.c.raw_data).where(bt.c.id == 1)).scalar_one()
    assert got == payload, f"raw_data was mangled: {got!r}"


# ===========================================================================
# Test 5 — stable upper bound / race: a row added AFTER run_max is captured is
# NOT copied this run, but IS copied next run (no torn read).
# ===========================================================================


def test_stable_upper_bound_race(src_engine, dst_engine):
    mfb = _import()
    _seed_src(src_engine, [_row(i) for i in range(1, 4)])

    # Simulate a scrape landing mid-backup: a hook fires AFTER run_max is
    # captured but BEFORE rows are read, inserting id=99 on a NEW trading day.
    def _mid(stage):
        if stage == "after_run_max":
            _seed_src(src_engine, [_row(99, data_date=date(2026, 6, 2), captured_at=_CAP_DAY2)])

    result = mfb.backup_money_flow(src_engine, dst_engine, _race_hook=_mid)
    # id=99 > run_max(3) -> excluded this run (not a torn read).
    assert result.run_max == 3
    assert result.copied == 3
    assert _dst_ids(dst_engine, mfb.backup_table()) == [1, 2, 3]

    # Next run reconciles the new day and picks up id=99.
    result2 = mfb.backup_money_flow(src_engine, dst_engine)
    assert result2.copied == 1
    assert _dst_ids(dst_engine, mfb.backup_table()) == [1, 2, 3, 99]


# ===========================================================================
# Test 7 — --verify strength
# ===========================================================================


def test_verify_passes_after_clean_copy(src_engine, dst_engine):
    mfb = _import()
    _seed_src(src_engine, [_row(i) for i in range(1, 6)])
    mfb.backup_money_flow(src_engine, dst_engine)

    report = mfb.verify_backup(src_engine, dst_engine, run_max=5)
    assert report.ok, report.failures


def test_verify_raw_data_key_reorder_still_matches(src_engine, dst_engine):
    """7(b) — a raw_data key reorder must STILL match after canonicalization."""
    mfb = _import()
    _seed_src(src_engine, [_row(1, raw_data={"z": 1, "a": 2, "m": [3, 4]})])
    mfb.backup_money_flow(src_engine, dst_engine)

    # Rewrite the backup row's raw_data with the SAME content, different key order.
    bt = mfb.backup_table()
    with dst_engine.begin() as conn:
        conn.execute(
            bt.update().where(bt.c.id == 1).values(raw_data={"a": 2, "m": [3, 4], "z": 1})
        )

    report = mfb.verify_backup(src_engine, dst_engine, run_max=1)
    assert report.ok, f"key-reorder must canonicalize equal: {report.failures}"


def test_canon_captured_at_normalizes_offset_to_utc():
    """Codex P2 — the same instant rendered with a DIFFERENT tz offset (different
    session TimeZone) must canonicalize to the SAME string.

    On real Postgres ``timestamptz`` the driver returns tz-aware datetimes whose
    OFFSET depends on the session ``TimeZone`` (e.g. local TimescaleDB vs Neon).
    The missing-row key, the checksum sort key, and the checksum cell all route
    through ``_canon_captured_at``, so a pure offset-rendering difference between
    the two engines is verify-clean instead of false-failing the nightly cron.

    (SQLite's ``DateTime(timezone=True)`` strips tz on round-trip, so this is a
    direct unit test of the normalization the engine harness can't reproduce.)
    """
    from datetime import datetime as _datetime
    from datetime import timedelta
    from datetime import timezone as _tz

    mfb = _import()
    utc = _datetime(2026, 6, 1, 22, 0, tzinfo=_tz.utc)
    pst = utc.astimezone(_tz(timedelta(hours=-7)))  # same instant, -07:00 offset
    assert utc == pst  # same instant ...
    assert str(utc) != str(pst)  # ... but different raw rendering (pre-fix bug)

    # Normalization collapses them to one canonical UTC string.
    assert mfb._canon_captured_at(utc) == mfb._canon_captured_at(pst)
    assert mfb._canon_captured_at(utc) == "2026-06-01T22:00:00+00:00"

    # The missing-row key and the checksum both agree across the offset diff.
    row_utc = _row(1, captured_at=utc)
    row_pst = _row(1, captured_at=pst)
    assert mfb._checksum([row_utc]) == mfb._checksum([row_pst])

    # A genuinely different instant still diverges.
    other = _datetime(2026, 6, 1, 23, 0, tzinfo=_tz.utc)
    assert mfb._canon_captured_at(utc) != mfb._canon_captured_at(other)
    assert mfb._checksum([row_utc]) != mfb._checksum([_row(1, captured_at=other)])


def test_verify_missing_row_fails(src_engine, dst_engine):
    """7(a) — a missing backup (id, captured_at) in (0, run_max] -> fail."""
    mfb = _import()
    _seed_src(src_engine, [_row(i) for i in range(1, 6)])
    mfb.backup_money_flow(src_engine, dst_engine)

    bt = mfb.backup_table()
    with dst_engine.begin() as conn:
        conn.execute(bt.delete().where(bt.c.id == 3))

    report = mfb.verify_backup(src_engine, dst_engine, run_max=5)
    assert not report.ok
    assert any("missing" in f.lower() or "max" in f.lower() for f in report.failures)


def test_verify_checksum_divergence_fails(src_engine, dst_engine):
    """7(b) — content-different but count-equal must NOT pass."""
    mfb = _import()
    _seed_src(src_engine, [_row(i) for i in range(1, 6)])
    mfb.backup_money_flow(src_engine, dst_engine)

    bt = mfb.backup_table()
    with dst_engine.begin() as conn:
        conn.execute(bt.update().where(bt.c.id == 2).values(rank=999))

    report = mfb.verify_backup(src_engine, dst_engine, run_max=5)
    assert not report.ok
    assert any("checksum" in f.lower() for f in report.failures)


def test_verify_raw_data_value_divergence_fails(src_engine, dst_engine):
    """7(b) — a genuine raw_data VALUE change (not a reorder) -> checksum fail."""
    mfb = _import()
    _seed_src(src_engine, [_row(1, raw_data={"k": 1})])
    mfb.backup_money_flow(src_engine, dst_engine)

    bt = mfb.backup_table()
    with dst_engine.begin() as conn:
        conn.execute(bt.update().where(bt.c.id == 1).values(raw_data={"k": 2}))

    report = mfb.verify_backup(src_engine, dst_engine, run_max=1)
    assert not report.ok
    assert any("checksum" in f.lower() for f in report.failures)


def test_reconcile_self_heals_edited_old_row(src_engine, dst_engine):
    """An edited OLD source row (same (id, captured_at), different value) is
    detected by the per-day checksum and RE-COPIED by the reconcile.

    Under the old HWM-by-id copy an in-place edit below the high-water mark was
    never recopied — verify could only flag it. The data_date-aware reconcile
    fingerprints each day and recopies any day whose checksum changed, so the
    backup self-heals and verify is green.
    """
    mfb = _import()
    _seed_src(src_engine, [_row(i) for i in range(1, 4)])
    mfb.backup_money_flow(src_engine, dst_engine)  # backup now has 1,2,3

    # Edit an OLD source row in place (id=2). Same data_date -> the day's checksum
    # changes, so the reconcile recopies the whole day.
    with src_engine.begin() as conn:
        conn.execute(_SRC.update().where(_SRC.c.id == 2).values(rank=777))
    mfb.backup_money_flow(src_engine, dst_engine)

    # Backup now matches source — verify is green (self-healed).
    report = mfb.verify_backup(src_engine, dst_engine, run_max=3)
    assert report.ok, report.failures
    bt = mfb.backup_table()
    with dst_engine.connect() as conn:
        rank = conn.execute(select(bt.c.rank).where(bt.c.id == 2)).scalar_one()
    assert rank == 777, "edited row must be recopied into the backup"


def test_verify_duplicate_source_id_fails(src_engine, dst_engine):
    """7(c) — a duplicate source id trips the uniqueness guard."""
    mfb = _import()
    _seed_src(src_engine, [_row(1)])
    # Same id, different captured_at (allowed by composite PK) -> duplicate id.
    other_cap = datetime(2026, 6, 2, 22, 0, tzinfo=timezone.utc)
    _seed_src(src_engine, [_row(1, captured_at=other_cap)])

    report = mfb.verify_backup(src_engine, dst_engine, run_max=1)
    assert not report.ok
    assert any(
        "unique" in f.lower() or "duplicate" in f.lower() for f in report.failures
    )


def test_verify_count_equal_content_different_does_not_pass(src_engine, dst_engine):
    """Explicit: count-equal but content-different must NOT pass verify."""
    mfb = _import()
    _seed_src(src_engine, [_row(i) for i in range(1, 4)])
    mfb.backup_money_flow(src_engine, dst_engine)

    bt = mfb.backup_table()
    with dst_engine.begin() as conn:
        conn.execute(bt.update().where(bt.c.id == 1).values(symbol="ZZZZ"))

    # counts still equal (3 == 3)
    assert _count(src_engine, "money_flow_snapshots") == _count(
        dst_engine, "backup_money_flow_snapshots"
    )
    report = mfb.verify_backup(src_engine, dst_engine, run_max=3)
    assert not report.ok


# ===========================================================================
# Test 8 — transaction safety: a forced mid-insert failure leaves the backup
# unchanged (hwm not advanced); next run retries cleanly.
# ===========================================================================


def test_transaction_safety_rolls_back(src_engine, dst_engine):
    mfb = _import()
    _seed_src(src_engine, [_row(i) for i in range(1, 6)])

    boom = RuntimeError("forced mid-insert failure")

    def _fail(stage):
        if stage == "before_commit":
            raise boom

    with pytest.raises(RuntimeError):
        mfb.backup_money_flow(src_engine, dst_engine, _race_hook=_fail)

    # Nothing committed -> backup unchanged (table may not even exist or is empty).
    bt = mfb.backup_table()
    bt.create(dst_engine, checkfirst=True)
    assert _count(dst_engine, "backup_money_flow_snapshots") == 0

    # Next run retries from the same hwm (0) and copies everything.
    result = mfb.backup_money_flow(src_engine, dst_engine)
    assert result.copied == 5
    assert _dst_ids(dst_engine, mfb.backup_table()) == [1, 2, 3, 4, 5]


# ===========================================================================
# Empty source — nothing to copy, no error.
# ===========================================================================


def test_empty_source_no_op(src_engine, dst_engine):
    mfb = _import()
    result = mfb.backup_money_flow(src_engine, dst_engine)
    assert result.copied == 0
    assert result.run_max == 0


# ===========================================================================
# Test 6 — CLI: DATABASE_URL unset -> fail loud; --skip-if-unconfigured -> exit 0
# ===========================================================================


@pytest.fixture
def _local_src_path(tmp_path):
    """A file-backed SQLite local source seeded with rows, wired as the legacy
    engine via core.database.get_engine monkeypatch."""
    db_path = tmp_path / "local.db"
    eng = create_engine(f"sqlite:///{db_path}")
    _SRC_META.create_all(eng)
    with eng.begin() as conn:
        conn.execute(insert(_SRC), [_row(i) for i in range(1, 4)])
    eng.dispose()
    return db_path


def test_cli_requires_database_url(monkeypatch, _local_src_path):
    """DATABASE_URL unset -> non-zero, clean ClickException, no traceback."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from rainier.cli import cli

    res = CliRunner().invoke(cli, ["db", "backup-money-flow"])
    assert res.exit_code != 0
    assert res.exception is None or isinstance(res.exception, SystemExit), (
        f"expected a clean ClickException, got {res.exception!r}"
    )
    assert "DATABASE_URL" in res.output


def test_cli_skip_if_unconfigured_exits_zero(monkeypatch, _local_src_path):
    """--skip-if-unconfigured turns unset DATABASE_URL into warn + exit 0."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from rainier.cli import cli

    res = CliRunner().invoke(cli, ["db", "backup-money-flow", "--skip-if-unconfigured"])
    assert res.exit_code == 0, res.output
    assert "skip" in res.output.lower() or "warn" in res.output.lower()


def test_cli_backup_copies_to_neon(monkeypatch, tmp_path, _local_src_path):
    """End-to-end CLI: local (legacy) -> dst (DATABASE_URL) copies rows."""
    from rainier.cli import cli
    from rainier.core import database

    # Wire the legacy engine to the seeded local SQLite source.
    local_eng = create_engine(f"sqlite:///{_local_src_path}")
    monkeypatch.setattr(database, "get_engine", lambda *a, **k: local_eng)

    # Point the canonical (Neon) side at a file SQLite via db.engine.get_engine.
    dst_path = tmp_path / "neon.db"
    from rainier.db import engine as db_engine

    monkeypatch.setattr(
        db_engine, "get_engine", lambda: create_engine(f"sqlite:///{dst_path}")
    )

    res = CliRunner().invoke(cli, ["db", "backup-money-flow", "--verify"])
    assert res.exit_code == 0, res.output
    assert "backed up 3" in res.output.lower() or "3 new" in res.output.lower()

    check = create_engine(f"sqlite:///{dst_path}")
    assert _count(check, "backup_money_flow_snapshots") == 3
    check.dispose()
    local_eng.dispose()


def test_cli_registered():
    from rainier.cli import cli

    res = CliRunner().invoke(cli, ["db", "--help"])
    assert res.exit_code == 0, res.output
    assert "backup-money-flow" in res.output


# ===========================================================================
# Cron — config/cron.yaml has the money-flow-backup job (design §2.4)
# ===========================================================================


def test_cron_money_flow_backup_job_present():
    from pathlib import Path

    from rainier.scheduler.jobs import load_config

    repo_root = Path(__file__).resolve().parents[1]
    jobs = load_config(repo_root / "config" / "cron.yaml")
    by_name = {j["name"]: j for j in jobs}
    assert "money-flow-backup" in by_name, "money-flow-backup cron job missing"

    job = by_name["money-flow-backup"]
    assert job["schedule"] == "30 15 * * 1-5", job["schedule"]
    # Runs --verify so the cron fires Discord on any integrity failure.
    assert "backup-money-flow" in job["command"]
    assert "--verify" in job["command"]
    # Cron must NOT pass --skip-if-unconfigured (prod stays loud on unset URL).
    assert "--skip-if-unconfigured" not in job["command"]
    assert job["enabled"] is True
    assert job["log"] == "data/money-flow-backup.log"
