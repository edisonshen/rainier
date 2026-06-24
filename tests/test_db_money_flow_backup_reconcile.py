"""WS C — backup is a `data_date`-aware reconcile, not HWM-by-id append.

The rebuild-the-day fix (WS A) makes a day's `money_flow_snapshots` rows change
AFTER first write: a later same-day scrape DELETEs the day's rows and re-INSERTs
fresh ones (new autoincrement ids). The old backup copy was high-water-mark by
`id` (cursor = `MAX(id)`), insert-only — it structurally cannot:

  * purge the orphaned OLD ids the rebuild deleted (they sit below the HWM), nor
  * re-copy a changed day whose new rows it already passed.

So `verify_backup`'s full-window checksum / missing-row checks would fail the
nightly cron after any same-day rebuild. WS C replaces the cursor with a
reconcile over the UNION of source+backup `data_date`s: a day that disagrees
(newer `captured_at`, or a row-count / checksum mismatch — incl. a backup-only
day the source shrank) is delete-by-day-then-recopied; new days still append; a
day gone-to-zero in source is left deleted. `verify_backup` stays green and
re-runnable after a rebuild.

Both engines are SQLite in-memory (engine-agnostic core).
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
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
    delete,
    insert,
    select,
)
from sqlalchemy.types import JSON

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

DAY1 = date(2026, 6, 1)
DAY2 = date(2026, 6, 2)
T1 = datetime(2026, 6, 1, 15, 45, tzinfo=timezone.utc)
T2 = datetime(2026, 6, 1, 22, 0, tzinfo=timezone.utc)
T_DAY2 = datetime(2026, 6, 2, 22, 0, tzinfo=timezone.utc)


def _row(i: int, *, data_date: date = DAY1, captured_at: datetime = T1, **over) -> dict:
    base = {
        "id": i,
        "captured_at": captured_at,
        "capture_session": "morning",
        "data_date": data_date,
        "view_type": "daily",
        "ranking_type": "top100",
        "symbol": f"S{i:03d}",
        "rank": i,
        "daily_change": i,
        "sector": "Tech",
        "industry": "SW",
        "long_short": "Long in",
        "raw_data": {"id": i},
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


def _import():
    from rainier.db import money_flow_backup as mfb

    return mfb


def _seed(eng, rows):
    with eng.begin() as conn:
        conn.execute(insert(_SRC), rows)


def _dst_ids(eng, bt) -> list[int]:
    with eng.connect() as conn:
        return sorted(conn.execute(select(bt.c.id)).scalars().all())


def test_same_day_rebuild_purges_orphans_and_recopies(src_engine, dst_engine):
    """Morning day1 (ids 1,2) backed up; a later same-day rebuild DELETEs 1,2 and
    INSERTs fresh ids 3,4 with flipped data. The backup must end holding ONLY
    {3,4} (orphans 1,2 purged) and verify must be green."""
    mfb = _import()
    _seed(src_engine, [_row(1), _row(2)])
    mfb.backup_money_flow(src_engine, dst_engine)
    bt = mfb.backup_table()
    assert _dst_ids(dst_engine, bt) == [1, 2]

    # Rebuild day1: delete the day's rows, insert fresh ones with a new captured_at
    # and flipped dominance (simulates WS A).
    with src_engine.begin() as conn:
        conn.execute(delete(_SRC).where(_SRC.c.data_date == DAY1))
    _seed(
        src_engine,
        [
            _row(3, captured_at=T2, capture_session="midday", long_short="No dominance"),
            _row(4, captured_at=T2, capture_session="midday", long_short="No dominance"),
        ],
    )

    mfb.backup_money_flow(src_engine, dst_engine)

    assert _dst_ids(dst_engine, bt) == [3, 4], "orphaned old ids 1,2 must be purged"

    run_max = 4
    report = mfb.verify_backup(src_engine, dst_engine, run_max=run_max)
    assert report.ok, report.failures


def test_reconcile_is_idempotent_after_rebuild(src_engine, dst_engine):
    """A second reconcile run after a rebuild copies nothing new and stays green."""
    mfb = _import()
    _seed(src_engine, [_row(1, data_date=DAY1)])
    mfb.backup_money_flow(src_engine, dst_engine)
    with src_engine.begin() as conn:
        conn.execute(delete(_SRC).where(_SRC.c.data_date == DAY1))
    _seed(src_engine, [_row(2, data_date=DAY1, captured_at=T2)])
    mfb.backup_money_flow(src_engine, dst_engine)

    r2 = mfb.backup_money_flow(src_engine, dst_engine)  # second run, no changes
    assert r2.copied == 0
    bt = mfb.backup_table()
    assert _dst_ids(dst_engine, bt) == [2]
    assert mfb.verify_backup(src_engine, dst_engine, run_max=2).ok


def test_new_day_still_appends(src_engine, dst_engine):
    """An unchanged old day + a brand-new day: only the new day is copied."""
    mfb = _import()
    _seed(src_engine, [_row(1, data_date=DAY1)])
    mfb.backup_money_flow(src_engine, dst_engine)

    _seed(src_engine, [_row(2, data_date=DAY2, captured_at=T_DAY2)])
    result = mfb.backup_money_flow(src_engine, dst_engine)

    bt = mfb.backup_table()
    assert _dst_ids(dst_engine, bt) == [1, 2]
    assert result.copied >= 1
    assert mfb.verify_backup(src_engine, dst_engine, run_max=2).ok


def test_day_shrunk_in_source_recopies_smaller(src_engine, dst_engine):
    """A day that SHRANK in source (3 rows -> 1 row, same/lower max-id) is a
    backup-only-larger day MAX(id) can't detect — the reconcile must still purge
    the dropped rows."""
    mfb = _import()
    _seed(src_engine, [_row(1), _row(2), _row(3)])
    mfb.backup_money_flow(src_engine, dst_engine)
    bt = mfb.backup_table()
    assert _dst_ids(dst_engine, bt) == [1, 2, 3]

    # Rebuild day1 smaller: keep only id=1 (delete 2,3), bump its captured_at.
    with src_engine.begin() as conn:
        conn.execute(delete(_SRC).where(_SRC.c.id.in_([2, 3])))
        conn.execute(
            _SRC.update().where(_SRC.c.id == 1).values(captured_at=T2)
        )

    mfb.backup_money_flow(src_engine, dst_engine)
    assert _dst_ids(dst_engine, bt) == [1], "dropped rows 2,3 must be purged from backup"
    assert mfb.verify_backup(src_engine, dst_engine, run_max=3).ok


def test_day_gone_to_zero_left_deleted(src_engine, dst_engine):
    """If a whole day disappears from source, the backup day is left deleted."""
    mfb = _import()
    _seed(src_engine, [_row(1, data_date=DAY1), _row(2, data_date=DAY2, captured_at=T_DAY2)])
    mfb.backup_money_flow(src_engine, dst_engine)

    with src_engine.begin() as conn:
        conn.execute(delete(_SRC).where(_SRC.c.data_date == DAY1))

    mfb.backup_money_flow(src_engine, dst_engine)
    bt = mfb.backup_table()
    assert _dst_ids(dst_engine, bt) == [2], "DAY1 rows must be purged from backup"
    assert mfb.verify_backup(src_engine, dst_engine, run_max=2).ok
