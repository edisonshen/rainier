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


def test_day_lost_in_source_is_retained_in_backup(src_engine, dst_engine):
    """Codex P1 regression — a historical day the SOURCE lost (bad restore, manual
    cleanup, corruption) must be RETAINED in the backup, NOT purged. The off-site
    copy is disaster recovery; it never propagates a source-side delete. verify
    stays green because it asserts the backup is a content-faithful SUPERSET of the
    source, not an exact mirror."""
    mfb = _import()
    _seed(src_engine, [_row(1, data_date=DAY1), _row(2, data_date=DAY2, captured_at=T_DAY2)])
    mfb.backup_money_flow(src_engine, dst_engine)

    with src_engine.begin() as conn:
        conn.execute(delete(_SRC).where(_SRC.c.data_date == DAY1))

    mfb.backup_money_flow(src_engine, dst_engine)
    bt = mfb.backup_table()
    assert _dst_ids(dst_engine, bt) == [1, 2], "lost DAY1 must be retained in the backup"
    assert mfb.verify_backup(src_engine, dst_engine, run_max=2).ok


def test_concurrent_rebuild_does_not_purge_day_from_backup(src_engine, dst_engine):
    """Codex P1 regression — a same-day rebuild landing AFTER the source snapshot
    must NOT cause the day to be purged from the backup.

    Day1 (ids 1,2) is backed up. Then, via the `after_run_max` race hook, we
    simulate WS A rebuilding day1 (delete 1,2 -> insert 3,4) the instant after the
    reconcile captured the source snapshot. The reconcile read run_max + src_rows
    from one consistent snapshot, so it still sees {1,2}; the backup must keep the
    day (NOT delete it and write nothing), and the rebuild is reconciled next run."""
    mfb = _import()
    _seed(src_engine, [_row(1), _row(2)])
    mfb.backup_money_flow(src_engine, dst_engine)
    bt = mfb.backup_table()
    assert _dst_ids(dst_engine, bt) == [1, 2]

    def _rebuild_after_snapshot(stage: str) -> None:
        if stage == "after_run_max":
            with src_engine.begin() as conn:
                conn.execute(delete(_SRC).where(_SRC.c.data_date == DAY1))
            _seed(
                src_engine,
                [_row(3, captured_at=T2, capture_session="midday"),
                 _row(4, captured_at=T2, capture_session="midday")],
            )

    mfb.backup_money_flow(src_engine, dst_engine, _race_hook=_rebuild_after_snapshot)
    assert _dst_ids(dst_engine, bt) == [1, 2], "day must not be purged on a torn read"

    # Next run sees the rebuilt day whole and reconciles to {3,4}, still green.
    mfb.backup_money_flow(src_engine, dst_engine)
    assert _dst_ids(dst_engine, bt) == [3, 4]
    assert mfb.verify_backup(src_engine, dst_engine, run_max=4).ok


def test_deleted_latest_day_with_high_ids_is_retained(src_engine, dst_engine):
    """When the day the source lost held the HIGHEST ids, dropping it lowers
    run_max below those ids. The backup retains that day (non-destructive
    invariant); the retained ids sit ABOVE the lowered run_max so verify — bounded
    to id<=run_max — stays green.

    day1=id1, day2=id2 backed up. Delete day2 (the high-id day) from source ->
    next run_max=1. day2 is backup-only -> retained, not purged."""
    mfb = _import()
    _seed(src_engine, [_row(1, data_date=DAY1), _row(2, data_date=DAY2, captured_at=T_DAY2)])
    mfb.backup_money_flow(src_engine, dst_engine)
    bt = mfb.backup_table()
    assert _dst_ids(dst_engine, bt) == [1, 2]

    with src_engine.begin() as conn:
        conn.execute(delete(_SRC).where(_SRC.c.data_date == DAY2))  # drop the high-id day

    result = mfb.backup_money_flow(src_engine, dst_engine)
    assert result.run_max == 1, "run_max drops to the remaining max source id"
    assert _dst_ids(dst_engine, bt) == [1, 2], "lost high-id day must be retained in backup"
    assert mfb.verify_backup(src_engine, dst_engine, run_max=1).ok


def test_empty_source_does_not_wipe_backup(src_engine, dst_engine):
    """Review regression — an EMPTY source must NOT purge the whole backup.

    The reconcile deletes any backup day absent from the source. A source with NO
    rows (fresh/rebuilt local DB, failed restore, or mis-pointed
    LEGACY_DATABASE_URL) would otherwise flag EVERY backup day as
    backup-only-and-gone and wipe the off-site copy of record on a routine cron.
    The empty-source guard aborts as a no-op and leaves the backup intact."""
    mfb = _import()
    _seed(src_engine, [_row(1, data_date=DAY1), _row(2, data_date=DAY2, captured_at=T_DAY2)])
    mfb.backup_money_flow(src_engine, dst_engine)
    bt = mfb.backup_table()
    assert _dst_ids(dst_engine, bt) == [1, 2]

    # Source goes fully empty (every row gone — the dangerous misconfiguration).
    with src_engine.begin() as conn:
        conn.execute(delete(_SRC))

    result = mfb.backup_money_flow(src_engine, dst_engine)
    assert result.copied == 0
    assert result.run_max == 0
    assert _dst_ids(dst_engine, bt) == [1, 2], "empty source must NOT wipe the backup"
