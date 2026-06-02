"""Nightly off-machine backup of ``money_flow_snapshots`` -> Neon.

Design: docs/DESIGN-money-flow-neon-backup.md §2.

``money_flow_snapshots`` (QU100 rankings history) lives only on the local
TimescaleDB — a single-machine SPOF, and the data is irreplaceable (QU's
date-keyed endpoint has no intraday/session replay). This module copies it,
incrementally and idempotently, into a plain ``backup.money_flow_snapshots``
table on Neon (which has managed backups). Local stays the primary; Neon holds
a backup copy.

Copy mechanism (high-water-mark by ``id``, insert-only, idempotent)
-------------------------------------------------------------------
The QU scraper is insert-only for money flow (it skips already-present
``(data_date, ranking_type)`` batches and only INSERTs; never UPDATE/DELETE),
``id`` is a monotonic sequence, so a HWM-by-``id`` copy is correct as long as
that invariant holds. ``verify_backup`` (full-window canonicalized checksum) is
the declared drift safety net if the invariant is ever violated.

    src = core.database.get_engine()    # legacy -> local TimescaleDB (PR #115)
    dst = db.engine.get_engine()        # canonical -> DATABASE_URL -> Neon
    run_max = MAX(id) on src            # STABLE upper bound, captured ONCE
    hwm     = MAX(id) on dst backup     # high-water mark
    copy WHERE id > hwm AND id <= run_max, ON CONFLICT (id, captured_at) DO NOTHING

A scrape landing rows mid-backup (id > run_max) is NOT torn-read — those rows
are picked up next run. The whole destination insert runs in ONE transaction so
a mid-run failure leaves the backup unchanged (next run retries from the same
hwm). JSONB ``raw_data`` is bound via a typed SQLAlchemy Core column (NOT raw
text interpolation).

    ASCII (one run):

        src.money_flow_snapshots            dst.backup.money_flow_snapshots
        ┌───────────────────────┐           ┌───────────────────────────┐
        │ id 1..run_max (stable)│  copy >hwm │ id 1..hwm  (already there)│
        │ id >run_max (ignored) │ ─────────▶ │ + id (hwm, run_max]       │
        └───────────────────────┘  one txn   └───────────────────────────┘
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.types import JSON

# Columns copied from core/models.py:MoneyFlowSnapshot (in order). ``raw_data``
# uses the generic JSON type for BINDING (a dict serializes identically whether
# the live column is JSON or JSONB); the migration declares the live column as
# JSONB on Postgres. The composite PK (id, captured_at) mirrors the source.
COLUMNS: tuple[str, ...] = (
    "id",
    "captured_at",
    "capture_session",
    "data_date",
    "view_type",
    "ranking_type",
    "symbol",
    "rank",
    "daily_change",
    "sector",
    "industry",
    "long_short",
    "raw_data",
)

SOURCE_TABLE = "money_flow_snapshots"
BACKUP_SCHEMA = "backup"
# Postgres: schema-qualified ``backup.money_flow_snapshots`` (per migration
# 0004). SQLite has no schemas, so the table is named ``backup_money_flow_
# snapshots`` (one word) there — used by the in-memory test harness only.
_BACKUP_TABLE_NO_SCHEMA = "backup_money_flow_snapshots"


def _build_table(name: str, schema: str | None) -> Table:
    return Table(
        name,
        MetaData(),
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
        # Composite PK mirrors the source (id alone is NOT DB-unique). This is
        # also the ON CONFLICT target for the insert-only copy.
        PrimaryKeyConstraint("id", "captured_at"),
        schema=schema,
    )


def backup_table(schema: str | None = None) -> Table:
    """Return the Core ``Table`` for the backup destination.

    ``schema=None`` (default, SQLite / tests) -> ``backup_money_flow_snapshots``
    with no schema. ``schema="backup"`` (Postgres / Neon) -> the schema-qualified
    ``backup.money_flow_snapshots`` matching migration 0004.
    """
    if schema is None:
        return _build_table(_BACKUP_TABLE_NO_SCHEMA, None)
    return _build_table(SOURCE_TABLE, schema)


def source_table() -> Table:
    """Return the Core ``Table`` for the source ``money_flow_snapshots`` read.

    Schema-unqualified so it resolves to ``public`` on Postgres and the only
    table on SQLite. A read-only projection of the copied columns.
    """
    return _build_table(SOURCE_TABLE, None)


def _backup_schema_for(engine: Engine) -> str | None:
    """``"backup"`` on Postgres, ``None`` on SQLite (no schema support)."""
    return BACKUP_SCHEMA if engine.dialect.name == "postgresql" else None


@dataclass
class BackupResult:
    """Outcome of one ``backup_money_flow`` run."""

    copied: int
    hwm_before: int
    run_max: int


@dataclass
class VerifyReport:
    """Outcome of ``verify_backup``: ``ok`` when every check passed."""

    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------


def backup_money_flow(
    src: Engine,
    dst: Engine,
    *,
    chunk_size: int = 5000,
    _race_hook: Callable[[str], None] | None = None,
) -> BackupResult:
    """Copy new ``money_flow_snapshots`` rows from ``src`` to ``dst`` (insert-only).

    Captures a STABLE ``run_max = MAX(id)`` on ``src`` once at the start, reads
    the backup high-water mark ``hwm`` from ``dst``, then copies
    ``id > hwm AND id <= run_max`` into the backup table within ONE transaction
    with ``ON CONFLICT (id, captured_at) DO NOTHING``. The destination table is
    created if absent (idempotent bootstrap from an empty backup).

    ``_race_hook`` is a test seam called with stage names
    (``"after_run_max"`` after the stable upper bound is captured;
    ``"before_commit"`` just before the txn commits) so tests can simulate a
    concurrent scrape or a forced mid-insert failure without sleeps.
    """
    src_tbl = source_table()
    schema = _backup_schema_for(dst)
    dst_tbl = backup_table(schema)

    # Ensure the destination table exists (bootstrap). On Postgres migration 0004
    # creates it; checkfirst makes this a no-op there and creates it on SQLite.
    dst_tbl.create(dst, checkfirst=True)

    # 1) Stable upper bound on the source, captured ONCE.
    with src.connect() as sconn:
        run_max = sconn.execute(select(func.coalesce(func.max(src_tbl.c.id), 0))).scalar_one()

    if _race_hook is not None:
        _race_hook("after_run_max")

    # 2) High-water mark on the destination backup.
    with dst.connect() as dconn:
        hwm = dconn.execute(
            select(func.coalesce(func.max(dst_tbl.c.id), 0))
        ).scalar_one()

    copied = 0
    if run_max <= hwm:
        return BackupResult(copied=0, hwm_before=hwm, run_max=run_max)

    # 3) Read the incremental window from the source, ordered by id.
    select_cols = [src_tbl.c[name] for name in COLUMNS]
    with src.connect() as sconn:
        rows = [
            dict(m)
            for m in sconn.execute(
                select(*select_cols)
                .where(src_tbl.c.id > hwm, src_tbl.c.id <= run_max)
                .order_by(src_tbl.c.id)
            ).mappings()
        ]

    # 4) One transaction on the destination: chunked typed Core insert with
    #    ON CONFLICT (id, captured_at) DO NOTHING. Commit once at the end so a
    #    mid-run failure leaves the backup unchanged.
    with dst.begin() as dconn:
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            if not chunk:
                continue
            stmt = _conflict_insert(dst, dst_tbl, chunk)
            dconn.execute(stmt, chunk)
            copied += len(chunk)
        if _race_hook is not None:
            _race_hook("before_commit")

    return BackupResult(copied=copied, hwm_before=hwm, run_max=run_max)


def _conflict_insert(dst: Engine, dst_tbl: Table, chunk: list[dict]):
    """Build an INSERT ... ON CONFLICT (id, captured_at) DO NOTHING statement.

    Postgres uses the native ``ON CONFLICT`` clause. SQLite (test harness) uses
    its own ``on_conflict_do_nothing`` — but a fresh in-memory backup never
    actually conflicts, so a plain insert is sufficient there.
    """
    if dst.dialect.name == "postgresql":
        return pg_insert(dst_tbl).on_conflict_do_nothing(
            index_elements=["id", "captured_at"]
        )
    if dst.dialect.name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        return sqlite_insert(dst_tbl).on_conflict_do_nothing(
            index_elements=["id", "captured_at"]
        )
    return dst_tbl.insert()


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def _canon_raw_data(value: object) -> str:
    """Deterministic canonical string for a ``raw_data`` cell.

    JSONB dict/list ordering is driver/PG-dependent, so the checksum MUST
    canonicalize before hashing (design §2.3 / Codex round 2). ``sort_keys``
    makes key order irrelevant; ``separators`` removes whitespace variance;
    ``default=str`` handles any non-JSON scalar (e.g. Decimal/date) the driver
    may surface. A genuine value change still changes the canonical string.
    """
    if value is None:
        return "null"
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _canon_cell(name: str, value: object) -> object:
    """Canonicalize one cell for the checksum.

    ``raw_data`` is canonicalized as sorted-key JSON; ``captured_at`` is
    normalized to a UTC ISO string so the same instant hashes identically
    regardless of the session timezone each engine rendered it in; everything
    else is tagged + stringified.
    """
    if value is None:
        return ("n",)
    if name == "raw_data":
        return ("j", _canon_raw_data(value))
    if name == "captured_at":
        import datetime as _dt

        if isinstance(value, _dt.datetime):
            if value.tzinfo is not None:
                value = value.astimezone(_dt.timezone.utc)
            return ("t", value.isoformat())
    return ("s", str(value))


def _checksum(rows: list[dict]) -> str:
    """Order-independent content hash over ``COLUMNS`` for ``rows``.

    Rows are sorted by the composite key ``(id, captured_at)`` so source/backup
    row order is irrelevant, then each row is rendered as a fixed-column-order
    tuple of canonical cells and BLAKE2b-hashed. ``raw_data`` is canonicalized
    (sorted-key JSON) so a key reorder hashes equal while a real value change
    diverges.
    """

    def sort_key(row: dict):
        return (row.get("id"), str(row.get("captured_at")))

    h = hashlib.blake2b(digest_size=16)
    for row in sorted(rows, key=sort_key):
        cells = tuple(_canon_cell(c, row.get(c)) for c in COLUMNS)
        h.update(repr(cells).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def _read_window(engine: Engine, tbl: Table, run_max: int) -> list[dict]:
    cols = [tbl.c[name] for name in COLUMNS]
    with engine.connect() as conn:
        return [
            dict(m)
            for m in conn.execute(
                select(*cols).where(tbl.c.id > 0, tbl.c.id <= run_max)
            ).mappings()
        ]


def verify_backup(src: Engine, dst: Engine, *, run_max: int) -> VerifyReport:
    """Strong integrity check over the full covered window ``id <= run_max``.

    Checks (design §2.3), any failure recorded loudly:

      (a) ``MAX(id)`` on src vs backup match for ``id <= run_max``;
      (b) missing-row reconciliation on the FULL composite key
          ``(id, captured_at)`` — no source row in ``(0, run_max]`` absent from
          the backup;
      (c) deterministic checksum over the full window ``id <= run_max`` (NOT
          just the incremental window) with ``raw_data`` canonicalized — catches
          an edited already-backed-up row (``id <= hwm``) that keeps its key;
      (d) ``id``-uniqueness guard: ``COUNT(*) == COUNT(DISTINCT id)`` over
          ``id <= run_max`` on the source (HWM-by-id relies on global ``id``
          uniqueness the composite PK does not enforce).
    """
    report = VerifyReport()
    src_tbl = source_table()
    dst_tbl = backup_table(_backup_schema_for(dst))
    # A standalone verify (no copy first) must not crash on a missing backup
    # table — create it if absent so the checks read an empty table and report
    # the drift (missing rows / max-id) rather than a raw OperationalError.
    dst_tbl.create(dst, checkfirst=True)

    # (d) id-uniqueness guard on the source. Done first: a duplicate id makes the
    # HWM unsound, so flag it before trusting the other checks.
    with src.connect() as sconn:
        total = sconn.execute(
            select(func.count()).select_from(src_tbl).where(src_tbl.c.id <= run_max)
        ).scalar_one()
        distinct = sconn.execute(
            select(func.count(func.distinct(src_tbl.c.id))).where(
                src_tbl.c.id <= run_max
            )
        ).scalar_one()
    if total != distinct:
        report.failures.append(
            f"source id-uniqueness guard FAILED: COUNT(*)={total} != "
            f"COUNT(DISTINCT id)={distinct} over id<={run_max} — a duplicate id "
            f"defeats the high-water-mark; the backup cannot be trusted."
        )

    # (a) MAX(id) match over id <= run_max.
    with src.connect() as sconn:
        src_max = sconn.execute(
            select(func.coalesce(func.max(src_tbl.c.id), 0)).where(
                src_tbl.c.id <= run_max
            )
        ).scalar_one()
    with dst.connect() as dconn:
        dst_max = dconn.execute(
            select(func.coalesce(func.max(dst_tbl.c.id), 0)).where(
                dst_tbl.c.id <= run_max
            )
        ).scalar_one()
    if src_max != dst_max:
        report.failures.append(
            f"MAX(id) mismatch over id<={run_max}: source={src_max} "
            f"backup={dst_max}"
        )

    # Read both full windows once for (b) + (c).
    src_rows = _read_window(src, src_tbl, run_max)
    dst_rows = _read_window(dst, dst_tbl, run_max)

    # (b) missing-row reconciliation on the full composite key.
    def _key(r: dict) -> tuple:
        return (r["id"], str(r["captured_at"]))

    src_keys = {_key(r) for r in src_rows}
    dst_keys = {_key(r) for r in dst_rows}
    missing = src_keys - dst_keys
    if missing:
        sample = sorted(missing)[:5]
        report.failures.append(
            f"{len(missing)} source row(s) in (0, {run_max}] missing from the "
            f"backup (composite key id, captured_at). sample={sample}"
        )

    # (c) full-window canonicalized checksum.
    if _checksum(src_rows) != _checksum(dst_rows):
        report.failures.append(
            f"checksum mismatch over the full window id<={run_max}: the backup "
            f"diverges from the source (an edited row, reordered non-raw_data "
            f"content, or torn copy). raw_data key-order is canonicalized, so "
            f"this is a genuine content drift."
        )

    return report
