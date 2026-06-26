"""Nightly off-machine backup of ``money_flow_snapshots`` -> Neon.

Design: docs/DESIGN-money-flow-neon-backup.md §2.

``money_flow_snapshots`` (QU100 rankings history) lives only on the local
TimescaleDB — a single-machine SPOF, and the data is irreplaceable (QU's
date-keyed endpoint has no intraday/session replay). This module copies it,
incrementally and idempotently, into a plain ``backup.money_flow_snapshots``
table on Neon (which has managed backups). Local stays the primary; Neon holds
a backup copy.

Copy mechanism (``data_date``-aware reconcile, idempotent)
----------------------------------------------------------
The QU scraper REBUILDS a day on each later same-day scrape: it DELETEs that
day's ``(data_date, ranking_type)`` rows and re-INSERTs fresh ones with new
autoincrement ids (see ``scrapers/qu/scraper.py:_persist_qu100``). So a day's
rows change after first write, and old ids vanish. The previous high-water-mark-
by-``id`` copy CANNOT cope: an id-forward cursor can neither purge the orphaned
old ids the rebuild deleted (they sit *below* the high-water mark) nor re-copy a
changed day whose new max-id it already passed — ``verify_backup``'s full-window
checksum would then fail the nightly cron.

Instead we reconcile per GENERATION ``(data_date, ranking_type)`` — the exact
unit the scraper rebuilds — over the UNION of source and backup generations, in
ONE destination transaction:

    src = core.database.get_engine()    # legacy -> local TimescaleDB (PR #115)
    dst = db.engine.get_engine()        # canonical -> DATABASE_URL -> Neon
    run_max = MAX(id) on src            # STABLE upper bound (id <= run_max only)
    for gen=(data_date, ranking_type) in source_gens ∪ backup_gens:
        source-only gen         -> copy the gen (append)
        backup-only gen         -> RETAIN (never propagate a source-side delete)
        present on both, DIFFER -> DELETE the gen in backup + re-copy from source
            (differ = row-count mismatch, OR a source captured_at newer than the
             backup's, OR a per-gen canonical checksum mismatch)
        present on both, equal  -> skip

Keying on ``(data_date, ranking_type)`` — NOT ``data_date`` alone — keeps the two
books independent: a partial source-side loss of one book (e.g. bottom100 dropped
by a bad restore while top100 survives) never disturbs the surviving book.

NON-DESTRUCTIVE INVARIANT: this function never propagates a source-side DELETE.
A backup DELETE is only ever issued to immediately RE-COPY the same generation in
the same txn (the same-day rebuild: the generation still exists in source with new
ids, so its stale backup rows — including the rebuild's orphaned old ids — are
replaced wholesale). A generation the source LOST entirely (bad restore, manual
cleanup, corruption) is LEFT INTACT in the backup — the off-site copy is disaster
recovery and must not erase the only recoverable copy of an irreplaceable day.

Only rows with ``id <= run_max`` are considered, so a scrape landing mid-backup
is not a torn read (it is reconciled next run). The whole destination mutation
runs in ONE transaction, so a mid-run failure leaves the backup unchanged.
JSONB ``raw_data`` is bound via a typed SQLAlchemy Core column (NOT raw text).

    ASCII (a same-day rebuild):

        src.money_flow_snapshots            dst.backup.money_flow_snapshots
        ┌───────────────────────┐           ┌───────────────────────────┐
        │ day D: ids {3,4} (new)│  D differs │ day D: ids {1,2} (stale)  │
        │ (ids 1,2 deleted)     │ ─────────▶ │ DELETE day D, re-copy {3,4}│
        └───────────────────────┘  one txn   └───────────────────────────┘
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
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
    tuple_,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.types import JSON

log = logging.getLogger(__name__)

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
    """Outcome of one ``backup_money_flow`` run.

    ``copied`` — rows written this run (new gens + recopied changed gens).
    ``hwm_before`` — pre-run ``MAX(id)`` in the backup (reported for logging; the
    reconcile keys on ``(data_date, ranking_type)``, not an id high-water mark).
    ``run_max`` — the stable ``MAX(id)`` upper bound the run reconciled up to.
    ``source_empty`` — the source had ZERO rows, so the reconcile was skipped
    (backup left intact). A configured backup whose source is empty is a
    misconfiguration (mispointed ``LEGACY_DATABASE_URL`` / failed restore), so the
    CLI turns this into a NON-ZERO exit — never a silent "0 rows" success.
    """

    copied: int
    hwm_before: int
    run_max: int
    source_empty: bool = False


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


def _gen_key(r: dict) -> tuple:
    """The reconcile/rebuild unit: ``(data_date, ranking_type)``.

    The QU scraper rebuilds a snapshot per ``(data_date, ranking_type)`` (each
    book independently — a midday top100 rebuild leaves bottom100 untouched), so
    the backup MUST reconcile on the same composite key. Keying on ``data_date``
    alone would lump both books: a partial source-side loss of ONE book (e.g.
    bottom100 dropped by a bad restore while top100 survives) would look like a
    generic "day changed", delete the whole day, and recopy only the survivor —
    erasing the backup-only book and defeating the non-destructive invariant.
    """
    return (r["data_date"], r["ranking_type"])


def _gen_fingerprints(rows: list[dict]) -> dict[tuple, dict]:
    """Group ``rows`` by ``(data_date, ranking_type)`` -> ``{count, max_cap,
    checksum}``.

    ``max_cap`` is the canonical (UTC-normalized) max ``captured_at`` over the
    generation; ``checksum`` is the order-independent content hash of its rows.
    Used to decide whether a generation differs between source and backup.
    """
    by_gen: dict[tuple, list[dict]] = {}
    for r in rows:
        by_gen.setdefault(_gen_key(r), []).append(r)
    fps: dict[tuple, dict] = {}
    for key, gen_rows in by_gen.items():
        max_cap = max(_canon_captured_at(r["captured_at"]) for r in gen_rows)
        fps[key] = {
            "count": len(gen_rows),
            "max_cap": max_cap,
            "checksum": _checksum(gen_rows),
        }
    return fps


def backup_money_flow(
    src: Engine,
    dst: Engine,
    *,
    chunk_size: int = 5000,
    _race_hook: Callable[[str], None] | None = None,
) -> BackupResult:
    """Reconcile ``money_flow_snapshots`` from ``src`` into ``dst`` per ``data_date``.

    Captures a STABLE ``run_max = MAX(id)`` on ``src`` once, then — considering
    only ``id <= run_max`` — reconciles each ``data_date`` in the union of source
    and backup days: a day present only in source is copied; a day present only in
    backup is RETAINED (the backup never propagates a source-side delete — see the
    non-destructive invariant in the module docstring); a day present in both but
    DIFFERING (row-count / newer ``captured_at`` / canonical-checksum mismatch —
    e.g. a same-day rebuild) is delete-then-recopied; an equal day is skipped. All
    mutations run in ONE destination transaction, so a mid-run failure leaves the
    backup unchanged. The destination table is created if absent.

    ``BackupResult.copied`` counts rows WRITTEN this run (new + recopied days).
    ``hwm_before`` is the pre-run ``MAX(id)`` in the backup (kept for callers /
    logging; the reconcile no longer uses an id high-water mark as the cursor).

    ``_race_hook`` is a test seam called with stage names (``"after_run_max"``
    after the stable upper bound is captured; ``"before_commit"`` just before the
    txn commits) so tests can simulate a concurrent scrape or a forced mid-run
    failure without sleeps.
    """
    src_tbl = source_table()
    schema = _backup_schema_for(dst)
    dst_tbl = backup_table(schema)

    # Ensure the destination table exists (bootstrap). On Postgres migration 0004
    # creates it; checkfirst makes this a no-op there and creates it on SQLite.
    dst_tbl.create(dst, checkfirst=True)

    # 1) Read the ENTIRE source in ONE snapshot, then derive run_max FROM that
    #    same read. A same-day rebuild (DELETE day D's ids <= run_max, re-INSERT
    #    ids > run_max) landing between "capture run_max" and "read src_rows" used
    #    to make D look source-empty -> the reconcile purged D from the backup
    #    (Codex P1: torn read). Reading rows and run_max from one consistent
    #    snapshot closes that window: either we see D's pre-rebuild rows (and
    #    run_max excludes the new ids), or we see the post-rebuild rows whole.
    src_all = _read_all(src, src_tbl)
    run_max = max((r["id"] for r in src_all), default=0)

    if _race_hook is not None:
        _race_hook("after_run_max")

    # 2) Pre-run backup high-water mark (reported, not used as the cursor).
    with dst.connect() as dconn:
        hwm = dconn.execute(
            select(func.coalesce(func.max(dst_tbl.c.id), 0))
        ).scalar_one()

    # 2a) Empty-source early-out + LOUD operational signal. The non-destructive
    #     invariant below (a backup-only day is never purged) already makes an
    #     empty source a safe no-op, but a source that went fully dark is a strong
    #     misconfiguration signal — a fresh/rebuilt local DB, a failed restore, or a
    #     mis-pointed LEGACY_DATABASE_URL (a confusion that already caused a prior
    #     P0). Log it loudly and return early so the operator notices rather than
    #     seeing a silent "0 rows" success.
    if not src_all:
        log.warning(
            "backup_money_flow: source is EMPTY — skipping reconcile (backup rows "
            "left intact, hwm=%s). Check LEGACY_DATABASE_URL / local DB health.",
            hwm,
        )
        return BackupResult(
            copied=0, hwm_before=hwm, run_max=run_max, source_empty=True
        )

    # 3) Read the FULL backup. Source rows are restricted to id <= run_max for the
    #    COPY (in-flight inserts past run_max are picked up next run).
    src_rows = [r for r in src_all if r["id"] <= run_max]
    dst_rows = _read_all(dst, dst_tbl)
    src_fps = _gen_fingerprints(src_rows)
    dst_fps = _gen_fingerprints(dst_rows)

    # Decide which generations to (re)copy and which to delete-then-recopy. The
    # unit is the GENERATION key ``(data_date, ranking_type)`` — the exact unit
    # the scraper rebuilds — so a partial source-side loss of one book never
    # disturbs the other (Codex P1).
    #
    # NON-DESTRUCTIVE INVARIANT: the backup NEVER propagates a source-side DELETE
    # of a whole generation. A generation present ONLY in the backup (gone from
    # source) is LEFT INTACT — the off-site copy is disaster recovery, so a source
    # that lost a historical book (bad restore, manual cleanup, partial
    # corruption) must not erase the only recoverable copy on the next cron
    # (Codex P1). We only ever DELETE a backup generation to immediately RE-COPY it
    # from source in the SAME txn (the same-day rebuild: the generation still
    # exists in source with new ids / content, so its stale backup rows — including
    # the rebuild's orphaned old ids — are replaced wholesale). Deletion is thus
    # always paired with a recopy; a backup generation is never left empty here.
    keys_to_delete: set = set()
    keys_to_copy: set = set()
    for key in set(src_fps) | set(dst_fps):
        in_src = key in src_fps
        in_dst = key in dst_fps
        if in_src and not in_dst:
            keys_to_copy.add(key)  # brand-new generation -> append
        elif in_dst and not in_src:
            continue  # backup-only generation: retain (never propagate a delete)
        else:
            s, d = src_fps[key], dst_fps[key]
            differs = (
                s["count"] != d["count"]
                or s["max_cap"] != d["max_cap"]
                or s["checksum"] != d["checksum"]
            )
            if differs:
                # Same-day rebuild: purge this generation's stale backup rows and
                # recopy the current source rows in one txn (delete paired w/ copy).
                keys_to_delete.add(key)
                keys_to_copy.add(key)

    rows_to_copy = [r for r in src_rows if _gen_key(r) in keys_to_copy]

    copied = 0
    # 4) One transaction on the destination: delete changed generations, then
    #    re-insert. Commit once so a mid-run failure leaves the backup unchanged.
    with dst.begin() as dconn:
        if keys_to_delete:
            dconn.execute(
                dst_tbl.delete().where(
                    tuple_(dst_tbl.c.data_date, dst_tbl.c.ranking_type).in_(
                        keys_to_delete
                    )
                )
            )
        for start in range(0, len(rows_to_copy), chunk_size):
            chunk = rows_to_copy[start : start + chunk_size]
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


def _canon_captured_at(value: object) -> str:
    """Canonical string for a ``captured_at`` value, tz-normalized to UTC.

    The same ``timestamptz`` instant can surface as different aware datetimes
    depending on each engine's session ``TimeZone`` (e.g. ``22:00+00:00`` vs
    ``15:00-07:00`` — Codex P2). Every comparison path (checksum, sort key, AND
    the missing-row reconciliation key) MUST normalize identically, or
    ``--verify`` can flag a copied row as missing and fail the nightly cron on a
    pure timezone-rendering difference. Aware datetimes are converted to UTC;
    naive datetimes and non-datetime values fall back to ``str()`` unchanged.
    """
    if isinstance(value, _dt.datetime) and value.tzinfo is not None:
        return value.astimezone(_dt.timezone.utc).isoformat()
    return str(value)


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
        if isinstance(value, _dt.datetime):
            return ("t", _canon_captured_at(value))
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
        return (row.get("id"), _canon_captured_at(row.get("captured_at")))

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


def _read_all(engine: Engine, tbl: Table) -> list[dict]:
    """Read every row of ``tbl`` (id > 0) in one connection.

    The source read derives ``run_max`` from this same snapshot so a concurrent
    same-day rebuild can't tear the window; the destination read is uncapped so a
    backup-only day whose ids exceed a lowered ``run_max`` is still seen (it is
    retained, not purged — see the non-destructive invariant).
    """
    cols = [tbl.c[name] for name in COLUMNS]
    with engine.connect() as conn:
        return [
            dict(m)
            for m in conn.execute(select(*cols).where(tbl.c.id > 0)).mappings()
        ]


def verify_backup(src: Engine, dst: Engine, *, run_max: int) -> VerifyReport:
    """Strong integrity check over the covered window ``id <= run_max``.

    The copy is a NON-DESTRUCTIVE ``data_date``-aware reconcile (new days appended,
    same-day rebuilds delete-then-recopied, days the source LOST left intact), so
    the backup must CONTAIN every source row over the window with matching content
    — a SUPERSET, not an exact mirror (it may retain a historical day the source
    dropped). Checks, any failure recorded loudly:

      (a) ``MAX(id)`` on src vs backup match for ``id <= run_max`` (source's max
          row is always mirrored; retained ids > run_max are excluded);
      (b) missing-row reconciliation on the FULL composite key
          ``(id, captured_at)`` — no source row in ``(0, run_max]`` absent from
          the backup;
      (c) content-faithful checksum: every source row in ``id <= run_max`` matches
          its backup copy (``raw_data`` canonicalized), comparing only the backup
          rows that share a source key so legitimately-retained extras are ignored
          — catches a copied row whose content drifted;
      (d) ``id``-uniqueness guard: ``COUNT(*) == COUNT(DISTINCT id)`` over
          ``id <= run_max`` on the source. Autoincrement keeps ``id`` globally
          unique (the composite PK ``(id, captured_at)`` alone does not enforce
          it); a duplicate id would make the per-day fingerprints ambiguous.
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

    # (b) missing-row reconciliation on the full composite key. captured_at is
    # tz-normalized identically to the checksum path (Codex P2) so a pure session
    # TimeZone-rendering difference between local and Neon cannot make a copied
    # row look missing and falsely fail the nightly cron.
    def _key(r: dict) -> tuple:
        return (r["id"], _canon_captured_at(r["captured_at"]))

    src_keys = {_key(r) for r in src_rows}
    dst_by_key = {_key(r): r for r in dst_rows}
    missing = src_keys - set(dst_by_key)
    if missing:
        sample = sorted(missing)[:5]
        report.failures.append(
            f"{len(missing)} source row(s) in (0, {run_max}] missing from the "
            f"backup (composite key id, captured_at). sample={sample}"
        )

    # (c) content-faithful checksum. The backup is a NON-DESTRUCTIVE archive: it
    # may RETAIN rows the source lost (a historical day a bad restore / corruption
    # dropped — see the module invariant), so the backup is a SUPERSET of the
    # source over the window, not an exact mirror. Verify therefore compares the
    # source against the backup rows that SHARE a source key — every mirrored row
    # must match content — and ignores retained extras. (b) already guarantees no
    # source row is missing; together they assert "the backup faithfully contains
    # every source row". A same-day rebuild's orphaned old ids cannot survive here:
    # an in-both-differing day is delete-then-recopied in ONE txn, so a successful
    # reconcile leaves no stale ids for a day still in source.
    dst_mirrored = [dst_by_key[k] for k in src_keys if k in dst_by_key]
    if _checksum(src_rows) != _checksum(dst_mirrored):
        report.failures.append(
            f"checksum mismatch over id<={run_max}: a source row's content "
            f"diverges from its backup copy (an edited row, reordered "
            f"non-raw_data content, or torn copy). raw_data key-order is "
            f"canonicalized, so this is a genuine content drift."
        )

    return report
