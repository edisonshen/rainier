"""Runner for the numbered ``migrations/*.sql`` files (the LEGACY DB track).

Why this exists
---------------
rainier has TWO migration tracks against TWO databases (see memory
``project_two_database_url_engines``):

  * Alembic (``db/alembic/``, ``rainier db migrate``) → the *Neon* canonical
    DB (``db.engine``, ``market.*``). NOT this module.
  * Numbered ``migrations/0001..NNNN.sql`` → the LEGACY local TimescaleDB
    (``core.database``, ``public.*``). THIS module.

The numbered files were applied **by hand** (``psql "$LEGACY_DATABASE_URL"
-f ...``) with no runner and no version table. The result: 0012/0013 silently
never ran on the local DB and the reclaim feature broke. This runner gives the
legacy track a real, idempotent apply mechanism with a ``schema_migrations``
version table, so "what's applied" is recorded instead of guessed.

How it applies a file
---------------------
Every up-migration file wraps its own ``BEGIN; ... COMMIT;``. To make the
file's DDL and the bookkeeping INSERT atomic (record a version ONLY if its DDL
committed) the runner strips that outer ``BEGIN;``/``COMMIT;`` and runs the
file body + the version INSERT inside ONE transaction it controls:

    ┌─ engine.begin() (one transaction per file) ─────────────┐
    │  CREATE TABLE IF NOT EXISTS schema_migrations …          │
    │  <stripped file body, e.g. CREATE TABLE IF NOT EXISTS …> │
    │  INSERT INTO schema_migrations (version) VALUES (<file>) │
    └─────────────────────────────────────────────────────────┘

If the DDL raises, the whole transaction rolls back and nothing is recorded —
the next run retries the same file. Already-recorded files are skipped, so a
second run is a no-op (idempotent).

``*_downgrade.sql`` files are NEVER applied by this runner; only the forward
(up) migrations are.
"""

from __future__ import annotations

import re
from pathlib import Path

import structlog
from sqlalchemy import text
from sqlalchemy.engine import Engine

log = structlog.get_logger()


def _resolve_migrations_dir() -> Path:
    """Locate the numbered ``migrations/*.sql`` directory.

    Resolved in two ways, in priority order — mirrors ``_resolve_alembic_config``
    in cli.py so the command works from both install shapes:

    1. **Wheel install** — ``importlib.resources.files("rainier") / "_db_assets"
       / "migrations"``. Hatchling's ``force-include`` ships the top-level
       ``migrations/`` tree into the wheel there, so a packaged CLI (no source
       checkout) still finds the files instead of returning ``[]`` and falsely
       reporting "up to date".
    2. **Editable / source checkout** — ``<repo>/migrations`` via ``__file__``
       (this module at ``src/rainier/core/legacy_migrate.py`` → ``parents[3]``
       is the repo root).
    """
    from importlib import resources

    try:
        anchor = resources.files("rainier") / "_db_assets" / "migrations"
        with resources.as_file(anchor) as path_obj:
            packaged = Path(path_obj)
        if packaged.is_dir():
            return packaged
    except (ModuleNotFoundError, FileNotFoundError):
        pass  # fall through to source-checkout path

    return Path(__file__).resolve().parents[3] / "migrations"


MIGRATIONS_DIR = _resolve_migrations_dir()

# Forward migrations are ``NNNN_<name>.sql``; downgrades are
# ``NNNN_<name>_downgrade.sql`` and must be excluded.
_UP_MIGRATION_RE = re.compile(r"^\d+_.*(?<!_downgrade)\.sql$")

# A line that is exactly ``BEGIN;`` / ``COMMIT;`` (optionally ``BEGIN``),
# case-insensitive, ignoring surrounding whitespace. Used to strip the file's
# own transaction wrapper so the runner can control the transaction.
_TXN_LINE_RE = re.compile(r"^\s*(BEGIN|COMMIT)\s*;?\s*$", re.IGNORECASE)


def discover_migrations(migrations_dir: Path | None = None) -> list[Path]:
    """Return the forward migration files, sorted by filename.

    Filename order is the apply order (zero-padded numeric prefixes sort
    lexicographically). ``*_downgrade.sql`` files are excluded.
    """
    directory = migrations_dir or MIGRATIONS_DIR
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and _UP_MIGRATION_RE.match(p.name)
    )


def _strip_outer_transaction(sql: str) -> str:
    """Remove a leading ``BEGIN;`` and a trailing ``COMMIT;`` line.

    The runner wraps each file in its own transaction, so the file's own outer
    wrapper must be removed to avoid a nested-transaction warning/abort. Only
    standalone ``BEGIN;``/``COMMIT;`` *lines* are stripped — DDL that merely
    contains those words inline is untouched.
    """
    lines = sql.splitlines()

    # Drop a leading standalone BEGIN; (skipping blank/comment lines before it).
    start = 0
    while start < len(lines) and (
        not lines[start].strip() or lines[start].lstrip().startswith("--")
    ):
        start += 1
    if start < len(lines) and _TXN_LINE_RE.match(lines[start]) and \
            lines[start].strip().upper().startswith("BEGIN"):
        lines = lines[:start] + lines[start + 1 :]

    # Drop a trailing standalone COMMIT; (skipping blank/comment lines after it).
    end = len(lines) - 1
    while end >= 0 and (not lines[end].strip() or lines[end].lstrip().startswith("--")):
        end -= 1
    if end >= 0 and _TXN_LINE_RE.match(lines[end]) and \
            lines[end].strip().upper().startswith("COMMIT"):
        lines = lines[:end] + lines[end + 1 :]

    return "\n".join(lines)


# The version table is ALWAYS schema-qualified ``public.schema_migrations``.
# Unqualified DDL/DML would resolve through the role's ``search_path``, which can
# put the CREATE in one schema and a later existence-check in another — then the
# runner thinks nothing is applied and replays every migration. The legacy ORM
# tables all live in ``public``, so the version table belongs there too.
_VERSION_TABLE = "public.schema_migrations"

# Executed by _apply_one (inside each per-file apply transaction) and
# baseline_migrations (inside the single all-or-nothing stamp transaction).
# NEVER in its own up-front transaction: a pre-committed empty version table
# surviving a failed first apply would disarm the UnversionedSchemaError guard.
_CREATE_VERSION_TABLE_SQL = (
    f"CREATE TABLE IF NOT EXISTS {_VERSION_TABLE} ("
    "  version TEXT PRIMARY KEY,"
    "  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()"
    ")"
)


def _pin_public(conn) -> None:
    """Pin this transaction's ``search_path`` to ``public``.

    The migration SQL bodies reference tables UNQUALIFIED (they were written to
    target the legacy public schema). If the engine was created with a custom
    ``search_path`` (e.g. test isolation uses ``search_path=<schema>,public``),
    that unqualified DDL would land in the wrong schema while the bookkeeping
    table is hardcoded to ``public`` — a split that desyncs the two. Pinning
    both to ``public`` for the duration of the apply keeps DDL and bookkeeping in
    the same schema. ``SET LOCAL`` is transaction-scoped, so it reverts on commit.
    """
    conn.exec_driver_sql("SET LOCAL search_path TO public")


class UnversionedSchemaError(RuntimeError):
    """Raised when a non-empty legacy schema has no RECORDED versions.

    Covers both a missing ``schema_migrations`` table and an
    existing-but-EMPTY one (a pre-created empty table must not disarm this
    guard). Replaying ``0001..N`` from scratch on such a DB is unsafe — some
    historical files don't re-run cleanly on an already-migrated schema. The
    caller must first ``--baseline`` to adopt the existing prefix.
    """


class EmptyDatabaseError(RuntimeError):
    """Raised when a plain run targets a truly EMPTY DB with the shipped files.

    The shipped ``migrations/*.sql`` assume an existing schema (0001 starts
    with ``ALTER TABLE analysis_results``, a table only ``db init`` creates),
    so a from-scratch replay dies at file 1 with a raw SQL error. The
    supported bootstrap is ``db init`` then ``--baseline``.
    """


class AlreadyVersionedError(RuntimeError):
    """Raised when baseline is invoked on a DB that already has recorded versions.

    Pending files on an already-versioned DB are genuinely NEW migrations that
    must be APPLIED (``run_migrations``), not stamped — stamping would record
    them as done and permanently skip their DDL (e.g. a constraint rebuild the
    tables/columns drift check cannot see). Baseline exists for exactly one
    state: adopting an UNVERSIONED pre-existing schema.
    """


def _legacy_schema_present(engine: Engine) -> bool:
    """True if any non-bookkeeping table already exists in ``public``.

    Used to distinguish a truly fresh DB (safe to run from 0001) from an
    existing, hand-migrated DB (must be baselined first). ``schema_migrations``
    itself is excluded so its presence/absence is judged separately.
    """
    sql = (
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name <> 'schema_migrations' "
        "LIMIT 1"
    )
    with engine.connect() as conn:
        return conn.execute(text(sql)).first() is not None


def applied_versions(engine: Engine) -> set[str]:
    """Return the set of already-applied migration filenames.

    Returns an empty set when ``public.schema_migrations`` does not exist yet (a
    fresh DB), so callers can compute "pending" without first creating it.
    """
    insp_sql = (
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = 'schema_migrations'"
    )
    with engine.connect() as conn:
        if conn.execute(text(insp_sql)).first() is None:
            return set()
        rows = (
            conn.execute(text(f"SELECT version FROM {_VERSION_TABLE}")).scalars().all()
        )
    return set(rows)


def pending_migrations(
    engine: Engine, migrations_dir: Path | None = None
) -> list[Path]:
    """Return discovered migrations not yet recorded in ``schema_migrations``."""
    done = applied_versions(engine)
    return [p for p in discover_migrations(migrations_dir) if p.name not in done]


def _apply_one(engine: Engine, path: Path) -> None:
    """Apply a single migration file and record it, atomically.

    The version-table CREATE IF NOT EXISTS, the file body (minus its own
    ``BEGIN;``/``COMMIT;``) and the bookkeeping INSERT run in ONE
    ``engine.begin()`` transaction. ``exec_driver_sql`` runs the
    multi-statement body as a single script via the DBAPI driver.

    Creating the version table HERE (not in a separate up-front transaction)
    is load-bearing (review 43f3 iter-2): if the very first apply on a fresh
    DB fails, a pre-committed empty ``schema_migrations`` would survive the
    rollback and permanently disarm ``run_migrations``'s
    ``UnversionedSchemaError`` guard — the operator's natural recovery
    (``db init``, retry) would then replay non-rerunnable historical SQL
    instead of being routed to ``--baseline``. In-transaction, a failed first
    apply leaves the DB exactly as it was.
    """
    body = _strip_outer_transaction(path.read_text())
    with engine.begin() as conn:
        _pin_public(conn)
        conn.execute(text(_CREATE_VERSION_TABLE_SQL))
        if body.strip():
            conn.exec_driver_sql(body)
        conn.execute(
            text(f"INSERT INTO {_VERSION_TABLE} (version) VALUES (:v)"),
            {"v": path.name},
        )


def baseline_migrations(
    engine: Engine, *, migrations_dir: Path | None = None
) -> list[str]:
    """Adopt an existing pre-versioned DB: RECORD pending files WITHOUT running.

    Some legacy DBs already carry the schema (created by ``db init`` or by hand
    via ``psql -f``) but have no ``schema_migrations`` table. Running
    ``run_migrations`` against them would replay 0001..N from scratch; a few
    historical files are not guaranteed to re-run cleanly on an already-migrated
    schema (e.g. 0001's non-IMMUTABLE index expression), so the first real run
    could crash before establishing a head.

    Baseline is the adoption escape hatch (alembic's ``stamp``): it creates
    ``public.schema_migrations`` and records every NOT-yet-recorded discovered
    file as applied, executing NONE of their SQL. After baselining, the operator
    runs ``run_migrations`` for any genuinely new files only.

    ATOMIC: the table creation and ALL stamps run in ONE transaction. A
    per-file (or ensure-then-stamp) split would let an interrupt leave a
    partial/empty ``schema_migrations`` behind — which defeats
    ``run_migrations``'s ``UnversionedSchemaError`` guard (the table now
    exists), so the next plain run would replay the un-stamped historical SQL
    onto the already-migrated live schema. All-or-nothing means an interrupted
    baseline leaves the DB exactly as it was, and the guard still fires.

    REFUSES on an already-versioned DB (codex 43f3 [P1]): when
    ``schema_migrations`` already has recorded versions, any pending file is a
    genuinely NEW migration that must be applied by ``run_migrations`` — a
    baseline there would stamp it as done without ever executing its DDL, and
    the tables/columns drift check cannot catch index/constraint-only files
    (e.g. 0008's ``ck_paper_skip_reason`` rebuild).

    Returns the filenames that were stamped (recorded without running).
    """
    pending = pending_migrations(engine, migrations_dir)
    if pending and applied_versions(engine):
        raise AlreadyVersionedError(
            "schema_migrations already has recorded versions; the "
            f"{len(pending)} pending file(s) are new migrations that must be "
            "APPLIED, not stamped. Run 'rainier db migrate-legacy' (without "
            "--baseline). Baseline only adopts an unversioned existing schema."
        )
    stamped: list[str] = []
    with engine.begin() as conn:
        _pin_public(conn)
        conn.execute(text(_CREATE_VERSION_TABLE_SQL))
        for path in pending:
            conn.execute(
                text(f"INSERT INTO {_VERSION_TABLE} (version) VALUES (:v)"),
                {"v": path.name},
            )
            stamped.append(path.name)
    for name in stamped:
        log.info("legacy_migration_baselined", version=name)
    return stamped


def run_migrations(
    engine: Engine,
    *,
    dry_run: bool = False,
    migrations_dir: Path | None = None,
) -> list[str]:
    """Apply all pending legacy migrations in filename order.

    Applies each not-yet-recorded forward migration in its own transaction
    (which also creates ``schema_migrations`` if absent — inside that same
    transaction, see ``_apply_one``), recording the filename on success.
    Already-applied files are skipped, so a second run is a no-op. A run that
    fails on its FIRST file leaves the DB untouched — no half-created version
    table survives to disarm the safety guard below.

    ``dry_run=True`` lists the pending filenames WITHOUT creating the version
    table or applying anything. The safety guards below run for dry-run too,
    so a dry-run is a TRUTHFUL preview of what a real run would do (codex 43f3
    [P2]: listing "13 pending" on a DB the real run refuses was misleading in
    the exact recovery scenario this runner targets).

    SAFETY GUARDS (both raise before any DB write):

    * ``UnversionedSchemaError`` — the legacy schema ALREADY has tables but no
      RECORDED versions (``schema_migrations`` missing OR existing-but-empty;
      an empty pre-created table must not disarm the guard). Refuses to
      blindly replay ``0001..N`` (historical files don't re-run cleanly on an
      already-migrated schema). Adopt with ``baseline_migrations`` /
      ``db migrate-legacy --baseline``, then run for the new tail only.
    * ``EmptyDatabaseError`` — the DB is truly EMPTY and the DEFAULT (shipped)
      migration set is in use. The shipped files assume an existing schema
      (0001 ALTERs tables only ``db init`` creates), so a from-scratch replay
      dies at file 1; steer to ``db init`` + ``--baseline`` instead. A custom
      ``migrations_dir`` skips this guard (self-contained sets bootstrap fine).

    Returns the list of filenames that were applied (dry-run: the filenames
    that WOULD be applied).
    """
    # NO recorded versions (the table is missing OR exists but is empty — an
    # empty pre-created table must not disarm the guard, codex 43f3 [P1]) means
    # this DB has never been adopted by the runner.
    if not applied_versions(engine):
        if _legacy_schema_present(engine):
            raise UnversionedSchemaError(
                "Legacy schema already has tables but no recorded versions in "
                "schema_migrations. Refusing to replay 0001..N (some "
                "historical files don't re-run cleanly on an existing "
                "schema). Run 'rainier db migrate-legacy --baseline' once to "
                "adopt the current schema, then re-run to apply any new "
                "migrations."
            )
        if migrations_dir is None:
            raise EmptyDatabaseError(
                "Empty legacy database: the shipped migrations/*.sql assume an "
                "existing schema (0001 ALTERs tables only 'rainier db init' "
                "creates) and cannot bootstrap from scratch. Run 'rainier db "
                "init' first, then 'rainier db migrate-legacy --baseline' to "
                "adopt the file history."
            )

    if dry_run:
        pending = pending_migrations(engine, migrations_dir)
        return [p.name for p in pending]

    pending = pending_migrations(engine, migrations_dir)
    applied: list[str] = []
    for path in pending:
        _apply_one(engine, path)
        log.info("legacy_migration_applied", version=path.name)
        applied.append(path.name)
    return applied
