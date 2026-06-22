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


def _ensure_version_table(engine: Engine) -> None:
    """Create ``public.schema_migrations`` if absent (idempotent)."""
    with engine.begin() as conn:
        _pin_public(conn)
        conn.execute(
            text(
                f"CREATE TABLE IF NOT EXISTS {_VERSION_TABLE} ("
                "  version TEXT PRIMARY KEY,"
                "  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                ")"
            )
        )


class UnversionedSchemaError(RuntimeError):
    """Raised when a non-empty legacy schema has no ``schema_migrations`` table.

    Replaying ``0001..N`` from scratch on such a DB is unsafe — some historical
    files don't re-run cleanly on an already-migrated schema. The caller must
    first ``--baseline`` to adopt the existing prefix.
    """


def _version_table_exists(engine: Engine) -> bool:
    sql = (
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = 'schema_migrations'"
    )
    with engine.connect() as conn:
        return conn.execute(text(sql)).first() is not None


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

    The file body (minus its own ``BEGIN;``/``COMMIT;``) and the bookkeeping
    INSERT run in ONE ``engine.begin()`` transaction. ``exec_driver_sql`` runs
    the multi-statement body as a single script via the DBAPI driver.
    """
    body = _strip_outer_transaction(path.read_text())
    with engine.begin() as conn:
        _pin_public(conn)
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

    Returns the filenames that were stamped (recorded without running).
    """
    _ensure_version_table(engine)
    pending = pending_migrations(engine, migrations_dir)
    stamped: list[str] = []
    for path in pending:
        with engine.begin() as conn:
            _pin_public(conn)
            conn.execute(
                text(f"INSERT INTO {_VERSION_TABLE} (version) VALUES (:v)"),
                {"v": path.name},
            )
        log.info("legacy_migration_baselined", version=path.name)
        stamped.append(path.name)
    return stamped


def run_migrations(
    engine: Engine,
    *,
    dry_run: bool = False,
    migrations_dir: Path | None = None,
) -> list[str]:
    """Apply all pending legacy migrations in filename order.

    Creates ``schema_migrations`` if absent, then applies each not-yet-recorded
    forward migration in its own transaction, recording the filename on success.
    Already-applied files are skipped, so a second run is a no-op.

    ``dry_run=True`` lists the pending filenames WITHOUT creating the version
    table or applying anything.

    SAFETY GUARD: if the legacy schema ALREADY has tables but no
    ``schema_migrations`` table, this raises ``UnversionedSchemaError`` instead
    of blindly replaying ``0001..N`` (which can fail on an already-migrated
    schema). Adopt such a DB with ``baseline_migrations`` /
    ``db migrate-legacy --baseline`` first, then run for the new tail only. A
    truly fresh DB (no tables) runs from 0001 normally.

    Returns the list of filenames that were applied (dry-run: the filenames
    that WOULD be applied).
    """
    if dry_run:
        pending = pending_migrations(engine, migrations_dir)
        return [p.name for p in pending]

    if not _version_table_exists(engine) and _legacy_schema_present(engine):
        raise UnversionedSchemaError(
            "Legacy schema already has tables but no schema_migrations table. "
            "Refusing to replay 0001..N (some historical files don't re-run "
            "cleanly on an existing schema). Run 'rainier db migrate-legacy "
            "--baseline' once to adopt the current schema, then re-run to apply "
            "any new migrations."
        )

    _ensure_version_table(engine)
    pending = pending_migrations(engine, migrations_dir)
    applied: list[str] = []
    for path in pending:
        _apply_one(engine, path)
        log.info("legacy_migration_applied", version=path.name)
        applied.append(path.name)
    return applied
