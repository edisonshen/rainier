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

# Repo-root-relative migrations directory. legacy_migrate.py lives at
# src/rainier/core/legacy_migrate.py → parents[3] is the repo root.
MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"

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


def _ensure_version_table(engine: Engine) -> None:
    """Create ``schema_migrations`` if absent (idempotent)."""
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "  version TEXT PRIMARY KEY,"
                "  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                ")"
            )
        )


def applied_versions(engine: Engine) -> set[str]:
    """Return the set of already-applied migration filenames.

    Returns an empty set when ``schema_migrations`` does not exist yet (a
    fresh DB), so callers can compute "pending" without first creating it.
    """
    insp_sql = (
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = 'schema_migrations'"
    )
    with engine.connect() as conn:
        if conn.execute(text(insp_sql)).first() is None:
            return set()
        rows = conn.execute(text("SELECT version FROM schema_migrations")).scalars().all()
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
        if body.strip():
            conn.exec_driver_sql(body)
        conn.execute(
            text("INSERT INTO schema_migrations (version) VALUES (:v)"),
            {"v": path.name},
        )


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

    Returns the list of filenames that were applied (dry-run: the filenames
    that WOULD be applied).
    """
    if dry_run:
        pending = pending_migrations(engine, migrations_dir)
        return [p.name for p in pending]

    _ensure_version_table(engine)
    pending = pending_migrations(engine, migrations_dir)
    applied: list[str] = []
    for path in pending:
        _apply_one(engine, path)
        log.info("legacy_migration_applied", version=path.name)
        applied.append(path.name)
    return applied
