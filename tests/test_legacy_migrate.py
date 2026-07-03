"""Tests for the legacy ``migrations/*.sql`` runner (``core/legacy_migrate``).

Guards the 2026-06-22 legacy-drift fix: the numbered migrations were applied by
hand with no runner and no version table, so 0012/0013 silently never ran. This
runner records what's applied in ``schema_migrations`` and is idempotent.

Two flavors:
  * SYNTHETIC migrations dir (``tmp_path``) — deterministic ordering /
    idempotency / missing-middle / dry-run, independent of real file contents.
  * REAL ``migrations/*.sql`` — the runner applies the shipped files end to end
    and reports zero schema drift afterwards (the chokepoint stays green).

Gated on ``requires_postgres``. Uses an ISOLATED per-process throwaway database
(CREATE DATABASE alongside the provided URL) so it never touches a live DB even
if ``RAINIER_TEST_DATABASE_URL`` points at one — same guard as
``tests/test_schema_check.py``.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, inspect, text

pytestmark = pytest.mark.requires_postgres

try:
    from pytest_postgresql import factories as _pg_factories

    _HAS_PYTEST_POSTGRESQL = True
except ImportError:  # pragma: no cover
    _HAS_PYTEST_POSTGRESQL = False


def _local_pg_binary_available() -> bool:
    import shutil

    return shutil.which("pg_config") is not None and shutil.which("initdb") is not None


if _HAS_PYTEST_POSTGRESQL and _local_pg_binary_available():
    postgresql_proc = _pg_factories.postgresql_proc(port=None, unixsocketdir="/tmp")
    postgresql = _pg_factories.postgresql("postgresql_proc")


@pytest.fixture
def base_db_url(request):
    """Clean Postgres URL. RAINIER_TEST_DATABASE_URL -> pytest-postgresql -> skip."""
    env_url = os.environ.get("RAINIER_TEST_DATABASE_URL")
    if env_url:
        return env_url
    if not _HAS_PYTEST_POSTGRESQL:
        pytest.skip("pytest-postgresql not installed")
    if not _local_pg_binary_available():
        pytest.skip("pg_config / initdb not on PATH; set RAINIER_TEST_DATABASE_URL")
    pg = request.getfixturevalue("postgresql")
    return (
        f"postgresql+psycopg://{pg.info.user}@{pg.info.host}:{pg.info.port}"
        f"/{pg.info.dbname}"
    )


@pytest.fixture
def throwaway_engine(base_db_url):
    """An empty engine on an ISOLATED per-process throwaway database.

    Never ``drop_all``s / mutates the provided URL: CREATE DATABASE a per-pid
    throwaway alongside it, yield an engine on it, DROP DATABASE on teardown.
    Mirrors tests/test_schema_check.py so even a live RAINIER_TEST_DATABASE_URL
    is safe.
    """
    from sqlalchemy.engine import make_url

    base = make_url(base_db_url)
    throwaway = f"legacymigrate_test_{os.getpid()}"
    admin = create_engine(base_db_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        can_create = conn.exec_driver_sql(
            "SELECT rolcreatedb OR rolsuper FROM pg_roles WHERE rolname = current_user"
        ).scalar()
        if not can_create:
            admin.dispose()
            pytest.skip("test role lacks CREATEDB; cannot isolate a throwaway database")
        conn.exec_driver_sql(f'DROP DATABASE IF EXISTS "{throwaway}"')
        conn.exec_driver_sql(f'CREATE DATABASE "{throwaway}"')

    engine = None
    try:
        engine = create_engine(base.set(database=throwaway))
        yield engine
    finally:
        if engine is not None:
            engine.dispose()
        with admin.connect() as conn:
            conn.exec_driver_sql(f'DROP DATABASE IF EXISTS "{throwaway}"')
        admin.dispose()


def _write_migration(directory, name: str, sql: str) -> None:
    (directory / name).write_text(sql)


def _make_synthetic_migrations(directory) -> None:
    """Three forward migrations + a downgrade decoy the runner must ignore."""
    _write_migration(
        directory,
        "0001_create_a.sql",
        "BEGIN;\nCREATE TABLE IF NOT EXISTS mig_a (id int PRIMARY KEY);\nCOMMIT;\n",
    )
    _write_migration(
        directory,
        "0002_create_b.sql",
        "BEGIN;\nCREATE TABLE IF NOT EXISTS mig_b (id int PRIMARY KEY);\nCOMMIT;\n",
    )
    _write_migration(
        directory,
        "0003_add_col.sql",
        "BEGIN;\nALTER TABLE mig_a ADD COLUMN IF NOT EXISTS extra text;\nCOMMIT;\n",
    )
    # Decoy: a downgrade file must NEVER be applied by the runner.
    _write_migration(
        directory,
        "0003_add_col_downgrade.sql",
        "BEGIN;\nDROP TABLE IF EXISTS mig_a;\nCOMMIT;\n",
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discover_excludes_downgrades_and_sorts(tmp_path):
    from rainier.core.legacy_migrate import discover_migrations

    _make_synthetic_migrations(tmp_path)
    names = [p.name for p in discover_migrations(tmp_path)]
    assert names == ["0001_create_a.sql", "0002_create_b.sql", "0003_add_col.sql"], names
    assert not any("downgrade" in n for n in names)


# ---------------------------------------------------------------------------
# Fresh DB: creates schema_migrations + applies in order
# ---------------------------------------------------------------------------


def test_fresh_db_applies_all_in_order(throwaway_engine, tmp_path):
    from rainier.core.legacy_migrate import applied_versions, run_migrations

    _make_synthetic_migrations(tmp_path)
    applied = run_migrations(throwaway_engine, migrations_dir=tmp_path)
    assert applied == ["0001_create_a.sql", "0002_create_b.sql", "0003_add_col.sql"]

    insp = inspect(throwaway_engine)
    assert "schema_migrations" in insp.get_table_names()
    assert {"mig_a", "mig_b"} <= set(insp.get_table_names())
    # 0003's ALTER ran (proves order: 0001 created mig_a before 0003 altered it).
    assert "extra" in {c["name"] for c in insp.get_columns("mig_a")}
    assert applied_versions(throwaway_engine) == set(applied)


def test_second_run_is_noop(throwaway_engine, tmp_path):
    from rainier.core.legacy_migrate import run_migrations

    _make_synthetic_migrations(tmp_path)
    run_migrations(throwaway_engine, migrations_dir=tmp_path)
    second = run_migrations(throwaway_engine, migrations_dir=tmp_path)
    assert second == [], "re-run must apply nothing (idempotent)"


# ---------------------------------------------------------------------------
# DB missing a later migration: only the missing one is applied + recorded
# (the 0012 scenario, generalized — applies whatever is not yet recorded).
# ---------------------------------------------------------------------------


def test_db_missing_latest_applies_only_that_one(throwaway_engine, tmp_path):
    from rainier.core.legacy_migrate import run_migrations

    _make_synthetic_migrations(tmp_path)
    # Simulate a DB that has 0001+0002 applied but is MISSING 0003 (the "0012
    # never ran" state): create the version table + record the first two, and
    # create their tables so the world matches the bookkeeping.
    with throwaway_engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE schema_migrations ("
                " version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now())"
            )
        )
        conn.execute(text("CREATE TABLE mig_a (id int PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE mig_b (id int PRIMARY KEY)"))
        conn.execute(
            text(
                "INSERT INTO schema_migrations (version) VALUES "
                "('0001_create_a.sql'), ('0002_create_b.sql')"
            )
        )

    applied = run_migrations(throwaway_engine, migrations_dir=tmp_path)
    assert applied == ["0003_add_col.sql"], "only the missing migration applies"

    insp = inspect(throwaway_engine)
    assert "extra" in {c["name"] for c in insp.get_columns("mig_a")}

    # Re-run records nothing new.
    assert run_migrations(throwaway_engine, migrations_dir=tmp_path) == []


# ---------------------------------------------------------------------------
# --dry-run lists pending, applies nothing
# ---------------------------------------------------------------------------


def test_dry_run_lists_pending_and_applies_nothing(throwaway_engine, tmp_path):
    from rainier.core.legacy_migrate import run_migrations

    _make_synthetic_migrations(tmp_path)
    pending = run_migrations(throwaway_engine, dry_run=True, migrations_dir=tmp_path)
    assert pending == ["0001_create_a.sql", "0002_create_b.sql", "0003_add_col.sql"]

    insp = inspect(throwaway_engine)
    # Nothing applied: no schema_migrations table, no migration tables.
    assert "schema_migrations" not in insp.get_table_names()
    assert "mig_a" not in insp.get_table_names()


# ---------------------------------------------------------------------------
# Failure mid-file rolls back the version record (atomicity)
# ---------------------------------------------------------------------------


def test_failed_migration_records_nothing(throwaway_engine, tmp_path):
    from rainier.core.legacy_migrate import applied_versions, run_migrations

    _write_migration(
        tmp_path,
        "0001_ok.sql",
        "BEGIN;\nCREATE TABLE IF NOT EXISTS mig_ok (id int PRIMARY KEY);\nCOMMIT;\n",
    )
    _write_migration(
        tmp_path,
        "0002_boom.sql",
        "BEGIN;\nCREATE TABLE mig_boom (id int);\nSELECT 1/0;\nCOMMIT;\n",
    )
    from sqlalchemy.exc import DBAPIError

    with pytest.raises(DBAPIError):
        run_migrations(throwaway_engine, migrations_dir=tmp_path)

    insp = inspect(throwaway_engine)
    # 0001 committed + recorded; 0002 rolled back entirely — not recorded, table absent.
    assert applied_versions(throwaway_engine) == {"0001_ok.sql"}
    assert "mig_boom" not in insp.get_table_names()


def test_failed_first_migration_leaves_guard_armed(throwaway_engine, tmp_path):
    """A run that fails on its FIRST file must leave the DB untouched
    (review 43f3 iter-2).

    The version table used to be committed in its own up-front transaction; a
    first-apply failure (the real 0001 does an ALTER on a table only ``db
    init`` creates, so this is the guaranteed fresh-DB outcome) left an EMPTY
    ``schema_migrations`` behind. That permanently disarmed the
    ``UnversionedSchemaError`` guard: after the operator's natural recovery
    (``db init``, retry) the runner would replay non-rerunnable historical SQL
    instead of routing to ``--baseline``.
    """
    from sqlalchemy.exc import ProgrammingError

    from rainier.core.legacy_migrate import UnversionedSchemaError, run_migrations

    # Mirrors real 0001 on an empty DB: ALTER on a table no migration creates.
    _write_migration(
        tmp_path,
        "0001_alter_missing.sql",
        "BEGIN;\nALTER TABLE not_there ADD COLUMN x int;\nCOMMIT;\n",
    )
    with pytest.raises(ProgrammingError):
        run_migrations(throwaway_engine, migrations_dir=tmp_path)

    # The failed run left NOTHING behind — especially no empty version table.
    assert "schema_migrations" not in inspect(throwaway_engine).get_table_names()

    # Operator recovery: schema gets created out-of-band (db init). The guard
    # must still fire and route to --baseline instead of replaying.
    with throwaway_engine.begin() as conn:
        conn.execute(text("CREATE TABLE not_there (id int)"))
    with pytest.raises(UnversionedSchemaError):
        run_migrations(throwaway_engine, migrations_dir=tmp_path)


# ---------------------------------------------------------------------------
# Baseline: adopt an existing schema WITHOUT replaying SQL (codex 43f3 [P1])
# ---------------------------------------------------------------------------


def test_baseline_records_without_running(throwaway_engine, tmp_path):
    from rainier.core.legacy_migrate import (
        applied_versions,
        baseline_migrations,
        run_migrations,
    )

    _make_synthetic_migrations(tmp_path)
    stamped = baseline_migrations(throwaway_engine, migrations_dir=tmp_path)
    assert stamped == ["0001_create_a.sql", "0002_create_b.sql", "0003_add_col.sql"]

    # All recorded, but NONE of the SQL ran — the tables don't exist.
    insp = inspect(throwaway_engine)
    assert applied_versions(throwaway_engine) == set(stamped)
    assert "mig_a" not in insp.get_table_names()
    assert "mig_b" not in insp.get_table_names()

    # After baseline, a normal run is a no-op (everything already stamped).
    assert run_migrations(throwaway_engine, migrations_dir=tmp_path) == []


def test_baseline_then_run_applies_only_new(throwaway_engine, tmp_path):
    """Baseline the existing set, add a NEW file, then run applies only the new
    one — the realistic adoption flow."""
    from rainier.core.legacy_migrate import baseline_migrations, run_migrations

    _make_synthetic_migrations(tmp_path)
    baseline_migrations(throwaway_engine, migrations_dir=tmp_path)

    _write_migration(
        tmp_path,
        "0004_new.sql",
        "BEGIN;\nCREATE TABLE IF NOT EXISTS mig_new (id int PRIMARY KEY);\nCOMMIT;\n",
    )
    applied = run_migrations(throwaway_engine, migrations_dir=tmp_path)
    assert applied == ["0004_new.sql"]
    assert "mig_new" in inspect(throwaway_engine).get_table_names()


def test_baseline_is_all_or_nothing(throwaway_engine, tmp_path):
    """A failed/interrupted baseline must stamp NOTHING (review 43f3 iter-1).

    A partially-stamped ``schema_migrations`` defeats the
    ``UnversionedSchemaError`` guard (the table now exists), so the next plain
    run would replay the un-stamped tail's SQL onto the already-migrated
    schema. Force a mid-stamp failure via a CHECK that rejects the LAST pending
    file and assert no prefix was recorded.
    """
    from sqlalchemy.exc import IntegrityError

    from rainier.core.legacy_migrate import applied_versions, baseline_migrations

    _make_synthetic_migrations(tmp_path)
    with throwaway_engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE public.schema_migrations ("
                " version TEXT PRIMARY KEY,"
                " applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
                " CHECK (version <> '0003_add_col.sql'))"
            )
        )

    with pytest.raises(IntegrityError):
        baseline_migrations(throwaway_engine, migrations_dir=tmp_path)

    assert applied_versions(throwaway_engine) == set(), (
        "a failed baseline must roll back every stamp, not leave a prefix"
    )


# ---------------------------------------------------------------------------
# search_path independence — version table is always public.schema_migrations
# (codex 43f3 [P2]): an unqualified table under a non-public search_path would
# put the CREATE in one schema and the existence-check in another → replay loop.
# ---------------------------------------------------------------------------


def test_idempotent_under_non_public_search_path(throwaway_engine, tmp_path):
    from sqlalchemy import event

    from rainier.core.legacy_migrate import applied_versions, run_migrations

    # Create a user schema that shadows public in the search_path. If the runner
    # wrote/read schema_migrations unqualified, the second run would not see the
    # first run's records and would replay everything.
    with throwaway_engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS shadow"))

    @event.listens_for(throwaway_engine, "connect")
    def _set_search_path(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("SET search_path TO shadow, public")
        cur.close()

    # The pool already holds the DBAPI connection that ran CREATE SCHEMA above,
    # and the "connect" hook fires only on NEW connections — without a dispose,
    # every later checkout reuses the un-hooked pooled connection and this test
    # passes vacuously even with _pin_public removed. Dispose so run_migrations
    # checks out fresh, shadow-first connections, and sanity-check the hook took.
    throwaway_engine.dispose()
    with throwaway_engine.connect() as conn:
        search_path = conn.exec_driver_sql("SHOW search_path").scalar()
    assert "shadow" in search_path, (
        f"test-harness bug: search_path hook did not apply ({search_path!r})"
    )

    _make_synthetic_migrations(tmp_path)
    first = run_migrations(throwaway_engine, migrations_dir=tmp_path)
    assert first == ["0001_create_a.sql", "0002_create_b.sql", "0003_add_col.sql"]

    second = run_migrations(throwaway_engine, migrations_dir=tmp_path)
    assert second == [], "re-run under a non-public search_path must be a no-op"
    assert applied_versions(throwaway_engine) == set(first)

    # Both the version table AND the migration DDL landed in public, not the
    # shadow schema — _pin_public keeps bookkeeping and DDL in the same schema.
    insp = inspect(throwaway_engine)
    assert "schema_migrations" in insp.get_table_names(schema="public")
    assert "schema_migrations" not in insp.get_table_names(schema="shadow")
    assert {"mig_a", "mig_b"} <= set(insp.get_table_names(schema="public"))
    assert "mig_a" not in insp.get_table_names(schema="shadow")


# ---------------------------------------------------------------------------
# REAL shipped migrations are discovered + driven by the runner
# ---------------------------------------------------------------------------
# These use the real migrations/*.sql (no temp dir) but do NOT depend on every
# file's DDL executing on a bare engine — a migration's *content* (e.g. 0001's
# non-IMMUTABLE index expression) is out of scope for the runner task, which
# may not even rewrite those files. They prove the runner sees the shipped
# files in order and records what it drives.


def test_real_migrations_discovered_in_order():
    """``discover_migrations`` finds the shipped numbered files, sorted, with
    the ``*_downgrade.sql`` files excluded."""
    from rainier.core.legacy_migrate import MIGRATIONS_DIR, discover_migrations

    names = [p.name for p in discover_migrations()]
    assert names, "expected shipped migrations/*.sql to be discovered"
    assert names == sorted(names), "shipped migrations must be filename-sorted"
    assert all(p.parent == MIGRATIONS_DIR for p in discover_migrations())
    assert not any("downgrade" in n for n in names), (
        f"downgrade files must never be discovered as forward migrations: {names}"
    )
    # Every discovered file has a zero-padded numeric prefix.
    assert all(n.split("_", 1)[0].isdigit() for n in names), names


def test_real_migrations_all_pending_on_fresh_db(throwaway_engine):
    """On a fresh DB, ``pending_migrations`` returns every shipped forward file
    (nothing recorded yet), and ``--dry-run`` lists them without applying."""
    from rainier.core.legacy_migrate import (
        discover_migrations,
        pending_migrations,
        run_migrations,
    )

    all_names = [p.name for p in discover_migrations()]
    pending = [p.name for p in pending_migrations(throwaway_engine)]
    assert pending == all_names

    dry = run_migrations(throwaway_engine, dry_run=True)
    assert dry == all_names

    # Dry-run created nothing.
    insp = inspect(throwaway_engine)
    assert "schema_migrations" not in insp.get_table_names()


def test_real_migration_recorded_when_driven(throwaway_engine):
    """The runner records a shipped file in ``schema_migrations`` after it
    records the shipped files via the baseline adoption flow.

    The realistic state for the shipped files is "schema already exists" (db
    init / hand psql), so the correct way to record them is baseline — which
    stamps every discovered file WITHOUT running its (possibly non-rerunnable)
    SQL. This proves the runner sees and records the real shipped set, then is a
    no-op."""
    from rainier.core.legacy_migrate import (
        applied_versions,
        baseline_migrations,
        discover_migrations,
        run_migrations,
    )
    from rainier.core.models import Base

    # Seed the ORM tables (the realistic "already migrated by db init" state).
    Base.metadata.create_all(throwaway_engine)

    all_names = [p.name for p in discover_migrations()]
    stamped = baseline_migrations(throwaway_engine)
    assert stamped == all_names, "baseline must record every shipped file in order"
    assert applied_versions(throwaway_engine) == set(all_names)

    # After baseline, a normal run is a no-op (everything recorded).
    assert run_migrations(throwaway_engine) == []


def test_run_refuses_unversioned_existing_schema(throwaway_engine):
    """run_migrations raises on an existing schema with no schema_migrations
    table — it must NOT replay 0001..N (codex 43f3 round-3 [P1])."""
    from rainier.core.legacy_migrate import (
        UnversionedSchemaError,
        applied_versions,
        run_migrations,
    )
    from rainier.core.models import Base

    Base.metadata.create_all(throwaway_engine)  # tables present, no version table
    with pytest.raises(UnversionedSchemaError):
        run_migrations(throwaway_engine)

    # Nothing was created or recorded — the guard fired before any work.
    assert applied_versions(throwaway_engine) == set()
    assert "schema_migrations" not in inspect(throwaway_engine).get_table_names()

    # Baseline adopts it; then a run is a clean no-op.
    from rainier.core.legacy_migrate import baseline_migrations

    baseline_migrations(throwaway_engine)
    assert run_migrations(throwaway_engine) == []


def test_fresh_empty_db_runs_from_0001_despite_guard(throwaway_engine, tmp_path):
    """The guard only blocks a NON-EMPTY unversioned schema; a truly fresh
    (empty) DB still runs from the first migration normally."""
    from rainier.core.legacy_migrate import run_migrations

    _make_synthetic_migrations(tmp_path)
    applied = run_migrations(throwaway_engine, migrations_dir=tmp_path)
    assert applied == ["0001_create_a.sql", "0002_create_b.sql", "0003_add_col.sql"]
