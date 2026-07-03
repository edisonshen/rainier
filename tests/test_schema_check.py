"""Regression tests for the schema-drift detector (``core/schema_check``).

Guards the 2026-06-03 QU100 P0: the shared local ``stocks`` table was silently
reduced to a 2-column stub (``id``, ``symbol``), so the scraper's
``INSERT INTO stocks (symbol, name, ...)`` failed with ``column "name" ...
does not exist`` and persisted 0 rows. ``db init`` couldn't detect it
(``create_all`` is additive only). ``check_schema_drift`` makes such drift
visible; these tests prove it flags exactly that incident.

Gated on ``requires_postgres`` — needs a real Postgres to ``create_all`` the
legacy ORM and introspect it (SQLite can't represent JSONB/TIMESTAMPTZ defs).
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text

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
def legacy_db_url(request):
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
def legacy_engine(legacy_db_url):
    """A legacy-ORM-initialized engine on an ISOLATED throwaway database.

    NEVER runs ``drop_all`` against the provided URL — that is exactly the
    shared-DB clobbering this hotfix exists to prevent (and how the original
    incident happened: the prior tests' ``RAINIER_TEST_DATABASE_URL`` pointed
    at the live ``rainier`` DB). Instead we ``CREATE DATABASE`` a per-process
    throwaway alongside it, ``create_all`` there, and ``DROP DATABASE`` it on
    teardown — so even if the URL points at the live DB, its tables are
    untouched.
    """
    from sqlalchemy.engine import make_url

    from rainier.core.models import Base

    base = make_url(legacy_db_url)
    throwaway = f"schemacheck_test_{os.getpid()}"
    # AUTOCOMMIT: CREATE/DROP DATABASE cannot run inside a transaction block.
    admin = create_engine(legacy_db_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        # Managed/CI roles often lack CREATEDB. Skip gracefully rather than
        # hard-failing setup when we can't isolate a throwaway database.
        can_create = conn.exec_driver_sql(
            "SELECT rolcreatedb OR rolsuper FROM pg_roles WHERE rolname = current_user"
        ).scalar()
        if not can_create:
            admin.dispose()
            pytest.skip("test role lacks CREATEDB; cannot isolate a throwaway database")
        conn.exec_driver_sql(f'DROP DATABASE IF EXISTS "{throwaway}"')
        conn.exec_driver_sql(f'CREATE DATABASE "{throwaway}"')

    # create_engine + create_all are INSIDE the try so a failure there still
    # drops the throwaway DB + disposes admin in finally (no leaked database /
    # connection pool). pid-based naming + DROP IF EXISTS self-heals a prior
    # same-pid crash leak on the next run.
    engine = None
    try:
        engine = create_engine(base.set(database=throwaway))
        Base.metadata.create_all(engine)
        yield engine
    finally:
        if engine is not None:
            engine.dispose()  # release connections so DROP DATABASE can proceed
        with admin.connect() as conn:
            conn.exec_driver_sql(f'DROP DATABASE IF EXISTS "{throwaway}"')
        admin.dispose()


def test_fresh_create_all_has_no_drift(legacy_engine):
    """A freshly create_all'd legacy schema must report zero drift."""
    from rainier.core.schema_check import check_schema_drift

    assert check_schema_drift(legacy_engine) == []


def test_detects_stub_stocks_missing_name(legacy_engine):
    """THE regression for the 2026-06-03 QU100 P0: a ``stocks`` table missing
    the ``name`` column (the exact stub state) must be flagged. Pre-detector
    this drift was invisible and the scraper failed silently."""
    from rainier.core.schema_check import check_schema_drift

    with legacy_engine.begin() as conn:
        conn.execute(text("ALTER TABLE stocks DROP COLUMN name"))

    findings = check_schema_drift(legacy_engine)
    assert "missing column: stocks.name" in findings


def test_detects_dropped_table(legacy_engine):
    """A wholly-dropped ORM table (e.g. the ``stock_prices`` the same incident
    deleted) must be flagged as a missing table, not silently ignored."""
    from rainier.core.schema_check import check_schema_drift

    with legacy_engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS stock_prices CASCADE"))

    findings = check_schema_drift(legacy_engine)
    assert "missing table: stock_prices" in findings


def test_detects_dropped_index(legacy_engine):
    """An ORM-declared named index that is absent live must be flagged
    (codex 43f3 review: an index-only migration that never ran was invisible
    to the tables/columns check, so --baseline could stamp it as applied)."""
    from rainier.core.schema_check import check_schema_drift

    with legacy_engine.begin() as conn:
        conn.execute(text("DROP INDEX ix_signals_symbol_ts"))

    findings = check_schema_drift(legacy_engine)
    assert "missing index: signals.ix_signals_symbol_ts" in findings


def test_detects_dropped_constraint(legacy_engine):
    """An ORM-declared named constraint that is absent live must be flagged
    (e.g. a constraint-only migration like 0008's ck_paper_skip_reason rebuild
    that never ran)."""
    from rainier.core.schema_check import check_schema_drift

    with legacy_engine.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE paper_reclaim_queue "
                "DROP CONSTRAINT ck_paper_reclaim_status"
            )
        )

    findings = check_schema_drift(legacy_engine)
    assert (
        "missing constraint: paper_reclaim_queue.ck_paper_reclaim_status" in findings
    )


def test_checks_public_regardless_of_search_path(legacy_engine):
    """The drift check must inspect ``public`` explicitly (codex 43f3 [P1]):
    with a non-public-first ``search_path``, unqualified inspection would
    validate the FRONT schema instead of the one the migration runner stamps
    and pins DDL to. Here public holds the full schema and the front (shadow)
    schema is empty — the check must still be clean."""
    from sqlalchemy import event

    from rainier.core.schema_check import check_schema_drift

    with legacy_engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS shadow"))

    @event.listens_for(legacy_engine, "connect")
    def _set_search_path(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("SET search_path TO shadow, public")
        cur.close()

    # The pool still holds the un-hooked connection that ran CREATE SCHEMA;
    # dispose so inspection checks out fresh, shadow-first connections.
    legacy_engine.dispose()
    with legacy_engine.connect() as conn:
        search_path = conn.exec_driver_sql("SHOW search_path").scalar()
    assert "shadow" in search_path, (
        f"test-harness bug: search_path hook did not apply ({search_path!r})"
    )

    assert check_schema_drift(legacy_engine) == [], (
        "public holds the full ORM schema; a shadow-first search_path must "
        "not make the check inspect (and fail against) the empty front schema"
    )


def test_multiple_stub_columns_all_reported(legacy_engine):
    """The full stub (all 6 non-id/symbol columns dropped) reports each one,
    so an operator sees the complete repair list, not just the first miss."""
    from rainier.core.schema_check import check_schema_drift

    with legacy_engine.begin() as conn:
        for col in ("name", "sector", "industry", "is_active", "created_at", "updated_at"):
            conn.execute(text(f"ALTER TABLE stocks DROP COLUMN {col}"))

    findings = set(check_schema_drift(legacy_engine))
    for col in ("name", "sector", "industry", "is_active", "created_at", "updated_at"):
        assert f"missing column: stocks.{col}" in findings
