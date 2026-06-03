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
        conn.exec_driver_sql(f'DROP DATABASE IF EXISTS "{throwaway}"')
        conn.exec_driver_sql(f'CREATE DATABASE "{throwaway}"')

    engine = create_engine(base.set(database=throwaway))
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
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
