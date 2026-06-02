"""Shared fixtures for the paper-trade tracker test suite.

Two execution lanes (TEST-SPEC §Engine):

* **logic lane** — sqlite in-memory + the PR #115 singleton-reset, for pure /
  ORM-logic tests that don't need Postgres-only features. `legacy_sqlite_session`.
* **postgres lane** (`requires_postgres`) — an ephemeral local Postgres with the
  0005 migration applied, for partial-unique / JSONB / CHECK / catalog tests
  that sqlite cannot model. `pg_legacy_engine` / `pg_legacy_session`. Skips
  cleanly when no Postgres is reachable (CI provides one).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_UP = REPO_ROOT / "migrations" / "0005_paper_tracker.sql"
MIGRATION_DOWN = REPO_ROOT / "migrations" / "0005_paper_tracker_downgrade.sql"

# Minimal DDL for the FK-target tables the paper migration references. The real
# schema lives in migrations/0001-0004; for an isolated paper-tracker test DB we
# only need the columns paper_trade FKs against (analysis_results.id,
# screened_stocks.id) plus the screened_stocks columns the additive migration
# touches and the stock_prices table the ingest/exit tests need.
_PREREQ_DDL = """
CREATE TABLE IF NOT EXISTS analysis_results (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    llm_model       VARCHAR(100) NOT NULL DEFAULT 'test',
    prompt_template VARCHAR(100) NOT NULL DEFAULT 'test',
    recommendation  VARCHAR(10),
    confidence      DOUBLE PRECISION,
    reasoning       TEXT,
    structured_output JSONB,
    session_name    VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS stocks (
    symbol VARCHAR(10) PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS screened_stocks (
    id              SERIAL PRIMARY KEY,
    captured_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    scan_date       DATE NOT NULL,
    session_name    VARCHAR(20) NOT NULL,
    symbol          VARCHAR(10) NOT NULL,
    rule_rank       INTEGER NOT NULL,
    composite_score DOUBLE PRECISION NOT NULL,
    money_flow_score DOUBLE PRECISION,
    sector          VARCHAR(50),
    pattern_type    VARCHAR(50),
    pattern_confidence DOUBLE PRECISION,
    llm_confidence  INTEGER,
    thesis_id       INTEGER,
    CONSTRAINT uq_screened_stocks_scan_session_symbol
        UNIQUE (scan_date, session_name, symbol)
);

CREATE TABLE IF NOT EXISTS stock_prices (
    id      BIGSERIAL,
    symbol  VARCHAR(10) NOT NULL,
    date    TIMESTAMP WITH TIME ZONE NOT NULL,
    open    DOUBLE PRECISION,
    high    DOUBLE PRECISION,
    low     DOUBLE PRECISION,
    close   DOUBLE PRECISION,
    volume  BIGINT,
    PRIMARY KEY (id, date),
    CONSTRAINT uq_stock_price_symbol_date UNIQUE (symbol, date)
);
"""


def _has_pytest_postgresql() -> bool:
    try:
        import pytest_postgresql  # noqa: F401

        return True
    except ImportError:
        return False


def _pg_binary_available() -> bool:
    import shutil

    return shutil.which("pg_config") is not None and shutil.which("initdb") is not None


if _has_pytest_postgresql() and _pg_binary_available():
    from pytest_postgresql import factories as _pg_factories

    postgresql_proc = _pg_factories.postgresql_proc(port=None, unixsocketdir="/tmp")
    postgresql = _pg_factories.postgresql("postgresql_proc")


def _resolve_pg_url(request) -> str:
    env_url = os.environ.get("RAINIER_TEST_DATABASE_URL")
    if env_url:
        return env_url
    if not _has_pytest_postgresql():
        pytest.skip("pytest-postgresql not installed")
    if not _pg_binary_available():
        pytest.skip("pg_config / initdb not on PATH — set RAINIER_TEST_DATABASE_URL")
    pg = request.getfixturevalue("postgresql")
    return (
        f"postgresql+psycopg://{pg.info.user}@{pg.info.host}:{pg.info.port}"
        f"/{pg.info.dbname}"
    )


def _apply_sql(engine, path: Path) -> None:
    sql = path.read_text()
    with engine.begin() as conn:
        conn.execute(text(sql))


@pytest.fixture
def pg_legacy_engine(request):
    """Ephemeral Postgres with prereq tables + the 0005 migration applied.

    Also binds the legacy `core.database` singleton to it so production code
    under test (ingest / fill / positions) hits this DB. Resets the singleton
    before and after (PR #115 discipline).
    """
    url = _resolve_pg_url(request)
    engine = create_engine(url, future=True)
    with engine.begin() as conn:
        conn.execute(text(_PREREQ_DDL))
    _apply_sql(engine, MIGRATION_UP)

    from rainier.core import config, database

    config._settings = None
    database._engine = engine
    database._session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield engine
    finally:
        database._engine = None
        database._session_factory = None
        config._settings = None
        engine.dispose()


@pytest.fixture
def pg_legacy_session(pg_legacy_engine):
    Session = sessionmaker(bind=pg_legacy_engine, expire_on_commit=False)
    sess = Session()
    try:
        yield sess
    finally:
        sess.rollback()
        sess.close()
