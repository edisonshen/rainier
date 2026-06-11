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
# Phase 2 (D7a): paper_calibration. Applied on top of 0005 in the same throwaway
# schema so calibration compute/persist tests have the table.
MIGRATION_0007_UP = REPO_ROOT / "migrations" / "0007_paper_calibration.sql"
MIGRATION_0007_DOWN = REPO_ROOT / "migrations" / "0007_paper_calibration_downgrade.sql"
# Phase 3 (miss-sweep): extends ck_paper_skip_reason with `zero_share_price`.
MIGRATION_0008_UP = REPO_ROOT / "migrations" / "0008_paper_skip_zero_share.sql"
MIGRATION_0008_DOWN = (
    REPO_ROOT / "migrations" / "0008_paper_skip_zero_share_downgrade.sql"
)
# R-A: paper_trade.reflection + outcome-embargo CHECK. Applied on top of 0005 so
# full-entity PaperTrade ORM reads (positions.py et al) see the column.
MIGRATION_0009_UP = REPO_ROOT / "migrations" / "0009_paper_reflection.sql"
MIGRATION_0009_DOWN = REPO_ROOT / "migrations" / "0009_paper_reflection_downgrade.sql"
# research_insights (0003, self-contained + idempotent) — the lessons tests
# (check_paper_lessons → emit_insight) round-trip ResearchInsight rows. Without
# this the tests only pass when a prior suite left the table in `public`
# (search_path fallback) — an order-dependent pass on a fresh test DB.
MIGRATION_0003_UP = REPO_ROOT / "migrations" / "0003_llm_thesis_pr3.sql"

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

-- Mirror the Stock ORM (models.py): id PK + symbol UNIQUE (the FK target for
-- stock_prices.symbol) + the descriptive columns. `session.add(Stock(symbol=…))`
-- emits an INSERT over ALL ORM columns, so they must all exist here.
CREATE TABLE IF NOT EXISTS stocks (
    id          SERIAL PRIMARY KEY,
    symbol      VARCHAR(10) NOT NULL UNIQUE,
    name        VARCHAR(255),
    sector      VARCHAR(100),
    industry    VARCHAR(200),
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Mirror the full ScreenedStockRecord ORM (models.py) MINUS the four trade-level
-- columns (entry_price/stop_loss/target_price/rr_ratio), which migration 0005
-- adds via `ADD COLUMN IF NOT EXISTS` — so the migration's additive step stays
-- under test. A `select(ScreenedStockRecord)` emits ALL ORM columns, so every
-- non-level column must exist here or the ORM read fails (ProgrammingError).
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
    shadow_combined_score DOUBLE PRECISION,
    would_be_combined_rank INTEGER,
    thesis_id       INTEGER,
    patterns_in_chart_not_in_indicators_count INTEGER,
    action_taken    VARCHAR(20),
    outcome_pct     DOUBLE PRECISION,
    outcome_recorded_at TIMESTAMP WITH TIME ZONE,
    notes           TEXT,
    forward_return_5d  DOUBLE PRECISION,
    forward_return_10d DOUBLE PRECISION,
    outcome_backfilled_at TIMESTAMP WITH TIME ZONE,
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

-- Mirror the MoneyFlowSnapshot ORM (models.py) for the Phase-3 miss-sweep
-- cohort selector (`get_current_qu100_cohort` reads data_date / ranking_type /
-- captured_at / rank). ORM inserts emit ALL columns, so all must exist.
CREATE TABLE IF NOT EXISTS money_flow_snapshots (
    id              BIGSERIAL,
    captured_at     TIMESTAMP WITH TIME ZONE NOT NULL,
    capture_session VARCHAR(20) NOT NULL,
    data_date       DATE NOT NULL,
    view_type       VARCHAR(10) NOT NULL DEFAULT 'daily',
    ranking_type    VARCHAR(10) NOT NULL,
    symbol          VARCHAR(10) NOT NULL REFERENCES stocks (symbol),
    rank            INTEGER NOT NULL,
    daily_change    INTEGER,
    sector          VARCHAR(100),
    industry        VARCHAR(200),
    long_short      VARCHAR(50),
    raw_data        JSONB,
    PRIMARY KEY (id, captured_at)
);

-- Mirror the ResearchInsight ORM (models.py) for the Phase-3 missed_winner
-- insight emission (emit_insight UPSERTs pending rows by (kind, subject)).
CREATE TABLE IF NOT EXISTS research_insights (
    id               SERIAL PRIMARY KEY,
    created_at       TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at       TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    kind             VARCHAR(40) NOT NULL,
    severity         VARCHAR(10) NOT NULL,
    subject          VARCHAR(100) NOT NULL,
    evidence         JSONB,
    action           JSONB,
    rationale        TEXT,
    recurrence_count INTEGER NOT NULL DEFAULT 1,
    status           VARCHAR(20) NOT NULL DEFAULT 'pending',
    decided_at       TIMESTAMP WITH TIME ZONE,
    decided_by       VARCHAR(200),
    applied_change   JSONB
);

-- Mirror the ThesisEvaluation ORM (models.py) enough for the D7a calibration
-- headline compute (reads horizon/verdict/llm_confidence/return_pct/scan_date).
CREATE TABLE IF NOT EXISTS thesis_evaluations (
    id                 SERIAL PRIMARY KEY,
    thesis_id          INTEGER NOT NULL REFERENCES analysis_results (id),
    screened_record_id INTEGER REFERENCES screened_stocks (id),
    evaluated_at       TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    horizon            VARCHAR(8) NOT NULL,
    scan_date          DATE NOT NULL,
    symbol             VARCHAR(10) NOT NULL,
    verdict            VARCHAR(20) NOT NULL,
    llm_confidence     INTEGER,
    entry_price        DOUBLE PRECISION NOT NULL,
    exit_price         DOUBLE PRECISION NOT NULL,
    return_pct         DOUBLE PRECISION NOT NULL,
    hit                BOOLEAN NOT NULL,
    signals_used       VARCHAR(50)[],
    notes              TEXT
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


# Disposable schema for the paper-tracker (0005) fixtures. Building the prereq
# tables + 0005 migration inside a throwaway schema (instead of shared `public`)
# gives each run a guaranteed-clean namespace (no residue collisions across tests
# on a reused `RAINIER_TEST_DATABASE_URL`) AND lets teardown drop ONLY this schema
# — never the real `public.*` ORM tables a reused DB may hold. `public` stays on
# the search_path as a fallback so the timescaledb extension functions (installed
# in `public`) stay resolvable. Mirrors the 0006 fixtures. The fixture exposes
# `engine.rainier_schema` so schema-scoped inspector calls in the 0005 tests can
# target the right namespace.
_PAPER_SCHEMA = "rainier_paper_test"


@pytest.fixture
def pg_legacy_engine(request):
    """Ephemeral Postgres with prereq tables + the 0005 migration applied.

    Builds everything in a disposable schema (`_PAPER_SCHEMA`) so teardown drops
    only that schema (never shared `public`) and each run starts clean. Also
    binds the legacy `core.database` singleton to it so production code under
    test (ingest / fill / positions) hits this DB. Resets the singleton before
    and after (PR #115 discipline).
    """
    url = _resolve_pg_url(request)
    # search_path-free admin connection so DROP/CREATE SCHEMA targets the right
    # namespace regardless of any pinned path.
    admin = create_engine(url, future=True)
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {_PAPER_SCHEMA} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {_PAPER_SCHEMA}"))
    admin.dispose()

    engine = create_engine(
        url,
        future=True,
        connect_args={"options": f"-csearch_path={_PAPER_SCHEMA},public"},
    )
    engine.rainier_schema = _PAPER_SCHEMA  # type: ignore[attr-defined]
    with engine.begin() as conn:
        conn.execute(text(_PREREQ_DDL))
    _apply_sql(engine, MIGRATION_UP)
    _apply_sql(engine, MIGRATION_0007_UP)  # D7a paper_calibration
    _apply_sql(engine, MIGRATION_0008_UP)  # Phase 3 zero_share_price skip reason
    _apply_sql(engine, MIGRATION_0009_UP)  # R-A paper_trade.reflection
    _apply_sql(engine, MIGRATION_0003_UP)  # research_insights (lessons tests)

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
        # Drop ONLY the throwaway schema — never shared `public`. A reused
        # `RAINIER_TEST_DATABASE_URL` keeps its real ORM tables intact.
        admin = create_engine(url, future=True)
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {_PAPER_SCHEMA} CASCADE"))
        admin.dispose()


@pytest.fixture
def pg_legacy_session(pg_legacy_engine):
    Session = sessionmaker(bind=pg_legacy_engine, expire_on_commit=False)
    sess = Session()
    try:
        yield sess
    finally:
        sess.rollback()
        sess.close()


# --------------------------------------------------------------------------
# Migration 0006 (stock_prices stock_id -> symbol realign) fixtures.
# --------------------------------------------------------------------------

MIGRATION_0006_UP = REPO_ROOT / "migrations" / "0006_stock_prices_symbol_key.sql"
MIGRATION_0006_DOWN = REPO_ROOT / "migrations" / "0006_stock_prices_symbol_key_downgrade.sql"

# The OLD, stock_id-keyed `stock_prices` shape that migration 0006 realigns. The
# baseline `_PREREQ_DDL` above already builds the NEW (symbol-keyed) shape, so the
# 0006 migration tests need to start from the pre-migration drift instead.
_OLD_STOCK_PRICES_DDL = """
CREATE TABLE IF NOT EXISTS stock_prices (
    id        BIGSERIAL,
    stock_id  INTEGER NOT NULL REFERENCES stocks (id),
    date      TIMESTAMP WITH TIME ZONE NOT NULL,
    open      DOUBLE PRECISION,
    high      DOUBLE PRECISION,
    low       DOUBLE PRECISION,
    close     DOUBLE PRECISION,
    volume    BIGINT,
    PRIMARY KEY (id, date),
    CONSTRAINT uq_stock_price_date UNIQUE (stock_id, date)
);
CREATE INDEX IF NOT EXISTS ix_stock_prices_stock_id ON stock_prices (stock_id);
"""

# Only the FK-target `stocks` is needed for the 0006 tests (not the whole paper
# prereq set / 0005 migration).
_STOCKS_ONLY_DDL = """
CREATE TABLE IF NOT EXISTS stocks (
    id          SERIAL PRIMARY KEY,
    symbol      VARCHAR(10) NOT NULL UNIQUE,
    name        VARCHAR(255),
    sector      VARCHAR(100),
    industry    VARCHAR(200),
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""


# Dedicated, disposable schema for the 0006 migration fixtures. Isolating these
# fixtures in their own schema means the `DROP SCHEMA ... CASCADE` setup/teardown
# can NEVER reach the shared `public` ORM tables — so pointing
# `RAINIER_TEST_DATABASE_URL` at a reusable Postgres with the full schema is safe
# (a CASCADE drop of `public.stocks` would otherwise strip FK constraints from
# every sibling table that references it). Codex review P2.
_MIG0006_SCHEMA = "rainier_mig0006_test"


def _isolated_engine(url: str):
    """Engine whose connections resolve names in the disposable 0006 test schema.

    `search_path` puts `_MIG0006_SCHEMA` FIRST, so `create_all` / `init_db()` /
    the migration's unqualified `stock_prices`/`stocks` all create and resolve
    there (shadowing any `public` namesakes in a reused DB). `public` stays on
    the path only as a fallback so the timescaledb extension functions
    (`create_hypertable`, installed in `public`) remain resolvable. The schema is
    (re)created fresh and dropped CASCADE around the fixture body — CASCADE is
    scoped to throwaway objects only, never the shared `public` ORM tables.
    """
    engine = create_engine(
        url,
        future=True,
        connect_args={"options": f"-csearch_path={_MIG0006_SCHEMA},public"},
    )
    # Recreate the schema fresh using a search_path-free connection so the DROP
    # CASCADE targets the right schema regardless of the pinned path.
    admin = create_engine(url, future=True)
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {_MIG0006_SCHEMA} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {_MIG0006_SCHEMA}"))
    admin.dispose()
    return engine


def _drop_isolated_schema(url: str) -> None:
    admin = create_engine(url, future=True)
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {_MIG0006_SCHEMA} CASCADE"))
    admin.dispose()


def _has_timescaledb(engine) -> bool:
    """True if the connected Postgres has the timescaledb extension installed."""
    with engine.connect() as conn:
        return bool(
            conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
            ).scalar()
        )


def _try_create_extension(engine) -> bool:
    """Best-effort `CREATE EXTENSION timescaledb`; returns whether it's present."""
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
    except Exception:
        pass
    return _has_timescaledb(engine)


@pytest.fixture
def pg_oldshape_engine(request):
    """Ephemeral Postgres seeded with the OLD stock_id-keyed `stock_prices`.

    Makes `stock_prices` a hypertable when the timescaledb extension is
    available (so the in-place ALTER is exercised against a real hypertable);
    falls back to a plain table otherwise (schema/idempotency/backfill checks
    still run; the hypertable-catalog asserts are `requires_timescaledb`-gated).

    Exposes `engine.rainier_has_timescaledb` (bool) for tests to gate on.
    """
    url = _resolve_pg_url(request)
    # Isolated schema: setup/teardown CASCADE can't touch shared `public` tables.
    engine = _isolated_engine(url)
    has_ts = _try_create_extension(engine)
    with engine.begin() as conn:
        conn.execute(text(_STOCKS_ONLY_DDL))
        conn.execute(text(_OLD_STOCK_PRICES_DDL))
    if has_ts:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "SELECT create_hypertable('stock_prices', 'date', "
                    "migrate_data => true, if_not_exists => true)"
                )
            )
    engine.rainier_has_timescaledb = has_ts  # type: ignore[attr-defined]
    try:
        yield engine
    finally:
        engine.dispose()
        _drop_isolated_schema(url)


@pytest.fixture
def pg_empty_engine(request):
    """Ephemeral Postgres with NOTHING created — for the fresh `init_db()` path.

    Binds the legacy `core.database` singleton so `init_db()` targets this DB,
    and resets it before/after (PR #115 discipline).
    """
    url = _resolve_pg_url(request)
    # Isolated, freshly-(re)created empty schema so init_db() exercises the create
    # path; teardown CASCADE is scoped to this schema, never shared `public`.
    engine = _isolated_engine(url)
    engine.rainier_has_timescaledb = _try_create_extension(engine)  # type: ignore[attr-defined]

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
        _drop_isolated_schema(url)
