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
import secrets
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_suffix() -> str:
    """A short per-run/per-worker token for throwaway schema names.

    Including the xdist worker id (``PYTEST_XDIST_WORKER``, e.g. ``gw3``) and a
    random token guarantees two concurrent runs against the SAME shared test DB
    never collide on a fixed schema name — the old fixed names
    (``rainier_paper_test``) could clobber each other and, if a run was SIGKILL'd,
    leak. The token stays within the gc allowlist regex
    (``^rainier_(paper|chartmig|mig0006)_test(_[a-z0-9]+)*$``): lowercase
    alphanumerics joined by underscores. ``rainier db gc-test-schemas`` reaps any
    that still leak.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "").lower()
    worker = "".join(c for c in worker if c.isalnum()) or "main"
    return f"{worker}_{secrets.token_hex(3)}"
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
# R-D chart archive (task qu100-chart-archive-77f3): chart_images
# source/as_of_date/superseded_by + paper_trade.chart_id. Applied on top of 0005
# so full-entity PaperTrade ORM reads see paper_trade.chart_id.
MIGRATION_0010_UP = REPO_ROOT / "migrations" / "0010_chart_archive.sql"
MIGRATION_0010_DOWN = REPO_ROOT / "migrations" / "0010_chart_archive_downgrade.sql"
# R-E (PR 5): qu100_daily_features — the daily feature-snapshot table the
# feature step upserts into (and the wiring/persist tests read back).
MIGRATION_0011_UP = REPO_ROOT / "migrations" / "0011_qu100_daily_features.sql"
MIGRATION_0011_DOWN = (
    REPO_ROOT / "migrations" / "0011_qu100_daily_features_downgrade.sql"
)
# WS B (P0 batch): screened_stocks.bearish_invalidation_level +
# paper_reclaim_queue. Applied on top of 0005 so reclaim-path tests have the
# column + table.
MIGRATION_0012_UP = REPO_ROOT / "migrations" / "0012_reclaim_queue.sql"
MIGRATION_0012_DOWN = REPO_ROOT / "migrations" / "0012_reclaim_queue_downgrade.sql"
# WS A (P0 batch): paper_trade.shadow + shadow-scoped active-symbol indexes.
MIGRATION_0013_UP = REPO_ROOT / "migrations" / "0013_paper_trade_shadow.sql"
MIGRATION_0013_DOWN = (
    REPO_ROOT / "migrations" / "0013_paper_trade_shadow_downgrade.sql"
)

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

-- Mirror the ChartImage ORM (models.py) at its PRE-0010 shape (0001 create +
-- 0004 additive columns/indexes) so migration 0010's additive step stays under
-- test. chart_images has no CREATE TABLE migration (it predates the numbered
-- migrations; prod got it via `db init` create_all).
CREATE TABLE IF NOT EXISTS chart_images (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(10) NOT NULL REFERENCES stocks (symbol),
    captured_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    timeframe_days  INTEGER DEFAULT 120,
    file_path       VARCHAR(500),
    file_size_bytes INTEGER,
    image_bytes     BYTEA,
    sha256          VARCHAR(64),
    scan_date       DATE,
    width           INTEGER,
    height          INTEGER
);
CREATE INDEX IF NOT EXISTS ix_chart_images_sha256 ON chart_images (sha256);
CREATE INDEX IF NOT EXISTS ix_chart_images_scan_date ON chart_images (scan_date);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chart_image_symbol_scan_sha
    ON chart_images (symbol, scan_date, sha256)
    WHERE sha256 IS NOT NULL;

-- Mirror the MoneyFlowSnapshot ORM (models.py) for the Phase-3 miss-sweep
-- cohort selector (`get_current_qu100_cohort` reads data_date / ranking_type /
-- captured_at / rank) AND the R-D appearance-query tests. ORM inserts emit ALL
-- columns, so all must exist. Plain table here (prod is a TimescaleDB
-- hypertable; the composite PK is the only hypertable-specific shape, preserved).
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
# Base name; the live schema gets a per-run suffix (see `_run_suffix`) so two
# concurrent runs against a shared test DB never collide. Both stay within the
# `rainier db gc-test-schemas` allowlist.
_PAPER_SCHEMA_BASE = "rainier_paper_test"


@pytest.fixture
def pg_legacy_engine(request):
    """Ephemeral Postgres with prereq tables + the 0005 migration applied.

    Builds everything in a per-run disposable schema so teardown drops only that
    schema (never shared `public`) and each run starts clean. Also binds the
    legacy `core.database` singleton to it so production code under test (ingest
    / fill / positions) hits this DB. Resets the singleton before and after (PR
    #115 discipline).

    Leak-hardening (WS C): the schema name carries a per-run/per-worker suffix
    (no fixed-name collisions) and ALL setup (CREATE SCHEMA + DDL + migrations +
    singleton bind) runs INSIDE the try whose `finally` drops the schema — so a
    failure partway through setup still reaps the schema instead of leaking it.
    """
    url = _resolve_pg_url(request)
    schema = f"{_PAPER_SCHEMA_BASE}_{_run_suffix()}"
    engine = None
    try:
        # search_path-free admin connection so DROP/CREATE SCHEMA targets the
        # right namespace regardless of any pinned path.
        admin = create_engine(url, future=True)
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
            conn.execute(text(f"CREATE SCHEMA {schema}"))
        admin.dispose()

        engine = create_engine(
            url,
            future=True,
            connect_args={"options": f"-csearch_path={schema},public"},
        )
        engine.rainier_schema = schema  # type: ignore[attr-defined]
        with engine.begin() as conn:
            conn.execute(text(_PREREQ_DDL))
        _apply_sql(engine, MIGRATION_UP)
        _apply_sql(engine, MIGRATION_0007_UP)  # D7a paper_calibration
        _apply_sql(engine, MIGRATION_0008_UP)  # Phase 3 zero_share_price skip
        _apply_sql(engine, MIGRATION_0009_UP)  # R-A paper_trade.reflection
        _apply_sql(engine, MIGRATION_0003_UP)  # research_insights (lessons tests)
        # R-D chart archive. `exists()` guard keeps the rest of the paper suite
        # runnable during the tdd-red phase before the migration file lands;
        # test_chart_archive.py::test_migration_0010_files_exist pins the file's
        # existence so a green run can never silently skip it.
        if MIGRATION_0010_UP.exists():
            _apply_sql(engine, MIGRATION_0010_UP)
        _apply_sql(engine, MIGRATION_0011_UP)  # R-E qu100_daily_features
        _apply_sql(engine, MIGRATION_0012_UP)  # WS B reclaim queue + column
        _apply_sql(engine, MIGRATION_0013_UP)  # WS A paper_trade.shadow

        from rainier.core import config, database

        config._settings = None
        database._engine = engine
        database._session_factory = sessionmaker(
            bind=engine, expire_on_commit=False
        )
        yield engine
    finally:
        from rainier.core import config, database

        database._engine = None
        database._session_factory = None
        config._settings = None
        if engine is not None:
            engine.dispose()
        # Drop ONLY the throwaway schema — never shared `public`. A reused
        # `RAINIER_TEST_DATABASE_URL` keeps its real ORM tables intact.
        admin = create_engine(url, future=True)
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
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
# Migration 0010 (R-D chart archive) fixtures — PRE-0010 state so the
# migration's backfill / index creation / idempotency / downgrade run under
# test. Own disposable schema; teardown can never reach shared `public`.
# --------------------------------------------------------------------------

_CHART_MIG_SCHEMA_BASE = "rainier_chartmig_test"


@pytest.fixture
def pg_chart_mig_engine(request):
    """Ephemeral Postgres at the PRE-0010 schema state (prereqs + 0005 only).

    Tests seed pre-existing `chart_images` rows (incl. same-day duplicate
    thesis charts), then apply 0010 themselves and assert backfill/index/
    downgrade behavior. The legacy `core.database` singleton is NOT bound —
    these tests drive raw SQL/ORM sessions against the returned engine.

    Leak-hardening (WS C): per-run schema name; all setup inside the try whose
    finally drops the schema.
    """
    url = _resolve_pg_url(request)
    schema = f"{_CHART_MIG_SCHEMA_BASE}_{_run_suffix()}"
    engine = None
    try:
        admin = create_engine(url, future=True)
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
            conn.execute(text(f"CREATE SCHEMA {schema}"))
        admin.dispose()

        engine = create_engine(
            url,
            future=True,
            connect_args={"options": f"-csearch_path={schema},public"},
        )
        engine.rainier_schema = schema  # type: ignore[attr-defined]
        with engine.begin() as conn:
            conn.execute(text(_PREREQ_DDL))
        _apply_sql(engine, MIGRATION_UP)  # 0005: paper_trade (0010 alters it)
        yield engine
    finally:
        if engine is not None:
            engine.dispose()
        admin = create_engine(url, future=True)
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        admin.dispose()


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
_MIG0006_SCHEMA_BASE = "rainier_mig0006_test"


def _isolated_engine(url: str, schema: str):
    """Engine whose connections resolve names in the disposable 0006 test schema.

    `search_path` puts `schema` FIRST, so `create_all` / `init_db()` / the
    migration's unqualified `stock_prices`/`stocks` all create and resolve there
    (shadowing any `public` namesakes in a reused DB). `public` stays on the path
    only as a fallback so the timescaledb extension functions
    (`create_hypertable`, installed in `public`) remain resolvable. The schema is
    (re)created fresh and dropped CASCADE around the fixture body — CASCADE is
    scoped to throwaway objects only, never the shared `public` ORM tables.

    `schema` carries a per-run/per-worker suffix (WS C) so concurrent runs never
    collide on a fixed name.
    """
    engine = create_engine(
        url,
        future=True,
        connect_args={"options": f"-csearch_path={schema},public"},
    )
    # Recreate the schema fresh using a search_path-free connection so the DROP
    # CASCADE targets the right schema regardless of the pinned path.
    admin = create_engine(url, future=True)
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    admin.dispose()
    return engine


def _drop_isolated_schema(url: str, schema: str) -> None:
    admin = create_engine(url, future=True)
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
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
    schema = f"{_MIG0006_SCHEMA_BASE}_{_run_suffix()}"
    engine = None
    try:
        # Isolated schema: setup/teardown CASCADE can't touch shared `public`.
        engine = _isolated_engine(url, schema)
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
        yield engine
    finally:
        if engine is not None:
            engine.dispose()
        _drop_isolated_schema(url, schema)


@pytest.fixture
def pg_empty_engine(request):
    """Ephemeral Postgres with NOTHING created — for the fresh `init_db()` path.

    Binds the legacy `core.database` singleton so `init_db()` targets this DB,
    and resets it before/after (PR #115 discipline).
    """
    url = _resolve_pg_url(request)
    schema = f"{_MIG0006_SCHEMA_BASE}_{_run_suffix()}"
    engine = None
    try:
        # Isolated, freshly-(re)created empty schema so init_db() exercises the
        # create path; teardown CASCADE is scoped to this schema, never `public`.
        engine = _isolated_engine(url, schema)
        engine.rainier_has_timescaledb = _try_create_extension(  # type: ignore[attr-defined]
            engine
        )

        from rainier.core import config, database

        config._settings = None
        database._engine = engine
        database._session_factory = sessionmaker(
            bind=engine, expire_on_commit=False
        )
        yield engine
    finally:
        from rainier.core import config, database

        database._engine = None
        database._session_factory = None
        config._settings = None
        if engine is not None:
            engine.dispose()
        _drop_isolated_schema(url, schema)
