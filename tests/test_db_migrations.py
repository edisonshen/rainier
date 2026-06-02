"""Tests for Alembic migration 0001 — Postgres canonical-store pivot Phase 1.

Verifies:
  * `alembic upgrade head` on an empty Postgres creates the `market` schema
    plus the 5 tables exactly as declared in src/rainier/db/schema.py.
  * `alembic downgrade base` returns to an empty state (schema dropped).
  * Re-running upgrade is idempotent (no-op when already at head).
  * Introspected schema matches schema.py exactly (catches the
    include_schemas=True footgun on day 1 — see task plan §3).

Skipped unless a local Postgres is reachable. Uses pytest-postgresql when
the binary is available; otherwise honors `RAINIER_TEST_DATABASE_URL` for
operators with a manual local Postgres handy.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, inspect, text

# Mark every test in this module — the schema is on Postgres, not SQLite.
pytestmark = pytest.mark.requires_postgres


# ---------------------------------------------------------------------------
# Fixture: database_url
# ---------------------------------------------------------------------------
# Priority:
#   1. If RAINIER_TEST_DATABASE_URL is set, use it (operator-provided sandbox).
#   2. Otherwise, try pytest-postgresql for an isolated per-test DB.
#   3. Otherwise, skip — no Postgres available.

try:
    from pytest_postgresql import factories as _pg_factories

    _HAS_PYTEST_POSTGRESQL = True
except ImportError:  # pragma: no cover
    _HAS_PYTEST_POSTGRESQL = False


def _local_pg_binary_available() -> bool:
    """pytest-postgresql needs pg_config + initdb on $PATH. If they're
    missing we can't spin up an isolated DB and must skip cleanly rather
    than letting the fixture raise ExecutableMissingException."""
    import shutil

    return shutil.which("pg_config") is not None and shutil.which("initdb") is not None


if _HAS_PYTEST_POSTGRESQL and _local_pg_binary_available():
    postgresql_proc = _pg_factories.postgresql_proc(port=None, unixsocketdir="/tmp")
    postgresql = _pg_factories.postgresql("postgresql_proc")


@pytest.fixture
def database_url(request, monkeypatch):
    """Return a DATABASE_URL for a clean Postgres + set it on the env.

    Resolution priority:
      1. ``RAINIER_TEST_DATABASE_URL`` — operator-supplied sandbox (docker, neon, etc).
      2. pytest-postgresql — ephemeral local Postgres (needs pg_config + initdb).
      3. Skip — no Postgres reachable.
    """
    env_url = os.environ.get("RAINIER_TEST_DATABASE_URL")
    if env_url:
        monkeypatch.setenv("DATABASE_URL", env_url)
        yield env_url
        return

    if not _HAS_PYTEST_POSTGRESQL:
        pytest.skip("pytest-postgresql not installed")
    if not _local_pg_binary_available():
        pytest.skip(
            "pg_config / initdb not on PATH — install postgres locally or set "
            "RAINIER_TEST_DATABASE_URL"
        )

    pg = request.getfixturevalue("postgresql")
    url = (
        f"postgresql+psycopg://{pg.info.user}@{pg.info.host}:{pg.info.port}"
        f"/{pg.info.dbname}"
    )
    monkeypatch.setenv("DATABASE_URL", url)
    yield url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _alembic_config():
    """Build an Alembic config pointing at db/alembic.ini with the script
    location resolved relative to the repo root (NOT cwd)."""
    from pathlib import Path

    from alembic.config import Config

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "db" / "alembic.ini"
    cfg = Config(str(cfg_path))
    # alembic.ini stores `script_location = %(here)s/alembic`, which already
    # resolves relative to the ini file. We override anyway as defense-in-depth
    # in case a future edit drops the %(here)s prefix — the regression test
    # `test_raw_alembic_ini_works_from_any_cwd` below catches that footgun.
    cfg.set_main_option("script_location", str(repo_root / "db" / "alembic"))
    return cfg


def _expected_tables() -> set[str]:
    """All market.* tables after migration head.

    Phase 1/2 (task plan §4): the 5 thematic tables.
    breadth-pg-write-path (0003): + breadth_indicator_daily + benchmark_ohlcv.
    """
    return {
        "tickers",
        "sectors",
        "thematic_ohlcv",
        "thematic_features_daily",
        "thematic_labels_daily",
        "breadth_indicator_daily",
        "benchmark_ohlcv",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_upgrade_head_creates_market_schema_and_tables(database_url):
    """`alembic upgrade head` on empty DB creates schema + 5 tables."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)
    schemas = set(insp.get_schema_names())
    assert "market" in schemas, f"market schema missing; found {schemas}"

    tables = set(insp.get_table_names(schema="market"))
    assert tables == _expected_tables(), (
        f"market.* tables mismatch — got {tables}, expected {_expected_tables()}"
    )

    eng.dispose()


def test_upgrade_is_idempotent(database_url):
    """Re-running upgrade head after head is a no-op."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    # Second call should not error.
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)
    tables = set(insp.get_table_names(schema="market"))
    assert tables == _expected_tables()
    eng.dispose()


def test_downgrade_base_drops_market_schema(database_url):
    """`alembic downgrade base` returns to empty state."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    eng = create_engine(database_url)
    insp = inspect(eng)
    schemas = set(insp.get_schema_names())
    # market schema is dropped; alembic_version table stays in public.
    assert "market" not in schemas, (
        f"market schema not dropped on downgrade; found {schemas}"
    )
    eng.dispose()


def test_alembic_version_table_lives_in_public(database_url):
    """Per task plan §5: alembic_version belongs in public, not market.
    Otherwise dropping market in downgrade also wipes alembic state."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT table_schema FROM information_schema.tables "
                "WHERE table_name = 'alembic_version'"
            )
        ).fetchall()
    eng.dispose()

    schemas = {r[0] for r in rows}
    assert schemas == {"public"}, (
        f"alembic_version should live in public only; found {schemas}"
    )


def test_schema_introspection_matches_schema_py(database_url):
    """After upgrade head, the introspected column shape for every table
    matches the SQLAlchemy `Table` definitions in src/rainier/db/schema.py.

    This catches the Alembic include_schemas=True footgun on day 1: if
    autogenerate hadn't been aware of market.*, the migration would skip
    half the columns and this test would fail."""
    from alembic import command

    from rainier.db.schema import metadata as target_metadata

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)

    for table in target_metadata.sorted_tables:
        if table.schema != "market":
            continue
        live_cols = {c["name"]: c for c in insp.get_columns(table.name, schema="market")}
        decl_cols = {c.name: c for c in table.columns}
        assert set(live_cols) == set(decl_cols), (
            f"column set mismatch for market.{table.name} — "
            f"live={set(live_cols)} declared={set(decl_cols)}"
        )

    eng.dispose()


def test_expected_indexes_exist(database_url):
    """Indexes on (date) / (asof_date) per task plan §4 schema DDL."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)

    # thematic_ohlcv: index on (date)
    ohlcv_idx = insp.get_indexes("thematic_ohlcv", schema="market")
    assert any(
        idx["column_names"] == ["date"] for idx in ohlcv_idx
    ), f"missing index on market.thematic_ohlcv(date); got {ohlcv_idx}"

    # thematic_features_daily: index on (asof_date)
    feat_idx = insp.get_indexes("thematic_features_daily", schema="market")
    assert any(
        idx["column_names"] == ["asof_date"] for idx in feat_idx
    ), f"missing index on market.thematic_features_daily(asof_date); got {feat_idx}"

    # thematic_labels_daily: index on (asof_date)
    lbl_idx = insp.get_indexes("thematic_labels_daily", schema="market")
    assert any(
        idx["column_names"] == ["asof_date"] for idx in lbl_idx
    ), f"missing index on market.thematic_labels_daily(asof_date); got {lbl_idx}"

    eng.dispose()


def test_foreign_keys_to_registries(database_url):
    """thematic_features_daily references market.tickers + market.sectors."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)
    fks = insp.get_foreign_keys("thematic_features_daily", schema="market")
    eng.dispose()

    referred = {(fk["referred_schema"], fk["referred_table"]) for fk in fks}
    assert ("market", "tickers") in referred, f"FK→market.tickers missing; got {fks}"
    assert ("market", "sectors") in referred, f"FK→market.sectors missing; got {fks}"


# ---------------------------------------------------------------------------
# downgrade base must not wipe foreign objects (task plan migrate-downgrade-base)
# ---------------------------------------------------------------------------
# codex iter-4 of PR #102 (deferred P2): 0001's downgrade() issued an
# unconditional `DROP SCHEMA market CASCADE`. Acceptable while rainier solely
# owns `market`, but if the schema ever holds a non-rainier object, downgrade
# base would silently wipe it. The guard: downgrade() drops only the objects
# THIS revision's upgrade() created, and removes the schema only when no
# foreign objects remain.


def _reset_market(eng) -> None:
    """Force the DB back to a pre-0001 state regardless of prior test runs.

    The ``database_url`` fixture gives a per-test isolated DB only via
    pytest-postgresql; when an operator points ``RAINIER_TEST_DATABASE_URL`` at
    a reused sandbox there is no isolation. Dropping ``market`` + the alembic
    bookkeeping row makes this test deterministic on either path."""
    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS market CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS public.alembic_version"))


def test_downgrade_base_preserves_foreign_objects(database_url):
    """A non-rainier object living in ``market`` survives ``downgrade base``.

    Regression for the unconditional ``DROP SCHEMA market CASCADE`` footgun:
    seed a foreign table into ``market`` AFTER upgrade head, then downgrade
    base. The foreign table (and the ``market`` schema it lives in) must
    survive; this revision's own tables must be gone.
    """
    from alembic import command

    eng = create_engine(database_url)
    _reset_market(eng)

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    # Seed a foreign object that this revision did NOT create.
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE market.foreign_owned (id integer primary key)"))
        conn.execute(text("INSERT INTO market.foreign_owned (id) VALUES (1), (2)"))

    command.downgrade(cfg, "base")

    insp = inspect(eng)
    schemas = set(insp.get_schema_names())
    assert "market" in schemas, (
        "downgrade base dropped the whole market schema while a foreign object "
        f"was present; surviving schemas={schemas}"
    )

    tables = set(insp.get_table_names(schema="market"))
    # Foreign object survives untouched.
    assert "foreign_owned" in tables, (
        f"foreign object market.foreign_owned was wiped by downgrade; tables={tables}"
    )
    with eng.connect() as conn:
        rows = conn.execute(text("SELECT id FROM market.foreign_owned ORDER BY id")).fetchall()
    assert [r[0] for r in rows] == [1, 2], "foreign object rows were not preserved"

    # This revision's own tables are gone.
    leftover = tables & _expected_tables()
    assert leftover == set(), (
        f"downgrade base left this revision's own tables behind: {leftover}"
    )

    _reset_market(eng)
    eng.dispose()


def test_downgrade_base_preserves_foreign_sequence(database_url):
    """The emptiness gate covers non-table objects too: a foreign SEQUENCE in
    ``market`` survives ``downgrade base`` and keeps the schema alive.

    A tables-only gate would think the schema empty, attempt the RESTRICT drop,
    and abort the downgrade with an error — this locks in the pg_class/pg_proc/
    pg_type probe."""
    from alembic import command

    eng = create_engine(database_url)
    _reset_market(eng)

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    with eng.begin() as conn:
        conn.execute(text("CREATE SEQUENCE market.foreign_seq"))

    command.downgrade(cfg, "base")

    insp = inspect(eng)
    assert "market" in set(insp.get_schema_names()), (
        "downgrade base dropped market while a foreign sequence was present"
    )
    with eng.connect() as conn:
        exists = conn.execute(
            text(
                "SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'market' AND c.relname = 'foreign_seq' AND c.relkind = 'S'"
            )
        ).fetchone()
    assert exists is not None, "foreign sequence market.foreign_seq was wiped by downgrade"
    assert set(insp.get_table_names(schema="market")) & _expected_tables() == set(), (
        "downgrade base left this revision's own tables behind"
    )

    _reset_market(eng)
    eng.dispose()


def test_downgrade_base_clean_round_trip(database_url):
    """On a DB where ``market`` is exclusively this revision's, upgrade ->
    downgrade -> upgrade still round-trips: the schema is fully removed on
    downgrade and re-created on the next upgrade."""
    from alembic import command

    eng = create_engine(database_url)
    _reset_market(eng)

    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    insp = inspect(eng)
    # No foreign objects → schema is fully removed, returning to empty state.
    assert "market" not in set(insp.get_schema_names()), (
        "downgrade base should drop the now-empty market schema on a clean DB"
    )

    # Re-upgrade rebuilds everything.
    command.upgrade(cfg, "head")
    insp = inspect(eng)
    assert set(insp.get_table_names(schema="market")) == _expected_tables()

    _reset_market(eng)
    eng.dispose()


def test_downgrade_base_fails_loudly_on_foreign_dependent(database_url):
    """A foreign object DEPENDING ON one of this revision's tables is never
    silently destroyed: the non-cascading ``DROP TABLE ... RESTRICT`` aborts
    the downgrade instead.

    codex iter-2 (P1): the original ``DROP TABLE ... CASCADE`` would silently
    drop a foreign view/FK built on top of ``market.tickers`` even though the
    schema is preserved — the exact data/DDL loss this revision is meant to
    prevent. With RESTRICT, the drop raises; the operator sees the foreign
    dependency and resolves it rather than losing it. We assert both the raise
    AND that the foreign view + our table are still intact afterward (the
    downgrade transaction rolled back).
    """
    import sqlalchemy.exc as sa_exc
    from alembic import command

    eng = create_engine(database_url)
    _reset_market(eng)

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    # Foreign view built on a revision-owned table. CASCADE would have dropped
    # it silently; RESTRICT must refuse.
    with eng.begin() as conn:
        conn.execute(
            text("CREATE VIEW market.foreign_view AS SELECT ticker_id FROM market.tickers")
        )

    with pytest.raises(sa_exc.DBAPIError):
        command.downgrade(cfg, "base")

    # The aborted downgrade left everything intact: the foreign view, the table
    # it depends on, and the schema all survive.
    insp = inspect(eng)
    assert "market" in set(insp.get_schema_names()), "schema was dropped despite a foreign dependent"
    assert "tickers" in set(insp.get_table_names(schema="market")), (
        "market.tickers was dropped despite a foreign view depending on it"
    )
    with eng.connect() as conn:
        view_exists = conn.execute(
            text(
                "SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'market' AND c.relname = 'foreign_view' AND c.relkind = 'v'"
            )
        ).fetchone()
    assert view_exists is not None, "foreign view market.foreign_view was wiped by downgrade"

    _reset_market(eng)
    eng.dispose()


# ---------------------------------------------------------------------------
# Migration 0002 — schema-seam fixes for the dual-write writers (task plan §3)
# ---------------------------------------------------------------------------


def _column(insp, table, name):
    """Return the introspected column dict for ``name`` or None."""
    for col in insp.get_columns(table, schema="market"):
        if col["name"] == name:
            return col
    return None


def test_0002_adds_trading_day_ordinal(database_url):
    """`upgrade head` (through 0002) adds market.thematic_features_daily.
    trading_day_ordinal as a nullable integer column."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)
    col = _column(insp, "thematic_features_daily", "trading_day_ordinal")
    eng.dispose()

    assert col is not None, "trading_day_ordinal column missing after 0002"
    assert col["nullable"] is True, f"trading_day_ordinal should be nullable; got {col}"
    # Integer (int4) — matches the int32 ordinal the feature compute emits.
    assert "INT" in str(col["type"]).upper(), f"unexpected type {col['type']}"


def test_0002_relaxes_label_complete_through_nullable(database_url):
    """After 0002, market.thematic_labels_daily.label_complete_through is
    nullable (was NOT NULL in 0001)."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)
    col = _column(insp, "thematic_labels_daily", "label_complete_through")
    eng.dispose()

    assert col is not None, "label_complete_through column missing"
    assert col["nullable"] is True, (
        f"label_complete_through should be nullable after 0002; got {col}"
    )


def test_0002_downgrade_reverts_both_seams(database_url):
    """`downgrade -1` (0002 -> 0001) drops trading_day_ordinal and re-tightens
    label_complete_through to NOT NULL."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "0001")

    eng = create_engine(database_url)
    insp = inspect(eng)
    ord_col = _column(insp, "thematic_features_daily", "trading_day_ordinal")
    lbl_col = _column(insp, "thematic_labels_daily", "label_complete_through")
    eng.dispose()

    assert ord_col is None, "trading_day_ordinal should be dropped on downgrade"
    assert lbl_col is not None
    assert lbl_col["nullable"] is False, (
        f"label_complete_through should be NOT NULL again after downgrade; got {lbl_col}"
    )


def test_0002_upgrade_downgrade_upgrade_round_trips(database_url):
    """0002 is fully reversible: head -> 0001 -> head leaves the column present
    and the constraint relaxed again."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "0001")
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)
    ord_col = _column(insp, "thematic_features_daily", "trading_day_ordinal")
    lbl_col = _column(insp, "thematic_labels_daily", "label_complete_through")
    eng.dispose()

    assert ord_col is not None and ord_col["nullable"] is True
    assert lbl_col is not None and lbl_col["nullable"] is True


# ---------------------------------------------------------------------------
# Migration 0003 — breadth_indicator_daily + benchmark_ohlcv (breadth-pg)
# ---------------------------------------------------------------------------


def test_0003_creates_breadth_indicator_daily(database_url):
    """`upgrade head` (through 0003) creates market.breadth_indicator_daily
    with the LONG shape (asof_date, indicator, value) and PK (asof_date,
    indicator)."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)
    cols = {c["name"]: c for c in insp.get_columns("breadth_indicator_daily", schema="market")}
    pk = insp.get_pk_constraint("breadth_indicator_daily", schema="market")
    eng.dispose()

    assert set(cols) == {"asof_date", "indicator", "value"}, f"got {set(cols)}"
    # value mirrors parquet float64 -> double precision, nullable for NaN.
    assert "DOUBLE" in str(cols["value"]["type"]).upper(), cols["value"]["type"]
    assert cols["value"]["nullable"] is True
    assert cols["asof_date"]["nullable"] is False
    assert cols["indicator"]["nullable"] is False
    assert set(pk["constrained_columns"]) == {"asof_date", "indicator"}, pk


def test_0003_creates_benchmark_ohlcv(database_url):
    """`upgrade head` creates market.benchmark_ohlcv mirroring thematic_ohlcv
    (symbol/date PK, OHLCV double precision, volume bigint, provenance cols)."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)
    cols = {c["name"]: c for c in insp.get_columns("benchmark_ohlcv", schema="market")}
    pk = insp.get_pk_constraint("benchmark_ohlcv", schema="market")
    eng.dispose()

    assert set(cols) == {
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "fetched_at",
        "yfinance_version",
    }, f"got {set(cols)}"
    assert "BIGINT" in str(cols["volume"]["type"]).upper(), cols["volume"]["type"]
    for c in ("open", "high", "low", "close"):
        assert "DOUBLE" in str(cols[c]["type"]).upper(), cols[c]["type"]
    assert set(pk["constrained_columns"]) == {"symbol", "date"}, pk


def test_0003_breadth_indicator_index(database_url):
    """breadth_indicator_daily has an index on (asof_date) like the other
    date-keyed tables (verify-coverage groups by asof_date)."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)
    idx = insp.get_indexes("breadth_indicator_daily", schema="market")
    eng.dispose()
    assert any(
        i["column_names"] == ["asof_date"] for i in idx
    ), f"missing index on breadth_indicator_daily(asof_date); got {idx}"


def test_0003_benchmark_ohlcv_index(database_url):
    """benchmark_ohlcv has an index on (date) like thematic_ohlcv."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)
    idx = insp.get_indexes("benchmark_ohlcv", schema="market")
    eng.dispose()
    assert any(
        i["column_names"] == ["date"] for i in idx
    ), f"missing index on benchmark_ohlcv(date); got {idx}"


def test_0003_no_stray_public_tables(database_url):
    """Acceptance gate (design D-6): 0003 must NOT leak tables into the public
    schema. The include_schemas=True footgun would put schema-unqualified
    tables in public; assert public holds only alembic_version."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)
    public_tables = set(insp.get_table_names(schema="public"))
    eng.dispose()
    # The only public table should be alembic's bookkeeping; the two new
    # market.* tables must live under market, not public.
    assert public_tables <= {"alembic_version"}, (
        f"0003 leaked tables into public: {public_tables - {'alembic_version'}}"
    )


def test_0003_downgrade_round_trips(database_url):
    """0003 is reversible: head -> 0002 drops the two new tables; -> head
    recreates them. Downgrade is guarded (does NOT unconditionally drop the
    market schema — that belongs to 0001)."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "0002")

    eng = create_engine(database_url)
    insp = inspect(eng)
    after_down = set(insp.get_table_names(schema="market"))
    # market schema still exists (0001 owns it); only the 0003 tables are gone.
    schemas = set(insp.get_schema_names())
    eng.dispose()
    assert "market" in schemas, "0003 downgrade must NOT drop the market schema"
    assert "breadth_indicator_daily" not in after_down
    assert "benchmark_ohlcv" not in after_down
    # thematic tables survive (owned by 0001).
    assert "thematic_ohlcv" in after_down

    command.upgrade(cfg, "head")
    eng = create_engine(database_url)
    insp = inspect(eng)
    after_up = set(insp.get_table_names(schema="market"))
    eng.dispose()
    assert "breadth_indicator_daily" in after_up
    assert "benchmark_ohlcv" in after_up


def test_alembic_does_not_disable_existing_loggers(database_url):
    """Regression: alembic env.py runs ``fileConfig(..., disable_existing_loggers
    =False)``. The default (True) tears down every logger created before alembic
    runs, silencing rainier's own loggers in-process — which makes any later
    caplog-based test that asserts on a rainier log message fail. Guard the flag.

    We register a logger BEFORE running alembic, then assert it still emits
    (is not disabled) afterward.
    """
    import logging

    from alembic import command

    canary = logging.getLogger("rainier._alembic_logging_canary")
    canary.disabled = False
    assert canary.disabled is False

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    # If env.py used disable_existing_loggers=True, this canary would now be
    # disabled (logging.config disables all loggers not named in the ini).
    assert canary.disabled is False, (
        "alembic fileConfig disabled a pre-existing logger — env.py must pass "
        "disable_existing_loggers=False"
    )


# ---------------------------------------------------------------------------
# Migration 0004 — backup.money_flow_snapshots (money-flow-neon-backup-b613)
# ---------------------------------------------------------------------------
# Hand-written (env.py:_include_only_market does NOT manage the ``backup``
# schema — intentional, same as it doesn't manage ``public``). 0004 creates a
# plain-table off-machine backup of the irreplaceable QU100 money-flow history.


def test_0004_creates_backup_schema_and_table(database_url):
    """`upgrade head` (through 0004) creates the ``backup`` schema +
    backup.money_flow_snapshots mirroring core/models.py columns with the
    composite PK (id, captured_at) and no stocks FK."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)
    schemas = set(insp.get_schema_names())
    assert "backup" in schemas, f"backup schema missing; found {schemas}"

    cols = {
        c["name"]: c
        for c in insp.get_columns("money_flow_snapshots", schema="backup")
    }
    pk = insp.get_pk_constraint("money_flow_snapshots", schema="backup")
    fks = insp.get_foreign_keys("money_flow_snapshots", schema="backup")
    eng.dispose()

    assert set(cols) == {
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
    }, f"got {set(cols)}"
    # Composite PK mirrors the source exactly (id alone is NOT DB-unique).
    assert set(pk["constrained_columns"]) == {"id", "captured_at"}, pk
    # raw_data is JSONB (not text).
    assert "JSON" in str(cols["raw_data"]["type"]).upper(), cols["raw_data"]["type"]
    # No stocks FK on the standalone backup table.
    assert fks == [], f"backup table must have no FK; got {fks}"


def test_0004_backup_index_on_data_date_ranking_type(database_url):
    """Index on (data_date, ranking_type) for ad-hoc restore queries."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)
    idx = insp.get_indexes("money_flow_snapshots", schema="backup")
    eng.dispose()
    assert any(
        i["column_names"] == ["data_date", "ranking_type"] for i in idx
    ), f"missing index on backup.money_flow_snapshots(data_date, ranking_type); got {idx}"


def test_0004_does_not_touch_market_schema(database_url):
    """0004 owns only the ``backup`` schema; the ``market`` tables are intact."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)
    market_tables = set(insp.get_table_names(schema="market"))
    eng.dispose()
    # All the market.* tables from 0001-0003 still present.
    assert _expected_tables() <= market_tables, (
        f"0004 disturbed market.*; got {market_tables}"
    )


def test_0004_downgrade_drops_backup_schema(database_url):
    """`downgrade` (0004 -> 0003) drops backup.money_flow_snapshots + the
    backup schema; the market schema + tables survive (0004 owns only backup)."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "0003")

    eng = create_engine(database_url)
    insp = inspect(eng)
    schemas = set(insp.get_schema_names())
    market_tables = set(insp.get_table_names(schema="market"))
    eng.dispose()

    assert "backup" not in schemas, (
        f"0004 downgrade must drop the backup schema; found {schemas}"
    )
    # market schema + tables untouched by the backup downgrade.
    assert "market" in schemas
    assert _expected_tables() <= market_tables


def test_0004_downgrade_round_trips(database_url):
    """0004 is reversible: head -> 0003 -> head re-creates the backup table."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "0003")
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)
    assert "backup" in set(insp.get_schema_names())
    assert "money_flow_snapshots" in set(
        insp.get_table_names(schema="backup")
    )
    eng.dispose()


def test_0004_downgrade_preserves_preexisting_schema_object(database_url):
    """Codex P2 — downgrade must NOT CASCADE-drop a PRE-EXISTING ``backup`` schema.

    Upgrade uses CREATE SCHEMA IF NOT EXISTS, so the schema can pre-exist with
    unrelated objects. A naive ``DROP SCHEMA backup CASCADE`` on downgrade would
    delete them (data loss). Downgrade must drop only its own table/index and
    leave a non-empty ``backup`` schema (plus the foreign object) intact.
    """
    from alembic import command
    from sqlalchemy import text

    cfg = _alembic_config()

    # Pre-seed the backup schema with an unrelated object BEFORE migrating.
    eng = create_engine(database_url)
    with eng.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS backup"))
        conn.execute(text("CREATE TABLE backup.unrelated_keepme (x int)"))
        conn.execute(text("INSERT INTO backup.unrelated_keepme (x) VALUES (42)"))
    eng.dispose()

    command.upgrade(cfg, "head")  # adopts the existing schema, adds our table
    command.downgrade(cfg, "0003")  # must drop ONLY our table/index

    eng = create_engine(database_url)
    insp = inspect(eng)
    schemas = set(insp.get_schema_names())
    backup_tables = set(insp.get_table_names(schema="backup"))
    # The foreign object + its data survive; our table is gone; schema preserved.
    with eng.connect() as conn:
        kept = conn.execute(text("SELECT x FROM backup.unrelated_keepme")).scalar_one()
    eng.dispose()

    assert "backup" in schemas, "non-empty backup schema must be preserved"
    assert "unrelated_keepme" in backup_tables, "foreign object must survive downgrade"
    assert "money_flow_snapshots" not in backup_tables, "our table must be dropped"
    assert kept == 42, "foreign object's data must be intact (no CASCADE wipe)"


def test_0004_no_stray_public_tables(database_url):
    """0004 must NOT leak tables into public (only alembic_version belongs)."""
    from alembic import command

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    insp = inspect(eng)
    public_tables = set(insp.get_table_names(schema="public"))
    eng.dispose()
    assert public_tables <= {"alembic_version"}, (
        f"0004 leaked tables into public: {public_tables - {'alembic_version'}}"
    )
