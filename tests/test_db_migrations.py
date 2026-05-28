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
    """The 5 tables declared in the design + task plan §4."""
    return {
        "tickers",
        "sectors",
        "thematic_ohlcv",
        "thematic_features_daily",
        "thematic_labels_daily",
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
