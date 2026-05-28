"""Tests for the ``market_upsert`` helper — Phase 2 dual-write (task plan §6).

Verifies the one shared UPSERT helper that all three dual-write call sites use:

  * insert then re-insert the same PK is idempotent (DO UPDATE, no duplicate);
  * a second run with changed non-PK columns updates in place;
  * batching across the batch boundary inserts every row exactly once;
  * FK-child insert succeeds when the parent rows are inserted first (ordering
    is the caller's job — the helper doesn't reorder, so the test documents
    the contract).

Gated on ``requires_postgres`` — ON CONFLICT is Postgres-dialect-specific.
Reuses the same database_url fixture shape as tests/test_db_migrations.py.
"""

from __future__ import annotations

import datetime as dt
import os

import pytest
from sqlalchemy import create_engine, select, text

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
def database_url(request, monkeypatch):
    """A DATABASE_URL for a clean Postgres + set it on the env.

    Priority: RAINIER_TEST_DATABASE_URL -> pytest-postgresql -> skip.
    """
    env_url = os.environ.get("RAINIER_TEST_DATABASE_URL")
    if env_url:
        monkeypatch.setenv("DATABASE_URL", env_url)
        yield env_url
        return

    if not _HAS_PYTEST_POSTGRESQL:
        pytest.skip("pytest-postgresql not installed")
    if not _local_pg_binary_available():
        pytest.skip("pg_config / initdb not on PATH; set RAINIER_TEST_DATABASE_URL")

    pg = request.getfixturevalue("postgresql")
    url = (
        f"postgresql+psycopg://{pg.info.user}@{pg.info.host}:{pg.info.port}"
        f"/{pg.info.dbname}"
    )
    monkeypatch.setenv("DATABASE_URL", url)
    yield url


@pytest.fixture
def migrated_engine(database_url):
    """Run alembic to head, yield a fresh engine, drop the schema on teardown."""
    from pathlib import Path

    from alembic import command
    from alembic.config import Config

    repo_root = Path(__file__).resolve().parents[1]
    cfg = Config(str(repo_root / "db" / "alembic.ini"))
    cfg.set_main_option("script_location", str(repo_root / "db" / "alembic"))
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    try:
        yield eng
    finally:
        # Reset so a re-run against the same DB starts clean. DROP CASCADE +
        # clear alembic bookkeeping (unconditional; doesn't depend on the
        # data-protecting 0002 downgrade succeeding).
        with eng.begin() as conn:
            conn.exec_driver_sql("DROP SCHEMA IF EXISTS market CASCADE")
            conn.exec_driver_sql("DROP TABLE IF EXISTS public.alembic_version")
        eng.dispose()


def _count(eng, table) -> int:
    with eng.connect() as conn:
        return conn.execute(
            text(f"SELECT count(*) FROM market.{table}")
        ).scalar_one()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_insert_then_upsert_is_idempotent(migrated_engine):
    """Re-running the same rows leaves the row count unchanged (DO UPDATE)."""
    from rainier.db import schema
    from rainier.db.upsert import market_upsert

    rows = [
        {"sector_id": 1, "sector_name": "tech", "first_seen": dt.date(2024, 1, 1)},
        {"sector_id": 2, "sector_name": "energy", "first_seen": dt.date(2024, 1, 1)},
    ]
    market_upsert(migrated_engine, schema.sectors, rows, ["sector_id"])
    assert _count(migrated_engine, "sectors") == 2

    # Same rows again — must not duplicate.
    market_upsert(migrated_engine, schema.sectors, rows, ["sector_id"])
    assert _count(migrated_engine, "sectors") == 2


def test_upsert_updates_non_pk_columns(migrated_engine):
    """A second upsert with a changed non-PK column updates the row in place."""
    from rainier.db import schema
    from rainier.db.upsert import market_upsert

    market_upsert(
        migrated_engine,
        schema.sectors,
        [{"sector_id": 1, "sector_name": "tech", "first_seen": dt.date(2024, 1, 1)}],
        ["sector_id"],
    )
    market_upsert(
        migrated_engine,
        schema.sectors,
        [{"sector_id": 1, "sector_name": "technology", "first_seen": dt.date(2024, 1, 1)}],
        ["sector_id"],
    )
    with migrated_engine.connect() as conn:
        name = conn.execute(
            select(schema.sectors.c.sector_name).where(schema.sectors.c.sector_id == 1)
        ).scalar_one()
    assert name == "technology"
    assert _count(migrated_engine, "sectors") == 1


def test_upsert_batches_across_boundary(migrated_engine):
    """Inserting more rows than the batch size inserts every row exactly once."""
    from rainier.db import schema
    from rainier.db.upsert import market_upsert

    n = 2500  # > default batch_size of 1000
    rows = [
        {"ticker_id": i, "symbol": f"SYM{i}", "first_seen": dt.date(2024, 1, 1)}
        for i in range(1, n + 1)
    ]
    market_upsert(migrated_engine, schema.tickers, rows, ["ticker_id"], batch_size=1000)
    assert _count(migrated_engine, "tickers") == n

    # Re-run idempotent across batches too.
    market_upsert(migrated_engine, schema.tickers, rows, ["ticker_id"], batch_size=1000)
    assert _count(migrated_engine, "tickers") == n


def test_upsert_empty_rows_is_noop(migrated_engine):
    """An empty row list must not error and must not write anything."""
    from rainier.db import schema
    from rainier.db.upsert import market_upsert

    market_upsert(migrated_engine, schema.sectors, [], ["sector_id"])
    assert _count(migrated_engine, "sectors") == 0


def test_upsert_immutable_first_seen_not_overwritten(migrated_engine):
    """immutable_cols keeps first_seen stable: a later-day re-run with a newer
    first_seen value must NOT re-stamp the original (registry provenance)."""
    from rainier.db import schema
    from rainier.db.upsert import market_upsert

    market_upsert(
        migrated_engine,
        schema.tickers,
        [{"ticker_id": 1, "symbol": "AAA", "first_seen": dt.date(2024, 1, 1)}],
        ["ticker_id"],
        immutable_cols=["first_seen"],
    )
    # Re-run on a later day: same key, newer first_seen — must be ignored.
    market_upsert(
        migrated_engine,
        schema.tickers,
        [{"ticker_id": 1, "symbol": "AAA", "first_seen": dt.date(2024, 2, 2)}],
        ["ticker_id"],
        immutable_cols=["first_seen"],
    )
    with migrated_engine.connect() as conn:
        fs = conn.execute(
            select(schema.tickers.c.first_seen).where(schema.tickers.c.ticker_id == 1)
        ).scalar_one()
    assert fs == dt.date(2024, 1, 1), "first_seen must stay at the original date"


def test_upsert_omitted_column_not_nulled_on_conflict(migrated_engine):
    """A column the caller omits on re-run must NOT be reset to NULL. The
    registry write omits last_seen; an existing last_seen value must survive."""
    from rainier.db import schema
    from rainier.db.upsert import market_upsert

    # Initial: explicitly set last_seen (simulates a later delist marker).
    market_upsert(
        migrated_engine,
        schema.tickers,
        [
            {
                "ticker_id": 1,
                "symbol": "AAA",
                "first_seen": dt.date(2024, 1, 1),
                "last_seen": dt.date(2024, 6, 1),
            }
        ],
        ["ticker_id"],
        immutable_cols=["first_seen"],
    )
    # Re-run as the registry writers do: omit last_seen entirely.
    market_upsert(
        migrated_engine,
        schema.tickers,
        [{"ticker_id": 1, "symbol": "AAA", "first_seen": dt.date(2024, 1, 1)}],
        ["ticker_id"],
        immutable_cols=["first_seen"],
    )
    with migrated_engine.connect() as conn:
        ls = conn.execute(
            select(schema.tickers.c.last_seen).where(schema.tickers.c.ticker_id == 1)
        ).scalar_one()
    assert ls == dt.date(2024, 6, 1), "omitted last_seen must not be clobbered to NULL"


def test_upsert_immutable_identity_not_remapped_on_conflict(migrated_engine):
    """Registry safety ([D-015]): with the identity name column immutable, a
    conflict on the stable id must NOT remap symbol. A diverged caller (id=1
    now claims a different symbol) leaves the original row untouched rather
    than silently repointing existing feature FK rows at the wrong ticker."""
    from rainier.db import schema
    from rainier.db.upsert import market_upsert

    market_upsert(
        migrated_engine,
        schema.tickers,
        [{"ticker_id": 1, "symbol": "AAA", "first_seen": dt.date(2024, 1, 1)}],
        ["ticker_id"],
        immutable_cols=["symbol", "first_seen"],
    )
    # Diverged re-run: id 1 now carries a DIFFERENT symbol — must be ignored.
    market_upsert(
        migrated_engine,
        schema.tickers,
        [{"ticker_id": 1, "symbol": "ZZZ", "first_seen": dt.date(2024, 2, 2)}],
        ["ticker_id"],
        immutable_cols=["symbol", "first_seen"],
    )
    with migrated_engine.connect() as conn:
        sym = conn.execute(
            select(schema.tickers.c.symbol).where(schema.tickers.c.ticker_id == 1)
        ).scalar_one()
    assert sym == "AAA", "stable ticker_id must not be remapped to a new symbol"
    assert _count(migrated_engine, "tickers") == 1


def test_upsert_heterogeneous_key_batch_preserves_omitted_protection(migrated_engine):
    """A batch mixing row dicts with different key-sets must not drop a column
    a later row supplies, nor NULL-clobber a column an earlier row omitted.
    Rows are grouped by key-set so each group's update set is exactly its keys."""
    from rainier.db import schema
    from rainier.db.upsert import market_upsert

    # Seed id=1 WITH last_seen, id=2 WITHOUT.
    market_upsert(
        migrated_engine,
        schema.tickers,
        [
            {"ticker_id": 1, "symbol": "AAA", "first_seen": dt.date(2024, 1, 1),
             "last_seen": dt.date(2024, 6, 1)},
            {"ticker_id": 2, "symbol": "BBB", "first_seen": dt.date(2024, 1, 1)},
        ],
        ["ticker_id"],
        immutable_cols=["first_seen"],
    )
    # Re-upsert a MIXED batch: id=1 omits last_seen (must keep 2024-06-01),
    # id=2 now supplies last_seen (must be written), in one call.
    market_upsert(
        migrated_engine,
        schema.tickers,
        [
            {"ticker_id": 1, "symbol": "AAA", "first_seen": dt.date(2024, 1, 1)},
            {"ticker_id": 2, "symbol": "BBB", "first_seen": dt.date(2024, 1, 1),
             "last_seen": dt.date(2024, 7, 7)},
        ],
        ["ticker_id"],
        immutable_cols=["first_seen"],
    )
    with migrated_engine.connect() as conn:
        ls1 = conn.execute(
            select(schema.tickers.c.last_seen).where(schema.tickers.c.ticker_id == 1)
        ).scalar_one()
        ls2 = conn.execute(
            select(schema.tickers.c.last_seen).where(schema.tickers.c.ticker_id == 2)
        ).scalar_one()
    assert ls1 == dt.date(2024, 6, 1), "id=1 omitted last_seen must survive"
    assert ls2 == dt.date(2024, 7, 7), "id=2 supplied last_seen must be written"


def test_upsert_fk_child_after_parents(migrated_engine):
    """thematic_ohlcv has no FK, but features does: parents (tickers/sectors)
    upserted first, then the FK child inserts without violation."""
    from rainier.db import schema
    from rainier.db.upsert import market_upsert

    market_upsert(
        migrated_engine,
        schema.tickers,
        [{"ticker_id": 1, "symbol": "AAA", "first_seen": dt.date(2024, 1, 1)}],
        ["ticker_id"],
    )
    market_upsert(
        migrated_engine,
        schema.sectors,
        [{"sector_id": 1, "sector_name": "tech", "first_seen": dt.date(2024, 1, 1)}],
        ["sector_id"],
    )
    feature_row = {
        "asof_date": dt.date(2024, 11, 8),
        "trading_day_ordinal": 42,
        "symbol": "AAA",
        "ticker_id": 1,
        "sector_id": 1,
        "close": 100.0,
        "ret_1d": 0.01,
        "rel_5": None,
        "rel_10": None,
        "rel_20": None,
        "ret_ytd": None,
        "relvol_5": None,
        "relvol_10": None,
        "relvol_20": None,
        "r_5": None,
        "r_10": None,
        "r_20": None,
        "rank": 50,
        "rank_delta_1d": 0,
        "rank_delta_5d": 0,
        "top15_streak": 0,
        "rank_minus_sector_mean": None,
        "universe_size": 1,
        "universe_yaml_sha": "deadbeef",
        "computed_at": dt.datetime(2024, 11, 8, 16, 30, tzinfo=dt.timezone.utc),
    }
    market_upsert(
        migrated_engine,
        schema.thematic_features_daily,
        [feature_row],
        ["asof_date", "symbol"],
    )
    assert _count(migrated_engine, "thematic_features_daily") == 1
