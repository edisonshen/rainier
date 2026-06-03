"""Migration 0006 — realign `stock_prices` stock_id -> symbol (TASK-PLAN T1-T7).

Lane split (TASK-PLAN §Tests):

* Pure schema / idempotency / backfill checks run under `requires_postgres` on
  plain Postgres.
* Hypertable-catalog assertions (still-a-hypertable after the in-place ALTER)
  are gated behind `requires_timescaledb` — they skip with a clear reason when
  the test Postgres has no timescaledb extension (the ephemeral pytest-postgresql
  build is plain PG).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import inspect, text

REPO_ROOT = Path(__file__).resolve().parents[2]
UP = REPO_ROOT / "migrations" / "0006_stock_prices_symbol_key.sql"
DOWN = REPO_ROOT / "migrations" / "0006_stock_prices_symbol_key_downgrade.sql"

D = datetime(2026, 1, 5, tzinfo=timezone.utc)


def _apply(engine, path: Path) -> None:
    with engine.begin() as conn:
        conn.execute(text(path.read_text()))


def _seed_old_row(engine, *, stock_id: int, symbol: str, close: float) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO stocks (id, symbol) VALUES (:i, :s) ON CONFLICT DO NOTHING"),
            {"i": stock_id, "s": symbol},
        )
        conn.execute(
            text(
                "INSERT INTO stock_prices (stock_id, date, open, high, low, close, volume) "
                "VALUES (:i, :d, 100, 110, 95, :c, 1000)"
            ),
            {"i": stock_id, "d": D, "c": close},
        )


def _columns(engine) -> dict:
    insp = inspect(engine)
    return {c["name"]: c for c in insp.get_columns("stock_prices")}


def _constraint_names(engine, contype: str | None = None) -> set[str]:
    sql = (
        "SELECT con.conname FROM pg_constraint con "
        "JOIN pg_class rel ON rel.oid = con.conrelid "
        "WHERE rel.relname = 'stock_prices'"
    )
    if contype:
        sql += f" AND con.contype = '{contype}'"
    with engine.connect() as conn:
        return {r[0] for r in conn.execute(text(sql)).all()}


def _index_names(engine) -> set[str]:
    with engine.connect() as conn:
        return {
            r[0]
            for r in conn.execute(
                text("SELECT indexname FROM pg_indexes WHERE tablename = 'stock_prices'")
            ).all()
        }


def _pk_columns(engine) -> list[str]:
    return inspect(engine).get_pk_constraint("stock_prices")["constrained_columns"]


def _is_hypertable(engine) -> bool:
    with engine.connect() as conn:
        return bool(
            conn.execute(
                text(
                    "SELECT count(*) FROM timescaledb_information.hypertables "
                    "WHERE hypertable_name = 'stock_prices'"
                )
            ).scalar()
        )


def _unique_index_columns(engine) -> list[list[str]]:
    """Column lists of every UNIQUE index/constraint on stock_prices."""
    sql = """
        SELECT array_agg(a.attname ORDER BY k.ord) AS cols
          FROM pg_index ix
          JOIN pg_class t ON t.oid = ix.indrelid
          JOIN LATERAL unnest(ix.indkey) WITH ORDINALITY AS k(attnum, ord) ON TRUE
          JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum
         WHERE t.relname = 'stock_prices' AND ix.indisunique
         GROUP BY ix.indexrelid
    """
    with engine.connect() as conn:
        return [list(r[0]) for r in conn.execute(text(sql)).all()]


# ---------------------------------------------------------------------------
# T1 — migration applies + schema
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_t1_schema_after_up(pg_oldshape_engine):
    engine = pg_oldshape_engine
    _apply(engine, UP)

    cols = _columns(engine)
    # symbol varchar(10) NOT NULL
    assert "symbol" in cols
    assert cols["symbol"]["nullable"] is False
    assert "VARCHAR(10)" in str(cols["symbol"]["type"]).upper().replace(" ", "")
    # stock_id gone
    assert "stock_id" not in cols
    # date still timestamptz NOT NULL
    assert cols["date"]["nullable"] is False
    assert "TIMESTAMP" in str(cols["date"]["type"]).upper()
    # id default/sequence preserved (BIGINT autoincrement)
    assert "nextval" in (cols["id"].get("default") or "")

    # FK symbol -> stocks.symbol
    fks = inspect(engine).get_foreign_keys("stock_prices")
    sym_fk = [f for f in fks if f["constrained_columns"] == ["symbol"]]
    assert sym_fk and sym_fk[0]["referred_table"] == "stocks"
    assert sym_fk[0]["referred_columns"] == ["symbol"]
    # no stock_id FK
    assert not any(f["constrained_columns"] == ["stock_id"] for f in fks)

    # uq_stock_price_symbol_date present; uq_stock_price_date gone
    cons = _constraint_names(engine)
    assert "uq_stock_price_symbol_date" in cons
    assert "uq_stock_price_date" not in cons

    # index on symbol; stock_id index gone
    idxs = _index_names(engine)
    assert "ix_stock_prices_symbol" in idxs
    assert "ix_stock_prices_stock_id" not in idxs

    # PK still (id, date)
    assert _pk_columns(engine) == ["id", "date"]

    # Timescale rule — no unique index omits `date`.
    for col_list in _unique_index_columns(engine):
        assert "date" in col_list, f"unique index {col_list} omits the partition column"


@pytest.mark.requires_timescaledb
def test_t1_still_hypertable_after_up(pg_oldshape_engine):
    if not getattr(pg_oldshape_engine, "rainier_has_timescaledb", False):
        pytest.skip("no timescaledb extension")
    _apply(pg_oldshape_engine, UP)
    assert _is_hypertable(pg_oldshape_engine)


# ---------------------------------------------------------------------------
# T2 — data preservation
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_t2_data_preserved(pg_oldshape_engine):
    engine = pg_oldshape_engine
    _seed_old_row(engine, stock_id=42, symbol="NVDA", close=120.85)
    _apply(engine, UP)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT symbol, date, close, open, high, low, volume FROM stock_prices")
        ).one()
    assert row.symbol == "NVDA"
    assert row.close == 120.85
    assert row.open == 100 and row.high == 110 and row.low == 95 and row.volume == 1000
    # FK satisfied (NVDA exists in stocks) — proven by the apply not failing.


# ---------------------------------------------------------------------------
# T3 — idempotent (re-apply + already-migrated no-op)
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_t3_idempotent_double_apply(pg_oldshape_engine):
    engine = pg_oldshape_engine
    _seed_old_row(engine, stock_id=42, symbol="NVDA", close=120.85)
    _apply(engine, UP)
    # Second apply (stock_id already gone) is a clean no-op.
    _apply(engine, UP)
    cols = _columns(engine)
    assert "stock_id" not in cols and cols["symbol"]["nullable"] is False
    with engine.connect() as conn:
        assert conn.execute(text("SELECT symbol FROM stock_prices")).scalar() == "NVDA"


@pytest.mark.requires_postgres
def test_t3_noop_on_fresh_symbol_keyed(pg_legacy_engine):
    # pg_legacy_engine builds the ALREADY symbol-keyed stock_prices (the fresh
    # `db init` shape). Applying 0006 must be a clean no-op.
    _apply(pg_legacy_engine, UP)
    cols = _columns(pg_legacy_engine)
    assert "symbol" in cols and "stock_id" not in cols
    assert "uq_stock_price_symbol_date" in _constraint_names(pg_legacy_engine)


# ---------------------------------------------------------------------------
# T4 — downgrade: symmetric + data-preserving
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_t4_downgrade_symmetric(pg_oldshape_engine):
    engine = pg_oldshape_engine
    _apply(engine, UP)
    # Insert symbol-keyed rows post-migration.
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO stocks (id, symbol) VALUES (42, 'NVDA'), (7, 'TSLA')")
        )
        conn.execute(
            text(
                "INSERT INTO stock_prices (symbol, date, close) "
                "VALUES ('NVDA', :d1, 120.85), ('TSLA', :d2, 250.5)"
            ),
            {"d1": D, "d2": datetime(2026, 1, 6, tzinfo=timezone.utc)},
        )

    _apply(engine, DOWN)

    cols = _columns(engine)
    # stock_id restored + NOT NULL; symbol gone.
    assert "stock_id" in cols
    assert cols["stock_id"]["nullable"] is False
    assert "symbol" not in cols
    # uq_stock_price_date restored; uq_stock_price_symbol_date gone.
    cons = _constraint_names(engine)
    assert "uq_stock_price_date" in cons
    assert "uq_stock_price_symbol_date" not in cons
    # stock_id index restored; symbol index gone.
    idxs = _index_names(engine)
    assert "ix_stock_prices_stock_id" in idxs
    assert "ix_stock_prices_symbol" not in idxs
    # stock_id FK restored.
    fks = inspect(engine).get_foreign_keys("stock_prices")
    assert any(
        f["constrained_columns"] == ["stock_id"] and f["referred_table"] == "stocks"
        for f in fks
    )
    # PK still (id, date).
    assert _pk_columns(engine) == ["id", "date"]
    # Rows survive with correct stock_id backfilled from symbol.
    with engine.connect() as conn:
        got = {
            (r.stock_id, float(r.close))
            for r in conn.execute(text("SELECT stock_id, close FROM stock_prices")).all()
        }
    assert got == {(42, 120.85), (7, 250.5)}


@pytest.mark.requires_timescaledb
def test_t4_downgrade_preserves_hypertable(pg_oldshape_engine):
    if not getattr(pg_oldshape_engine, "rainier_has_timescaledb", False):
        pytest.skip("no timescaledb extension")
    _apply(pg_oldshape_engine, UP)
    _apply(pg_oldshape_engine, DOWN)
    assert _is_hypertable(pg_oldshape_engine)


# ---------------------------------------------------------------------------
# T5 — fresh db init
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_t5_fresh_db_init_symbol_keyed(pg_empty_engine):
    from rainier.core.database import init_db

    init_db()
    cols = _columns(pg_empty_engine)
    assert "symbol" in cols and "stock_id" not in cols
    fks = inspect(pg_empty_engine).get_foreign_keys("stock_prices")
    assert any(
        f["constrained_columns"] == ["symbol"] and f["referred_columns"] == ["symbol"]
        for f in fks
    )
    assert "uq_stock_price_symbol_date" in _constraint_names(pg_empty_engine)


@pytest.mark.requires_timescaledb
def test_t5_fresh_db_init_hypertable(pg_empty_engine):
    if not getattr(pg_empty_engine, "rainier_has_timescaledb", False):
        pytest.skip("no timescaledb extension")
    from rainier.core.database import init_db

    init_db()
    assert _is_hypertable(pg_empty_engine)


# ---------------------------------------------------------------------------
# T6 — ingest smoke (regression for `column stock_prices.symbol does not exist`)
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_t6_ingest_smoke(pg_legacy_session):
    """With the symbol-keyed table, ingest_prices upserts a (symbol, date) row
    and a re-run is a no-op. yfinance mocked (fetch_fn injected, no network)."""
    from datetime import date

    from rainier.core.models import StockPrice
    from rainier.paper.ingest import ingest_prices

    as_of = date(2026, 1, 9)  # Friday
    target = date(2026, 1, 9)
    bar = {
        "date": target,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": 1000,
    }

    def fetch_fn(syms, start, end):
        return {s: [bar] for s in syms}

    res = ingest_prices(["AAA"], as_of=as_of, fetch_fn=fetch_fn, window_days=5)
    assert res["upserted"] >= 1
    rows = (
        pg_legacy_session.query(StockPrice)
        .filter(StockPrice.symbol == "AAA")
        .all()
    )
    assert len(rows) == 1 and rows[0].close == 10.5

    # Re-run with the same single bar: idempotent on the (symbol, date) key — no
    # NEW row is created (the design re-confirms in-window bars via upsert, so
    # `upserted` may re-count the existing pair; the invariant is "no new rows").
    ingest_prices(["AAA"], as_of=as_of, fetch_fn=fetch_fn, window_days=5)
    rows2 = (
        pg_legacy_session.query(StockPrice)
        .filter(StockPrice.symbol == "AAA")
        .all()
    )
    assert len(rows2) == 1 and rows2[0].close == 10.5


# ---------------------------------------------------------------------------
# T7 — no stock_id readers (static)
# ---------------------------------------------------------------------------


def test_t7_no_stock_id_readers():
    """No src/ code references stock_prices.stock_id / StockPrice.stock_id."""
    import re

    src = REPO_ROOT / "src"
    offenders: list[str] = []
    # `StockPrice.stock_id` (ORM attr) or `stock_prices.stock_id` (raw SQL).
    pat = re.compile(r"StockPrice\.stock_id|stock_prices\.stock_id")
    for py in src.rglob("*.py"):
        text_ = py.read_text()
        for i, line in enumerate(text_.splitlines(), 1):
            if pat.search(line):
                offenders.append(f"{py.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
    assert not offenders, "stale stock_id readers:\n" + "\n".join(offenders)
