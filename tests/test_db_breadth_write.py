"""Dual-write + backfill + verify tests for the breadth PG write path.

breadth-pg-write-path-6c54 (one PR): market.breadth_indicator_daily (LONG) +
market.benchmark_ohlcv. Mirrors the Phase 2/3 thematic dual-write tests
(tests/test_db_dual_write.py): each writer driven through its real CLI entry
point against a temp Postgres migrated to head.

Asserted:
  * LONG dual-write: parquet (asof_date, indicator, value) -> identical PG rows,
    NO transform (frame_to_pg_rows projects the three columns directly).
  * UPSERT idempotency: re-running compute updates value, no duplicate
    (asof_date, indicator).
  * FULL-HISTORY upsert (load-bearing): a late OHLCV correction that changes
    cumulative-indicator rows for many past dates is reflected for ALL affected
    dates in PG, not just the latest.
  * SPY backfill-spy -> benchmark_ohlcv with fetched_at/yfinance_version
    immutable on conflict.
  * mirror_guard non-fatal: DATABASE_URL unset -> parquet still writes, exit 0.
  * Backfill parity (requires_postgres): counts + checksums match parquet for
    both tables.
  * verify-coverage reports the two new tables.

PG-backed tests gated on ``requires_postgres``; skip-path test runs always.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Synthetic frame builders
# ---------------------------------------------------------------------------


def _build_breadth_long(asof_dates: list[date], indicators: list[str]) -> pd.DataFrame:
    """Build a LONG breadth frame matching sp500_breadth_daily.parquet shape.

    (asof_date, indicator, value) with value as float64. Deterministic values
    so the parquet<->PG checksum is meaningful.
    """
    rows = []
    for di, d in enumerate(asof_dates):
        for ii, ind in enumerate(indicators):
            rows.append(
                {
                    "asof_date": d,
                    "indicator": ind,
                    "value": float(di * 100 + ii) + 0.5,
                }
            )
    df = pd.DataFrame(rows)
    df["value"] = df["value"].astype("float64")
    return df.sort_values(["asof_date", "indicator"]).reset_index(drop=True)


def _build_spy_history(asof_dates: list[date]) -> pd.DataFrame:
    """Build a SPY OHLCV frame matching spy_history.parquet / benchmark_ohlcv."""
    rows = []
    close = 400.0
    for i, d in enumerate(asof_dates):
        close = close * (1.0 + 0.001 * (i % 5 - 2))
        rows.append(
            {
                "symbol": "SPY",
                "date": d,
                "open": close,
                "high": close * 1.004,
                "low": close * 0.996,
                "close": close,
                "volume": 70_000_000 + i,
                "fetched_at": pd.Timestamp("2024-11-08T16:30:00Z"),
                "yfinance_version": "1.2.0",
            }
        )
    return pd.DataFrame(rows)


def _trading_days(start: date, n: int) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d = d + timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# Postgres fixtures (mirror tests/test_db_dual_write.py resolution)
# ---------------------------------------------------------------------------


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
        with eng.begin() as conn:
            conn.exec_driver_sql("DROP SCHEMA IF EXISTS market CASCADE")
            conn.exec_driver_sql("DROP TABLE IF EXISTS public.alembic_version")
        eng.dispose()


def _count(eng, table) -> int:
    with eng.connect() as conn:
        return conn.execute(text(f"SELECT count(*) FROM market.{table}")).scalar_one()


# ---------------------------------------------------------------------------
# LONG dual-write: parquet -> identical PG rows (no transform)
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_breadth_dual_write_long_identity(migrated_engine, database_url):
    """_dual_write_breadth_pg mirrors the LONG frame 1:1 — same (asof_date,
    indicator, value) rows in PG, no pivot/transform."""
    from rainier.cli import _dual_write_breadth_pg

    days = _trading_days(date(2024, 1, 2), 4)
    inds = ["ad_cumulative", "pct_above_ma_50", "new_52w_high"]
    df = _build_breadth_long(days, inds)

    _dual_write_breadth_pg(df)

    assert _count(migrated_engine, "breadth_indicator_daily") == len(df)
    with migrated_engine.connect() as conn:
        got = pd.DataFrame(
            conn.execute(
                text(
                    "SELECT asof_date, indicator, value FROM "
                    "market.breadth_indicator_daily ORDER BY asof_date, indicator"
                )
            ).mappings().all()
        )
    # Same shape, same values (no transform layer changed anything).
    assert list(got["indicator"]) == list(df["indicator"])
    for a, b in zip(got["value"], df["value"], strict=True):
        assert a == pytest.approx(b)


@pytest.mark.requires_postgres
def test_breadth_dual_write_idempotent_updates_value(migrated_engine, database_url):
    """Re-running with a changed value UPSERTs in place — no duplicate
    (asof_date, indicator)."""
    from rainier.cli import _dual_write_breadth_pg

    days = _trading_days(date(2024, 1, 2), 3)
    inds = ["ad_cumulative", "pct_above_ma_50"]
    df = _build_breadth_long(days, inds)
    _dual_write_breadth_pg(df)
    n = len(df)
    assert _count(migrated_engine, "breadth_indicator_daily") == n

    # Bump every value; same keys -> UPDATE, count unchanged.
    df2 = df.copy()
    df2["value"] = df2["value"] + 1000.0
    _dual_write_breadth_pg(df2)
    assert _count(migrated_engine, "breadth_indicator_daily") == n

    expected = df2.loc[
        (df2["asof_date"] == days[0]) & (df2["indicator"] == "ad_cumulative"),
        "value",
    ].iloc[0]
    with migrated_engine.connect() as conn:
        v = conn.execute(
            text(
                "SELECT value FROM market.breadth_indicator_daily "
                "WHERE asof_date=:d AND indicator='ad_cumulative'"
            ),
            {"d": days[0]},
        ).scalar_one()
    assert v == pytest.approx(expected)


@pytest.mark.requires_postgres
def test_breadth_full_history_upsert_late_correction(migrated_engine, database_url):
    """LOAD-BEARING: a late OHLCV correction rewrites cumulative-indicator rows
    for many PAST dates. Writing the FULL recomputed history each compute must
    update ALL affected dates in PG, not just the latest.

    We simulate: first compute writes history h1; a correction recomputes
    cumulative values that differ on EVERY date >= some early date; the second
    full-history dual-write must reflect the new value on those early dates too.
    """
    from rainier.cli import _dual_write_breadth_pg

    days = _trading_days(date(2024, 1, 2), 6)
    inds = ["ad_cumulative", "mcclellan_summation"]
    h1 = _build_breadth_long(days, inds)
    _dual_write_breadth_pg(h1)

    # Late correction: cumulative indicators shift on every date from index 1
    # onward (a corrected early bar propagates forward through the cumsum).
    h2 = h1.copy()
    shifted = h2["asof_date"] >= days[1]
    h2.loc[shifted, "value"] = h2.loc[shifted, "value"] + 7.0
    _dual_write_breadth_pg(h2)

    # PG must now match h2 on EVERY affected date, including the early ones.
    with migrated_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT asof_date, indicator, value FROM "
                "market.breadth_indicator_daily ORDER BY asof_date, indicator"
            )
        ).mappings().all()
    pg = {(r["asof_date"], r["indicator"]): r["value"] for r in rows}
    for _, r in h2.iterrows():
        assert pg[(r["asof_date"], r["indicator"])] == pytest.approx(r["value"]), (
            f"stale row at {r['asof_date']}/{r['indicator']} — full-history "
            "upsert did not update a past date"
        )
    # And specifically: an EARLY date (index 1) reflects the +7 correction.
    early = days[1]
    assert pg[(early, "ad_cumulative")] == pytest.approx(
        h1.loc[(h1["asof_date"] == early) & (h1["indicator"] == "ad_cumulative"), "value"].iloc[0] + 7.0
    )


# ---------------------------------------------------------------------------
# SPY -> benchmark_ohlcv
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_benchmark_dual_write_ohlcv(migrated_engine, database_url):
    """_dual_write_benchmark_pg mirrors the SPY frame into benchmark_ohlcv."""
    from rainier.cli import _dual_write_benchmark_pg

    days = _trading_days(date(2024, 1, 2), 5)
    df = _build_spy_history(days)
    _dual_write_benchmark_pg(df)

    assert _count(migrated_engine, "benchmark_ohlcv") == len(df)
    with migrated_engine.connect() as conn:
        close = conn.execute(
            text(
                "SELECT close FROM market.benchmark_ohlcv "
                "WHERE symbol='SPY' AND date=:d"
            ),
            {"d": days[2]},
        ).scalar_one()
    assert close == pytest.approx(df.iloc[2]["close"])


@pytest.mark.requires_postgres
def test_benchmark_provenance_mutable_for_parquet_parity(migrated_engine, database_url):
    """fetched_at / yfinance_version are MUTABLE on conflict (codex iter-1 P1).

    The parquet ``ohlcv_backfill._upsert`` is latest-write-wins, so an
    incremental backfill-spy run that overlaps existing dates re-stamps those
    rows' provenance in the parquet. PG is a byte-for-parity mirror checked by
    verify-coverage (design D-5), so it must adopt the SAME new provenance —
    pinning these columns immutable would leave PG holding stale metadata while
    the parquet moved on, failing the checksum. Mirrors thematic_ohlcv (which
    also keeps provenance mutable)."""
    from rainier.cli import _dual_write_benchmark_pg

    days = _trading_days(date(2024, 1, 2), 3)
    df = _build_spy_history(days)
    _dual_write_benchmark_pg(df)

    df2 = df.copy()
    df2["close"] = df2["close"] + 5.0  # price corrections update
    df2["fetched_at"] = pd.Timestamp("2025-01-01T00:00:00Z")  # provenance updates too
    df2["yfinance_version"] = "9.9.9"
    _dual_write_benchmark_pg(df2)

    with migrated_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT close, yfinance_version FROM market.benchmark_ohlcv "
                "WHERE symbol='SPY' AND date=:d"
            ),
            {"d": days[0]},
        ).mappings().one()
    assert row["close"] == pytest.approx(df2.iloc[0]["close"]), "price should update"
    assert row["yfinance_version"] == "9.9.9", (
        "provenance must update to match the latest-write-wins parquet so PG "
        "stays byte-for-parity under verify-coverage"
    )


# ---------------------------------------------------------------------------
# CLI integration: compute-indicators + backfill-spy dual-write
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_compute_indicators_cli_dual_writes(migrated_engine, tmp_path, database_url):
    """`market-breadth compute-indicators` writes parquet AND mirrors the full
    breadth history into market.breadth_indicator_daily."""
    from rainier.cli import cli

    # Minimal OHLCV universe so compute produces a small breadth parquet.
    days = _trading_days(date(2024, 1, 2), 30)
    ohlcv_rows = []
    for sym in ("AAA", "BBB", "CCC"):
        close = 100.0
        for i, d in enumerate(days):
            close = close * (1.0 + 0.002 * (i % 3 - 1))
            ohlcv_rows.append(
                {
                    "symbol": sym, "date": d, "open": close, "high": close * 1.01,
                    "low": close * 0.99, "close": close, "volume": 1_000_000,
                }
            )
    universe = pd.DataFrame(ohlcv_rows)
    cache = tmp_path / "cache"
    cache.mkdir()
    uni_path = cache / "sp500_universe.parquet"
    universe.to_parquet(uni_path)
    out_path = cache / "sp500_breadth_daily.parquet"

    res = CliRunner().invoke(
        cli,
        [
            "market-breadth", "compute-indicators",
            "--input", str(uni_path), "--output", str(out_path),
            "--epoch", "2024-01-01",
        ],
    )
    assert res.exit_code == 0, res.output
    breadth = pd.read_parquet(out_path)
    assert _count(migrated_engine, "breadth_indicator_daily") == len(breadth)


@pytest.mark.requires_postgres
def test_backfill_spy_cli_dual_writes(migrated_engine, tmp_path, database_url):
    """`market-breadth backfill-spy` writes parquet AND mirrors into
    benchmark_ohlcv."""
    from rainier.cli import cli

    days = _trading_days(date(2024, 1, 2), 6)

    def fake_fetch(syms, start, end):
        out = {}
        for s in syms:
            out[s] = pd.DataFrame(
                {
                    "date": days,
                    "open": [400.0 + i for i in range(len(days))],
                    "high": [401.0 + i for i in range(len(days))],
                    "low": [399.0 + i for i in range(len(days))],
                    "close": [400.5 + i for i in range(len(days))],
                    "volume": [70_000_000 + i for i in range(len(days))],
                }
            )
        return out

    cache = tmp_path / "cache"
    cache.mkdir()
    out_path = cache / "spy_history.parquet"

    # Inject the fake fetcher via monkeypatch on ohlcv_backfill.backfill's
    # default fetch_fn path: the CLI calls backfill(symbols=["SPY"], ...). We
    # patch the module-level network fetcher.
    import rainier.market_breadth.ohlcv_backfill as obf

    orig = obf._yfinance_fetch
    obf._yfinance_fetch = fake_fetch
    try:
        res = CliRunner().invoke(
            cli,
            [
                "market-breadth", "backfill-spy",
                "--since", "2024-01-01", "--output", str(out_path),
            ],
        )
    finally:
        obf._yfinance_fetch = orig

    assert res.exit_code == 0, res.output
    spy = pd.read_parquet(out_path)
    assert _count(migrated_engine, "benchmark_ohlcv") == len(spy)


@pytest.mark.requires_postgres
def test_backfill_spy_incremental_cli_idempotent(migrated_engine, tmp_path, database_url):
    """REGRESSION (benchmark-ohlcv-spy-stale): the daily cron runs
    `market-breadth backfill-spy --incremental`, which is the ONLY writer of
    market.benchmark_ohlcv. Re-running it (a missed-cron catch-up, or two runs in
    a day) must UPSERT on (symbol, date) — no duplicate rows. This guards the
    table fengshen's /trading SPY price pane reads against silent dupes when the
    scheduled job re-fires.
    """
    from rainier.cli import cli

    days = _trading_days(date(2024, 1, 2), 6)

    def fake_fetch(syms, start, end):
        return {
            s: pd.DataFrame(
                {
                    "date": days,
                    "open": [400.0 + i for i in range(len(days))],
                    "high": [401.0 + i for i in range(len(days))],
                    "low": [399.0 + i for i in range(len(days))],
                    "close": [400.5 + i for i in range(len(days))],
                    "volume": [70_000_000 + i for i in range(len(days))],
                }
            )
            for s in syms
        }

    cache = tmp_path / "cache"
    cache.mkdir()
    out_path = cache / "spy_history.parquet"

    import rainier.market_breadth.ohlcv_backfill as obf

    orig = obf._yfinance_fetch
    obf._yfinance_fetch = fake_fetch
    try:
        for _ in range(2):  # run the incremental cron command twice
            res = CliRunner().invoke(
                cli,
                [
                    "market-breadth", "backfill-spy",
                    "--incremental", "--output", str(out_path),
                ],
            )
            assert res.exit_code == 0, res.output
    finally:
        obf._yfinance_fetch = orig

    spy = pd.read_parquet(out_path)
    # No duplicate (symbol, date) in PG, and PG matches the (deduped) parquet.
    assert _count(migrated_engine, "benchmark_ohlcv") == len(spy)
    with migrated_engine.connect() as conn:
        dupes = conn.execute(
            text(
                "SELECT count(*) FROM (SELECT symbol, date FROM market.benchmark_ohlcv "
                "GROUP BY symbol, date HAVING count(*) > 1) d"
            )
        ).scalar_one()
    assert dupes == 0, "re-running incremental backfill-spy created duplicate rows"


# ---------------------------------------------------------------------------
# One-shot backfill parity + verify-coverage (both new tables)
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_backfill_from_parquet_seeds_both_tables(migrated_engine, tmp_path, database_url):
    """db backfill-from-parquet seeds breadth_indicator_daily + benchmark_ohlcv
    from their parquet caches, and verify-coverage reports parity for both."""
    from rainier.db.backfill import backfill_from_parquet
    from rainier.db.verify import verify_coverage

    cache = tmp_path / "cache"
    cache.mkdir()

    days = _trading_days(date(2024, 1, 2), 5)
    breadth = _build_breadth_long(days, ["ad_cumulative", "pct_above_ma_50"])
    breadth.to_parquet(cache / "sp500_breadth_daily.parquet")
    spy = _build_spy_history(days)
    spy.to_parquet(cache / "spy_history.parquet")

    counts = backfill_from_parquet(migrated_engine, cache)
    assert counts["breadth_indicator_daily"] == len(breadth)
    assert counts["benchmark_ohlcv"] == len(spy)

    report = verify_coverage(migrated_engine, cache)
    drift_tables = {d.table for d in report.drift}
    assert "breadth_indicator_daily" not in drift_tables, report.drift
    assert "benchmark_ohlcv" not in drift_tables, report.drift
    assert report.ok, report.drift


@pytest.mark.requires_postgres
def test_benchmark_incremental_overlap_keeps_verify_parity(
    migrated_engine, tmp_path, database_url
):
    """REGRESSION (codex iter-1 P1): an incremental backfill-spy that overlaps
    existing dates re-stamps fetched_at/yfinance_version in the parquet
    (latest-write-wins). The PG mirror MUST adopt the same provenance so
    verify-coverage stays clean — if benchmark provenance were pinned immutable,
    PG would keep stale metadata and the checksum would drift.

    Simulates the production sequence: parquet on disk -> dual-write -> a later
    overlapping run rewrites the same dates' provenance in BOTH parquet and PG.
    """
    from rainier.cli import _dual_write_benchmark_pg
    from rainier.db.verify import verify_coverage

    cache = tmp_path / "cache"
    cache.mkdir()
    spy_path = cache / "spy_history.parquet"

    days = _trading_days(date(2024, 1, 2), 5)
    spy = _build_spy_history(days)
    spy.to_parquet(spy_path)
    _dual_write_benchmark_pg(spy)
    assert verify_coverage(migrated_engine, cache).ok

    # Overlapping incremental refresh: same dates, new fetch provenance (and a
    # tiny price correction). Parquet is latest-write-wins, so on disk these
    # dates now carry the NEW fetched_at/version.
    spy2 = spy.copy()
    spy2["close"] = spy2["close"] + 0.25
    spy2["fetched_at"] = pd.Timestamp("2025-03-01T16:30:00Z")
    spy2["yfinance_version"] = "9.9.9"
    spy2.to_parquet(spy_path)
    _dual_write_benchmark_pg(spy2)

    report = verify_coverage(migrated_engine, cache)
    assert report.ok, (
        f"benchmark_ohlcv drifted from parquet after an overlapping incremental "
        f"refresh — provenance must be mutable: {report.drift}"
    )


@pytest.mark.requires_postgres
def test_backfill_from_parquet_breadth_idempotent(migrated_engine, tmp_path, database_url):
    """Re-running backfill is idempotent for the LONG breadth table."""
    from rainier.db.backfill import backfill_from_parquet

    cache = tmp_path / "cache"
    cache.mkdir()
    days = _trading_days(date(2024, 1, 2), 4)
    breadth = _build_breadth_long(days, ["ad_cumulative"])
    breadth.to_parquet(cache / "sp500_breadth_daily.parquet")

    backfill_from_parquet(migrated_engine, cache)
    backfill_from_parquet(migrated_engine, cache)
    assert _count(migrated_engine, "breadth_indicator_daily") == len(breadth)


# ---------------------------------------------------------------------------
# mirror_guard non-fatal — NO Postgres needed
# ---------------------------------------------------------------------------


def test_breadth_dual_write_noop_without_database_url(monkeypatch):
    """DATABASE_URL unset -> _dual_write_breadth_pg warns + skips PG, no raise."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from rainier.cli import _dual_write_breadth_pg

    days = _trading_days(date(2024, 1, 2), 2)
    df = _build_breadth_long(days, ["ad_cumulative"])
    # Must not raise — the parquet pipeline is load-bearing, PG is a mirror.
    _dual_write_breadth_pg(df)


def test_benchmark_dual_write_noop_without_database_url(monkeypatch):
    """DATABASE_URL unset -> _dual_write_benchmark_pg warns + skips PG, no raise."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from rainier.cli import _dual_write_benchmark_pg

    days = _trading_days(date(2024, 1, 2), 2)
    df = _build_spy_history(days)
    _dual_write_benchmark_pg(df)


def test_breadth_dual_write_empty_frame_noop(monkeypatch):
    """An empty breadth frame is a no-op (no PG touch even if URL set)."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://nouser@127.0.0.1:1/nodb")
    from rainier.cli import _dual_write_breadth_pg

    _dual_write_breadth_pg(pd.DataFrame(columns=["asof_date", "indicator", "value"]))
