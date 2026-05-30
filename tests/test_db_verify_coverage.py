"""Integration tests for Phase 3 ``rainier db verify-coverage`` (task plan §5).

verify-coverage proves parquet and Postgres agree per ``(asof_date, table)`` —
row count + an order-independent content checksum. We assert:

  * after a clean backfill the report is all-match and exits 0;
  * an injected missing row -> nonzero exit naming the offending (date, table);
  * an injected checksum drift (mutated value) -> nonzero exit naming it;
  * float round-trip parity does NOT false-positive (parquet float64 vs PG
    DOUBLE_PRECISION/REAL is within the documented rounding tolerance).

PG-backed tests gated on ``requires_postgres``; the URL-unset CLI test runs
always. Fixtures + the synthetic cache builder are shared with
tests/test_db_backfill.py.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine, text

from tests.test_db_backfill import _write_cache

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


# ---------------------------------------------------------------------------
# verify_coverage (module API): clean backfill -> all-match
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_verify_clean_backfill_all_match(migrated_engine, tmp_path, database_url):
    from rainier.db.backfill import backfill_from_parquet
    from rainier.db.verify import verify_coverage

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA", "BBB", "CCC"], n_days=10)
    backfill_from_parquet(migrated_engine, cache)

    report = verify_coverage(migrated_engine, cache)
    assert report.ok, f"expected all-match, drift: {report.drift}"
    assert report.drift == []


@pytest.mark.requires_postgres
def test_verify_detects_missing_row(migrated_engine, tmp_path, database_url):
    """Delete a PG row -> verify reports drift naming the (date, table)."""
    from rainier.db.backfill import backfill_from_parquet
    from rainier.db.verify import verify_coverage

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA", "BBB"], n_days=8)
    backfill_from_parquet(migrated_engine, cache)

    # Pick a known asof_date and drop one feature row for it.
    with migrated_engine.begin() as conn:
        victim = conn.execute(
            text(
                "SELECT asof_date FROM market.thematic_features_daily "
                "ORDER BY asof_date LIMIT 1"
            )
        ).scalar_one()
        conn.execute(
            text(
                "DELETE FROM market.thematic_features_daily "
                "WHERE asof_date=:d AND symbol='AAA'"
            ),
            {"d": victim},
        )

    report = verify_coverage(migrated_engine, cache)
    assert not report.ok
    offenders = {(d.table, d.asof_date) for d in report.drift}
    assert ("thematic_features_daily", victim) in offenders, report.drift


@pytest.mark.requires_postgres
def test_verify_detects_checksum_drift(migrated_engine, tmp_path, database_url):
    """Mutate a PG value (same row count) -> checksum mismatch is caught."""
    from rainier.db.backfill import backfill_from_parquet
    from rainier.db.verify import verify_coverage

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA", "BBB"], n_days=8)
    backfill_from_parquet(migrated_engine, cache)

    with migrated_engine.begin() as conn:
        victim = conn.execute(
            text(
                "SELECT asof_date FROM market.thematic_labels_daily "
                "ORDER BY asof_date LIMIT 1"
            )
        ).scalar_one()
        # Change a value without changing the row count.
        conn.execute(
            text(
                "UPDATE market.thematic_labels_daily SET fwd_3d_ret = 99.0 "
                "WHERE asof_date=:d AND symbol='AAA'"
            ),
            {"d": victim},
        )

    report = verify_coverage(migrated_engine, cache)
    assert not report.ok
    offenders = {(d.table, d.asof_date) for d in report.drift}
    assert ("thematic_labels_daily", victim) in offenders, report.drift


@pytest.mark.requires_postgres
def test_verify_float_roundtrip_no_false_positive(migrated_engine, tmp_path, database_url):
    """Parquet float64 vs PG REAL/DOUBLE_PRECISION round-trip must not trip the
    checksum. A clean backfill of float-heavy feature/label tables verifies
    clean — the tolerance absorbs the storage-precision delta."""
    from rainier.db.backfill import backfill_from_parquet
    from rainier.db.verify import verify_coverage

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA", "BBB", "CCC", "DDD"], n_days=15)
    backfill_from_parquet(migrated_engine, cache)

    report = verify_coverage(migrated_engine, cache)
    feat_drift = [d for d in report.drift if d.table == "thematic_features_daily"]
    label_drift = [d for d in report.drift if d.table == "thematic_labels_daily"]
    assert feat_drift == [], f"float features falsely flagged: {feat_drift}"
    assert label_drift == [], f"float labels falsely flagged: {label_drift}"
    assert report.ok


@pytest.mark.requires_postgres
def test_verify_real_column_float32_boundary_no_false_positive(
    migrated_engine, tmp_path, database_url
):
    """Regression (real-data smoke): a parquet float64 in a PG REAL column whose
    float32 round-trip straddles a significant-figure rounding boundary must NOT
    be flagged. Earlier sig-fig rounding gave parquet 0.07677094638... -> 0.0767709
    but PG's float32 0.07677095 -> 0.076771, a false positive. Casting both sides
    through float32 makes them bit-identical. Also exercises NaN -> NULL parity
    (recent asof dates have incomplete forward returns)."""
    import math

    import numpy as np
    import pandas as pd
    from sqlalchemy import text

    from rainier.db import schema
    from rainier.db.rows import frame_to_pg_rows
    from rainier.db.upsert import market_upsert
    from rainier.db.verify import verify_coverage

    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    asof = date(2026, 5, 8)
    # Values picked from the real-data smoke that broke sig-fig rounding, plus a
    # NaN forward-return row (incomplete horizon).
    boundary = [
        0.07677094638347626,
        0.007865045219659805,
        0.0001843344944063574,
        0.022905850782990456,
        -0.03227955102920532,
    ]
    rows = []
    for i, sym in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
        v = boundary[i]
        rows.append(
            {
                "asof_date": asof, "symbol": sym,
                "fwd_3d_ret": v, "fwd_5d_ret": v * 1.5, "fwd_10d_ret": v * 2.0,
                "fwd_20d_ret": float("nan"), "fwd_30d_ret": float("nan"),
                "fwd_5d_excess_ret": v * 0.5, "fwd_10d_excess_ret": v * 0.25,
                "fwd_20d_excess_ret": float("nan"),
                "fwd_10d_max_drawdown": abs(v), "fwd_10d_max_runup": abs(v) * 0.3,
                "label_complete_through": None,
            }
        )
    label_df = pd.DataFrame(rows)
    label_df.to_parquet(cache / "thematic_labels_daily.parquet")

    # Backfill ONLY labels (write directly; no FK on labels).
    label_cols = list(schema.thematic_labels_daily.columns.keys())
    market_upsert(
        migrated_engine, schema.thematic_labels_daily,
        frame_to_pg_rows(label_df, label_cols), ["asof_date", "symbol"],
    )

    # Sanity: the parquet float64 and the PG REAL round-trip really do differ in
    # the low bits (so the test would catch a naive repr/hash regression).
    with migrated_engine.connect() as conn:
        pg_v = conn.execute(
            text(
                "SELECT fwd_3d_ret FROM market.thematic_labels_daily "
                "WHERE asof_date=:a AND symbol='AAA'"
            ),
            {"a": asof},
        ).scalar_one()
    assert pg_v != boundary[0], "PG REAL must lose precision vs parquet float64"
    assert math.isclose(float(np.float32(boundary[0])), float(np.float32(pg_v)))

    report = verify_coverage(migrated_engine, cache)
    drift = [d for d in report.drift if d.table == "thematic_labels_daily"]
    assert drift == [], f"float32-boundary values falsely flagged drift: {drift}"
    assert report.ok


# ---------------------------------------------------------------------------
# Float-width + NaN/NULL parity (live-Neon labels artifact, verify#8aab)
# ---------------------------------------------------------------------------
#
# Live verify against Neon flagged ~100 spurious checksum mismatches on
# market.thematic_labels_daily while every other table was TRUE parity. Root
# cause: the parquet stores forward-return columns as float32 while the live PG
# column is DOUBLE PRECISION (the live DDL diverged from the static REAL schema).
# Keying float32-casting off the STATIC schema's REAL membership missed it.
#
# The fix derives the float32 column set from the PARQUET dtypes (the
# source-of-precision) and applies the float32 round-trip ONLY to those columns
# — on both sides. A float64-stored column (OHLC prices, breadth value) keeps
# FULL float64 precision, so genuine sub-float32 drift in a DOUBLE column is
# still caught (codex P1: an unconditional float32 cast would blind the gate to
# real drift in DOUBLE columns). NaN<->NULL collapses for every column.


def _float_truncated_by_f32() -> tuple[float, float]:
    """A float64 whose float32 cast changes its bit value, plus that cast.

    Returns ``(orig64, parquet_f32)`` where ``orig64`` is what a DOUBLE PRECISION
    PG column would hold from a full-precision load, and ``parquet_f32`` is the
    same number after a float32 parquet round-trip. They differ in float64 bits;
    a float32-canonicalizing checksum must collapse them to equal.
    """
    import numpy as np

    orig64 = 0.06948674738744653
    parquet_f32 = float(np.float32(orig64))
    assert orig64 != parquet_f32, "value must actually lose bits under float32"
    return orig64, parquet_f32


def test_checksum_float32_column_no_false_positive():
    """A column the parquet stores as float32 must hash equal across the float32
    parquet value and its widened-to-float64 PG mirror (REAL *or* a
    schema-diverged DOUBLE). The column is named in ``float32_cols`` so both
    sides take the float32 round-trip and collapse to the same 32-bit value."""
    from rainier.db.verify import _checksum

    orig64, parquet_f32 = _float_truncated_by_f32()
    cols = ["symbol", "fwd_20d_ret"]
    pk = ("symbol",)
    f32_cols = frozenset({"fwd_20d_ret"})
    parquet_rows = [{"symbol": "AAA", "fwd_20d_ret": parquet_f32}]
    pg_rows = [{"symbol": "AAA", "fwd_20d_ret": orig64}]
    assert _checksum(parquet_rows, cols, pk, f32_cols) == _checksum(
        pg_rows, cols, pk, f32_cols
    ), "float32-stored column: parquet f32 vs widened PG f64 must hash equal"


def test_checksum_double_column_catches_subfloat32_drift():
    """codex P1 regression: a float64-stored (DOUBLE) column must STILL catch
    real drift below a float32 ULP. ``100.00000001`` vs ``100.00000002`` are
    distinct float64s that collapse to the same float32 bucket — an
    unconditional float32 cast would silently pass corrupted data. Because the
    column is NOT in ``float32_cols`` it is hashed at full float64 precision, so
    the two values mismatch (drift detected)."""
    import numpy as np

    from rainier.db.verify import _checksum

    cols = ["symbol", "close"]
    pk = ("symbol",)
    a64, b64 = 100.00000001, 100.00000002
    # Precondition: these collapse to one float32 bucket (the blinding hazard).
    assert np.float32(a64) == np.float32(b64), "test premise: same float32 bucket"
    assert a64 != b64, "but distinct as float64"
    # close is a DOUBLE-stored column -> NOT in float32_cols -> full precision.
    a = [{"symbol": "AAA", "close": a64}]
    b = [{"symbol": "AAA", "close": b64}]
    assert _checksum(a, cols, pk, frozenset()) != _checksum(
        b, cols, pk, frozenset()
    ), "DOUBLE column drift below float32 ULP must STILL be detected"


def test_checksum_nan_equals_sql_null():
    """A parquet float NaN and a SQL NULL (None) for the same cell must hash
    equal — for every column, float32 or not. Recent asof dates legitimately
    carry NaN forward returns (window not elapsed) written as NULL in PG; that
    representation difference must not be flagged as drift."""
    from rainier.db.verify import _checksum

    cols = ["symbol", "fwd_20d_ret"]
    pk = ("symbol",)
    f32_cols = frozenset({"fwd_20d_ret"})
    nan_rows = [{"symbol": "AAA", "fwd_20d_ret": float("nan")}]
    null_rows = [{"symbol": "AAA", "fwd_20d_ret": None}]
    assert _checksum(nan_rows, cols, pk, f32_cols) == _checksum(
        null_rows, cols, pk, f32_cols
    ), "NaN (parquet) and NULL (PG) must canonicalize to one token"


def test_canon_cell_numpy_nan_collapses_to_null_token():
    """RED (verify#0d00): a numpy-dtype NaN must collapse to the SAME NULL token
    as None and python-float NaN — for both is_float32 settings.

    The forward-return columns are float32 dtype, so their NaN cells arrive as
    ``numpy.float32`` NaN. The pre-fix ``isinstance(v, float)`` guard missed
    those (``np.float32`` is NOT a Python ``float`` subclass), leaking them to
    ``('s', 'nan')`` while the PG NULL side hashed ``('n',)`` — ~100 spurious
    ``thematic_labels_daily`` mismatches on recent dates. ``np.float64`` happens
    to subclass ``float`` so it already collapsed; the fix must cover both."""
    import numpy as np

    from rainier.db.verify import _canon_cell

    null_token = _canon_cell(None, True)
    for is_f32 in (True, False):
        assert _canon_cell(np.float32("nan"), is_f32) == _canon_cell(None, is_f32), (
            f"np.float32 NaN must equal NULL token (is_float32={is_f32})"
        )
        assert _canon_cell(np.float64("nan"), is_f32) == _canon_cell(None, is_f32), (
            f"np.float64 NaN must equal NULL token (is_float32={is_f32})"
        )
        assert _canon_cell(float("nan"), is_f32) == _canon_cell(None, is_f32), (
            f"python float NaN must equal NULL token (is_float32={is_f32})"
        )
    # And the None token itself is the canonical NULL token.
    assert null_token == ("n",)


def test_canon_cell_pandas_nat_collapses_to_null_token():
    """RED (verify#0d00): pandas NaT (the datetime NULL) must also collapse to
    the NULL token, not stringify to ``('s', 'NaT')`` — TIMESTAMP columns with
    an absent value would otherwise false-positive against PG NULL."""
    import pandas as pd

    from rainier.db.verify import _canon_cell

    assert _canon_cell(pd.NaT, False) == _canon_cell(None, False)
    assert _canon_cell(pd.NaT, True) == _canon_cell(None, True)


def test_checksum_numpy_float32_nan_equals_null_end_to_end():
    """RED (verify#0d00) end-to-end: a frame with a float32 column carrying NaN
    cells, pushed through the SAME path verify uses (``frame_to_pg_rows`` ->
    ``_checksum`` with that column in ``float32_cols``), must hash EQUAL to the
    None/NULL counterpart row.

    This is the exact gap that let #110 through: that fix's tests used
    python-float NaN (already collapsed). Here the parquet-origin column is a
    real numpy float32 dtype, so its NaN is ``numpy.float32`` NaN — the leaking
    representation. The PG-side NULL comes back as Python ``None``."""
    import numpy as np
    import pandas as pd

    from rainier.db.rows import frame_to_pg_rows
    from rainier.db.verify import _checksum

    cols = ["symbol", "fwd_20d_ret"]
    pk = ("symbol",)
    f32_cols = frozenset({"fwd_20d_ret"})

    # Parquet side: a genuine float32 column whose cell is NaN (forward window
    # not yet elapsed). df.to_dict yields numpy.float32 NaN for this cell.
    parquet_df = pd.DataFrame(
        {
            "symbol": pd.Series(["AAA"], dtype="object"),
            "fwd_20d_ret": pd.Series([np.float32("nan")], dtype=np.float32),
        }
    )
    # PG side: SQL NULL -> Python None.
    pg_df = pd.DataFrame(
        {
            "symbol": pd.Series(["AAA"], dtype="object"),
            "fwd_20d_ret": pd.Series([None], dtype="object"),
        }
    )

    parquet_rows = frame_to_pg_rows(parquet_df, cols)
    pg_rows = frame_to_pg_rows(pg_df, cols)

    assert _checksum(parquet_rows, cols, pk, f32_cols) == _checksum(
        pg_rows, cols, pk, f32_cols
    ), "float32-dtype NaN cell (parquet) must hash equal to NULL (PG) end-to-end"

    # Also feed a raw numpy.float32 NaN straight into _checksum (bypassing the
    # pg_value coercion) so this guards _canon_cell directly: even if a numpy NaN
    # reaches the canon layer un-coerced, it must collapse to the NULL token.
    raw_f32_nan_rows = [{"symbol": "AAA", "fwd_20d_ret": np.float32("nan")}]
    assert _checksum(raw_f32_nan_rows, cols, pk, f32_cols) == _checksum(
        pg_rows, cols, pk, f32_cols
    ), "raw numpy.float32 NaN must hash equal to NULL at the _checksum layer"


def test_checksum_different_values_still_mismatch():
    """NEGATIVE gate: genuinely different numbers must STILL mismatch. The
    width/NaN canonicalization must not blind the gate to real drift — two
    float32-column values that differ beyond float32 precision (and a
    NaN-vs-real-value pair) must hash differently."""
    from rainier.db.verify import _checksum

    cols = ["symbol", "fwd_20d_ret"]
    pk = ("symbol",)
    f32_cols = frozenset({"fwd_20d_ret"})
    a = [{"symbol": "AAA", "fwd_20d_ret": 0.05}]
    b = [{"symbol": "AAA", "fwd_20d_ret": 0.06}]
    assert _checksum(a, cols, pk, f32_cols) != _checksum(
        b, cols, pk, f32_cols
    ), "0.05 vs 0.06 is real drift and must mismatch"
    # NaN/NULL must NOT collapse into a real value: a present number != absent.
    real_val = [{"symbol": "AAA", "fwd_20d_ret": 0.0}]
    nan_val = [{"symbol": "AAA", "fwd_20d_ret": float("nan")}]
    assert _checksum(real_val, cols, pk, f32_cols) != _checksum(
        nan_val, cols, pk, f32_cols
    ), "a real 0.0 and a missing (NaN/NULL) cell must remain distinguishable"


def test_float32_columns_derives_from_parquet_dtypes():
    """``_float32_columns`` reports exactly the float32-dtype columns of a frame
    (the live source-of-precision), so a float64 column is excluded and stays
    full precision in the checksum."""
    import numpy as np
    import pandas as pd

    from rainier.db.verify import _float32_columns

    df = pd.DataFrame(
        {
            "symbol": pd.Series(["AAA"], dtype="object"),
            "fwd_20d_ret": pd.Series([0.1], dtype=np.float32),
            "close": pd.Series([100.0], dtype=np.float64),
        }
    )
    assert _float32_columns(df) == frozenset({"fwd_20d_ret"})


def test_real_columns_from_schema():
    """``_real_columns`` reports the PG ``REAL`` columns of a table spec — the
    second normalization signal (float64-parquet-into-REAL direction)."""
    from rainier.db import rows as rows_mod
    from rainier.db.verify import _real_columns

    labels_spec = next(
        s for s in rows_mod.TABLE_SPECS if s.name == "thematic_labels_daily"
    )
    real_cols = _real_columns(labels_spec)
    # The schema declares the forward-return columns REAL; assert a representative
    # one is present and a non-float key column is absent.
    assert "fwd_3d_ret" in real_cols
    assert "symbol" not in real_cols


def test_checksum_float64_parquet_into_real_pg_no_false_positive():
    """codex iter-1 P1 regression: a column stored float64 in parquet but
    ``REAL`` in PG must NOT false-positive. PG rounds the value to float32 on
    round-trip while the parquet float64 keeps full bits; because the column is
    in the schema-``REAL`` set the checksum normalizes both sides through float32
    and they hash equal. (Parquet-dtype alone returns empty here and would
    falsely flag drift.)"""
    import numpy as np

    from rainier.db.verify import _checksum

    cols = ["symbol", "fwd_3d_ret"]
    pk = ("symbol",)
    orig64 = 0.06948674738744653  # what the float64 parquet holds
    pg_real = float(np.float32(orig64))  # what PG REAL returns after rounding
    assert orig64 != pg_real
    # Union set includes the schema-REAL column even though parquet dtype is f64.
    f32_cols = frozenset({"fwd_3d_ret"})
    parquet_rows = [{"symbol": "AAA", "fwd_3d_ret": orig64}]
    pg_rows = [{"symbol": "AAA", "fwd_3d_ret": pg_real}]
    assert _checksum(parquet_rows, cols, pk, f32_cols) == _checksum(
        pg_rows, cols, pk, f32_cols
    ), "float64 parquet vs rounded PG REAL must hash equal (schema-REAL signal)"


# ---------------------------------------------------------------------------
# Timezone-aware datetime parity (TIMESTAMPTZ columns)
# ---------------------------------------------------------------------------


def test_checksum_tz_aware_same_instant_matches():
    """Same instant in different tz offsets must hash identically.

    PG returns TIMESTAMPTZ (fetched_at/computed_at) in the session timezone, so
    a clean backfill can yield ``08:30-08:00`` on the PG side vs ``16:30+00:00``
    in parquet — the SAME instant. The checksum must normalize to UTC so this
    is NOT flagged as drift. (Regression: a session TZ != UTC would otherwise
    false-positive a clean parity run.)"""
    import datetime as _dt

    from rainier.db.verify import _checksum

    cols = ["symbol", "fetched_at"]
    pk = ("symbol",)
    utc = _dt.datetime(2024, 11, 8, 16, 30, tzinfo=_dt.timezone.utc)
    pacific = utc.astimezone(_dt.timezone(_dt.timedelta(hours=-8)))
    assert utc != pacific or str(utc) != str(pacific)  # reprs differ pre-norm
    parquet_rows = [{"symbol": "AAA", "fetched_at": utc}]
    pg_rows = [{"symbol": "AAA", "fetched_at": pacific}]
    assert _checksum(parquet_rows, cols, pk) == _checksum(
        pg_rows, cols, pk
    ), "same instant in a different tz offset must hash equal"


@pytest.mark.requires_postgres
def test_verify_clean_under_non_utc_session_tz(migrated_engine, tmp_path, database_url):
    """A clean backfill verifies clean even when the PG session timezone is not
    UTC (TIMESTAMPTZ columns then render in that zone). End-to-end guard for the
    tz-normalization fix."""
    from sqlalchemy import event

    from rainier.db.backfill import backfill_from_parquet
    from rainier.db.verify import verify_coverage

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA", "BBB"], n_days=6)
    backfill_from_parquet(migrated_engine, cache)

    # Force a non-UTC timezone on EVERY connection verify_coverage opens (it
    # opens its own connection in _read_pg, so a one-off SET on a separate
    # connection would not reach it). With this, TIMESTAMPTZ columns render in
    # US/Pacific while parquet stays UTC — the case the fix must absorb.
    pac_engine = create_engine(database_url)

    @event.listens_for(pac_engine, "connect")
    def _set_pacific(dbapi_conn, _rec):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("SET TIME ZONE 'America/Los_Angeles'")
        cur.close()

    try:
        # Sanity: a TIMESTAMPTZ really renders in the session zone now.
        with pac_engine.connect() as conn:
            rendered = conn.exec_driver_sql(
                "SELECT fetched_at::text FROM market.thematic_ohlcv LIMIT 1"
            ).scalar_one()
        assert "+00" not in rendered, f"expected non-UTC render, got {rendered!r}"

        report = verify_coverage(pac_engine, cache)
        assert report.ok, f"non-UTC session must not false-positive drift: {report.drift}"
    finally:
        pac_engine.dispose()


# ---------------------------------------------------------------------------
# CLI surface: exit codes
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_cli_verify_exit_zero_on_match(migrated_engine, tmp_path, database_url):
    from rainier.cli import cli
    from rainier.db.backfill import backfill_from_parquet

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA", "BBB"], n_days=8)
    backfill_from_parquet(migrated_engine, cache)

    res = CliRunner().invoke(
        cli, ["db", "verify-coverage", "--cache-dir", str(cache)]
    )
    assert res.exit_code == 0, res.output


@pytest.mark.requires_postgres
def test_cli_verify_nonzero_on_drift(migrated_engine, tmp_path, database_url):
    from rainier.cli import cli
    from rainier.db.backfill import backfill_from_parquet

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA", "BBB"], n_days=8)
    backfill_from_parquet(migrated_engine, cache)

    with migrated_engine.begin() as conn:
        victim = conn.execute(
            text("SELECT asof_date FROM market.thematic_features_daily LIMIT 1")
        ).scalar_one()
        conn.execute(
            text(
                "DELETE FROM market.thematic_features_daily "
                "WHERE asof_date=:d AND symbol='AAA'"
            ),
            {"d": victim},
        )

    res = CliRunner().invoke(
        cli, ["db", "verify-coverage", "--cache-dir", str(cache)]
    )
    assert res.exit_code != 0
    assert "thematic_features_daily" in res.output
    assert str(victim) in res.output


def test_cli_verify_requires_database_url(tmp_path, monkeypatch):
    """DATABASE_URL unset -> a clean ClickException, not a raw traceback."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from rainier.cli import cli

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA"], n_days=3)

    res = CliRunner().invoke(
        cli, ["db", "verify-coverage", "--cache-dir", str(cache)]
    )
    assert res.exit_code != 0
    assert res.exception is None or isinstance(res.exception, SystemExit), (
        f"expected a clean ClickException, got {res.exception!r}"
    )
    assert "DATABASE_URL" in res.output


def test_cli_verify_rejects_reversed_window(tmp_path, monkeypatch):
    """--asof-start > --asof-end -> clean ClickException, never a silent CLEAN
    pass. A reversed window compares an empty parquet slice against an empty PG
    slice and both 'match', so verify-coverage would falsely report parity. The
    guard must reject it before any comparison (and before DATABASE_URL is even
    required)."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from rainier.cli import cli

    res = CliRunner().invoke(
        cli,
        [
            "db", "verify-coverage", "--cache-dir", str(tmp_path / "cache"),
            "--asof-start", "2026-05-10", "--asof-end", "2026-05-01",
        ],
    )
    assert res.exit_code != 0
    assert res.exception is None or isinstance(res.exception, SystemExit), (
        f"expected a clean ClickException, got {res.exception!r}"
    )
    assert "asof-start" in res.output and "asof-end" in res.output


def test_cli_verify_registered():
    from rainier.cli import cli

    res = CliRunner().invoke(cli, ["db", "--help"])
    assert res.exit_code == 0, res.output
    assert "verify-coverage" in res.output
