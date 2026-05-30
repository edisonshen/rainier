"""Unit tests for scripts/backfill_thematic_universe.py.

Mirrors the shape of `tests/test_backfill_macro_context.py` (vix worker).
Same fail-modes — empty fetch, partial coverage, --force cohort collision,
atomic-write tmp leak. Difference: SYMBOLS is loaded from
`config/thematic_universe.yaml` at call time, not hard-coded.

The script never hits the network in tests — `fetch_fn` injection point
keeps CI offline.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "backfill_thematic_universe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "backfill_thematic_universe", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture
def backfill_mod():
    return _load_module()


def _stub_fetch(symbols, start, end):
    """Deterministic fixture: 5 trading days x N symbols, monotonic prices.

    Mirrors the vix worker's stub: a real dict so the script does not depend
    on yfinance's multi-index DataFrame internally.
    """
    days = pd.date_range("2024-10-01", periods=5, freq="B").date.tolist()
    out: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        base = 100.0 + i * 10.0
        rows = []
        for j, d in enumerate(days):
            rows.append(
                {
                    "date": d,
                    "open": base + j,
                    "high": base + j + 0.5,
                    "low": base + j - 0.5,
                    "close": base + j + 0.25,
                    "volume": 1000 * (j + 1),
                }
            )
        out[sym] = pd.DataFrame(rows)
    return out


# ---------------------------------------------------------------------------
# Determinism + atomic write
# ---------------------------------------------------------------------------


def test_stub_determinism(backfill_mod, tmp_path):
    """Backfill on a stub fixture is byte-deterministic modulo `fetched_at`."""
    out_a = tmp_path / "a.parquet"
    out_b = tmp_path / "b.parquet"

    backfill_mod.backfill(
        symbols=["XLK", "XLE", "XLF"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out_a,
        force=False,
        fetch_fn=_stub_fetch,
        min_coverage=0.0,
    )
    backfill_mod.backfill(
        symbols=["XLK", "XLE", "XLF"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out_b,
        force=False,
        fetch_fn=_stub_fetch,
        min_coverage=0.0,
    )

    df_a = pd.read_parquet(out_a)
    df_b = pd.read_parquet(out_b)

    # Sort key (symbol, date) ascending.
    assert list(df_a["symbol"].unique()) == sorted(df_a["symbol"].unique().tolist())
    for sym in df_a["symbol"].unique():
        dates = df_a.loc[df_a["symbol"] == sym, "date"].tolist()
        assert dates == sorted(dates), f"date order broken for {sym}"

    stable_cols = [c for c in df_a.columns if c not in {"fetched_at", "yfinance_version"}]
    pd.testing.assert_frame_equal(
        df_a[stable_cols].reset_index(drop=True),
        df_b[stable_cols].reset_index(drop=True),
    )


def test_atomic_write_no_tmp_leak(backfill_mod, tmp_path):
    out = tmp_path / "thematic.parquet"
    backfill_mod.backfill(
        symbols=["XLK", "SMH"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=False,
        fetch_fn=_stub_fetch,
        min_coverage=0.0,
    )
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == [], f"tmp leak: {leftovers}"


def test_parquet_schema_columns(backfill_mod, tmp_path):
    out = tmp_path / "thematic.parquet"
    backfill_mod.backfill(
        symbols=["XLK", "SMH", "XLE"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=False,
        fetch_fn=_stub_fetch,
        min_coverage=0.0,
    )

    schema = pq.read_schema(out)
    names = set(schema.names)
    required = {
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "fetched_at",
        "yfinance_version",
    }
    assert required.issubset(names), f"missing columns: {required - names}"
    # adjusted_close intentionally NOT stored (look-ahead-leaky per DESIGN).
    assert "adjusted_close" not in names, (
        "adjusted_close is back-adjusted by yfinance — must not be in the "
        "thematic_universe cache (DESIGN-thematic-ranks-dashboard §5.0)."
    )


# ---------------------------------------------------------------------------
# --force cohort + overwrite refusal
# ---------------------------------------------------------------------------


def test_refuses_overwrite_without_force(backfill_mod, tmp_path):
    out = tmp_path / "thematic.parquet"
    backfill_mod.backfill(
        symbols=["XLK"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=False,
        fetch_fn=_stub_fetch,
        min_coverage=0.0,
    )
    with pytest.raises(FileExistsError):
        backfill_mod.backfill(
            symbols=["XLK"],
            start="2024-10-01",
            end="2024-10-08",
            out_path=out,
            force=False,
            fetch_fn=_stub_fetch,
        )


def test_force_writes_dated_cohort(backfill_mod, tmp_path):
    """--force does NOT overwrite in place; writes a sibling cohort file."""
    out = tmp_path / "thematic.parquet"
    backfill_mod.backfill(
        symbols=["XLK"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=False,
        fetch_fn=_stub_fetch,
        min_coverage=0.0,
    )
    cohort = backfill_mod.backfill(
        symbols=["XLK"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=True,
        fetch_fn=_stub_fetch,
        min_coverage=0.0,
    )

    assert out.exists()
    assert cohort != out
    assert cohort.exists()
    assert cohort.parent == out.parent


def test_dry_run_does_not_write(backfill_mod, tmp_path):
    out = tmp_path / "thematic.parquet"
    result = backfill_mod.backfill(
        symbols=["XLK", "XLF"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=False,
        dry_run=True,
        fetch_fn=_stub_fetch,
    )
    assert not out.exists()
    assert result is not None


# ---------------------------------------------------------------------------
# YAML-driven SYMBOLS
# ---------------------------------------------------------------------------


def test_load_symbols_from_yaml(backfill_mod):
    """`load_symbols_from_yaml(path)` returns the flat ticker list from YAML."""
    from io import StringIO

    import yaml as _yaml

    sample = """\
version: 1
schema: thematic_universe.v1
asof_seeded: 2026-05-22
universe:
  technology:
    - XLK
    - SMH
  energy:
    - XLE
"""
    # Write to a temp path because the loader takes a Path.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(sample)
        path = Path(fh.name)
    try:
        symbols = backfill_mod.load_symbols_from_yaml(path)
        # Round-trip parse to confirm the loader matches yaml.safe_load shape.
        parsed = _yaml.safe_load(StringIO(sample))
        expected = []
        for buckets in parsed["universe"].values():
            expected.extend(buckets)
        assert sorted(symbols) == sorted(expected)
    finally:
        path.unlink()


def test_default_yaml_drives_seed_universe(backfill_mod):
    """When called with no --symbols, the script reads
    `config/thematic_universe.yaml` and the seed has 94 tickers per DESIGN §4.
    """
    default_yaml = ROOT / "config" / "thematic_universe.yaml"
    symbols = backfill_mod.load_symbols_from_yaml(default_yaml)
    assert len(symbols) == 94


# ---------------------------------------------------------------------------
# Empty-symbol fail-fast (mirrors vix worker codex iter-1)
# ---------------------------------------------------------------------------


def test_empty_symbol_raises_unless_allowlisted(backfill_mod, tmp_path):
    def partial_fetch(symbols, start, end):
        full = _stub_fetch(["XLK"], start, end)
        full["DEAD"] = pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume"]
        )
        return full

    out = tmp_path / "thematic.parquet"
    with pytest.raises(ValueError, match="DEAD"):
        backfill_mod.backfill(
            symbols=["XLK", "DEAD"],
            start="2024-10-01",
            end="2024-10-08",
            out_path=out,
            force=False,
            fetch_fn=partial_fetch,
        )
    assert not out.exists()


def test_empty_symbol_allowed_via_allowlist(backfill_mod, tmp_path):
    def partial_fetch(symbols, start, end):
        full = _stub_fetch(["XLK"], start, end)
        full["DXYZ"] = pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume"]
        )
        return full

    out = tmp_path / "thematic.parquet"
    backfill_mod.backfill(
        symbols=["XLK", "DXYZ"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=False,
        fetch_fn=partial_fetch,
        allow_empty=["DXYZ"],
        min_coverage=0.0,
    )
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df["symbol"].unique()) == {"XLK"}


# ---------------------------------------------------------------------------
# --incremental — recent-window refresh, in-place upsert (mirrors breadth
# backfill-ohlcv --incremental). Refreshes the existing cache without --force
# cohort so the stale-OHLCV guard in `run-daily` is satisfied.
# ---------------------------------------------------------------------------


def _windowed_fetch_factory(per_symbol_dates):
    """Return a fetch_fn that records its (symbols, start, end) args and
    returns rows ONLY for the dates inside [start, end] (inclusive), so a test
    can assert the incremental path fetches the recent window, not full history.
    """

    captured: dict[str, object] = {}

    def _fetch(symbols, start, end):
        captured["symbols"] = list(symbols)
        captured["start"] = start
        captured["end"] = end
        start_d = pd.to_datetime(start).date()
        end_d = pd.to_datetime(end).date()
        out: dict[str, pd.DataFrame] = {}
        for i, sym in enumerate(symbols):
            base = 100.0 + i * 10.0
            rows = []
            for j, d in enumerate(per_symbol_dates):
                if d < start_d or d > end_d:
                    continue
                rows.append(
                    {
                        "date": d,
                        "open": base + j,
                        "high": base + j + 0.5,
                        "low": base + j - 0.5,
                        "close": base + j + 0.25,
                        "volume": 1000 * (j + 1),
                    }
                )
            out[sym] = pd.DataFrame(
                rows,
                columns=["date", "open", "high", "low", "close", "volume"],
            )
        return out

    _fetch.captured = captured  # type: ignore[attr-defined]
    return _fetch


def test_incremental_fetches_recent_window_only(backfill_mod, tmp_path):
    """`incremental=True` ignores start/end and fetches a fixed recent window
    (today - INCREMENTAL_WINDOW_DAYS .. today), not the full history.
    """
    from datetime import date, timedelta

    out = tmp_path / "thematic.parquet"
    # Seed an existing cache with old history so we can prove the incremental
    # fetch window is the recent gap, not a full re-fetch.
    backfill_mod.backfill(
        symbols=["XLK", "SMH"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=False,
        fetch_fn=_stub_fetch,
        min_coverage=0.0,
    )

    today = date(2026, 5, 29)
    recent_dates = [today - timedelta(days=2), today - timedelta(days=1), today]
    fetch = _windowed_fetch_factory(recent_dates)

    backfill_mod.backfill(
        symbols=["XLK", "SMH"],
        start="2024-10-01",  # ignored under incremental
        end="2024-10-08",  # ignored under incremental
        out_path=out,
        incremental=True,
        fetch_fn=fetch,
        today=today,
        min_coverage=0.0,
    )

    cap = fetch.captured  # type: ignore[attr-defined]
    expected_start = (today - timedelta(days=backfill_mod.INCREMENTAL_WINDOW_DAYS)).isoformat()
    assert cap["start"] == expected_start
    assert cap["end"] == today.isoformat()


def test_incremental_upserts_in_place_no_cohort(backfill_mod, tmp_path):
    """`incremental=True` merges the recent window INTO the existing parquet
    on (symbol, date) — no FileExistsError, no sibling cohort file written.
    """
    from datetime import date, timedelta

    out = tmp_path / "thematic.parquet"
    backfill_mod.backfill(
        symbols=["XLK", "SMH"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=False,
        fetch_fn=_stub_fetch,
        min_coverage=0.0,
    )
    old_rows = len(pd.read_parquet(out))
    siblings_before = set(tmp_path.glob("thematic_*.parquet"))

    today = date(2026, 5, 29)
    recent_dates = [today - timedelta(days=1), today]
    fetch = _windowed_fetch_factory(recent_dates)

    result = backfill_mod.backfill(
        symbols=["XLK", "SMH"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        incremental=True,
        fetch_fn=fetch,
        today=today,
        min_coverage=0.0,
    )

    # Wrote IN PLACE, not a cohort sibling.
    assert result == out
    siblings_after = set(tmp_path.glob("thematic_*.parquet"))
    assert siblings_after == siblings_before, "incremental must not write a cohort"

    df = pd.read_parquet(out)
    # Old rows preserved + new recent rows appended (2 symbols x 2 new dates).
    assert len(df) == old_rows + 2 * len(recent_dates)
    assert not df.duplicated(subset=["symbol", "date"]).any()
    # The new dates are present.
    present_dates = set(df["date"].unique())
    for d in recent_dates:
        assert d in present_dates


def test_incremental_no_gap_is_noop(backfill_mod, tmp_path):
    """Re-running incremental with the same window produces no new rows
    (idempotent upsert on (symbol, date))."""
    from datetime import date, timedelta

    out = tmp_path / "thematic.parquet"
    backfill_mod.backfill(
        symbols=["XLK"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=False,
        fetch_fn=_stub_fetch,
        min_coverage=0.0,
    )

    today = date(2026, 5, 29)
    recent_dates = [today - timedelta(days=1), today]
    fetch = _windowed_fetch_factory(recent_dates)

    backfill_mod.backfill(
        symbols=["XLK"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        incremental=True,
        fetch_fn=fetch,
        today=today,
        min_coverage=0.0,
    )
    rows_after_first = len(pd.read_parquet(out))

    # Second incremental over the identical window: no new rows.
    backfill_mod.backfill(
        symbols=["XLK"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        incremental=True,
        fetch_fn=fetch,
        today=today,
        min_coverage=0.0,
    )
    df = pd.read_parquet(out)
    assert len(df) == rows_after_first
    assert not df.duplicated(subset=["symbol", "date"]).any()


def test_incremental_on_missing_cache_writes_fresh(backfill_mod, tmp_path):
    """`incremental=True` with no existing parquet just writes the recent
    window (no FileExistsError, no cohort) — first-run safety net."""
    from datetime import date, timedelta

    out = tmp_path / "thematic.parquet"
    today = date(2026, 5, 29)
    recent_dates = [today - timedelta(days=1), today]
    fetch = _windowed_fetch_factory(recent_dates)

    result = backfill_mod.backfill(
        symbols=["XLK"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        incremental=True,
        fetch_fn=fetch,
        today=today,
        min_coverage=0.0,
    )
    assert result == out
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df["date"].unique()) == set(recent_dates)
