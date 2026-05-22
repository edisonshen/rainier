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
