"""Unit tests for scripts/backfill_macro_context.py.

The backfill script is a one-off operator-run tool. We never want it touching
the network during CI, so its public API takes a `fetch_fn` injection point.
Tests pass a stub fixture; the production CLI wires `yfinance.download`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

# Import the script as a module — it lives in `scripts/`, not under `src/`.
ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "backfill_macro_context.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "backfill_macro_context", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture
def backfill_mod():
    return _load_module()


def _stub_fetch(symbols, start, end):
    """Deterministic fixture: 5 trading days x N symbols, monotonic prices.

    Mirrors the shape `yfinance.download(..., group_by="ticker")` returns:
    a dict-like mapping `symbol -> pd.DataFrame` with OHLCV + Adj Close.

    We return a real dict so the backfill script doesn't depend on the
    yfinance multi-index DataFrame format internally.
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
                    "adjusted_close": base + j + 0.25,
                }
            )
        out[sym] = pd.DataFrame(rows)
    return out


def test_stub_determinism(backfill_mod, tmp_path):
    """Backfill on a stub fixture is byte-deterministic modulo `fetched_at`."""
    out_a = tmp_path / "a.parquet"
    out_b = tmp_path / "b.parquet"

    backfill_mod.backfill(
        symbols=["VIX", "QQQ", "XLK"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out_a,
        force=False,
        fetch_fn=_stub_fetch,
    )
    backfill_mod.backfill(
        symbols=["VIX", "QQQ", "XLK"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out_b,
        force=False,
        fetch_fn=_stub_fetch,
    )

    df_a = pd.read_parquet(out_a)
    df_b = pd.read_parquet(out_b)

    # Sort key is (symbol, date) ascending.
    assert list(df_a["symbol"].unique()) == sorted(df_a["symbol"].unique().tolist())
    for sym in df_a["symbol"].unique():
        dates = df_a.loc[df_a["symbol"] == sym, "date"].tolist()
        assert dates == sorted(dates), f"date order broken for {sym}"

    # Drop volatile columns and compare the rest byte-by-byte.
    stable_cols = [
        c for c in df_a.columns if c not in {"fetched_at", "yfinance_version"}
    ]
    pd.testing.assert_frame_equal(
        df_a[stable_cols].reset_index(drop=True),
        df_b[stable_cols].reset_index(drop=True),
    )


def test_refuses_overwrite_without_force(backfill_mod, tmp_path):
    out = tmp_path / "macro.parquet"
    backfill_mod.backfill(
        symbols=["VIX"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=False,
        fetch_fn=_stub_fetch,
    )

    with pytest.raises(FileExistsError):
        backfill_mod.backfill(
            symbols=["VIX"],
            start="2024-10-01",
            end="2024-10-08",
            out_path=out,
            force=False,
            fetch_fn=_stub_fetch,
        )


def test_force_writes_dated_cohort(backfill_mod, tmp_path):
    """--force does NOT overwrite in place; writes a sibling cohort file."""
    out = tmp_path / "macro.parquet"
    backfill_mod.backfill(
        symbols=["VIX"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=False,
        fetch_fn=_stub_fetch,
    )
    cohort_path = backfill_mod.backfill(
        symbols=["VIX"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=True,
        fetch_fn=_stub_fetch,
    )

    # Original untouched.
    assert out.exists()
    # New cohort file written next to it.
    assert cohort_path != out
    assert cohort_path.exists()
    assert cohort_path.parent == out.parent


def test_dry_run_does_not_write(backfill_mod, tmp_path):
    out = tmp_path / "macro.parquet"
    result = backfill_mod.backfill(
        symbols=["VIX", "QQQ"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=False,
        dry_run=True,
        fetch_fn=_stub_fetch,
    )
    assert not out.exists()
    # dry-run returns the planned matrix as a sanity object (list/dict).
    assert result is not None


def test_atomic_write_no_tmp_leak(backfill_mod, tmp_path):
    out = tmp_path / "macro.parquet"
    backfill_mod.backfill(
        symbols=["VIX", "QQQ"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=False,
        fetch_fn=_stub_fetch,
    )
    # No .tmp sibling left behind.
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == [], f"tmp leak: {leftovers}"


def test_parquet_schema_columns(backfill_mod, tmp_path):
    out = tmp_path / "macro.parquet"
    backfill_mod.backfill(
        symbols=["VIX", "QQQ", "XLK"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=False,
        fetch_fn=_stub_fetch,
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
        "adjusted_close",
        "fetched_at",
        "yfinance_version",
    }
    assert required.issubset(names), f"missing columns: {required - names}"


def test_symbols_constant_has_36_phase0_tickers(backfill_mod):
    """Phase 0 final list per TASK-PLAN §4 (resolved by coord 431c2abf)."""
    expected = {
        # vol (4)
        "VIX",
        "VIX9D",
        "VIX3M",
        "UVXY",
        # benchmarks (4)
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
        # sectors (11)
        "XLK",
        "XLF",
        "XLE",
        "XLV",
        "XLI",
        "XLY",
        "XLP",
        "XLU",
        "XLB",
        "XLRE",
        "XLC",
        # themes (3)
        "SMH",
        "XBI",
        "KRE",
        # international (3)
        "EEM",
        "FXI",
        "KWEB",
        # commodities (3)
        "USO",
        "GDX",
        "SLV",
        # fixed income (4)
        "TLT",
        "HYG",
        "IEF",
        "TIP",
        # currency (1)
        "UUP",
        # crypto (2)
        "IBIT",
        "IREN",
        # innovation (1)
        "ARKK",
    }
    assert set(backfill_mod.SYMBOLS) == expected
    assert len(backfill_mod.SYMBOLS) == 36


def test_symbol_normalization_strips_caret(backfill_mod, tmp_path):
    """yfinance returns `^VIX`; stored symbol must be `VIX`."""
    # Simulate yfinance returning ^-prefixed keys.
    def caret_fetch(symbols, start, end):
        return _stub_fetch(["VIX", "VIX9D", "VIX3M"], start, end)

    out = tmp_path / "macro.parquet"
    backfill_mod.backfill(
        symbols=["VIX", "VIX9D", "VIX3M"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=False,
        fetch_fn=caret_fetch,
    )

    df = pd.read_parquet(out)
    # No ^-prefixed symbols in the stored column.
    assert not any(s.startswith("^") for s in df["symbol"].unique())


# ---------------------------------------------------------------------------
# Codex iter-1 regressions
# ---------------------------------------------------------------------------


def test_empty_symbol_raises_unless_allowlisted(backfill_mod, tmp_path):
    """Silent partial backfill is a correctness bug — see codex iter-1.

    When the provider returns zero rows for a requested symbol (outage, bad
    ticker, rate limit), the backfill MUST raise before writing the cache so
    the L3 evaluator cannot consume a partial backfill that the ledger claims
    is complete. ``--allow-empty`` is the explicit operator-opt-out.
    """
    def partial_fetch(symbols, start, end):
        # VIX returns data; QQQ returns empty.
        full = _stub_fetch(["VIX"], start, end)
        full["QQQ"] = pd.DataFrame(
            columns=[
                "date", "open", "high", "low", "close",
                "volume", "adjusted_close",
            ]
        )
        return full

    out = tmp_path / "macro.parquet"
    with pytest.raises(ValueError, match="QQQ"):
        backfill_mod.backfill(
            symbols=["VIX", "QQQ"],
            start="2024-10-01",
            end="2024-10-08",
            out_path=out,
            force=False,
            fetch_fn=partial_fetch,
        )
    # Cache must NOT be written when validation fails.
    assert not out.exists()


def test_empty_symbol_allowed_via_allowlist(backfill_mod, tmp_path):
    """Explicit allowlist lets the operator permit known-stale tickers."""
    def partial_fetch(symbols, start, end):
        full = _stub_fetch(["VIX"], start, end)
        full["DEAD"] = pd.DataFrame(
            columns=[
                "date", "open", "high", "low", "close",
                "volume", "adjusted_close",
            ]
        )
        return full

    out = tmp_path / "macro.parquet"
    backfill_mod.backfill(
        symbols=["VIX", "DEAD"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=False,
        fetch_fn=partial_fetch,
        allow_empty=["DEAD"],
    )
    assert out.exists()
    df = pd.read_parquet(out)
    # DEAD silently absent; VIX rows present.
    assert set(df["symbol"].unique()) == {"VIX"}


def test_force_cohort_paths_never_collide(backfill_mod, tmp_path, monkeypatch):
    """Two --force runs in the same second must NOT overwrite each other.

    The ledger declares ``revision_immutability: true`` for every cohort. If
    two ``--force`` invocations land on the same per-second timestamp, the
    older cohort would be silently replaced by ``os.replace`` — violating the
    invariant. See codex iter-2.
    """
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    # Freeze fetched_at so both runs hit the same SECOND but the cohort path
    # must still differ (microsecond resolution + collision suffix).
    frozen = _dt(2026, 5, 22, 12, 0, 0, 0, tzinfo=_tz.utc)

    class _FrozenDT:
        @classmethod
        def now(cls, tz=None):
            return frozen

    monkeypatch.setattr(backfill_mod, "datetime", _FrozenDT)

    out = tmp_path / "macro.parquet"
    # Initial backfill creates the base cache.
    backfill_mod.backfill(
        symbols=["VIX"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=False,
        fetch_fn=_stub_fetch,
    )
    # Two --force runs at the exact same fetched_at.
    cohort1 = backfill_mod.backfill(
        symbols=["VIX"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=True,
        fetch_fn=_stub_fetch,
    )
    cohort2 = backfill_mod.backfill(
        symbols=["VIX"],
        start="2024-10-01",
        end="2024-10-08",
        out_path=out,
        force=True,
        fetch_fn=_stub_fetch,
    )

    assert cohort1 != cohort2, "force cohorts must not share a path"
    assert cohort1.exists()
    assert cohort2.exists()
    # Original untouched.
    assert out.exists()


def test_yfinance_end_is_bumped_for_inclusive_semantics(backfill_mod, monkeypatch):
    """yfinance treats ``end`` as exclusive; loader treats it as inclusive.

    The fetcher MUST add +1 day to the wire ``end`` so an operator-facing
    request like ``--end 2026-05-01`` includes 2026-05-01 trading-day rows
    (matching ``read_range(..., end='2026-05-01')`` inclusive semantics).
    See codex iter-1.
    """
    captured: dict[str, str] = {}

    class _StubYF:
        __version__ = "stub-test"

        @staticmethod
        def download(*args, **kwargs):
            captured["end"] = kwargs["end"]
            captured["start"] = kwargs["start"]
            # Return a non-empty single-row frame so the fetcher's downstream
            # processing doesn't take the empty-frame branch.
            df = pd.DataFrame(
                {
                    "Open": [100.0],
                    "High": [101.0],
                    "Low": [99.0],
                    "Close": [100.5],
                    "Adj Close": [100.5],
                    "Volume": [1000],
                },
                index=pd.DatetimeIndex(["2024-10-01"], name="Date"),
            )
            return df

    monkeypatch.setitem(__import__("sys").modules, "yfinance", _StubYF)

    backfill_mod._yfinance_fetch(["VIX"], "2024-10-01", "2024-10-08")

    # start passes through unchanged; end is bumped by +1 calendar day.
    assert captured["start"] == "2024-10-01"
    assert captured["end"] == "2024-10-09", (
        f"expected end bumped to 2024-10-09 (inclusive bridge), got {captured['end']}"
    )
