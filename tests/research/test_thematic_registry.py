"""Stable-ID registry invariants (per DESIGN §5.4 + [D-015]).

Ticker and sector registries are append-only parquets that assign integer
IDs the FIRST time a symbol or sector_name is observed. Once an ID is
assigned, it is NEVER reused or remapped — even if the symbol is removed
from `config/thematic_universe.yaml`, the registry row stays so model
embeddings keep referring to the same ID.

Invariants tested here:
  1. ``get_or_assign_ticker_id("XLK")`` called twice returns the same int.
  2. After assigning ids 1..5 to {A,B,C,D,E}, removing A and adding F
     leaves ids 1..5 untouched and gives F id=6 (next-available).
  3. Sector registry obeys the same invariants on `sector_name`.
  4. Registry parquet round-trips identically.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from rainier.research.breadth import registry as reg


def test_first_assignment_returns_int(tmp_path: Path):
    out = tmp_path / "ticker_registry.parquet"
    tid = reg.get_or_assign_ticker_id("XLK", asof=date(2026, 5, 22), registry_path=out)
    assert isinstance(tid, int)
    assert tid >= 1


def test_idempotent_same_symbol_same_id(tmp_path: Path):
    out = tmp_path / "ticker_registry.parquet"
    first = reg.get_or_assign_ticker_id("XLK", asof=date(2026, 5, 22), registry_path=out)
    second = reg.get_or_assign_ticker_id("XLK", asof=date(2026, 5, 23), registry_path=out)
    assert first == second, "second call must return the same id (stable-ID invariant)"


def test_distinct_symbols_get_distinct_ids(tmp_path: Path):
    out = tmp_path / "ticker_registry.parquet"
    ids = {
        sym: reg.get_or_assign_ticker_id(sym, asof=date(2026, 5, 22), registry_path=out)
        for sym in ("AAA", "BBB", "CCC", "DDD", "EEE")
    }
    assert len(set(ids.values())) == 5
    # First-seen ordering preserved → ids assigned in call order starting at 1.
    assert ids == {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4, "EEE": 5}


def test_removed_symbol_keeps_id_new_symbol_gets_next(tmp_path: Path):
    """Operator removes A from the YAML; A's registry row must NOT be deleted.
    A new symbol F joining later gets id=6, not id=1 (no reuse).
    """
    out = tmp_path / "ticker_registry.parquet"
    for sym in ("AAA", "BBB", "CCC", "DDD", "EEE"):
        reg.get_or_assign_ticker_id(sym, asof=date(2026, 5, 22), registry_path=out)

    # Simulate next backfill: only B..E + new F (A removed from YAML).
    aid_before = reg.get_or_assign_ticker_id("AAA", asof=date(2026, 5, 22), registry_path=out)
    fid = reg.get_or_assign_ticker_id("FFF", asof=date(2026, 5, 23), registry_path=out)

    # AAA's row still present, same id.
    assert aid_before == 1
    # FFF gets next-available id, never recycled.
    assert fid == 6


def test_registry_round_trips_to_parquet(tmp_path: Path):
    out = tmp_path / "ticker_registry.parquet"
    reg.get_or_assign_ticker_id("XLK", asof=date(2026, 5, 22), registry_path=out)
    reg.get_or_assign_ticker_id("XLF", asof=date(2026, 5, 23), registry_path=out)

    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df.columns) >= {"ticker_id", "symbol", "first_seen"}
    assert df["symbol"].is_unique
    # ints are int16-storable (within [-32768, 32767]); registry uses int16 per design.
    assert df["ticker_id"].max() < 32768


def test_load_registry_helper(tmp_path: Path):
    """`load_ticker_registry` returns the current mapping as a dict."""
    out = tmp_path / "ticker_registry.parquet"
    reg.get_or_assign_ticker_id("XLK", asof=date(2026, 5, 22), registry_path=out)
    reg.get_or_assign_ticker_id("XLF", asof=date(2026, 5, 22), registry_path=out)

    mapping = reg.load_ticker_registry(out)
    assert mapping == {"XLK": 1, "XLF": 2}


def test_empty_registry_yields_empty_mapping(tmp_path: Path):
    out = tmp_path / "missing.parquet"
    assert reg.load_ticker_registry(out) == {}


# ---------------------------------------------------------------------------
# Sector registry — same invariants
# ---------------------------------------------------------------------------


def test_sector_registry_idempotent_and_appends(tmp_path: Path):
    out = tmp_path / "sector_registry.parquet"
    sids = {
        s: reg.get_or_assign_sector_id(s, asof=date(2026, 5, 22), registry_path=out)
        for s in ("energy", "technology", "healthcare")
    }
    assert sids == {"energy": 1, "technology": 2, "healthcare": 3}

    # Re-assignment is stable.
    again = reg.get_or_assign_sector_id("technology", asof=date(2026, 6, 1), registry_path=out)
    assert again == 2


def test_sector_remove_then_add_keeps_old_ids(tmp_path: Path):
    out = tmp_path / "sector_registry.parquet"
    for s in ("energy", "technology", "healthcare"):
        reg.get_or_assign_sector_id(s, asof=date(2026, 5, 22), registry_path=out)
    # Simulate sector rename: "thematic" is brand-new → gets id=4.
    new_id = reg.get_or_assign_sector_id("thematic", asof=date(2026, 6, 1), registry_path=out)
    assert new_id == 4


def test_seed_registries_helper_handles_all_universe(tmp_path: Path):
    """`seed_registries_from_universe` registers every (sector → tickers) pair.

    Idempotent: second call adds nothing new.
    """
    universe = {
        "energy": ["XLE", "XOP"],
        "technology": ["XLK", "SMH"],
    }
    ticker_out = tmp_path / "ticker_registry.parquet"
    sector_out = tmp_path / "sector_registry.parquet"

    reg.seed_registries_from_universe(
        universe,
        asof=date(2026, 5, 22),
        ticker_registry_path=ticker_out,
        sector_registry_path=sector_out,
    )
    t1 = reg.load_ticker_registry(ticker_out)
    s1 = reg.load_sector_registry(sector_out)
    assert set(t1.keys()) == {"XLE", "XOP", "XLK", "SMH"}
    assert set(s1.keys()) == {"energy", "technology"}

    # Second call with identical universe → no new rows, no id changes.
    reg.seed_registries_from_universe(
        universe,
        asof=date(2026, 5, 23),
        ticker_registry_path=ticker_out,
        sector_registry_path=sector_out,
    )
    t2 = reg.load_ticker_registry(ticker_out)
    s2 = reg.load_sector_registry(sector_out)
    assert t1 == t2
    assert s1 == s2
