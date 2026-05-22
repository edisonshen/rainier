"""Schema validation for config/thematic_universe.yaml.

The universe YAML is operator-edited; the backfill + ranking pipeline keys
off it. The shape is load-bearing: bad shape → silent miscompute. Tests pin
the contract so an accidental indent or stray "BATS:" prefix fails CI loudly.

Design ref: docs/DESIGN-thematic-ranks-dashboard.md §4 ([D-001]).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = ROOT / "config" / "thematic_universe.yaml"

# YFinance accepts plain symbols only — no `BATS:`, `NYSE:`, `NASDAQ:` prefixes.
EXCHANGE_PREFIX_RE = re.compile(r"^[A-Z]+:")
# YFinance ETF symbol shape: 1-5 alphanumerics, optionally one dot (e.g., BRK.B).
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,5}$")


@pytest.fixture
def universe() -> dict:
    with UNIVERSE_PATH.open("r") as fh:
        return yaml.safe_load(fh)


def test_universe_file_exists():
    assert UNIVERSE_PATH.exists(), f"missing universe yaml at {UNIVERSE_PATH}"


def test_top_level_shape(universe):
    """Required keys: version, schema, asof_seeded, universe."""
    assert universe.get("version") == 1
    assert universe.get("schema") == "thematic_universe.v1"
    assert "asof_seeded" in universe
    assert "universe" in universe
    assert isinstance(universe["universe"], dict)


def test_universe_has_94_tickers(universe):
    """Phase A seed per DESIGN §4 = 94 tickers spread across sector buckets."""
    all_tickers: list[str] = []
    for sector, tickers in universe["universe"].items():
        all_tickers.extend(tickers)
    assert len(all_tickers) == 94, (
        f"expected 94 tickers in seed universe, got {len(all_tickers)}"
    )


def test_no_duplicate_tickers_across_sectors(universe):
    """A ticker MUST live in exactly one sector bucket.

    Duplicates would inflate the universe size for ranking (centile
    denominator) without adding a real symbol — silent miscompute.
    """
    seen: dict[str, str] = {}
    for sector, tickers in universe["universe"].items():
        for t in tickers:
            assert t not in seen, (
                f"duplicate ticker {t!r}: in both {seen[t]!r} and {sector!r}"
            )
            seen[t] = sector


def test_all_tickers_uppercase_no_prefix(universe):
    """No `BATS:` / `NYSE:` / lowercase. yfinance is symbol-only."""
    for sector, tickers in universe["universe"].items():
        for t in tickers:
            assert isinstance(t, str), f"non-string ticker in {sector}: {t!r}"
            assert t == t.upper(), f"ticker not uppercase: {t!r}"
            assert not EXCHANGE_PREFIX_RE.match(t), (
                f"ticker has exchange prefix (strip it): {t!r}"
            )
            assert SYMBOL_RE.match(t), (
                f"ticker doesn't match yfinance symbol shape: {t!r}"
            )


def test_no_empty_sector_buckets(universe):
    """A sector key with an empty list is editor noise; drop the key instead."""
    for sector, tickers in universe["universe"].items():
        assert isinstance(tickers, list), f"sector {sector!r} not a list"
        assert len(tickers) > 0, f"sector {sector!r} has zero tickers"


def test_seed_includes_anchor_tickers(universe):
    """Sanity: a few well-known SPDRs and themes from DESIGN §4 must be in the
    seed (catches accidental deletes during YAML hand-edits).
    """
    all_tickers: set[str] = set()
    for sector, tickers in universe["universe"].items():
        all_tickers.update(tickers)

    anchors = {
        # Sector SPDRs
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
        "XLRE", "XLC",
        # Themes called out in DESIGN §4
        "SMH", "XBI", "KRE", "ARKK", "DXYZ",
    }
    missing = anchors - all_tickers
    assert not missing, f"seed missing anchor tickers from DESIGN §4: {sorted(missing)}"
