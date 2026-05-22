"""Schema test for src/rainier/research/data_availability.yaml.

The ledger is the look-ahead control surface: every (symbol, field) pair the
L3 evaluator can read MUST be registered with an observability timestamp rule
and a revision-immutability claim (per DESIGN-qu100-llm-backtest [D-014]).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "src" / "rainier" / "research" / "data_availability.yaml"
BACKFILL_SCRIPT = ROOT / "scripts" / "backfill_macro_context.py"

REQUIRED_FIELDS = {"open", "high", "low", "close", "volume"}
# adjusted_close is INTENTIONALLY excluded from the ledger because yfinance
# back-adjusts it for future splits/dividends — i.e., the value at parquet
# read-time is NOT what was observable at the historical date's close.
# Registering it under end_of_trading_day would leak look-ahead. The column
# still exists in the parquet for non-look-ahead analyses, but the L3
# evaluator MUST restrict reads to REQUIRED_FIELDS. Per codex iter-6.
LOOK_AHEAD_LEAKY_FIELDS = {"adjusted_close"}


def _load_backfill_symbols() -> list[str]:
    spec = importlib.util.spec_from_file_location(
        "backfill_macro_context", BACKFILL_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return list(module.SYMBOLS)


@pytest.fixture
def ledger():
    with LEDGER_PATH.open("r") as fh:
        return yaml.safe_load(fh)


def test_ledger_file_exists():
    assert LEDGER_PATH.exists(), f"missing ledger at {LEDGER_PATH}"


def test_ledger_top_level_shape(ledger):
    """Top-level keys: `schema_version` + `entries`."""
    assert "schema_version" in ledger
    assert "entries" in ledger
    assert isinstance(ledger["entries"], list)


def test_macro_context_fields_registered(ledger):
    """Every (SYMBOL, field) pair from the backfill SYMBOLS constant is registered."""
    expected_symbols = set(_load_backfill_symbols())

    # Build the (symbol, field) set from the ledger.
    registered: set[tuple[str, str]] = set()
    for entry in ledger["entries"]:
        if entry.get("dataset") != "macro_context":
            continue
        sym = entry["symbol"]
        for field in entry["fields"]:
            registered.add((sym, field))

    expected: set[tuple[str, str]] = {
        (s, f) for s in expected_symbols for f in REQUIRED_FIELDS
    }
    missing = expected - registered
    assert not missing, f"missing ledger entries: {sorted(missing)[:10]}..."


def test_observability_rule_and_immutability(ledger):
    """Every macro_context entry declares end_of_trading_day + immutability=True."""
    for entry in ledger["entries"]:
        if entry.get("dataset") != "macro_context":
            continue
        assert entry["observability_timestamp_rule"] == "end_of_trading_day", (
            f"entry {entry['symbol']} has wrong observability rule: "
            f"{entry['observability_timestamp_rule']!r}"
        )
        assert entry["revision_immutability"] is True, (
            f"entry {entry['symbol']} must declare revision_immutability=true"
        )


def test_no_unexpected_symbols(ledger):
    """Ledger doesn't register symbols outside the Phase 0 list."""
    expected_symbols = set(_load_backfill_symbols())
    registered_symbols = {
        entry["symbol"]
        for entry in ledger["entries"]
        if entry.get("dataset") == "macro_context"
    }
    extra = registered_symbols - expected_symbols
    assert not extra, f"unexpected ledger symbols: {sorted(extra)}"


def test_provenance_fields_present(ledger):
    """Each macro_context entry records source + version provenance."""
    for entry in ledger["entries"]:
        if entry.get("dataset") != "macro_context":
            continue
        assert entry.get("source") == "yfinance", (
            f"entry {entry['symbol']} missing source=yfinance"
        )
        # `yfinance_version` recorded per-row in parquet; ledger records the
        # minimum compatible version pinned at fetch time.
        assert "yfinance_version_at_backfill" in entry, (
            f"entry {entry['symbol']} missing yfinance_version_at_backfill"
        )


def test_no_look_ahead_leaky_fields_registered(ledger):
    """No entry may register a field that violates the observability rule.

    yfinance's ``Adj Close`` is back-adjusted for future splits/dividends,
    so the value at parquet read-time is NOT what was observable at the
    historical date's close. Registering it under ``end_of_trading_day``
    would let the L3 evaluator silently consume look-ahead-adjusted data.
    See codex iter-6 [P1].
    """
    for entry in ledger["entries"]:
        if entry.get("dataset") != "macro_context":
            continue
        leaky = set(entry["fields"]) & LOOK_AHEAD_LEAKY_FIELDS
        assert not leaky, (
            f"entry {entry['symbol']} registers look-ahead-leaky field(s) "
            f"{sorted(leaky)} under {entry['observability_timestamp_rule']!r}. "
            f"Remove from `fields:` or use a distinct non-as-of rule."
        )
