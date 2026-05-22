"""Schema-aware loader for the data-availability ledger.

The ledger is the look-ahead control surface (D-014). Slice 0 introduces
non-macro_context entries (qu100_history, ohlcv, signals, chart_image,
etc.) so the loader gets a typed API that knows about per-dataset
observability rules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rainier.research import data_availability_loader as loader

LEDGER_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "rainier" / "research" / "data_availability.yaml"
)


def test_load_returns_entries_list():
    ledger = loader.load(LEDGER_PATH)
    assert ledger.schema_version >= 1
    assert ledger.entries  # non-empty


def test_every_entry_carries_observability_rule():
    ledger = loader.load(LEDGER_PATH)
    valid_rules = loader.VALID_OBSERVABILITY_RULES
    for entry in ledger.entries:
        assert entry.observability_timestamp_rule in valid_rules


def test_revision_immutability_present_for_every_entry():
    ledger = loader.load(LEDGER_PATH)
    for entry in ledger.entries:
        assert entry.revision_immutability in (True, False)


def test_qu100_history_entry_present():
    """Slice 0 extends the ledger with qu100_history (D-016)."""
    ledger = loader.load(LEDGER_PATH)
    datasets = {e.dataset for e in ledger.entries}
    assert "qu100_history" in datasets


def test_chart_image_entry_present():
    """Slice 0 ledger declares chart_image (D-025) so cost-pilot calls
    can fingerprint the multimodal payload."""
    ledger = loader.load(LEDGER_PATH)
    datasets = {e.dataset for e in ledger.entries}
    assert "chart_image" in datasets


def test_signals_entries_present():
    """Per-symbol signals (pinbar/sr/pivots/bias) are ledger-registered."""
    ledger = loader.load(LEDGER_PATH)
    datasets = {e.dataset for e in ledger.entries}
    for sig_dataset in ("signals_pinbar", "signals_sr", "signals_pivots", "signals_bias"):
        assert sig_dataset in datasets, f"missing dataset {sig_dataset}"


def test_lookup_by_dataset_returns_matching_entries():
    ledger = loader.load(LEDGER_PATH)
    macro = ledger.by_dataset("macro_context")
    assert macro
    assert all(e.dataset == "macro_context" for e in macro)


def test_pending_status_field_present():
    """Slice 0 introduces a `pending` flag for fields tracked in sibling tasks."""
    ledger = loader.load(LEDGER_PATH)
    # The macro_context field is currently NOT pending (vix-backfill landed).
    macro = ledger.by_dataset("macro_context")
    assert all(not e.pending for e in macro)


def test_unknown_rule_raises():
    bad = {
        "schema_version": 1,
        "entries": [
            {
                "dataset": "macro_context",
                "symbol": "VIX",
                "fields": ["close"],
                "observability_timestamp_rule": "wishful_thinking",
                "revision_immutability": True,
                "source": "yfinance",
            },
        ],
    }
    with pytest.raises(loader.LedgerError):
        loader._parse(bad)  # noqa: SLF001 — exercising the parser directly
