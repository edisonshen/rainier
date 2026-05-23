"""Schema-aware loader for ``data_availability.yaml``.

The ledger is the look-ahead control surface (D-014). vix-sector-backfill
shipped the YAML; Slice 0 adds the typed loader + extra datasets
(qu100_history, chart_image, signals_*).

The loader does just enough work to keep callers honest:
  - parse the ledger into typed Entry records
  - reject unknown observability rules
  - provide ``by_dataset`` / ``by_symbol`` lookups

It deliberately does NOT enforce "every field referenced by the
evaluator is registered" — that is the L3 evaluator's responsibility
(it walks the EvidencePack at construction time and grep-matches each
read against the ledger).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class LedgerError(ValueError):
    """Raised when the ledger YAML is malformed."""


VALID_OBSERVABILITY_RULES: frozenset[str] = frozenset(
    {
        # T close fully observable at market-close T.
        "end_of_trading_day",
        # Available at the QU scrape session (intraday; per-source detail in notes).
        "qu_session_close",
        # Available immediately on the day it's computed (signals derived from
        # T-1 close inputs; output is T-1-close itself).
        "derived_from_t_minus_1_close",
        # Computed locally from end-of-day inputs (e.g. thematic features).
        # Output is observable at T-close + compute latency.
        "derived_from_eod",
        # Forward-looking label: observable T+n trading days after the as-of
        # date (e.g. thematic_labels_daily ret_fwd_5d only knows its value at
        # T+5). The L3 evaluator must NOT serve these for as-of <= T+n-1.
        "t_plus_n_close",
        # Pending an external dependency (sibling task). Treated as "unobservable
        # by default" — the L3 evaluator rejects look-ups against pending fields.
        "pending_sibling_task",
    }
)


@dataclass(frozen=True)
class LedgerEntry:
    dataset: str
    symbol: str | None
    fields: tuple[str, ...]
    observability_timestamp_rule: str
    revision_immutability: bool
    source: str | None = None
    yfinance_version_at_backfill: str | None = None
    notes: str | None = None
    pending: bool = False  # Slice 0 extension; defaults to False for legacy rows


@dataclass(frozen=True)
class Ledger:
    schema_version: int
    entries: tuple[LedgerEntry, ...] = field(default_factory=tuple)

    def by_dataset(self, dataset: str) -> list[LedgerEntry]:
        return [e for e in self.entries if e.dataset == dataset]

    def by_symbol(self, symbol: str) -> list[LedgerEntry]:
        return [e for e in self.entries if e.symbol == symbol]


def _parse(data: dict[str, Any]) -> Ledger:
    if "schema_version" not in data:
        raise LedgerError("ledger missing `schema_version`")
    raw_entries = data.get("entries") or []
    if not isinstance(raw_entries, list):
        raise LedgerError("ledger `entries` must be a list")
    entries: list[LedgerEntry] = []
    for idx, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise LedgerError(f"entry {idx} is not a mapping")
        rule = raw.get("observability_timestamp_rule")
        if rule not in VALID_OBSERVABILITY_RULES:
            raise LedgerError(
                f"entry {idx} (dataset={raw.get('dataset')!r}) has unknown "
                f"observability_timestamp_rule: {rule!r}"
            )
        if "revision_immutability" not in raw:
            raise LedgerError(
                f"entry {idx} missing `revision_immutability` claim"
            )
        fields_val = raw.get("fields") or []
        entries.append(
            LedgerEntry(
                dataset=raw["dataset"],
                symbol=raw.get("symbol"),
                fields=tuple(fields_val),
                observability_timestamp_rule=rule,
                revision_immutability=bool(raw["revision_immutability"]),
                source=raw.get("source"),
                yfinance_version_at_backfill=raw.get("yfinance_version_at_backfill"),
                notes=raw.get("notes"),
                pending=bool(raw.get("pending", False)),
            )
        )
    return Ledger(
        schema_version=int(data["schema_version"]),
        entries=tuple(entries),
    )


def load(path: str | Path) -> Ledger:
    """Parse ``data_availability.yaml`` into a typed :class:`Ledger`."""
    with Path(path).open("r") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise LedgerError(f"{path} did not parse to a mapping")
    return _parse(data)
