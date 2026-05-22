"""EvidencePack assembler — STUB. Full implementation lands in Slice 1.

Slice 0 exposes only :func:`assemble_dry_run_pack` so the CLI's
``llm-research call --dry-run`` path can produce a deterministic prompt
without touching the real signal pipeline. Slice 1 swaps this for the
actual builder that reads OHLCV + signals + chart image.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DryRunPack:
    skill: str
    ticker: str
    day: date

    def as_prompt(self) -> str:
        """Deterministic prompt for the dry-run path.

        The shape is intentionally minimal — Slice 1 replaces this with
        the real prompt template + injected signals. We pin the exact
        text here so the determinism test (same `(skill, ticker, day)`
        → byte-identical output) holds without depending on Slice 1.
        """
        return (
            f"[DRY-RUN] skill={self.skill} ticker={self.ticker} "
            f"day={self.day.isoformat()}\n"
            "WHAT YOU CAN DO:\n"
            "- Emit `setup_long` if your analysis supports entry within next 3 trading days.\n"
            "- Emit `no_setup` if no clear edge today.\n"
            "- Emit `watch` if you'd track this ticker but don't enter yet.\n"
            "\n"
            "WHAT YOU CANNOT DO:\n"
            "- Reference any data dated after the T-1 close.\n"
            "- Quote now_price more than 2% off the T-1 close.\n"
            "- Emit verdicts not in {setup_long, no_setup, watch}.\n"
            "- Skip the price-ordering check (entry > stop, target > entry).\n"
        )


def assemble_dry_run_pack(*, skill: str, ticker: str, day: date) -> DryRunPack:
    return DryRunPack(skill=skill, ticker=ticker, day=day)
