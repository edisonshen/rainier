"""LLM thesis layer (PR1) — public surface.

The four pieces a caller usually needs:
  - `compute_theses_and_persist` — scheduler entry point
  - `assemble_evidence`          — exposed for the `rainier thesis ticker` CLI
  - `TradeThesis`                — Pydantic schema for the LLM output
  - `REGISTRY`                   — signal name → class lookup
"""

from __future__ import annotations

from .persistence import persist_screened_stocks, update_with_thesis
from .prompt import PROMPT_VERSION
from .schemas import EvidencePack, TradeThesis, compute_input_hash
from .service import (
    assemble_evidence,
    compute_theses_and_persist,
    generate_thesis,
)
from .signals import REGISTRY, SignalContext, ThesisSignal

__all__ = [
    "PROMPT_VERSION",
    "REGISTRY",
    "EvidencePack",
    "SignalContext",
    "ThesisSignal",
    "TradeThesis",
    "assemble_evidence",
    "compute_input_hash",
    "compute_theses_and_persist",
    "generate_thesis",
    "persist_screened_stocks",
    "update_with_thesis",
]
