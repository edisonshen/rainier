"""Signal framework — `ThesisSignal` Protocol + `SignalContext` shared dataclass.

Each piece of evidence the LLM reads is a Signal. Signals are pure functions of
(symbol, scan_date, candidate, ohlcv_df, params) → JSON-serializable value.
The framework runs them in parallel via asyncio.gather + asyncio.to_thread
(see service.assemble_evidence). One bad signal does not kill the thesis.

Adding a new signal: drop a file under `signals/`, add it to REGISTRY in
`signals/__init__.py`, add a YAML toggle under `llm_thesis.signals.<name>`.
Zero scheduler/prompt changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from rainier.core.types import StockCandidate

# Signal output: must be JSON-serializable so it can land in JSONB columns.
SignalValue = dict[str, Any]


@dataclass
class SignalContext:
    """Context passed to every Signal.compute() call."""

    symbol: str
    scan_date: date
    session_name: str
    candidate: StockCandidate
    ohlcv_df: pd.DataFrame | None = None
    params: dict[str, Any] = field(default_factory=dict)
    # Optional DB session — signals open their own via get_session() when None,
    # but tests can inject a sessionmaker callable for isolation.


@runtime_checkable
class ThesisSignal(Protocol):
    """Protocol every signal in the registry must satisfy.

    Class attributes (set on subclasses):
      - `name`: registry key, must match settings.yaml entry
      - `version`: bump to invalidate cached values when logic changes
      - `cost_estimate_ms`: rough latency budget hint, surfaced by dashboards
    """

    name: str
    version: str
    cost_estimate_ms: int

    def compute(self, ctx: SignalContext) -> SignalValue | None:
        """Pure compute — return JSON-serializable value or None to skip ticker."""
        ...

    def render_for_prompt(self, value: SignalValue) -> str:
        """Convert value to a short text snippet stitched into the LLM prompt."""
        ...
