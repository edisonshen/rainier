"""LLM-thesis signal registry.

Adding a new signal:
  1. Drop a file under `signals/` implementing `ThesisSignal`.
  2. Add it to REGISTRY below.
  3. Add a YAML toggle under `llm_thesis.signals.<name>`.

The scheduler imports REGISTRY at scan time and instantiates whichever
signals the YAML enables — no other code change required.
"""

from __future__ import annotations

from .base import SignalContext, SignalValue, ThesisSignal
from .capital_flow_streak import CapitalFlowStreakSignal
from .fundamentals import FundamentalsSignal
from .rank_trajectory import RankTrajectorySignal
from .sector_momentum import SectorMomentumSignal

REGISTRY: dict[str, type[ThesisSignal]] = {
    "rank_trajectory": RankTrajectorySignal,
    "capital_flow_streak": CapitalFlowStreakSignal,
    "sector_momentum": SectorMomentumSignal,
    "fundamentals": FundamentalsSignal,
}


__all__ = [
    "REGISTRY",
    "SignalContext",
    "SignalValue",
    "ThesisSignal",
    "RankTrajectorySignal",
    "CapitalFlowStreakSignal",
    "SectorMomentumSignal",
    "FundamentalsSignal",
]
