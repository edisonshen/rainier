"""Regime-aware signal filter — wraps any SignalEmitter and drops signals
when the current market regime is not in the allowed set.

Satisfies the SignalEmitter protocol (decorator pattern).
"""

from __future__ import annotations

import pandas as pd

from rainier.analysis.regime import RegimeDetector
from rainier.core.protocols import SignalEmitter
from rainier.core.types import MarketRegime, Signal, Timeframe


class RegimeFilter:
    """Wraps a SignalEmitter, only passing signals in allowed regimes."""

    def __init__(
        self,
        inner: SignalEmitter,
        detector: RegimeDetector,
        allowed_regimes: set[MarketRegime],
    ) -> None:
        self._inner = inner
        self._detector = detector
        self._allowed = allowed_regimes

    def emit(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
    ) -> list[Signal]:
        """Emit signals only if current bar's regime is in allowed set."""
        regime = self._detector.detect_at(df, len(df) - 1)
        if regime not in self._allowed:
            return []
        return self._inner.emit(df, symbol, timeframe)
