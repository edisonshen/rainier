"""SignalEmitter implementations — adapters that satisfy the SignalEmitter protocol.

Each emitter wraps a specific signal generation strategy (pin bar, pattern, ML, etc.)
and exposes the uniform emit(df, symbol, timeframe) → list[Signal] interface.
"""

from __future__ import annotations

import pandas as pd

from rainier.analysis.analyzer import analyze
from rainier.core.config import AnalysisConfig, SignalConfig
from rainier.core.types import Signal, Timeframe
from rainier.signals.generator import generate_signals


class PinBarSignalEmitter:
    """Emits signals using the Xiaojiang pin bar methodology.

    Wraps analyze() + generate_signals() behind the SignalEmitter protocol.
    """

    def __init__(
        self,
        analysis_config: AnalysisConfig | None = None,
        signal_config: SignalConfig | None = None,
    ) -> None:
        self.analysis_config = analysis_config or AnalysisConfig()
        self.signal_config = signal_config or SignalConfig()

    def emit(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
    ) -> list[Signal]:
        """Analyze OHLCV data and generate pin bar signals."""
        analysis = analyze(df, symbol, timeframe, self.analysis_config)
        return generate_signals(analysis, df, self.signal_config)
