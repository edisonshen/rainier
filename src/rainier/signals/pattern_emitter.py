"""PatternSignalEmitter — adapts 蔡森 pattern detection to SignalEmitter protocol.

Detects chart patterns (W底, M头, 破底翻, etc.) on daily OHLCV data and converts
PatternSignal -> Signal for the backtest engine. Entry at bar close price (market
order at close, modeling last-10-minutes execution).
"""

from __future__ import annotations

import pandas as pd

from rainier.analysis.stock_patterns import detect_patterns
from rainier.core.config import PatternEmitterConfig, StockScreenerConfig
from rainier.core.types import Direction, PatternSignal, Signal, Timeframe


class PatternSignalEmitter:
    """Emits signals from 蔡森 chart pattern detection.

    Trading style: daily timeframe, enter at market close.
    Converts PatternSignal -> Signal with entry_price = bar's close price.
    """

    def __init__(
        self,
        screener_config: StockScreenerConfig | None = None,
        emitter_config: PatternEmitterConfig | None = None,
    ) -> None:
        self.screener_config = screener_config or StockScreenerConfig()
        self.config = emitter_config or PatternEmitterConfig()

    def emit(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
    ) -> list[Signal]:
        """Detect patterns and convert to signals.

        Only emits patterns whose breakout_idx matches the last bar
        (current bar), ensuring we don't re-emit old patterns.
        """
        if len(df) < self.screener_config.min_pattern_bars:
            return []

        patterns = detect_patterns(symbol, df, self.screener_config)

        last_idx = len(df) - 1
        last_bar = df.iloc[-1]
        timestamp = pd.Timestamp(last_bar["timestamp"]).to_pydatetime()
        close_price = float(last_bar["close"])

        signals: list[Signal] = []
        for ps in patterns:
            # Filter: only patterns breaking out on current bar
            if ps.breakout_idx != last_idx:
                continue

            # Filter: status
            if ps.status not in self.config.status_filter:
                continue

            # Filter: confidence
            if ps.confidence < self.config.min_confidence:
                continue

            # Filter: R:R ratio
            if ps.rr_ratio < self.config.min_rr_ratio:
                continue

            signal = self._convert(ps, symbol, timeframe, timestamp, close_price)
            if signal is not None:
                signals.append(signal)

        return signals

    def _convert(
        self,
        ps: PatternSignal,
        symbol: str,
        timeframe: Timeframe,
        timestamp: object,
        close_price: float,
    ) -> Signal | None:
        """Convert PatternSignal -> Signal."""
        # Direction mapping
        if ps.direction == "bullish":
            direction = Direction.LONG
        elif ps.direction == "bearish":
            direction = Direction.SHORT
        else:
            return None

        # Take-profit: wave2 if configured and available, else wave1
        if self.config.wave_target == "wave2" and ps.target_wave2 is not None:
            take_profit = ps.target_wave2
        else:
            take_profit = ps.target_wave1

        # Entry at close price (market order at close)
        return Signal(
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry_price=close_price,
            stop_loss=ps.stop_loss,
            take_profit=take_profit,
            confidence=ps.confidence,
            timestamp=timestamp,
            notes=f"pattern:{ps.pattern_type}",
        )
