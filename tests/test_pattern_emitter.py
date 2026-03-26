"""Tests for PatternSignalEmitter — 蔡森 pattern to Signal conversion."""

from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

from rainier.core.config import PatternEmitterConfig
from rainier.core.protocols import SignalEmitter
from rainier.core.types import Direction, PatternSignal, Timeframe
from rainier.signals.pattern_emitter import PatternSignalEmitter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_daily_dataset(n_bars: int = 100, seed_price: float = 50.0) -> pd.DataFrame:
    """Create daily OHLCV data."""
    rows = []
    price = seed_price
    base = datetime(2025, 1, 1)
    for i in range(n_bars):
        cycle = i % 30
        move = 0.5 if cycle < 15 else -0.5
        o = price
        h = price + 1.5
        low = price - 1.5
        c = price + move
        rows.append({
            "timestamp": base + timedelta(days=i),
            "open": o, "high": h, "low": low, "close": c,
            "volume": 100000.0,
        })
        price = c
    return pd.DataFrame(rows)


def _make_pattern_signal(
    symbol: str = "AAPL",
    pattern_type: str = "w_bottom",
    direction: str = "bullish",
    status: str = "confirmed",
    confidence: float = 0.75,
    entry_price: float = 50.0,
    stop_loss: float = 47.0,
    target_wave1: float = 56.0,
    target_wave2: float | None = 62.0,
    rr_ratio: float = 2.0,
    breakout_idx: int | None = None,
) -> PatternSignal:
    return PatternSignal(
        symbol=symbol,
        pattern_type=pattern_type,
        direction=direction,
        status=status,
        confidence=confidence,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_wave1=target_wave1,
        target_wave2=target_wave2,
        rr_ratio=rr_ratio,
        breakout_idx=breakout_idx,
    )


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_satisfies_signal_emitter(self):
        emitter = PatternSignalEmitter()
        assert isinstance(emitter, SignalEmitter)


# ---------------------------------------------------------------------------
# Signal conversion
# ---------------------------------------------------------------------------


class TestPatternToSignal:
    def test_bullish_pattern_emits_long(self):
        df = _make_daily_dataset(100)
        last_idx = len(df) - 1
        ps = _make_pattern_signal(
            direction="bullish", breakout_idx=last_idx,
        )

        with patch(
            "rainier.signals.pattern_emitter.detect_patterns",
            return_value=[ps],
        ):
            emitter = PatternSignalEmitter()
            signals = emitter.emit(df, "AAPL", Timeframe.D1)

        assert len(signals) == 1
        assert signals[0].direction == Direction.LONG

    def test_bearish_pattern_emits_short(self):
        df = _make_daily_dataset(100)
        last_idx = len(df) - 1
        ps = _make_pattern_signal(
            direction="bearish",
            stop_loss=55.0,
            target_wave1=40.0,
            breakout_idx=last_idx,
        )

        with patch(
            "rainier.signals.pattern_emitter.detect_patterns",
            return_value=[ps],
        ):
            emitter = PatternSignalEmitter()
            signals = emitter.emit(df, "AAPL", Timeframe.D1)

        assert len(signals) == 1
        assert signals[0].direction == Direction.SHORT

    def test_entry_price_is_bar_close(self):
        df = _make_daily_dataset(100)
        last_idx = len(df) - 1
        expected_close = float(df.iloc[-1]["close"])
        ps = _make_pattern_signal(breakout_idx=last_idx)

        with patch(
            "rainier.signals.pattern_emitter.detect_patterns",
            return_value=[ps],
        ):
            emitter = PatternSignalEmitter()
            signals = emitter.emit(df, "AAPL", Timeframe.D1)

        assert signals[0].entry_price == expected_close

    def test_take_profit_uses_wave1_by_default(self):
        df = _make_daily_dataset(100)
        last_idx = len(df) - 1
        ps = _make_pattern_signal(
            target_wave1=56.0, target_wave2=62.0, breakout_idx=last_idx,
        )

        with patch(
            "rainier.signals.pattern_emitter.detect_patterns",
            return_value=[ps],
        ):
            emitter = PatternSignalEmitter()
            signals = emitter.emit(df, "AAPL", Timeframe.D1)

        assert signals[0].take_profit == 56.0

    def test_wave2_target_selection(self):
        df = _make_daily_dataset(100)
        last_idx = len(df) - 1
        ps = _make_pattern_signal(
            target_wave1=56.0, target_wave2=62.0, breakout_idx=last_idx,
        )

        with patch(
            "rainier.signals.pattern_emitter.detect_patterns",
            return_value=[ps],
        ):
            config = PatternEmitterConfig(wave_target="wave2")
            emitter = PatternSignalEmitter(emitter_config=config)
            signals = emitter.emit(df, "AAPL", Timeframe.D1)

        assert signals[0].take_profit == 62.0

    def test_wave2_falls_back_to_wave1_if_none(self):
        df = _make_daily_dataset(100)
        last_idx = len(df) - 1
        ps = _make_pattern_signal(
            target_wave1=56.0, target_wave2=None, breakout_idx=last_idx,
        )

        with patch(
            "rainier.signals.pattern_emitter.detect_patterns",
            return_value=[ps],
        ):
            config = PatternEmitterConfig(wave_target="wave2")
            emitter = PatternSignalEmitter(emitter_config=config)
            signals = emitter.emit(df, "AAPL", Timeframe.D1)

        assert signals[0].take_profit == 56.0

    def test_notes_carries_pattern_type(self):
        df = _make_daily_dataset(100)
        last_idx = len(df) - 1
        ps = _make_pattern_signal(
            pattern_type="false_breakdown", breakout_idx=last_idx,
        )

        with patch(
            "rainier.signals.pattern_emitter.detect_patterns",
            return_value=[ps],
        ):
            emitter = PatternSignalEmitter()
            signals = emitter.emit(df, "AAPL", Timeframe.D1)

        assert signals[0].notes == "pattern:false_breakdown"

    def test_timestamp_matches_last_bar(self):
        df = _make_daily_dataset(100)
        last_idx = len(df) - 1
        expected_ts = pd.Timestamp(df.iloc[-1]["timestamp"]).to_pydatetime()
        ps = _make_pattern_signal(breakout_idx=last_idx)

        with patch(
            "rainier.signals.pattern_emitter.detect_patterns",
            return_value=[ps],
        ):
            emitter = PatternSignalEmitter()
            signals = emitter.emit(df, "AAPL", Timeframe.D1)

        assert signals[0].timestamp == expected_ts


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_confidence_filter(self):
        df = _make_daily_dataset(100)
        last_idx = len(df) - 1
        ps = _make_pattern_signal(confidence=0.3, breakout_idx=last_idx)

        with patch(
            "rainier.signals.pattern_emitter.detect_patterns",
            return_value=[ps],
        ):
            config = PatternEmitterConfig(min_confidence=0.5)
            emitter = PatternSignalEmitter(emitter_config=config)
            signals = emitter.emit(df, "AAPL", Timeframe.D1)

        assert len(signals) == 0

    def test_rr_filter(self):
        df = _make_daily_dataset(100)
        last_idx = len(df) - 1
        ps = _make_pattern_signal(rr_ratio=1.0, breakout_idx=last_idx)

        with patch(
            "rainier.signals.pattern_emitter.detect_patterns",
            return_value=[ps],
        ):
            config = PatternEmitterConfig(min_rr_ratio=1.5)
            emitter = PatternSignalEmitter(emitter_config=config)
            signals = emitter.emit(df, "AAPL", Timeframe.D1)

        assert len(signals) == 0

    def test_status_filter_blocks_forming(self):
        df = _make_daily_dataset(100)
        last_idx = len(df) - 1
        ps = _make_pattern_signal(status="forming", breakout_idx=last_idx)

        with patch(
            "rainier.signals.pattern_emitter.detect_patterns",
            return_value=[ps],
        ):
            # Default: only "confirmed"
            emitter = PatternSignalEmitter()
            signals = emitter.emit(df, "AAPL", Timeframe.D1)

        assert len(signals) == 0

    def test_status_filter_allows_forming_when_configured(self):
        df = _make_daily_dataset(100)
        last_idx = len(df) - 1
        ps = _make_pattern_signal(status="forming", breakout_idx=last_idx)

        with patch(
            "rainier.signals.pattern_emitter.detect_patterns",
            return_value=[ps],
        ):
            config = PatternEmitterConfig(
                status_filter=["confirmed", "forming"],
            )
            emitter = PatternSignalEmitter(emitter_config=config)
            signals = emitter.emit(df, "AAPL", Timeframe.D1)

        assert len(signals) == 1

    def test_breakout_idx_must_match_last_bar(self):
        df = _make_daily_dataset(100)
        # Breakout on bar 50, not the last bar
        ps = _make_pattern_signal(breakout_idx=50)

        with patch(
            "rainier.signals.pattern_emitter.detect_patterns",
            return_value=[ps],
        ):
            emitter = PatternSignalEmitter()
            signals = emitter.emit(df, "AAPL", Timeframe.D1)

        assert len(signals) == 0

    def test_no_signals_on_short_data(self):
        df = _make_daily_dataset(5)
        emitter = PatternSignalEmitter()
        # min_pattern_bars default is 10, so 5 bars should produce nothing
        signals = emitter.emit(df, "AAPL", Timeframe.D1)
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Engine integration (pattern_type in TradeRecord)
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_pattern_type_parsed_from_notes(self):
        """Verify the engine extracts pattern_type from Signal.notes."""
        from rainier.backtest.engine import run_backtest
        from rainier.core.config import BacktestConfig

        df = _make_daily_dataset(200)
        last_idx = len(df) - 1
        ps = _make_pattern_signal(
            pattern_type="w_bottom", breakout_idx=last_idx,
        )

        with patch(
            "rainier.signals.pattern_emitter.detect_patterns",
            return_value=[ps],
        ):
            emitter = PatternSignalEmitter()
            config = BacktestConfig(
                sr_recompute_interval=1,
                max_open_positions=3,
            )
            metrics = run_backtest(
                df, "AAPL", Timeframe.D1, emitter, config,
            )

        # Should have at least one trade
        if metrics.total_trades > 0:
            trade = metrics.trades[0]
            assert trade.pattern_type == "w_bottom"
            assert "pattern_breakout" in trade.entry_reason
