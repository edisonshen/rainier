"""Tests for PinBarSignalEmitter — the pin bar SignalEmitter adapter."""

from datetime import datetime

from rainier.core.config import AnalysisConfig, SignalConfig
from rainier.core.protocols import SignalEmitter
from rainier.core.types import (
    AnalysisResult,
    Direction,
    Signal,
    SignalStatus,
    Timeframe,
)
from rainier.signals.emitter import PinBarSignalEmitter


class TestPinBarSignalEmitter:
    def test_satisfies_signal_emitter_protocol(self):
        assert isinstance(PinBarSignalEmitter(), SignalEmitter)

    def test_default_configs_created(self):
        emitter = PinBarSignalEmitter()
        assert isinstance(emitter.analysis_config, AnalysisConfig)
        assert isinstance(emitter.signal_config, SignalConfig)

    def test_custom_configs_stored(self):
        a_cfg = AnalysisConfig(max_sr_levels=7)
        s_cfg = SignalConfig(min_rr_ratio=3.0)
        emitter = PinBarSignalEmitter(analysis_config=a_cfg, signal_config=s_cfg)
        assert emitter.analysis_config is a_cfg
        assert emitter.signal_config is s_cfg

    def test_emit_on_flat_data_returns_empty_list(self, flat_candles):
        emitter = PinBarSignalEmitter()
        signals = emitter.emit(flat_candles, "NQ", Timeframe.H1)
        assert signals == []

    def test_emit_wires_analyze_and_generate_signals(self, monkeypatch, pin_bar_candles):
        """emit() must pass df/symbol/timeframe + configs through and return the signals."""
        import rainier.signals.emitter as emitter_mod

        a_cfg = AnalysisConfig()
        s_cfg = SignalConfig()
        emitter = PinBarSignalEmitter(analysis_config=a_cfg, signal_config=s_cfg)

        fake_result = AnalysisResult(symbol="NQ", timeframe=Timeframe.H1)
        fake_signal = Signal(
            symbol="NQ", timeframe=Timeframe.H1, direction=Direction.LONG,
            entry_price=100.0, stop_loss=99.0, take_profit=102.0,
            confidence=0.8, timestamp=datetime(2025, 1, 1),
            status=SignalStatus.PENDING,
        )
        calls = {}

        def fake_analyze(df, symbol, timeframe, config):
            calls["analyze"] = (df, symbol, timeframe, config)
            return fake_result

        def fake_generate(analysis, df, config):
            calls["generate"] = (analysis, df, config)
            return [fake_signal]

        monkeypatch.setattr(emitter_mod, "analyze", fake_analyze)
        monkeypatch.setattr(emitter_mod, "generate_signals", fake_generate)

        signals = emitter.emit(pin_bar_candles, "NQ", Timeframe.H1)

        assert signals == [fake_signal]
        assert calls["analyze"] == (pin_bar_candles, "NQ", Timeframe.H1, a_cfg)
        assert calls["generate"] == (fake_result, pin_bar_candles, s_cfg)

    def test_emit_produces_only_signal_instances(self, pin_bar_candles):
        emitter = PinBarSignalEmitter()
        signals = emitter.emit(pin_bar_candles, "NQ", Timeframe.H1)
        assert isinstance(signals, list)
        for sig in signals:
            assert isinstance(sig, Signal)
            assert sig.symbol == "NQ"
            assert sig.timeframe == Timeframe.H1
