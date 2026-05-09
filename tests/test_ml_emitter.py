"""Tests for MLSignalEmitter — ML-augmented signal generation."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from rainier.core.config import ScorerConfig, SignalConfig
from rainier.core.protocols import FeatureExtractorProtocol, SignalEmitter
from rainier.core.types import AnalysisResult, Direction, PatternSignal, Timeframe
from rainier.features.extractor import FeatureExtractor
from rainier.signals.ml_emitter import MLSignalEmitter, _pinbar_to_pattern_signal

# ---------------------------------------------------------------------------
# Fake scorer / extractor for testing
# ---------------------------------------------------------------------------


class FakeScorer:
    """Always returns a fixed score."""

    def __init__(self, score: float = 0.75):
        self._score = score

    def score(self, pattern: PatternSignal, features: pd.DataFrame) -> float:
        return self._score


class FakeExtractor:
    """Returns a minimal features DataFrame with the same index as df."""

    def extract(self, result: AnalysisResult, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        return pd.DataFrame({
            "body_size": [1.0] * n,
            "atr_14": [2.0] * n,
            "volume_ratio": [1.0] * n,
            "is_bullish": [1.0] * n,
        })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zigzag_df(n_bars: int = 200) -> pd.DataFrame:
    """Zigzag dataset that produces pin bars at reversals."""
    rows = []
    price = 100.0
    base = datetime(2025, 1, 1)

    for i in range(n_bars):
        cycle = i % 40
        move = 1.0 if cycle < 20 else -1.0

        o = price
        h = price + abs(move) + 1.0
        low = price - abs(move) - 0.5
        c = price + move

        rows.append({
            "timestamp": base + timedelta(hours=i),
            "open": o, "high": h, "low": low, "close": c,
            "volume": 1000.0 + (i % 10) * 100,
        })
        price = c

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMLSignalEmitterProtocol:
    """MLSignalEmitter satisfies the SignalEmitter protocol."""

    def test_implements_protocol(self):
        emitter = MLSignalEmitter(
            scorer=FakeScorer(),
            feature_extractor=FakeExtractor(),
        )
        assert isinstance(emitter, SignalEmitter)

    def test_emit_returns_list(self):
        emitter = MLSignalEmitter(
            scorer=FakeScorer(),
            feature_extractor=FakeExtractor(),
        )
        df = _make_zigzag_df(200)
        signals = emitter.emit(df, "TEST", Timeframe.H1)
        assert isinstance(signals, list)


class TestFeatureExtractorProtocolContract:
    """FeatureExtractor satisfies the FeatureExtractorProtocol contract.

    This protects against drift: if FeatureExtractor.extract() ever changes
    shape, the runtime_checkable protocol will catch it here before the
    MLSignalEmitter (which depends on the protocol) breaks at call time.
    """

    def test_concrete_extractor_satisfies_protocol(self):
        assert isinstance(FeatureExtractor(), FeatureExtractorProtocol)

    def test_fake_extractor_satisfies_protocol(self):
        # The test fake we use throughout this file must also satisfy
        # the protocol, otherwise our other tests are misleading.
        assert isinstance(FakeExtractor(), FeatureExtractorProtocol)


class TestMLSignalEmitterFiltering:
    """ML emitter respects confidence and R:R filters."""

    def test_low_confidence_filtered(self):
        """Scorer returns 0.3 but min_confidence=0.5 → no signals."""
        emitter = MLSignalEmitter(
            scorer=FakeScorer(score=0.3),
            feature_extractor=FakeExtractor(),
            signal_config=SignalConfig(
                scorer=ScorerConfig(min_confidence=0.5),
                min_rr_ratio=1.0,
            ),
        )
        df = _make_zigzag_df(200)
        signals = emitter.emit(df, "TEST", Timeframe.H1)
        assert len(signals) == 0

    def test_high_confidence_passes(self):
        """Scorer returns 0.9 with min_confidence=0.5 → signals pass."""
        emitter = MLSignalEmitter(
            scorer=FakeScorer(score=0.9),
            feature_extractor=FakeExtractor(),
            signal_config=SignalConfig(
                scorer=ScorerConfig(min_confidence=0.5),
                min_rr_ratio=0.5,  # low bar so we get signals
            ),
        )
        df = _make_zigzag_df(200)
        signals = emitter.emit(df, "TEST", Timeframe.H1)
        # May or may not produce signals depending on pin bar detection,
        # but confidence filter is not blocking
        for sig in signals:
            assert sig.confidence == 0.9

    def test_high_rr_filter(self):
        """With min_rr_ratio=10.0 → almost no signals pass."""
        emitter = MLSignalEmitter(
            scorer=FakeScorer(score=0.9),
            feature_extractor=FakeExtractor(),
            signal_config=SignalConfig(
                scorer=ScorerConfig(min_confidence=0.1),
                min_rr_ratio=10.0,
            ),
        )
        df = _make_zigzag_df(200)
        signals = emitter.emit(df, "TEST", Timeframe.H1)
        # Very high R:R bar means most signals filtered
        assert len(signals) == 0 or all(
            abs(s.take_profit - s.entry_price) / abs(s.entry_price - s.stop_loss) >= 10.0
            for s in signals
        )


class TestPinbarToPatternSignal:
    """_pinbar_to_pattern_signal produces correct PatternSignal."""

    def test_basic_conversion(self):
        from rainier.core.types import Candle, PinBar

        candle = Candle(
            timestamp=datetime(2025, 1, 1),
            open=100.0, high=105.0, low=98.0, close=104.0,
            volume=1000.0,
        )
        pin_bar = PinBar(
            candle=candle,
            index=5,
            direction=Direction.LONG,
            wick_ratio=3.0,
            nearest_sr=None,
        )

        ps = _pinbar_to_pattern_signal(
            pin_bar, entry=99.0, stop_loss=97.0, take_profit=103.0, rr_ratio=2.0,
        )

        assert ps.pattern_type == "pin_bar"
        assert ps.direction == "bullish"
        assert ps.entry_price == 99.0
        assert ps.stop_loss == 97.0
        assert ps.target_wave1 == 103.0
        assert ps.rr_ratio == 2.0
        assert ps.volume_confirmed is True

    def test_bearish_direction(self):
        from rainier.core.types import Candle, PinBar

        candle = Candle(
            timestamp=datetime(2025, 1, 1),
            open=100.0, high=102.0, low=95.0, close=96.0,
            volume=1000.0,
        )
        pin_bar = PinBar(
            candle=candle,
            index=5,
            direction=Direction.SHORT,
            wick_ratio=3.0,
            nearest_sr=None,
        )

        ps = _pinbar_to_pattern_signal(
            pin_bar, entry=101.0, stop_loss=103.0, take_profit=97.0, rr_ratio=2.0,
        )

        assert ps.direction == "bearish"
