"""ML-augmented signal emitter — uses a trained model for confidence scoring.

Replaces the rule-based score_setup() with an ML model's probability output
while reusing the same pin bar detection and S/R level logic.

Dependency injection: receives ScoringStrategy and FeatureExtractorProtocol
via constructor so that signals/ does not import ml/ or features/ directly.
"""

from __future__ import annotations

import pandas as pd

from rainier.analysis.analyzer import analyze
from rainier.core.config import AnalysisConfig, SignalConfig
from rainier.core.protocols import FeatureExtractorProtocol, ScoringStrategy
from rainier.core.types import (
    Direction,
    PatternSignal,
    PinBar,
    Signal,
    SignalStatus,
    Timeframe,
)
from rainier.signals.generator import _compute_levels, _dedup_signals


class MLSignalEmitter:
    """Emits signals using ML model confidence instead of rule-based scoring.

    Same pin bar detection as PinBarSignalEmitter, but replaces the
    weighted sub-score with ScoringStrategy.score() (typically MLScorer).
    """

    def __init__(
        self,
        scorer: ScoringStrategy,
        feature_extractor: FeatureExtractorProtocol,
        analysis_config: AnalysisConfig | None = None,
        signal_config: SignalConfig | None = None,
    ) -> None:
        self.scorer = scorer
        self.feature_extractor = feature_extractor
        self.analysis_config = analysis_config or AnalysisConfig()
        self.signal_config = signal_config or SignalConfig()

    def emit(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
    ) -> list[Signal]:
        """Analyze OHLCV data and generate ML-scored pin bar signals."""
        config = self.signal_config

        # Step 1: Run analysis pipeline (same as PinBarSignalEmitter)
        analysis = analyze(df, symbol, timeframe, self.analysis_config)

        # Step 2: Extract features for the entire lookback window once
        features = self.feature_extractor.extract(analysis, df)

        # Step 3: Score each pin bar with ML model
        all_signals: list[Signal] = []

        for pin_bar in analysis.pin_bars:
            # Compute entry/SL/TP levels
            entry, sl, tp = _compute_levels(pin_bar, analysis.sr_levels, config)
            if entry is None:
                continue

            # Check minimum R:R
            risk = abs(entry - sl)
            if risk <= 0:
                continue
            rr = abs(tp - entry) / risk
            if rr < config.min_rr_ratio:
                continue

            # Build PatternSignal wrapper for the scorer
            pattern = _pinbar_to_pattern_signal(pin_bar, entry, sl, tp, rr)

            # Slice features up to pin bar's bar index
            bar_idx = pin_bar.index
            if bar_idx < len(features):
                features_slice = features.iloc[: bar_idx + 1]
            else:
                features_slice = features

            # Score with ML model
            confidence = self.scorer.score(pattern, features_slice)

            if confidence < config.scorer.min_confidence:
                continue

            all_signals.append(
                Signal(
                    symbol=analysis.symbol,
                    timeframe=analysis.timeframe,
                    direction=pin_bar.direction,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    confidence=confidence,
                    timestamp=pin_bar.candle.timestamp,
                    status=SignalStatus.PENDING,
                    pin_bar=pin_bar,
                    sr_level=pin_bar.nearest_sr,
                )
            )

        return _dedup_signals(all_signals)


def _pinbar_to_pattern_signal(
    pin_bar: PinBar,
    entry: float,
    stop_loss: float,
    take_profit: float,
    rr_ratio: float,
) -> PatternSignal:
    """Convert a PinBar + computed levels into a PatternSignal for the scorer."""
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)

    return PatternSignal(
        symbol="",
        pattern_type="pin_bar",
        direction="bullish" if pin_bar.direction == Direction.LONG else "bearish",
        status="confirmed",
        confidence=0.0,  # will be set by scorer
        entry_price=entry,
        stop_loss=stop_loss,
        target_wave1=take_profit,
        risk_pct=risk / entry if entry > 0 else 0.0,
        reward_pct=reward / entry if entry > 0 else 0.0,
        rr_ratio=rr_ratio,
        volume_confirmed=pin_bar.candle.volume > 0 if pin_bar.candle.volume else False,
        pattern_start_idx=pin_bar.index,
    )
