"""Core dataclasses and enums used at all module boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1H"
    H4 = "4H"
    D1 = "1D"
    W1 = "1W"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class SRType(str, Enum):
    HORIZONTAL = "horizontal"
    DIAGONAL = "diagonal"


class SRRole(str, Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"


class SignalStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    STOPPED = "stopped"
    TARGET_HIT = "target_hit"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    timeframe: Timeframe = Timeframe.H1

    @property
    def body_top(self) -> float:
        return max(self.open, self.close)

    @property
    def body_bottom(self) -> float:
        return min(self.open, self.close)

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def upper_wick(self) -> float:
        return self.high - self.body_top

    @property
    def lower_wick(self) -> float:
        return self.body_bottom - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


@dataclass(frozen=True, slots=True)
class Pivot:
    index: int
    price: float
    timestamp: datetime
    is_high: bool


@dataclass(slots=True)
class SRLevel:
    price: float
    sr_type: SRType
    role: SRRole
    strength: float  # 0.0 - 1.0
    touches: int = 0
    first_seen: datetime | None = None
    last_tested: datetime | None = None
    # Diagonal-specific
    slope: float = 0.0  # price change per bar
    anchor_index: int = 0  # bar index where the line starts
    # Multi-TF source
    source_tf: Timeframe | None = None  # which timeframe this level was derived from

    def price_at(self, bar_index: int) -> float:
        """Get the price of this level at a given bar index (for diagonals)."""
        if self.sr_type == SRType.HORIZONTAL:
            return self.price
        return self.price + self.slope * (bar_index - self.anchor_index)


@dataclass(frozen=True, slots=True)
class PinBar:
    candle: Candle
    index: int
    direction: Direction  # LONG = bullish pin bar (long lower wick), SHORT = bearish
    wick_ratio: float  # dominant wick / body
    nearest_sr: SRLevel | None = None
    sr_distance_pct: float = 0.0  # distance to nearest S/R as % of price


@dataclass(frozen=True, slots=True)
class InsideBar:
    candle: Candle
    index: int
    mother_candle: Candle
    mother_index: int
    compression_ratio: float  # inside bar range / mother bar range


@dataclass(slots=True)
class Signal:
    symbol: str
    timeframe: Timeframe
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0.0 - 1.0
    timestamp: datetime
    status: SignalStatus = SignalStatus.PENDING
    rr_ratio: float = 0.0
    pin_bar: PinBar | None = None
    sr_level: SRLevel | None = None
    notes: str = ""

    def __post_init__(self):
        risk = abs(self.entry_price - self.stop_loss)
        if risk > 0:
            self.rr_ratio = abs(self.take_profit - self.entry_price) / risk


@dataclass(slots=True)
class AnalysisResult:
    symbol: str
    timeframe: Timeframe
    candles: list[Candle] = field(default_factory=list)
    pivots: list[Pivot] = field(default_factory=list)
    sr_levels: list[SRLevel] = field(default_factory=list)
    pin_bars: list[PinBar] = field(default_factory=list)
    inside_bars: list[InsideBar] = field(default_factory=list)
    bias: Direction | None = None
    signals: list[Signal] = field(default_factory=list)
