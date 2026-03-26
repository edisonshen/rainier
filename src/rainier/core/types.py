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


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"


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


@dataclass(frozen=True, slots=True)
class StockCandidate:
    """A screened QU100 stock candidate for notification."""
    symbol: str
    rank: int                       # QU100 rank (1-100)
    rank_change: int                # Daily rank change
    long_short: str                 # "Long in" / "Short in"
    capital_flow_direction: str     # "+", "-", "N"
    sector: str
    signal_strength: float          # 0-1 composite from money flow screener

    # Pattern data (from 蔡森 pattern detection)
    pattern_type: str | None = None        # "w_bottom", "bull_flag", etc.
    pattern_direction: str | None = None   # "bullish" / "bearish"
    pattern_status: str | None = None      # "forming" / "confirmed"
    pattern_confidence: float | None = None
    entry_price: float | None = None
    stop_loss: float | None = None
    target_price: float | None = None
    rr_ratio: float | None = None
    volume_confirmed: bool = False

    # Actionability context (today's price vs pattern levels)
    current_price: float | None = None
    distance_to_entry_pct: float | None = None  # % from current price to entry
    bars_since_breakout: int | None = None       # how many bars since confirmation


@dataclass(frozen=True, slots=True)
class PatternSignal:
    """A detected chart pattern from 蔡森 methodology."""
    symbol: str
    pattern_type: str           # "w_bottom", "false_breakdown", "bull_flag", etc.
    direction: str              # "bullish" or "bearish"
    status: str                 # "forming", "confirmed", "target_reached"
    confidence: float           # 0-1

    # Key price levels
    entry_price: float
    stop_loss: float
    target_wave1: float
    target_wave2: float | None = None
    risk_pct: float = 0.0      # (entry - SL) / entry as percentage
    reward_pct: float = 0.0    # (target - entry) / entry as percentage
    rr_ratio: float = 0.0      # reward / risk

    # Pattern components
    neckline: float = 0.0
    key_points: dict | None = None
    volume_confirmed: bool = False

    # Timestamps
    pattern_start_idx: int = 0
    pattern_end_idx: int | None = None
    breakout_idx: int | None = None


@dataclass(frozen=True, slots=True)
class MoneyFlowSignal:
    """A QU100 stock with money flow scoring."""
    symbol: str
    stock_id: int
    rank: int
    rank_change: int
    long_short: str
    capital_flow_direction: str
    days_in_top100: int
    sector: str
    industry: str
    signal_strength: float      # 0-1 composite score


@dataclass(frozen=True, slots=True)
class SectorTrend:
    """Sector-level trend from QU100 data."""
    sector: str
    long_in_count: int
    short_in_count: int
    net_sentiment: float        # (long - short) / total
    top_stocks: list[str]
    trend_direction: str        # "bullish", "bearish", "neutral"
    sector_rank: int


@dataclass(slots=True)
class StockScreenResult:
    """Full screening result for a single stock."""
    symbol: str
    name: str
    sector: str

    # Layer 1: Money flow
    money_flow_score: float
    long_short: str
    qu100_rank: int

    # Layer 2: Sector
    sector_trend: str
    sector_boost: float

    # Layer 3: Technical
    patterns: list[PatternSignal]
    best_pattern: PatternSignal | None

    # Composite
    composite_score: float
    recommendation: str         # "strong_buy", "buy", "watch", "avoid"

    # Action
    entry_price: float | None = None
    stop_loss: float | None = None
    target: float | None = None
    risk_pct: float | None = None


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
