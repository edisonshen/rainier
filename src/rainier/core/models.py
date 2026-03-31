"""ORM models — futures trading (3 tables) + stock money flow (7 tables)."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Futures trading tables
# ---------------------------------------------------------------------------


class CandleRecord(Base):
    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (
        Index("ix_candles_symbol_tf_ts", "symbol", "timeframe", "timestamp", unique=True),
    )


class SignalRecord(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss: Mapped[float] = mapped_column(Float, nullable=False)
    take_profit: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    rr_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_signals_symbol_ts", "symbol", "timestamp"),
    )


class TradeRecord(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    exit_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="open")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Stock money flow tables (from rainier)
# ---------------------------------------------------------------------------


class Stock(Base):
    __tablename__ = "stocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), unique=True, nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(String(255))
    sector: Mapped[str | None] = mapped_column(String(100))
    industry: Mapped[str | None] = mapped_column(String(200))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    money_flow_snapshots: Mapped[list[MoneyFlowSnapshot]] = relationship(
        back_populates="stock"
    )
    capital_flows: Mapped[list[StockCapitalFlow]] = relationship(back_populates="stock")
    capital_flow_bars: Mapped[list[CapitalFlowBar]] = relationship(back_populates="stock")
    prices: Mapped[list[StockPrice]] = relationship(back_populates="stock")
    chart_images: Mapped[list[ChartImage]] = relationship(back_populates="stock")


class MoneyFlowSnapshot(Base):
    __tablename__ = "money_flow_snapshots"
    # Logical key: (data_date, ranking_type, rank) — one stock per rank per day.
    # TimescaleDB hypertable requires captured_at in unique constraints,
    # so upsert logic in scraper.py enforces this at the application level.
    __table_args__ = (PrimaryKeyConstraint("id", "captured_at"),)

    id: Mapped[int] = mapped_column(BigInteger, autoincrement=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    capture_session: Mapped[str] = mapped_column(String(20), nullable=False)
    data_date: Mapped[date] = mapped_column(Date, nullable=False)
    view_type: Mapped[str] = mapped_column(String(10), nullable=False, server_default="daily")
    ranking_type: Mapped[str] = mapped_column(String(10), nullable=False)
    symbol: Mapped[str] = mapped_column(
        String(10), ForeignKey("stocks.symbol"), nullable=False, index=True
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    daily_change: Mapped[int | None] = mapped_column(Integer)
    sector: Mapped[str | None] = mapped_column(String(100))
    industry: Mapped[str | None] = mapped_column(String(200))
    long_short: Mapped[str | None] = mapped_column(String(50))
    raw_data: Mapped[dict | None] = mapped_column(JSONB)

    stock: Mapped[Stock] = relationship(
        back_populates="money_flow_snapshots",
        foreign_keys=[symbol],
        primaryjoin="MoneyFlowSnapshot.symbol == Stock.symbol",
    )


class StockCapitalFlow(Base):
    __tablename__ = "stock_capital_flow"
    __table_args__ = (PrimaryKeyConstraint("id", "flow_date"),)

    id: Mapped[int] = mapped_column(BigInteger, autoincrement=True)
    symbol: Mapped[str] = mapped_column(
        String(10), ForeignKey("stocks.symbol"), nullable=False, index=True
    )
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    flow_date: Mapped[date] = mapped_column(Date, nullable=False)
    period_type: Mapped[str] = mapped_column(String(10), nullable=False)
    week_start: Mapped[date | None] = mapped_column(Date)
    week_end: Mapped[date | None] = mapped_column(Date)
    capital_flow_direction: Mapped[str | None] = mapped_column(String(5))
    long_short: Mapped[str | None] = mapped_column(String(50))
    rank: Mapped[int | None] = mapped_column(Integer)
    rank_total: Mapped[int | None] = mapped_column(Integer)
    raw_data: Mapped[dict | None] = mapped_column(JSONB)

    stock: Mapped[Stock] = relationship(
        back_populates="capital_flows",
        foreign_keys=[symbol],
        primaryjoin="StockCapitalFlow.symbol == Stock.symbol",
    )


class CapitalFlowBar(Base):
    __tablename__ = "capital_flow_bars"
    __table_args__ = (PrimaryKeyConstraint("id", "bar_time"),)

    id: Mapped[int] = mapped_column(BigInteger, autoincrement=True)
    symbol: Mapped[str] = mapped_column(
        String(10), ForeignKey("stocks.symbol"), nullable=False, index=True
    )
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    bar_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    bar_type: Mapped[str] = mapped_column(String(10), nullable=False)
    total_flow: Mapped[float | None] = mapped_column(Float)
    near_term_flow: Mapped[float | None] = mapped_column(Float)
    raw_data: Mapped[dict | None] = mapped_column(JSONB)

    stock: Mapped[Stock] = relationship(
        back_populates="capital_flow_bars",
        foreign_keys=[symbol],
        primaryjoin="CapitalFlowBar.symbol == Stock.symbol",
    )


class StockPrice(Base):
    __tablename__ = "stock_prices"
    __table_args__ = (
        PrimaryKeyConstraint("id", "date"),
        UniqueConstraint("symbol", "date", name="uq_stock_price_symbol_date"),
    )

    id: Mapped[int] = mapped_column(BigInteger, autoincrement=True)
    symbol: Mapped[str] = mapped_column(
        String(10), ForeignKey("stocks.symbol"), nullable=False, index=True
    )
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[int | None] = mapped_column(BigInteger)

    stock: Mapped[Stock] = relationship(
        back_populates="prices",
        foreign_keys=[symbol],
        primaryjoin="StockPrice.symbol == Stock.symbol",
    )


class ChartImage(Base):
    __tablename__ = "chart_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(
        String(10), ForeignKey("stocks.symbol"), nullable=False, index=True
    )
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    timeframe_days: Mapped[int] = mapped_column(Integer, default=120)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer)

    stock: Mapped[Stock] = relationship(
        back_populates="chart_images",
        foreign_keys=[symbol],
        primaryjoin="ChartImage.symbol == Stock.symbol",
    )


# ---------------------------------------------------------------------------
# Monitor tables
# ---------------------------------------------------------------------------


class MonitorReadingRecord(Base):
    """Time-series readings from web monitors (TimescaleDB hypertable)."""

    __tablename__ = "monitor_readings"
    __table_args__ = (PrimaryKeyConstraint("id", "recorded_at"),)

    id: Mapped[int] = mapped_column(BigInteger, autoincrement=True)
    monitor_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    field_name: Mapped[str] = mapped_column(String(100), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_value: Mapped[str] = mapped_column(Text, nullable=False)
    numeric_value: Mapped[float | None] = mapped_column(Float)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)


class MonitorAlertRecord(Base):
    """Alert history from monitor checks."""

    __tablename__ = "monitor_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    monitor_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    field_name: Mapped[str] = mapped_column(String(100), nullable=False, server_default="")
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    check_details: Mapped[dict | None] = mapped_column(JSONB)
    reading_value: Mapped[str | None] = mapped_column(Text)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")


# ---------------------------------------------------------------------------
# LLM analysis tables
# ---------------------------------------------------------------------------


class LLMAnalysisRecord(Base):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    llm_provider: Mapped[str | None] = mapped_column(String(50))
    llm_model: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_template: Mapped[str] = mapped_column(String(100), nullable=False)
    money_flow_snapshot_ids: Mapped[list[int] | None] = mapped_column(ARRAY(BigInteger))
    chart_image_ids: Mapped[list[int] | None] = mapped_column(ARRAY(Integer))
    recommendation: Mapped[str | None] = mapped_column(String(10))
    confidence: Mapped[float | None] = mapped_column(Float)
    target_symbols: Mapped[list[str] | None] = mapped_column(ARRAY(String(10)))
    reasoning: Mapped[str | None] = mapped_column(Text)
    structured_output: Mapped[dict | None] = mapped_column(JSONB)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    total_cost_usd: Mapped[float | None] = mapped_column(Float)


# Tables to convert to TimescaleDB hypertables
HYPERTABLES = {
    "money_flow_snapshots": "captured_at",
    "stock_capital_flow": "flow_date",
    "capital_flow_bars": "bar_time",
    "stock_prices": "date",
    "monitor_readings": "recorded_at",
}
