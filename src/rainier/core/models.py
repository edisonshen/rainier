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
# Backtest tables
# ---------------------------------------------------------------------------


class BacktestTradingLog(Base):
    """Trading log for portfolio backtests — tracks each position with entry/exit reasoning."""

    __tablename__ = "backtest_trading_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    backtest_run_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(
        String(10), ForeignKey("stocks.symbol"), nullable=False, index=True
    )
    pattern_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entry_date: Mapped[date] = mapped_column(Date, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_date: Mapped[date | None] = mapped_column(Date)
    exit_price: Mapped[float | None] = mapped_column(Float)
    shares: Mapped[float] = mapped_column(Float, nullable=False)
    allocated_amount: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss: Mapped[float] = mapped_column(Float, nullable=False)
    target_price: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    exit_reason: Mapped[str | None] = mapped_column(String(30))
    return_pct: Mapped[float | None] = mapped_column(Float)
    pnl: Mapped[float | None] = mapped_column(Float)
    qu100_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)


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
    # PR1: idempotency hash + per-call signal name list (for SQL performance queries).
    # Migration for existing databases lives in migrations/0001_llm_thesis_pr1.sql —
    # `db init` (Base.metadata.create_all) is additive only, not ALTER.
    input_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    signals_used: Mapped[list[str] | None] = mapped_column(ARRAY(String(50)))
    # PR2 carry-over [P3]: first-class session column so Tier-1 cache lookup
    # filters in the WHERE clause instead of post-filtering on a JSONB key.
    # Avoids the bug where a cross-session row hides a same-session row from
    # the SELECT (the previous query used .order_by(id desc).first() and
    # post-filtered on _session_name, which dropped the lookup to None even
    # when a same-session row existed earlier in the partition). Migration
    # in migrations/0002_llm_thesis_pr2.sql.
    session_name: Mapped[str | None] = mapped_column(String(20), index=True)


class ScreenedStockRecord(Base):
    """Per-scan screener output + LLM augmentation + outcome tracking.

    Captures every screened candidate (~20 rows × every scan, ~80/day) plus
    LLM thesis fields on the top-5 of `afternoon`/`close` scans plus manual
    outcome tracking via `rainier thesis log`.

    Plain Postgres (NOT a TimescaleDB hypertable) — ~28K rows/year is small.
    """

    __tablename__ = "screened_stocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    scan_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    session_name: Mapped[str] = mapped_column(String(20), nullable=False)

    # From screener (always populated)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    rule_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False)
    money_flow_score: Mapped[float | None] = mapped_column(Float)
    sector: Mapped[str | None] = mapped_column(String(50))
    pattern_type: Mapped[str | None] = mapped_column(String(50))
    pattern_confidence: Mapped[float | None] = mapped_column(Float)

    # From LLM (nullable — only afternoon/close top-5 get this)
    llm_confidence: Mapped[int | None] = mapped_column(Integer)
    shadow_combined_score: Mapped[float | None] = mapped_column(Float)
    would_be_combined_rank: Mapped[int | None] = mapped_column(Integer)
    thesis_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("analysis_results.id")
    )
    patterns_in_chart_not_in_indicators_count: Mapped[int | None] = mapped_column(Integer)

    # Manual outcome tracking (filled in via `rainier thesis log`)
    action_taken: Mapped[str | None] = mapped_column(String(20))
    outcome_pct: Mapped[float | None] = mapped_column(Float)
    outcome_recorded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)

    # Auto outcome backfill (PR2 — columns reserved here so PR1 schema is forward-compat)
    forward_return_5d: Mapped[float | None] = mapped_column(Float)
    forward_return_10d: Mapped[float | None] = mapped_column(Float)
    outcome_backfilled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint(
            "scan_date",
            "session_name",
            "symbol",
            name="uq_screened_stocks_scan_session_symbol",
        ),
    )


class ThesisEvaluation(Base):
    """Per-thesis forward-return outcome at a fixed horizon (PR2).

    One row per (thesis_id, horizon). Populated by the daily eval job
    (`llm_thesis.eval.evaluate_horizon`) which runs nightly at 17:00 PT.
    Idempotent: re-running fills only missing rows.

    Plain Postgres (NOT a hypertable) — small row count (5 picks/day x 3
    horizons x ~252 trading days = ~3.8K/yr).
    """

    __tablename__ = "thesis_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thesis_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("analysis_results.id"), nullable=False, index=True
    )
    screened_record_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("screened_stocks.id")
    )
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    horizon: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    scan_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    verdict: Mapped[str] = mapped_column(String(20), nullable=False)
    llm_confidence: Mapped[int | None] = mapped_column(Integer)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float] = mapped_column(Float, nullable=False)
    return_pct: Mapped[float] = mapped_column(Float, nullable=False)
    # `hit` is True when the thesis direction matched the realized return sign.
    # For setup_long: hit iff return_pct > 0. For watch / no_setup: hit iff
    # return_pct <= 0 (i.e. the LLM was right NOT to buy).
    hit: Mapped[bool] = mapped_column(Boolean, nullable=False)
    # Denormalized from LLMAnalysisRecord so per-signal contribution queries
    # avoid the join entirely.
    signals_used: Mapped[list[str] | None] = mapped_column(ARRAY(String(50)))
    notes: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint(
            "thesis_id", "horizon", name="uq_thesis_evaluations_thesis_horizon"
        ),
    )


# Tables to convert to TimescaleDB hypertables
HYPERTABLES = {
    "money_flow_snapshots": "captured_at",
    "stock_capital_flow": "flow_date",
    "capital_flow_bars": "bar_time",
    "stock_prices": "date",
    "monitor_readings": "recorded_at",
}
