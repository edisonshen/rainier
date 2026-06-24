"""ORM models — futures trading (3 tables) + stock money flow (7 tables)."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
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
    # Logical key: (data_date, ranking_type, symbol) — one row per stock per
    # ranking type per day. (Rank is NOT unique within a day: a carried-forward
    # symbol can share a rank with a freshly-scraped one — see _persist_qu100's
    # rebuild-with-carry-forward.) TimescaleDB hypertables require captured_at in
    # unique constraints, so scraper.py enforces this key at the application level.
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
    """Persisted chart PNG per (symbol, scan_date).

    Originally PR1's ChartImage used a `file_path` to point at chart files on disk.
    PR5 introduces inline byte storage so the Discord renderer (and dashboard)
    can serve the exact PNG that went to the LLM without depending on the
    filesystem layout. New columns are nullable for backward compatibility:
    legacy rows keep using ``file_path``; new rows fill ``image_bytes`` +
    ``sha256`` and leave ``file_path`` empty.

    The partial unique index on ``(symbol, scan_date, sha256)`` (declared in
    migrations/0004_llm_thesis_pr5.sql) makes the persist path idempotent —
    two re-runs that produce identical PNG bytes collapse to one row.

    R-D chart archive (migration 0010): ``source`` ('thesis' / 'trade_close' /
    'miss_sweep'), ``as_of_date``, and ``superseded_by`` give archive rows an
    append-only logical identity ``(symbol, as_of_date, source)``; the CURRENT
    row is the one with ``superseded_by IS NULL`` (partial unique
    ``idx_chart_images_logical``). Thesis rows stay OUTSIDE that contract:
    ``as_of_date`` is NULL for ``source='thesis'`` forever (existing backfilled
    + future persists) — NULLs never collide in the partial unique, so the
    up-to-4-renders/day thesis pattern is unaffected. Archive rows keep
    ``scan_date`` NULL (they live on ``as_of_date``) so a flip-flop re-render
    can never collide with a superseded row in the 0004 partial unique.
    """

    __tablename__ = "chart_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(
        String(10), ForeignKey("stocks.symbol"), nullable=False, index=True
    )
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    timeframe_days: Mapped[int] = mapped_column(Integer, default=120)
    # Nullable now: PR5 rows persist bytes inline and leave file_path empty.
    file_path: Mapped[str | None] = mapped_column(String(500))
    file_size_bytes: Mapped[int | None] = mapped_column(Integer)
    # PR5: store the PNG bytes that went to the LLM so Discord/dashboard can
    # serve the exact image without re-rendering or hitting the filesystem.
    image_bytes: Mapped[bytes | None] = mapped_column(LargeBinary)
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    scan_date: Mapped[date | None] = mapped_column(Date, index=True)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    # R-D archive columns (0010). source/as_of_date are nullable because the
    # 3-step supersession staging inserts them NULL and promotes in-transaction;
    # thesis rows keep as_of_date NULL permanently.
    source: Mapped[str | None] = mapped_column(String(20))
    as_of_date: Mapped[date | None] = mapped_column(Date)
    superseded_by: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("chart_images.id")
    )

    stock: Mapped[Stock] = relationship(
        back_populates="chart_images",
        foreign_keys=[symbol],
        primaryjoin="ChartImage.symbol == Stock.symbol",
    )

    __table_args__ = (
        # Partial unique index — only one PR5 row per (symbol, scan_date,
        # sha256) at a time. Legacy rows (sha256 IS NULL) are excluded so the
        # constraint never fires on PR1-vintage data.
        Index(
            "idx_chart_image_symbol_scan_sha",
            "symbol",
            "scan_date",
            "sha256",
            unique=True,
            postgresql_where="sha256 IS NOT NULL",
        ),
        # R-D logical identity (0010): one CURRENT archive row per
        # (symbol, as_of_date, source). Thesis rows (as_of_date NULL) and
        # staged rows (source NULL) never collide — NULLS DISTINCT default.
        Index(
            "idx_chart_images_logical",
            "symbol",
            "as_of_date",
            "source",
            unique=True,
            postgresql_where="superseded_by IS NULL",
        ),
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

    # Paper-tracker (PR qu100-paper-tracker-p01): the pattern-derived trade
    # levels drawn on the chart shown to the LLM. Previously dropped before
    # persistence; now stored so `paper_trade` can read them from the durable
    # screened row (Tier-1 cache-replay safety, design D4). All nullable
    # (additive migration `ADD COLUMN IF NOT EXISTS`). `pattern_type` already
    # exists above — NOT re-added.
    entry_price: Mapped[float | None] = mapped_column(Float)
    stop_loss: Mapped[float | None] = mapped_column(Float)
    target_price: Mapped[float | None] = mapped_column(Float)
    rr_ratio: Mapped[float | None] = mapped_column(Float)

    # WS B (reclaim queue, migration 0012): the trap-high invalidation level for
    # an exact `false_breakout` (the pattern's `false_high` key-point — NOT
    # `stop_loss`, which sits a buffer above it). Persisted ONLY for
    # `pattern_type == 'false_breakout'`; NULL for every other pattern (incl.
    # `false_breakout_hs`, which has no `false_high`). A later close above this
    # level invalidates the bear setup and queues the symbol for a fresh long
    # thesis. Additive (`ADD COLUMN IF NOT EXISTS`).
    bearish_invalidation_level: Mapped[float | None] = mapped_column(Float)

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


class ResearchInsight(Base):
    """One row per finding emitted by the weekly auto-research job (PR3).

    Schema (design Section F item 22 + eng review D3 + D6):
      * `kind`            — one of signal_underperform / signal_overperform /
                            verdict_drift / calibration_off / prompt_regression /
                            new_pattern_discovered.
      * `severity`        — info / warn / critical.
      * `subject`         — the entity the insight is about (signal name,
                            verdict label, prompt_version, pattern text).
      * `evidence` JSONB  — raw stats (sample size, lift, p-value, etc.).
      * `action` JSONB    — STRUCTURED `{kind, target, params}` shape so the
                            accept handler dispatches via ACTION_EXECUTORS.
                            NOT free-text. See research.py for executor map.
      * `rationale`       — human-readable narrative for Discord/UI rendering,
                            separate from the executable action.
      * `recurrence_count`— D6 dedupe: when the same (kind, subject) fires
                            again while still pending, the existing row's
                            evidence/rationale are refreshed and this counter
                            increments instead of inserting a duplicate.
      * `status`          — pending / accepted / rejected / auto_applied / stale.
      * `applied_change`  — JSONB diff written when accepted/auto_applied.

    Plain Postgres (NOT a hypertable) — small row count.

    The partial unique index `idx_research_insight_pending_kind_subject` on
    (kind, subject) WHERE status='pending' makes UPSERT a single statement.
    Created in migrations/0003_llm_thesis_pr3.sql; the Index() declaration
    here marks intent for `db init` on fresh databases.
    """

    __tablename__ = "research_insights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    kind: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(10), nullable=False)
    subject: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    evidence: Mapped[dict | None] = mapped_column(JSONB)
    action: Mapped[dict | None] = mapped_column(JSONB)
    rationale: Mapped[str | None] = mapped_column(Text)
    recurrence_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending", server_default="pending"
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decided_by: Mapped[str | None] = mapped_column(String(200))
    applied_change: Mapped[dict | None] = mapped_column(JSONB)

    __table_args__ = (
        # Partial unique index — only ONE pending row per (kind, subject) at
        # a time so emit_insight() can UPSERT in a single statement.
        Index(
            "idx_research_insight_pending_kind_subject",
            "kind",
            "subject",
            unique=True,
            postgresql_where="status = 'pending'",
        ),
    )


# ---------------------------------------------------------------------------
# Paper-trade tracker tables (design DESIGN-qu100-llm-feedback-loop §4)
# ---------------------------------------------------------------------------


class PaperTrade(Base):
    """One row per opened paper position; pending → open → closed/expired.

    PLAIN Postgres table (NOT a TimescaleDB hypertable, design D10) so we can
    freely add `UNIQUE(thesis_id)` and a partial-unique index on
    `symbol WHERE status IN ('pending','open')` (one open position per symbol)
    without the hypertable rule that every unique index include the partition
    column. Volume is tiny (~5 picks/day), like `thesis_evaluations`.

    Booked fill/exit fields are immutable once set (design D5 split-freeze):
    `price_basis` records the adjustment basis the fill was booked under so
    `evaluate_exit` walks a consistent OHLC series for an open position.
    """

    __tablename__ = "paper_trade"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Provenance / linkage
    thesis_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("analysis_results.id"), nullable=False
    )
    screened_record_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("screened_stocks.id")
    )
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    scan_date: Mapped[date] = mapped_column(Date, nullable=False)
    session_name: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    status: Mapped[str] = mapped_column(
        String(10), nullable=False, default="pending", server_default="pending"
    )

    # Plan (set at creation, from the persisted screened row). `planned_entry_price`
    # is the pattern entry level — NOT overloaded onto `entry_price` (design D4).
    planned_entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss: Mapped[float] = mapped_column(Float, nullable=False)
    target_price: Mapped[float] = mapped_column(Float, nullable=False)
    rr_ratio: Mapped[float | None] = mapped_column(Float)
    pattern_type: Mapped[str | None] = mapped_column(String(50))
    llm_confidence: Mapped[int | None] = mapped_column(Integer)
    verdict: Mapped[str | None] = mapped_column(String(20))

    # Fill (all NULL while status='pending'; set at fill time)
    entry_date: Mapped[date | None] = mapped_column(Date)
    entry_price: Mapped[float | None] = mapped_column(Float)
    shares: Mapped[int | None] = mapped_column(Integer)
    allocated_amount: Mapped[float | None] = mapped_column(Float)
    residual_cash: Mapped[float | None] = mapped_column(Float)
    # Snapshot of `learned_time_stop_days` config at fill (NULL = no time-exit,
    # design D6). Applied per-position by evaluate_exit, never the live config.
    time_stop_days: Mapped[int | None] = mapped_column(Integer)
    # Adjustment basis the fill was booked under (design D5 split-freeze).
    price_basis: Mapped[str | None] = mapped_column(String(20))

    # Exit
    exit_date: Mapped[date | None] = mapped_column(Date)
    exit_price: Mapped[float | None] = mapped_column(Float)
    exit_reason: Mapped[str | None] = mapped_column(String(20))
    return_pct: Mapped[float | None] = mapped_column(Float)
    pnl: Mapped[float | None] = mapped_column(Float)

    # R-A (design DESIGN-qu100-llm-feedback-loop Appendix B): post-exit LLM
    # post-mortem. The CHECK below is the outcome embargo at the schema level —
    # a reflection can only exist once the trade has resolved (exit_reason set).
    # Written once by paper/reflection.py (NULL-only update, idempotent); the
    # last K feed the thesis prompt's calibration section.
    reflection: Mapped[str | None] = mapped_column(Text)
    # R-D chart archive (0010): the trade's CURRENT close chart. Moves to the
    # new row on supersession; superseded charts stay readable by id forever.
    chart_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("chart_images.id")
    )

    # WS A (shadow WATCH-buy, migration 0013): a shadow row is a measurement-only
    # position opened by a WATCH verdict. It runs through the REAL fill/exit
    # engine but is EXCLUDED from every live read (book, paper_skip, calibration,
    # reports, charts, research) and never consumes a LIVE symbol's active slot
    # (the partial-unique index below is scoped to shadow=false). The live book
    # is byte-identical whether or not shadow rows exist.
    shadow: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )

    __table_args__ = (
        UniqueConstraint("thesis_id", name="uq_paper_trade_thesis_id"),
        CheckConstraint(
            "status IN ('pending','open','closed','expired')",
            name="ck_paper_trade_status",
        ),
        CheckConstraint(
            "exit_reason IS NULL OR exit_reason IN "
            "('stop_loss','target','time_stop','manual')",
            name="ck_paper_trade_exit_reason",
        ),
        # R-A outcome embargo: reflections only exist for resolved trades.
        CheckConstraint(
            "reflection IS NULL OR exit_reason IS NOT NULL",
            name="ck_paper_trade_reflection_after_exit",
        ),
        # Partial unique index — one pending/open LIVE position per symbol
        # (design D1). Scoped to shadow=false (WS A) so a shadow position never
        # consumes a live symbol's active slot. Closed/expired rows fall out, so
        # a symbol can be re-traded later.
        Index(
            "idx_paper_trade_active_symbol",
            "symbol",
            unique=True,
            postgresql_where="status IN ('pending', 'open') AND shadow = false",
        ),
        # The shadow book honors one-active-per-symbol independently (WS A).
        Index(
            "idx_paper_trade_active_symbol_shadow",
            "symbol",
            unique=True,
            postgresql_where="status IN ('pending', 'open') AND shadow = true",
        ),
    )


class PaperSkip(Base):
    """Skip ledger — keeps the feedback denominator honest (design §4).

    Written when a `setup_long` passing the confidence gate is NOT opened.
    Session/confidence-gated rejects are NOT skips (they're "not a buy signal").
    Idempotent per skipped thesis via `UNIQUE(thesis_id)`.
    """

    __tablename__ = "paper_skip"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    thesis_id: Mapped[int] = mapped_column(Integer, nullable=False)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    scan_date: Mapped[date] = mapped_column(Date, nullable=False)
    reason: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("thesis_id", name="uq_paper_skip_thesis_id"),
        CheckConstraint(
            # zero_share_price: T+1 open > $10k notional floors shares to 0
            # (Phase 3 zero-share fill guard; migration 0008 for existing DBs).
            "reason IN ('symbol_already_active','missing_levels',"
            "'missing_screened_record','gap_invalidated',"
            "'same_symbol_lower_conviction','zero_share_price')",
            name="ck_paper_skip_reason",
        ),
    )


class PaperReportSnapshot(Base):
    """Persisted, regenerable report payload keyed by (report_type, as_of_date).

    The immutable record of what a daily/weekly report said. `rainier paper
    report` re-renders from this snapshot only; `--regenerate` recomputes from
    raw inputs and upserts (design D11). Plain Postgres, small row count.
    """

    __tablename__ = "paper_report_snapshot"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    report_type: Mapped[str] = mapped_column(String(10), nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "report_type", "as_of_date", name="uq_paper_report_snapshot_type_date"
        ),
        CheckConstraint(
            "report_type IN ('daily','weekly')",
            name="ck_paper_report_snapshot_type",
        ),
    )


class PaperCalibration(Base):
    """Bounded calibration block injected into the LLM thesis prompt (D7a).

    One row per as-of date, written by the daily eval job AFTER the paper
    daily report. `payload` is a bounded JSONB blob whose HEADLINE stats are
    the UNBIASED `ThesisEvaluation` fixed-horizon (1d/5d/10d) close-to-close
    numbers (survivorship-free), with the realized paper-trade record (last
    K=10 closed + MTM-including-open) carried as labeled supplementary
    context. `build_user_message()` reads the latest row and renders it into
    the prompt so the LLM sees how its prior calls actually graded out.

    Plain Postgres, tiny row count (one per trading day). NOT a hypertable.
    """

    __tablename__ = "paper_calibration"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("as_of_date", name="uq_paper_calibration_as_of_date"),
    )


class QU100DailyFeatures(Base):
    """R-E — one JSONB feature row per QU100 member per day (design Appendix B).

    ``features`` is JSONB (``{vwap, sma5, sma22, sma44, sma60, fractal,
    volume, vrvp: {...}, price_basis, feature_version, data_gap?}``) so new
    attributes ship without a table migration; the dataset joins against
    trades and misses for learning. ``rank`` is denormalized: that day's rank
    from the latest capture of the day (same dedup rule as the appearance
    query). Written by the daily feature step (``paper/features.py``) with an
    idempotent upsert on UNIQUE(symbol, data_date, ranking_type);
    ``ranking_type`` future-proofs the key.

    Plain Postgres table, NOT a hypertable (D10): ~100 rows/day, and a
    hypertable would block the unique constraint. Created in
    migrations/0011_qu100_daily_features.sql.
    """

    __tablename__ = "qu100_daily_features"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    data_date: Mapped[date] = mapped_column(Date, nullable=False)
    ranking_type: Mapped[str] = mapped_column(
        String(10), nullable=False, default="top100", server_default="top100"
    )
    rank: Mapped[int | None] = mapped_column(Integer)
    features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "data_date",
            "ranking_type",
            name="uq_qu100_daily_features_symbol_date_ranking",
        ),
    )


class PaperReclaimQueue(Base):
    """WS B — durable queue of invalidated bull-traps awaiting re-evaluation.

    A `false_breakout` is a correct bearish setup; it declines as designed. When
    a later close reclaims the trap high (`bearish_invalidation_level`), the bear
    thesis is dead and the symbol may be a fresh LONG. The scheduled LLM run only
    sees today's top-5, so a reclaimed name outside it would be lost — this queue
    persists it. The daily check (`paper/reclaim.py`) enforces idempotency
    (UNIQUE(symbol, invalidated_thesis_id)) and once-per-K-days dedup; the
    audit gate decides whether a fresh thesis is actually spawned.

    Status lifecycle:  queued -> reenqueued -> done | expired
    Created in migrations/0012_reclaim_queue.sql.
    """

    __tablename__ = "paper_reclaim_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    invalidated_thesis_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("analysis_results.id"), nullable=False
    )
    invalidation_level: Mapped[float] = mapped_column(Float, nullable=False)
    reclaim_date: Mapped[date] = mapped_column(Date, nullable=False)
    reclaim_close: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="queued", server_default="queued",
        index=True,
    )
    invalidated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    reenqueued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fresh_thesis_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("analysis_results.id")
    )

    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "invalidated_thesis_id",
            name="uq_paper_reclaim_symbol_thesis",
        ),
        CheckConstraint(
            "status IN ('queued', 'reenqueued', 'done', 'expired')",
            name="ck_paper_reclaim_status",
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
