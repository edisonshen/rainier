"""SQLAlchemy Core Table definitions for the ``market.*`` canonical schema.

These definitions are the single source of truth for the schema shape, and
they back the Alembic ``target_metadata`` used by autogenerate diffs (the
initial migration 0001 is hand-authored — see task plan §5 — but follow-up
migrations will autogenerate against this metadata).

We use SQLAlchemy Core (Table) not ORM (declarative_base). Reason from
task plan §6: future write paths will use Core's bulk UPSERT semantics
(``insert(...).on_conflict_do_update(...)``); introducing ORM here would
force a later switch.

Schema layout (see design doc §D-4):

    market.tickers                  registry (PK ticker_id)
    market.sectors                  registry (PK sector_id)
    market.thematic_ohlcv           Layer-0 raw daily OHLCV
    market.thematic_features_daily  Layer-A features (FKs → tickers, sectors)
    market.thematic_labels_daily    Layer-B forward returns
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    SmallInteger,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, REAL, TIMESTAMP

# All canonical tables live under the `market` schema (domain-named, NOT
# producer-named — design §D-2). Alembic env.py also passes
# include_schemas=True so autogenerate picks this up.
SCHEMA = "market"

metadata = MetaData(schema=SCHEMA)


tickers = Table(
    "tickers",
    metadata,
    Column("ticker_id", SmallInteger, primary_key=True),
    Column("symbol", Text, nullable=False, unique=True),
    Column("first_seen", Date, nullable=False),
    Column("last_seen", Date),  # NULL while active in the universe
)


sectors = Table(
    "sectors",
    metadata,
    Column("sector_id", SmallInteger, primary_key=True),
    Column("sector_name", Text, nullable=False, unique=True),
    Column("first_seen", Date, nullable=False),
)


thematic_ohlcv = Table(
    "thematic_ohlcv",
    metadata,
    Column("symbol", Text, nullable=False),
    Column("date", Date, nullable=False),
    Column("open", DOUBLE_PRECISION, nullable=False),
    Column("high", DOUBLE_PRECISION, nullable=False),
    Column("low", DOUBLE_PRECISION, nullable=False),
    Column("close", DOUBLE_PRECISION, nullable=False),
    Column("volume", BigInteger, nullable=False),
    Column("fetched_at", TIMESTAMP(timezone=True), nullable=False),
    Column("yfinance_version", Text, nullable=False),
    PrimaryKeyConstraint("symbol", "date", name="thematic_ohlcv_pkey"),
    Index("ix_thematic_ohlcv_date", "date"),
)


thematic_features_daily = Table(
    "thematic_features_daily",
    metadata,
    Column("asof_date", Date, nullable=False),
    # Panel-relative trading-day index emitted by the feature compute. Added
    # in migration 0002 to mirror the parquet schema (see task plan §3). It is
    # nullable: short backfills can produce rows without an ordinal, and the
    # value shifts on re-backfill so it is provenance, not a key.
    Column("trading_day_ordinal", Integer, nullable=True),
    Column("symbol", Text, nullable=False),
    # FK referent uses schema-qualified name so Alembic emits it correctly.
    Column(
        "ticker_id",
        SmallInteger,
        ForeignKey(f"{SCHEMA}.tickers.ticker_id"),
        nullable=False,
    ),
    Column(
        "sector_id",
        SmallInteger,
        ForeignKey(f"{SCHEMA}.sectors.sector_id"),
        nullable=False,
    ),
    Column("close", DOUBLE_PRECISION, nullable=False),
    Column("ret_1d", REAL),
    Column("rel_5", REAL),
    Column("rel_10", REAL),
    Column("rel_20", REAL),
    Column("ret_ytd", REAL),
    Column("relvol_5", REAL),
    Column("relvol_10", REAL),
    Column("relvol_20", REAL),
    Column("r_5", SmallInteger),
    Column("r_10", SmallInteger),
    Column("r_20", SmallInteger),
    Column("rank", SmallInteger, nullable=False),
    Column("rank_delta_1d", SmallInteger, nullable=False),
    Column("rank_delta_5d", SmallInteger, nullable=False),
    Column("top15_streak", SmallInteger, nullable=False),
    Column("rank_minus_sector_mean", REAL),
    Column("universe_size", SmallInteger, nullable=False),
    Column("universe_yaml_sha", Text, nullable=False),
    Column("computed_at", TIMESTAMP(timezone=True), nullable=False),
    PrimaryKeyConstraint("asof_date", "symbol", name="thematic_features_daily_pkey"),
    Index("ix_thematic_features_daily_asof_date", "asof_date"),
)


thematic_labels_daily = Table(
    "thematic_labels_daily",
    metadata,
    Column("asof_date", Date, nullable=False),
    Column("symbol", Text, nullable=False),
    Column("fwd_3d_ret", REAL),
    Column("fwd_5d_ret", REAL),
    Column("fwd_10d_ret", REAL),
    Column("fwd_20d_ret", REAL),
    Column("fwd_30d_ret", REAL),
    Column("fwd_5d_excess_ret", REAL),
    Column("fwd_10d_excess_ret", REAL),
    Column("fwd_20d_excess_ret", REAL),
    Column("fwd_10d_max_drawdown", REAL),
    Column("fwd_10d_max_runup", REAL),
    # Relaxed to nullable in migration 0002: a panel too short to complete any
    # forward horizon yields label_complete_through=None for valid rows.
    Column("label_complete_through", Date, nullable=True),
    PrimaryKeyConstraint("asof_date", "symbol", name="thematic_labels_daily_pkey"),
    Index("ix_thematic_labels_daily_asof_date", "asof_date"),
)


# market.breadth_indicator_daily — S&P 500 market-breadth indicators, LONG.
#
# Mirrors data/cache/sp500_breadth_daily.parquet 1:1 (asof_date, indicator,
# value) so the Phase 2/3 dual-write + verify machinery works with NO transform
# layer (design D-1). ``value`` is DOUBLE_PRECISION + nullable to mirror the
# parquet float64 column EXACTLY, including warm-up-tainted early rows (MAs
# before N days; new_52w_high/low use min_periods=1 "max so far" before 252
# days). We deliberately store whatever the parquet holds — divergent NULL
# logic here would break parquet<->PG checksum parity (design D-4).
breadth_indicator_daily = Table(
    "breadth_indicator_daily",
    metadata,
    Column("asof_date", Date, nullable=False),
    Column("indicator", Text, nullable=False),
    Column("value", DOUBLE_PRECISION, nullable=True),
    PrimaryKeyConstraint("asof_date", "indicator", name="breadth_indicator_daily_pkey"),
    Index("ix_breadth_indicator_daily_asof_date", "asof_date"),
)


# market.benchmark_ohlcv — benchmark/index OHLCV (SPY today; generalizes to
# QQQ/DIA/IWM/sector ETFs/VIX). Schema mirrors thematic_ohlcv +
# spy_history.parquet (design D-2). Named benchmark, not index, because SPY is
# an ETF benchmark. ``fetched_at``/``yfinance_version`` are insert-only
# provenance at the write call sites (immutable_cols), as in thematic_ohlcv.
benchmark_ohlcv = Table(
    "benchmark_ohlcv",
    metadata,
    Column("symbol", Text, nullable=False),
    Column("date", Date, nullable=False),
    Column("open", DOUBLE_PRECISION, nullable=False),
    Column("high", DOUBLE_PRECISION, nullable=False),
    Column("low", DOUBLE_PRECISION, nullable=False),
    Column("close", DOUBLE_PRECISION, nullable=False),
    Column("volume", BigInteger, nullable=False),
    Column("fetched_at", TIMESTAMP(timezone=True), nullable=False),
    Column("yfinance_version", Text, nullable=False),
    PrimaryKeyConstraint("symbol", "date", name="benchmark_ohlcv_pkey"),
    Index("ix_benchmark_ohlcv_date", "date"),
)


# Public surface — callers reach for the Table objects + the shared metadata.
__all__ = [
    "SCHEMA",
    "benchmark_ohlcv",
    "breadth_indicator_daily",
    "metadata",
    "sectors",
    "thematic_features_daily",
    "thematic_labels_daily",
    "thematic_ohlcv",
    "tickers",
]
