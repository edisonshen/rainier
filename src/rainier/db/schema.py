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
    Float,
    ForeignKey,
    Index,
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
    Column("label_complete_through", Date, nullable=False),
    PrimaryKeyConstraint("asof_date", "symbol", name="thematic_labels_daily_pkey"),
    Index("ix_thematic_labels_daily_asof_date", "asof_date"),
)


# Sentinel-style export so callers don't reach into module-internal names.
__all__ = [
    "Float",  # re-export for type hints
    "SCHEMA",
    "metadata",
    "sectors",
    "thematic_features_daily",
    "thematic_labels_daily",
    "thematic_ohlcv",
    "tickers",
]
