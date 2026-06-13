-- Migration 0011 — R-E: QU100 daily feature snapshot (design Appendix B)
--
-- Numbered 0011 after 0008 (paper skip zero-share), 0009 (paper reflection),
-- and 0010 (chart archive) landed on main.
--
-- Adds (on the LEGACY local-TimescaleDB engine — NOT Neon, see memory
-- project_two_database_url_engines):
--   * qu100_daily_features — one JSONB feature row per QU100 member per day
--     (vwap, sma5/22/44/60, fractal, volume, vrvp summary, price_basis,
--     feature_version, data_gap?). JSONB so new attributes ship without a
--     table migration; joins against trades and misses for learning.
--
--     Plain Postgres table (NOT a hypertable, D10): ~100 rows/day, and a
--     hypertable would block the UNIQUE(symbol, data_date, ranking_type)
--     key the idempotent daily upsert needs. `rank` is denormalized: that
--     day's rank from the latest capture of the day (same dedup rule as the
--     appearance query). Written by the daily feature step
--     (paper/features.py), failure-isolated from the trading steps.
--
-- Apply with (POST-#115 the legacy public-schema tables live on
-- LEGACY_DATABASE_URL, NOT DATABASE_URL which now points at Neon):
--   psql "$LEGACY_DATABASE_URL" -f migrations/0011_qu100_daily_features.sql
--
-- Idempotent: re-applying is safe (IF NOT EXISTS throughout). market.* and
-- every other public table are untouched.

BEGIN;

CREATE TABLE IF NOT EXISTS qu100_daily_features (
    id           BIGSERIAL PRIMARY KEY,
    symbol       VARCHAR(10) NOT NULL,
    data_date    DATE NOT NULL,
    ranking_type VARCHAR(10) NOT NULL DEFAULT 'top100',
    rank         INTEGER,
    features     JSONB NOT NULL,
    -- NOT NULL matches the ORM (Mapped[datetime], non-Optional) so a
    -- db-init-created table and a migration-created table agree.
    computed_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_qu100_daily_features_symbol_date_ranking
        UNIQUE (symbol, data_date, ranking_type)
);

COMMIT;

-- Downgrade lives in migrations/0011_qu100_daily_features_downgrade.sql.
