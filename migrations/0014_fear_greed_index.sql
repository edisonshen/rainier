-- Migration 0014 — CNN Fear & Greed Index (point-in-time sentiment table).
--
-- Numbered 0014 after 0013 (paper trade shadow) landed on main.
--
-- Adds (on the LEGACY local-TimescaleDB engine — NOT Neon, see memory
-- project_two_database_url_engines) so Phase 2 can join it to stock_prices:
--   * fear_greed_index — one row PER OBSERVATION of the CNN Fear & Greed
--     Index. Append-only-on-change: a new (date, observed_at) row is inserted
--     only when the pulled value differs from the latest stored one for that
--     date, so a cron double-fire is a no-op while a genuine source revision
--     appends a new immutable observation.
--
--     PLAIN Postgres table (NOT a hypertable, D10): ~1.5k rows, and a
--     hypertable would force the partition column into every unique index
--     (composite PK) and defeat the append-on-change MIN/MAX(observed_at) PIT
--     scan the design relies on. Deliberately absent from core/models.py
--     HYPERTABLES.
--
--     The 9 CNN component *scores* are columns (never drop sp125 / vix_50);
--     the composite rating plus each component's rating label live in `raw`
--     JSONB (display-oriented, recoverable). `source_version` is 'daily'
--     (live capture) or 'backfill' (revised values pulled after the fact) —
--     the true-PIT boundary is DERIVED, never stored:
--       SELECT MIN(observed_at) FROM fear_greed_index WHERE source_version='daily';
--
-- Apply with (POST-#115 the legacy public-schema tables live on
-- LEGACY_DATABASE_URL, NOT DATABASE_URL which now points at Neon):
--   psql "$LEGACY_DATABASE_URL" -f migrations/0014_fear_greed_index.sql
--
-- Idempotent: re-applying is safe (IF NOT EXISTS throughout). market.* and
-- every other public table are untouched.

BEGIN;

CREATE TABLE IF NOT EXISTS fear_greed_index (
    id                       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    date                     DATE             NOT NULL,
    observed_at              TIMESTAMPTZ      NOT NULL,
    score                    DOUBLE PRECISION NOT NULL,   -- composite 0-100
    rating                   TEXT,                        -- extreme fear … extreme greed
    momentum_sp500_score     DOUBLE PRECISION,
    momentum_sp125_score     DOUBLE PRECISION,
    price_strength_score     DOUBLE PRECISION,
    price_breadth_score      DOUBLE PRECISION,
    put_call_score           DOUBLE PRECISION,
    volatility_vix_score     DOUBLE PRECISION,
    volatility_vix_50_score  DOUBLE PRECISION,
    junk_bond_demand_score   DOUBLE PRECISION,
    safe_haven_demand_score  DOUBLE PRECISION,
    raw                      JSONB            NOT NULL,    -- full slice incl. all 9 rating labels
    source_version           TEXT             NOT NULL,    -- 'daily' | 'backfill'
    ingested_at              TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_fng_date     ON fear_greed_index (date);
-- Serves the append-on-change "latest observation" lookup; the Phase-2
-- earliest-observation read is a backward MIN scan of the same index.
CREATE INDEX IF NOT EXISTS ix_fng_date_obs ON fear_greed_index (date, observed_at DESC);

COMMIT;

-- Downgrade lives in migrations/0014_fear_greed_index_downgrade.sql.
