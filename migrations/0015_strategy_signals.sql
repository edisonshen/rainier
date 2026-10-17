-- Migration 0015 — persist daily strategy decisions on the legacy database.
--
-- strategy_signals stores one replaceable signal per strategy and session.
-- It is a plain table; daily strategy state does not belong in a hypertable.
--
-- Apply to LEGACY_DATABASE_URL (not the canonical DATABASE_URL):
--   psql "$LEGACY_DATABASE_URL" -f migrations/0015_strategy_signals.sql
--
-- Idempotent: re-applying is safe (IF NOT EXISTS).

BEGIN;

CREATE TABLE IF NOT EXISTS strategy_signals (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    strategy    TEXT             NOT NULL,
    signal_date DATE             NOT NULL,
    fg          DOUBLE PRECISION,
    qqq_close   DOUBLE PRECISION,
    qqq_sma     DOUBLE PRECISION,
    entry_price DOUBLE PRECISION,
    stop_price  DOUBLE PRECISION,
    regime_sma  DOUBLE PRECISION,
    holding     BOOLEAN,
    armed       BOOLEAN,
    bear        BOOLEAN,
    locked      BOOLEAN,
    action      TEXT             NOT NULL,
    params      JSONB            NOT NULL,
    created_at  TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT uq_strategy_signal_date UNIQUE (strategy, signal_date)
);

COMMIT;

-- Downgrade lives in migrations/0015_strategy_signals_downgrade.sql.
