-- Downgrade for migration 0013 (WS A shadow column). Idempotent.
--
-- Restores the original all-rows active-symbol unique index and drops the
-- shadow column + its scoped index.
--
--   psql "$LEGACY_DATABASE_URL" -f migrations/0013_paper_trade_shadow_downgrade.sql

BEGIN;

DROP INDEX IF EXISTS idx_paper_trade_active_symbol_shadow;
DROP INDEX IF EXISTS idx_paper_trade_active_symbol;
CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_trade_active_symbol
    ON paper_trade (symbol)
    WHERE status IN ('pending', 'open');

ALTER TABLE paper_trade
    DROP COLUMN IF EXISTS shadow;

COMMIT;
