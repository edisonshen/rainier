-- Migration 0014 DOWNGRADE — reverses 0014_fear_greed_index.sql.
--
-- Drops EXACTLY: fear_greed_index (+ its two indexes, dropped with the table).
-- market.* and every other public table are untouched.
--
-- Apply with:
--   psql "$LEGACY_DATABASE_URL" -f migrations/0014_fear_greed_index_downgrade.sql
--
-- Idempotent: re-applying is safe (IF EXISTS).

BEGIN;

DROP TABLE IF EXISTS fear_greed_index;

COMMIT;
