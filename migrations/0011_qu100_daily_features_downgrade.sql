-- Migration 0011 DOWNGRADE — reverses 0011_qu100_daily_features.sql.
--
-- Drops EXACTLY: qu100_daily_features (+ its unique constraint). market.*
-- and every other public table are untouched.
--
-- Apply with:
--   psql "$LEGACY_DATABASE_URL" -f migrations/0011_qu100_daily_features_downgrade.sql
--
-- Idempotent: re-applying is safe (IF EXISTS).

BEGIN;

DROP TABLE IF EXISTS qu100_daily_features;

COMMIT;
