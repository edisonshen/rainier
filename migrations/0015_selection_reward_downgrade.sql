-- Migration 0015 DOWNGRADE — reverses 0015_selection_reward.sql.
--
-- Drops EXACTLY: selection_reward (+ its constraints and indexes). market.*
-- and every other public table are untouched.
--
-- Apply with:
--   psql "$LEGACY_DATABASE_URL" -f migrations/0015_selection_reward_downgrade.sql
--
-- Idempotent: re-applying is safe (IF EXISTS).

BEGIN;

DROP TABLE IF EXISTS selection_reward;

COMMIT;
