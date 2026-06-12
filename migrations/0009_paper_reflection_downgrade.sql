-- Migration 0009 DOWNGRADE — reverses 0009_paper_reflection.sql.
--
-- Drops EXACTLY: paper_trade.reflection + its CHECK constraint. Every other
-- column/row of paper_trade (and every other table) is untouched.
--
-- Apply with:
--   psql "$LEGACY_DATABASE_URL" -f migrations/0009_paper_reflection_downgrade.sql
--
-- Idempotent: re-applying is safe (IF EXISTS).

BEGIN;

ALTER TABLE paper_trade
    DROP CONSTRAINT IF EXISTS ck_paper_trade_reflection_after_exit;
ALTER TABLE paper_trade DROP COLUMN IF EXISTS reflection;

COMMIT;

-- DATA LOSS WARNING: dropping the column destroys all LLM-generated
-- reflection text permanently. Re-applying 0008 only regenerates trades
-- exited within the trailing 30-day generation window (and at fresh LLM
-- cost); older reflections are unrecoverable. To snapshot first:
--   \copy (SELECT id, reflection FROM paper_trade WHERE reflection IS NOT NULL)
--     TO 'paper_trade_reflection_backup.csv' CSV HEADER
