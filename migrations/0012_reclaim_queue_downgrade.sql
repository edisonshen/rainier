-- Downgrade for migration 0012 (WS B reclaim queue). Idempotent.
--
--   psql "$LEGACY_DATABASE_URL" -f migrations/0012_reclaim_queue_downgrade.sql

BEGIN;

DROP TABLE IF EXISTS paper_reclaim_queue;
ALTER TABLE screened_stocks
    DROP COLUMN IF EXISTS bearish_invalidation_level;

COMMIT;
