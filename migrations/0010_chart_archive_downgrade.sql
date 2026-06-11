-- Migration 0010 DOWNGRADE — reverses 0010_chart_archive.sql.
--
-- Drops EXACTLY: chart_images.{source, as_of_date, superseded_by},
-- idx_chart_images_logical, and paper_trade.chart_id. Chart BYTES are never
-- touched — superseded rows simply lose the supersession pointer (they were
-- append-only duplicates, so the pre-0010 reader semantics are restored).
--
-- Apply with:
--   psql "$LEGACY_DATABASE_URL" -f migrations/0010_chart_archive_downgrade.sql
--
-- Idempotent: re-applying is safe (IF EXISTS).

BEGIN;

ALTER TABLE paper_trade DROP COLUMN IF EXISTS chart_id;

DROP INDEX IF EXISTS idx_chart_images_logical;

ALTER TABLE chart_images DROP COLUMN IF EXISTS superseded_by;
ALTER TABLE chart_images DROP COLUMN IF EXISTS as_of_date;
ALTER TABLE chart_images DROP COLUMN IF EXISTS source;

COMMIT;
