-- Migration 0007 DOWNGRADE — reverses 0007_paper_calibration.sql.
--
-- Drops EXACTLY: paper_calibration (+ its unique constraint). market.* and
-- every other public table are untouched.
--
-- Apply with:
--   psql "$LEGACY_DATABASE_URL" -f migrations/0007_paper_calibration_downgrade.sql
--
-- Idempotent: re-applying is safe (IF EXISTS).

BEGIN;

DROP TABLE IF EXISTS paper_calibration;

COMMIT;
