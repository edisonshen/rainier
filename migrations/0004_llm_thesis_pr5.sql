-- Migration 0004 — LLM Thesis PR5 schema additions
--
-- Extends `chart_images` to support inline PNG byte storage so the Discord
-- renderer (and Streamlit dashboard) can serve the exact chart that was
-- handed to the LLM without depending on the filesystem layout.
--
-- New columns are all nullable so legacy `file_path`-only rows from PR1 keep
-- working unchanged; PR5 rows fill `image_bytes` + `sha256` and leave
-- `file_path` empty.
--
-- Apply with:
--   psql "$DATABASE_URL" -f migrations/0004_llm_thesis_pr5.sql
--
-- Idempotent: re-applying is safe.

BEGIN;

-- Original chart_images.file_path was NOT NULL. PR5 persists bytes inline,
-- so file_path becomes optional. Drop the constraint if it exists.
ALTER TABLE chart_images
    ALTER COLUMN file_path DROP NOT NULL;

ALTER TABLE chart_images
    ADD COLUMN IF NOT EXISTS image_bytes BYTEA;

ALTER TABLE chart_images
    ADD COLUMN IF NOT EXISTS sha256 VARCHAR(64);

ALTER TABLE chart_images
    ADD COLUMN IF NOT EXISTS scan_date DATE;

ALTER TABLE chart_images
    ADD COLUMN IF NOT EXISTS width INTEGER;

ALTER TABLE chart_images
    ADD COLUMN IF NOT EXISTS height INTEGER;

CREATE INDEX IF NOT EXISTS ix_chart_images_sha256
    ON chart_images (sha256);

CREATE INDEX IF NOT EXISTS ix_chart_images_scan_date
    ON chart_images (scan_date);

-- Partial unique index — at most one PR5 row per (symbol, scan_date, sha256)
-- so the persist path is idempotent: re-running the same scan with identical
-- chart bytes collapses to one row instead of inserting duplicates. Legacy
-- rows (sha256 IS NULL) are excluded from the constraint.
CREATE UNIQUE INDEX IF NOT EXISTS idx_chart_image_symbol_scan_sha
    ON chart_images (symbol, scan_date, sha256)
    WHERE sha256 IS NOT NULL;

COMMIT;
