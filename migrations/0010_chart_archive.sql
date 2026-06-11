-- Migration 0010 — R-D chart archive (DESIGN-qu100-llm-feedback-loop, PR 2)
--
-- NUMBERING NOTE: 0008 is claimed by BOTH open PRs #138 (miss-sweep) and #139
-- at the time this ships; this file takes 0010 and the final number is
-- reconciled at merge time.
--
-- Adds (on the LEGACY local-TimescaleDB engine — NOT Neon, see memory
-- project_two_database_url_engines):
--   * chart_images.source        — 'thesis' (backfilled onto ALL existing rows,
--                                  which are thesis-path renders) / 'trade_close'
--                                  / 'miss_sweep'.
--   * chart_images.as_of_date    — the archive chart's logical as-of date.
--                                  STAYS NULL for source='thesis' forever
--                                  (existing AND future thesis persists): thesis
--                                  rows live OUTSIDE the logical-identity
--                                  contract, and NULLs never collide in the
--                                  partial unique below, so the up-to-4-renders/
--                                  day thesis pattern is unaffected.
--   * chart_images.superseded_by — self-FK; append-only supersession. The
--                                  current row for a logical identity is the one
--                                  with superseded_by IS NULL; superseded rows
--                                  keep their bytes forever (LLM/snapshot
--                                  references stay valid).
--   * idx_chart_images_logical   — partial unique on (symbol, as_of_date,
--                                  source) WHERE superseded_by IS NULL: one
--                                  CURRENT row per logical identity. Governs
--                                  only trade_close/miss_sweep rows (thesis rows
--                                  carry NULL as_of_date).
--   * paper_trade.chart_id       — FK to the trade's CURRENT close chart.
--
-- Apply with:
--   psql "$LEGACY_DATABASE_URL" -f migrations/0010_chart_archive.sql
--
-- Idempotent: re-applying is safe (IF NOT EXISTS throughout; the backfill
-- only touches rows whose source is still NULL, so trade_close/miss_sweep
-- rows written after the first apply are never clobbered).

BEGIN;

ALTER TABLE chart_images
    ADD COLUMN IF NOT EXISTS source VARCHAR(20);

ALTER TABLE chart_images
    ADD COLUMN IF NOT EXISTS as_of_date DATE;

ALTER TABLE chart_images
    ADD COLUMN IF NOT EXISTS superseded_by BIGINT REFERENCES chart_images (id);

-- Backfill: every pre-0010 row is a thesis-path render. as_of_date stays NULL.
UPDATE chart_images SET source = 'thesis' WHERE source IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_chart_images_logical
    ON chart_images (symbol, as_of_date, source)
    WHERE superseded_by IS NULL;

ALTER TABLE paper_trade
    ADD COLUMN IF NOT EXISTS chart_id BIGINT REFERENCES chart_images (id);

COMMIT;

-- Downgrade lives in migrations/0010_chart_archive_downgrade.sql.
