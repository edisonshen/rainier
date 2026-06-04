-- Migration 0007 — QU100 paper-trade calibration block (Phase 2, D7a)
--
-- Adds (on the LEGACY local-TimescaleDB engine — NOT Neon, see memory
-- project_two_database_url_engines):
--   * paper_calibration — one bounded calibration payload per as-of date.
--     Written by the daily eval job AFTER run_paper_daily_report
--     (scheduler/service.py). Read by build_user_message() to inject the
--     calibration section into the LLM thesis prompt.
--
--     Plain Postgres table (NOT a hypertable): tiny row count (one row per
--     trading day), keyed UNIQUE(as_of_date) so a same-day re-run upserts.
--     The HEADLINE stats inside `payload` are the UNBIASED ThesisEvaluation
--     fixed-horizon (1d/5d/10d) numbers (survivorship-free); the realized
--     paper-trade record (K=10 closed + MTM-including-open) is labeled
--     supplementary context. The compute lives in paper/calibration.py.
--
-- Apply with (POST-#115 the legacy public-schema tables live on
-- LEGACY_DATABASE_URL, NOT DATABASE_URL which now points at Neon):
--   psql "$LEGACY_DATABASE_URL" -f migrations/0007_paper_calibration.sql
--
-- Idempotent: re-applying is safe (IF NOT EXISTS throughout). market.* and
-- every other public table are untouched.

BEGIN;

CREATE TABLE IF NOT EXISTS paper_calibration (
    id          BIGSERIAL PRIMARY KEY,
    as_of_date  DATE NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_paper_calibration_as_of_date UNIQUE (as_of_date)
);

COMMIT;

-- Downgrade lives in migrations/0007_paper_calibration_downgrade.sql.
