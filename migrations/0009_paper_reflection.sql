-- Migration 0009 — R-A per-trade post-exit LLM reflections
-- (design DESIGN-qu100-llm-feedback-loop Appendix B, task qu100-ra-reflections)
--
-- Adds (on the LEGACY local-TimescaleDB engine — NOT Neon, see memory
-- project_two_database_url_engines):
--   * paper_trade.reflection TEXT — the LLM's 2–3 sentence post-mortem,
--     written once per closed trade by the daily job (paper/reflection.py)
--     AFTER step (v) (the report/chart-capture step). The last K=10 feed the
--     thesis prompt's calibration section.
--   * CHECK ck_paper_trade_reflection_after_exit — the outcome embargo at the
--     schema level: a reflection may only exist once the trade has resolved
--     (exit_reason set). No interim/open-position reflections, ever.
--
-- Apply with (POST-#115 the legacy public-schema tables live on
-- LEGACY_DATABASE_URL, NOT DATABASE_URL which now points at Neon):
--   psql "$LEGACY_DATABASE_URL" -f migrations/0009_paper_reflection.sql
--
-- Idempotent: re-applying is safe (IF NOT EXISTS / duplicate_object guard).
-- Additive only: no existing row or column is touched.

BEGIN;

ALTER TABLE paper_trade ADD COLUMN IF NOT EXISTS reflection TEXT;

DO $$
BEGIN
    ALTER TABLE paper_trade
        ADD CONSTRAINT ck_paper_trade_reflection_after_exit
        CHECK (reflection IS NULL OR exit_reason IS NOT NULL);
EXCEPTION
    WHEN duplicate_object THEN NULL;  -- already applied
END $$;

COMMIT;

-- Downgrade lives in migrations/0009_paper_reflection_downgrade.sql.
