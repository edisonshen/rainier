-- Migration 0002 — LLM Thesis PR2 schema additions
--
-- Adds:
--   - analysis_results.session_name (PR2 carry-over P3 fix #6) so the Tier-1
--     cache lookup filters in SQL instead of post-Python on a JSONB key.
--   - thesis_evaluations table (PR2 daily evaluation, design Section E item 19).
--
-- Apply with:
--   psql "$DATABASE_URL" -f migrations/0002_llm_thesis_pr2.sql
--
-- Idempotent: re-applying is safe.

BEGIN;

-- analysis_results.session_name — first-class per-row session marker. Backfill
-- from structured_output->>'_session_name' when available so PR1 rows don't
-- silently become uncacheable; rows older than the original session-marker
-- introduction will be NULL and the lookup falls back to the JSONB key path.
ALTER TABLE analysis_results
    ADD COLUMN IF NOT EXISTS session_name VARCHAR(20);

UPDATE analysis_results
   SET session_name = structured_output->>'_session_name'
 WHERE session_name IS NULL
   AND structured_output ? '_session_name';

CREATE INDEX IF NOT EXISTS ix_analysis_results_session_name
    ON analysis_results (session_name);

-- thesis_evaluations — created cleanly by `db init` on fresh DBs, but
-- explicit DDL here so this migration alone bootstraps an existing
-- production database without needing a separate `db init` step.
CREATE TABLE IF NOT EXISTS thesis_evaluations (
    id                 SERIAL PRIMARY KEY,
    thesis_id          INTEGER NOT NULL REFERENCES analysis_results(id),
    screened_record_id INTEGER REFERENCES screened_stocks(id),
    evaluated_at       TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    horizon            VARCHAR(8) NOT NULL,
    scan_date          DATE NOT NULL,
    symbol             VARCHAR(10) NOT NULL,
    verdict            VARCHAR(20) NOT NULL,
    llm_confidence     INTEGER,
    entry_price        DOUBLE PRECISION NOT NULL,
    exit_price         DOUBLE PRECISION NOT NULL,
    return_pct         DOUBLE PRECISION NOT NULL,
    hit                BOOLEAN NOT NULL,
    signals_used       VARCHAR(50)[],
    notes              TEXT,
    CONSTRAINT uq_thesis_evaluations_thesis_horizon UNIQUE (thesis_id, horizon)
);

CREATE INDEX IF NOT EXISTS ix_thesis_evaluations_thesis_id
    ON thesis_evaluations (thesis_id);
CREATE INDEX IF NOT EXISTS ix_thesis_evaluations_horizon
    ON thesis_evaluations (horizon);
CREATE INDEX IF NOT EXISTS ix_thesis_evaluations_scan_date
    ON thesis_evaluations (scan_date);
CREATE INDEX IF NOT EXISTS ix_thesis_evaluations_symbol
    ON thesis_evaluations (symbol);

COMMIT;
