-- Migration 0001 — LLM Thesis PR1 schema additions
--
-- Why a hand-written SQL migration instead of Alembic / `db init`?
--   `rainier db init` runs `Base.metadata.create_all`, which is ADDITIVE only:
--   it creates tables that don't exist but never alters existing tables. Adding
--   columns or new partial indexes to `analysis_results` (which already exists
--   in production) therefore requires this manual migration.
--
-- New tables in PR1 (created cleanly by `db init` on fresh DBs):
--   - screened_stocks
--
-- Existing tables touched (apply the ALTERs below on existing DBs):
--   - analysis_results — adds input_hash + signals_used columns and indexes.
--
-- Apply with:
--   psql "$DATABASE_URL" -f migrations/0001_llm_thesis_pr1.sql

BEGIN;

-- analysis_results.input_hash — sha256 of canonical EvidencePack JSON + image bytes
ALTER TABLE analysis_results
    ADD COLUMN IF NOT EXISTS input_hash VARCHAR(64);

-- analysis_results.signals_used — list of signal registry keys used in this thesis
ALTER TABLE analysis_results
    ADD COLUMN IF NOT EXISTS signals_used VARCHAR(50)[];

-- Btree index for cheap (date, target_symbols, prompt_template) cache lookup —
-- the Tier-1 path runs `WHERE date(created_at)=? AND target_symbols=[symbol]
-- AND prompt_template=?` and benefits from this composite index.
CREATE INDEX IF NOT EXISTS ix_llm_analysis_date_target_template
    ON analysis_results (
        date(created_at AT TIME ZONE 'UTC'),
        target_symbols,
        prompt_template
    );

-- Btree index on input_hash so Tier-2 idempotent inserts can probe quickly
CREATE INDEX IF NOT EXISTS ix_analysis_results_input_hash
    ON analysis_results (input_hash);

-- Idempotency unique partial index — guarantees ON CONFLICT DO NOTHING behaves
-- as expected: same (day, ticker, model, prompt_version, input_hash) at most once.
-- llm_model is included so v2 multi-model A/B drops in with no further migration.
CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_analysis_idempotent
    ON analysis_results (
        date_trunc('day', created_at),
        target_symbols,
        llm_model,
        prompt_template,
        input_hash
    )
    WHERE input_hash IS NOT NULL;

COMMIT;

-- screened_stocks — new table, created on fresh DBs by `db init`. On existing
-- DBs, run `uv run rainier db init` once after deploying this migration; the
-- ORM's create_all will materialize the new table without touching the others.
