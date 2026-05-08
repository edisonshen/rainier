-- Migration 0003 — LLM Thesis PR3 schema additions
--
-- Adds:
--   * research_insights table (design Section F item 22) — auto-research
--     findings from the weekly job. Carries kind, severity, structured
--     action JSONB (eng review D3), recurrence_count + partial unique index
--     for UPSERT (eng review D6).
--
-- Apply with:
--   psql "$DATABASE_URL" -f migrations/0003_llm_thesis_pr3.sql
--
-- Idempotent: re-applying is safe.

BEGIN;

CREATE TABLE IF NOT EXISTS research_insights (
    id                 SERIAL PRIMARY KEY,
    created_at         TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at         TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    kind               VARCHAR(40) NOT NULL,
    severity           VARCHAR(10) NOT NULL,
    subject            VARCHAR(100) NOT NULL,
    evidence           JSONB,
    action             JSONB,
    rationale          TEXT,
    recurrence_count   INTEGER NOT NULL DEFAULT 1,
    status             VARCHAR(20) NOT NULL DEFAULT 'pending',
    decided_at         TIMESTAMP WITH TIME ZONE,
    decided_by         VARCHAR(200),
    applied_change     JSONB
);

CREATE INDEX IF NOT EXISTS ix_research_insights_kind
    ON research_insights (kind);
CREATE INDEX IF NOT EXISTS ix_research_insights_subject
    ON research_insights (subject);

-- Partial unique index — at most ONE pending row per (kind, subject) so
-- emit_insight() can UPSERT in a single statement and the recurrence_count
-- counter increments deterministically. Rows that move to accepted/rejected/
-- stale fall out of the constraint, allowing a fresh pending row later.
CREATE UNIQUE INDEX IF NOT EXISTS idx_research_insight_pending_kind_subject
    ON research_insights (kind, subject)
    WHERE status = 'pending';

COMMIT;
