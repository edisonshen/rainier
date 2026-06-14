-- Migration 0012 — WS B: reconsider invalidated bull-traps (false_breakout)
--
-- A `false_breakout` is a CORRECT bearish setup (price pokes above resistance,
-- then rejects). It declines as designed. But when a LATER close reclaims the
-- trap high (`false_high`), the bear thesis is dead — the symbol may now be a
-- fresh LONG. The scheduled LLM run only sees today's top-5, so a reclaimed
-- name outside it would be lost. This migration adds:
--
--   1. screened_stocks.bearish_invalidation_level — the trap high (`false_high`,
--      from the pattern's key_points), persisted ONLY for exact `false_breakout`
--      (NOT `false_breakout_hs`, which has no false_high). NULL otherwise. This
--      is the level a later close must reclaim to invalidate the bear setup.
--      NOTE: this is `false_high`, NOT `stop_loss` (= false_high*(1+buffer),
--      higher) — the detector's stop sits above the trap high.
--
--   2. paper_reclaim_queue — a durable queue so a reclaim detected on a symbol
--      not in today's top-5 isn't dropped. One row per (symbol,
--      invalidated_thesis_id). Status: queued -> reenqueued -> done | expired.
--      K-day dedup + idempotency live in the daily check (paper/reclaim.py).
--
-- Applies to the LEGACY local-TimescaleDB engine (NOT Neon — see memory
-- project_two_database_url_engines). Idempotent (IF NOT EXISTS throughout).
--
--   psql "$LEGACY_DATABASE_URL" -f migrations/0012_reclaim_queue.sql
--
-- Downgrade: migrations/0012_reclaim_queue_downgrade.sql.

BEGIN;

ALTER TABLE screened_stocks
    ADD COLUMN IF NOT EXISTS bearish_invalidation_level DOUBLE PRECISION;

CREATE TABLE IF NOT EXISTS paper_reclaim_queue (
    id                     SERIAL PRIMARY KEY,
    symbol                 VARCHAR(10) NOT NULL,
    -- The declined bearish thesis whose trap was reclaimed. FK kept loose
    -- (no ON DELETE) — the stale declined row is never mutated by this path.
    invalidated_thesis_id  INTEGER NOT NULL REFERENCES analysis_results (id),
    invalidation_level     DOUBLE PRECISION NOT NULL,
    -- The close that reclaimed the level (drives K-day dedup).
    reclaim_date           DATE NOT NULL,
    reclaim_close          DOUBLE PRECISION NOT NULL,
    status                 VARCHAR(20) NOT NULL DEFAULT 'queued',
    invalidated_at         TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    reenqueued_at          TIMESTAMP WITH TIME ZONE,
    -- The fresh long thesis spawned from this reclaim (NULL until the audit
    -- gate passes and a new thesis is created).
    fresh_thesis_id        INTEGER REFERENCES analysis_results (id),
    CONSTRAINT uq_paper_reclaim_symbol_thesis
        UNIQUE (symbol, invalidated_thesis_id),
    CONSTRAINT ck_paper_reclaim_status
        CHECK (status IN ('queued', 'reenqueued', 'done', 'expired'))
);

CREATE INDEX IF NOT EXISTS ix_paper_reclaim_status
    ON paper_reclaim_queue (status);
CREATE INDEX IF NOT EXISTS ix_paper_reclaim_symbol
    ON paper_reclaim_queue (symbol);

COMMIT;
