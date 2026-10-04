-- Migration 0015 — R1: selection reward ledger (selection → track → reward)
--
-- Adds (on the LEGACY local-TimescaleDB engine — NOT Neon, see memory
-- project_two_database_url_engines):
--   * selection_reward — one row per (thesis, reward_name): the R-multiple
--     (`r_multiple`) and sleeve $P&L (`pnl_usd`) of the plan each QU100-LLM
--     decision committed to, scored against what the market did.
--       - `decision` classifies EVERY thesis (filled / pending / expired /
--         skipped / confidence-gated / watch / no_setup / gap_invalidated) —
--         the honest denominator for win-rate.
--       - Declined decisions with a valid long plan are scored COUNTERFACTUALLY
--         via the same pure evaluate_exit the live book uses
--         (`counterfactual = true`).
--       - `provisional = true` = mark-to-market (path unresolved at as_of_date),
--         re-upserted in place daily; a matured row is final and only rewritten
--         when `reward_version` changes.
--       - `value IS NULL` rows carry a `reason` explaining why the decision
--         could not be scored, so unscoreable decisions still count.
--       - `lever_context` snapshots the lever values live at decision time
--         (prompt_version, model, llm_confidence, confidence_gate, session,
--         pattern_type, signal_set_hash, time_stop_days, price_basis).
--
--     Written by daily step (vii) in scheduler/service.py (after calibration)
--     and by `rainier reward compute`. Compute lives in paper/rewards.py;
--     reward bodies in research/rewards/selection.py.
--
-- Apply with:
--   psql "$LEGACY_DATABASE_URL" -f migrations/0015_selection_reward.sql
--
-- Idempotent: re-applying is safe (IF NOT EXISTS throughout). market.* and
-- every other public table are untouched.

BEGIN;

CREATE TABLE IF NOT EXISTS selection_reward (
    id              BIGSERIAL PRIMARY KEY,
    thesis_id       INTEGER NOT NULL REFERENCES analysis_results (id),
    symbol          VARCHAR(10) NOT NULL,
    scan_date       DATE NOT NULL,
    session_name    VARCHAR(20) NOT NULL,
    decision        VARCHAR(30) NOT NULL,
    lever_context   JSONB NOT NULL DEFAULT '{}'::jsonb,
    reward_name     VARCHAR(40) NOT NULL,
    reward_version  INTEGER NOT NULL,
    value           DOUBLE PRECISION,
    reason          VARCHAR(40),
    provisional     BOOLEAN NOT NULL DEFAULT FALSE,
    counterfactual  BOOLEAN NOT NULL DEFAULT FALSE,
    outcome_date    DATE,
    as_of_date      DATE NOT NULL,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_selection_reward_thesis_reward UNIQUE (thesis_id, reward_name),
    CONSTRAINT ck_selection_reward_decision CHECK (
        decision IN ('setup_long_filled','setup_long_pending',
                     'setup_long_expired','setup_long_skipped','setup_long_gated',
                     'watch','no_setup','gap_invalidated')
    ),
    CONSTRAINT ck_selection_reward_value_or_reason CHECK (
        value IS NOT NULL OR reason IS NOT NULL
    ),
    CONSTRAINT ck_selection_reward_reason CHECK (
        reason IS NULL OR reason IN ('invalid_levels','no_plan',
                                     'missing_prices','gap_invalidated','basis_mismatch')
    )
);

CREATE INDEX IF NOT EXISTS ix_selection_reward_thesis_id ON selection_reward (thesis_id);
CREATE INDEX IF NOT EXISTS ix_selection_reward_symbol ON selection_reward (symbol);
CREATE INDEX IF NOT EXISTS ix_selection_reward_scan_date ON selection_reward (scan_date);
CREATE INDEX IF NOT EXISTS ix_selection_reward_decision ON selection_reward (decision);
CREATE INDEX IF NOT EXISTS ix_selection_reward_as_of_date ON selection_reward (as_of_date);

COMMIT;

-- Downgrade lives in migrations/0015_selection_reward_downgrade.sql.
