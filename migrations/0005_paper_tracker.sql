-- Migration 0005 — QU100 paper-trade tracker (Phase 0+1)
--
-- Adds (all on the LEGACY local-TimescaleDB engine, design DESIGN-qu100-llm-feedback-loop §4):
--   * paper_trade           — one row per opened paper position (PLAIN table,
--                             NOT a hypertable: id PK, UNIQUE(thesis_id),
--                             partial-unique UNIQUE(symbol) WHERE status IN
--                             ('pending','open'); fill fields incl. price_basis
--                             and time_stop_days nullable while pending).
--   * paper_skip            — skip ledger (thesis_id UNIQUE, reason check).
--   * paper_report_snapshot — regenerable report payload, unique(report_type, as_of_date).
--   * screened_stocks       — ADD COLUMN IF NOT EXISTS entry_price/stop_loss/
--                             target_price/rr_ratio (pattern_type already exists,
--                             NOT re-added).
--
-- Apply with (POST-#115 the legacy public-schema tables live on LEGACY_DATABASE_URL,
-- NOT DATABASE_URL which now points at Neon — see memory project_two_database_url_engines):
--   psql "$LEGACY_DATABASE_URL" -f migrations/0005_paper_tracker.sql
--
-- Idempotent: re-applying is safe (IF NOT EXISTS / DO NOTHING throughout).
-- market.* and all other public tables are untouched.

BEGIN;

-- --------------------------------------------------------------------------
-- screened_stocks: additive level columns (D4). pattern_type already exists.
-- --------------------------------------------------------------------------
ALTER TABLE screened_stocks ADD COLUMN IF NOT EXISTS entry_price  DOUBLE PRECISION;
ALTER TABLE screened_stocks ADD COLUMN IF NOT EXISTS stop_loss    DOUBLE PRECISION;
ALTER TABLE screened_stocks ADD COLUMN IF NOT EXISTS target_price DOUBLE PRECISION;
ALTER TABLE screened_stocks ADD COLUMN IF NOT EXISTS rr_ratio     DOUBLE PRECISION;

-- --------------------------------------------------------------------------
-- paper_trade (PLAIN table — created AFTER its FK targets analysis_results /
-- screened_stocks already exist).
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS paper_trade (
    id                  BIGSERIAL PRIMARY KEY,
    thesis_id           INTEGER NOT NULL REFERENCES analysis_results (id),
    screened_record_id  INTEGER REFERENCES screened_stocks (id),
    symbol              VARCHAR(10) NOT NULL,
    scan_date           DATE NOT NULL,
    session_name        VARCHAR(20) NOT NULL,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status              VARCHAR(10) NOT NULL DEFAULT 'pending',
    -- plan (set at creation, from the persisted screened row)
    planned_entry_price DOUBLE PRECISION NOT NULL,
    stop_loss           DOUBLE PRECISION NOT NULL,
    target_price        DOUBLE PRECISION NOT NULL,
    rr_ratio            DOUBLE PRECISION,
    pattern_type        VARCHAR(50),
    llm_confidence      INTEGER,
    verdict             VARCHAR(20),
    -- fill (NULL while pending)
    entry_date          DATE,
    entry_price         DOUBLE PRECISION,
    shares              INTEGER,
    allocated_amount    DOUBLE PRECISION,
    residual_cash       DOUBLE PRECISION,
    time_stop_days      INTEGER,
    price_basis         VARCHAR(20),
    -- exit
    exit_date           DATE,
    exit_price          DOUBLE PRECISION,
    exit_reason         VARCHAR(20),
    return_pct          DOUBLE PRECISION,
    pnl                 DOUBLE PRECISION,
    CONSTRAINT uq_paper_trade_thesis_id UNIQUE (thesis_id),
    CONSTRAINT ck_paper_trade_status
        CHECK (status IN ('pending','open','closed','expired')),
    CONSTRAINT ck_paper_trade_exit_reason
        CHECK (exit_reason IS NULL
               OR exit_reason IN ('stop_loss','target','time_stop','manual'))
);

CREATE INDEX IF NOT EXISTS ix_paper_trade_symbol ON paper_trade (symbol);

-- Partial unique index — one pending/open position per symbol (D1). Closed /
-- expired rows fall out so a symbol can be re-traded later. Predicate is
-- EXACTLY status IN ('pending','open').
CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_trade_active_symbol
    ON paper_trade (symbol)
    WHERE status IN ('pending', 'open');

-- --------------------------------------------------------------------------
-- paper_skip (skip ledger)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS paper_skip (
    id          BIGSERIAL PRIMARY KEY,
    thesis_id   INTEGER NOT NULL,
    symbol      VARCHAR(10) NOT NULL,
    scan_date   DATE NOT NULL,
    reason      VARCHAR(40) NOT NULL,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_paper_skip_thesis_id UNIQUE (thesis_id),
    CONSTRAINT ck_paper_skip_reason
        CHECK (reason IN ('symbol_already_active','missing_levels',
                          'missing_screened_record','gap_invalidated',
                          'same_symbol_lower_conviction'))
);

CREATE INDEX IF NOT EXISTS ix_paper_skip_symbol ON paper_skip (symbol);

-- --------------------------------------------------------------------------
-- paper_report_snapshot (regenerable report record)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS paper_report_snapshot (
    id           BIGSERIAL PRIMARY KEY,
    report_type  VARCHAR(10) NOT NULL,
    as_of_date   DATE NOT NULL,
    payload      JSONB NOT NULL,
    created_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_paper_report_snapshot_type_date UNIQUE (report_type, as_of_date),
    CONSTRAINT ck_paper_report_snapshot_type
        CHECK (report_type IN ('daily','weekly'))
);

COMMIT;

-- Downgrade lives in migrations/0005_paper_tracker_downgrade.sql — it reverses
-- exactly these objects (the three tables + the four screened_stocks columns +
-- their indexes/checks) and leaves market.* and all other public tables intact.
