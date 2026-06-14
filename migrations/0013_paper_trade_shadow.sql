-- Migration 0013 — WS A: shadow WATCH-buy measurement
--
-- The loop has never bought a stock (0 setup_long). The review found WATCH
-- verdicts outperform NO_SETUP, but acting naively is risky — the headline
-- return is a stops-free point-to-point estimate, not real P&L. So we MEASURE
-- first: a WATCH verdict (conf >= T, valid long-shape levels) opens a SHADOW
-- paper_trade that runs through the REAL fill/exit engine but is excluded from
-- every live read. The live book is unchanged. The live buy-flip (A2) is a
-- separate, gated change — NOT this migration.
--
-- Adds:
--   * paper_trade.shadow BOOLEAN NOT NULL DEFAULT FALSE — flags a shadow row.
--   * The existing one-active-position-per-symbol partial unique index is
--     re-scoped to `... AND shadow = false`, so a shadow position can NEVER
--     consume a LIVE symbol's active slot (isolation invariant). A new partial
--     unique index does the same for shadow rows so the shadow book itself
--     still honors one-active-per-symbol.
--
-- Applies to the LEGACY local-TimescaleDB engine (NOT Neon — see memory
-- project_two_database_url_engines). Idempotent.
--
--   psql "$LEGACY_DATABASE_URL" -f migrations/0013_paper_trade_shadow.sql
--
-- Downgrade: migrations/0013_paper_trade_shadow_downgrade.sql.

BEGIN;

ALTER TABLE paper_trade
    ADD COLUMN IF NOT EXISTS shadow BOOLEAN NOT NULL DEFAULT FALSE;

-- Re-scope the live active-symbol uniqueness to NON-shadow rows only. Drop the
-- old index (it covered all rows regardless of shadow) and recreate scoped.
DROP INDEX IF EXISTS idx_paper_trade_active_symbol;
CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_trade_active_symbol
    ON paper_trade (symbol)
    WHERE status IN ('pending', 'open') AND shadow = false;

-- The shadow book honors one-active-per-symbol independently.
CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_trade_active_symbol_shadow
    ON paper_trade (symbol)
    WHERE status IN ('pending', 'open') AND shadow = true;

COMMIT;
