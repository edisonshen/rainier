-- repair_local_stocks_schema.sql
-- ONE-OFF LOCAL REPAIR — not a numbered migration (the damage is local-DB only,
-- caused by the stock-prices migration's pre-fix test fixtures clobbering the
-- shared public schema; the isolation fix already shipped in PR #119).
--
-- Target: $LEGACY_DATABASE_URL (default local TimescaleDB localhost:5432/rainier).
-- Do NOT run against Neon ($DATABASE_URL) — that store is unaffected.
--
-- Problem: public.stocks was reduced to a 2-column stub (id, symbol; 0 rows),
-- so the QU100 scraper's `INSERT INTO stocks (symbol, name, sector, industry,
-- is_active)` fails with `column "name" ... does not exist` and persists 0 rows.
-- The actual QU100 history (money_flow_snapshots, 8300 rows) is intact.
--
-- This script restores stocks to the ORM shape (core/models.py:99 Stock).
-- Idempotent: ADD COLUMN IF NOT EXISTS + guarded NOT NULL/DEFAULT, safe to
-- re-run. Non-destructive: stocks is empty (0 rows) so backfill UPDATEs are
-- no-ops; no existing data is touched.
--
-- Current stub state (verified 2026-06-03):
--   columns: id INTEGER (PK stocks_pkey, NOT NULL), symbol VARCHAR(10) NULLABLE
--   unique:  stocks_symbol_key on (symbol)   <- already present, FK-target ready
--   missing: name, sector, industry, is_active, created_at, updated_at
--
-- ORM target (core/models.py Stock):
--   name VARCHAR(255) NULL | sector VARCHAR(100) NULL | industry VARCHAR(200) NULL
--   is_active BOOLEAN NOT NULL server_default true
--   created_at TIMESTAMPTZ NOT NULL server_default now()
--   updated_at TIMESTAMPTZ NOT NULL server_default now() (onupdate now() is ORM-side)
--   symbol VARCHAR(10) NOT NULL unique index

BEGIN;

-- 1. add the missing columns (nullable first; defaults/NOT NULL applied below)
ALTER TABLE stocks
    ADD COLUMN IF NOT EXISTS name       VARCHAR(255),
    ADD COLUMN IF NOT EXISTS sector     VARCHAR(100),
    ADD COLUMN IF NOT EXISTS industry   VARCHAR(200),
    ADD COLUMN IF NOT EXISTS is_active  BOOLEAN,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;

-- 2. is_active: default true, backfill any NULLs (0 rows), enforce NOT NULL
ALTER TABLE stocks ALTER COLUMN is_active SET DEFAULT true;
UPDATE stocks SET is_active = true WHERE is_active IS NULL;
ALTER TABLE stocks ALTER COLUMN is_active SET NOT NULL;

-- 3. created_at / updated_at: default now(), backfill NULLs, enforce NOT NULL
ALTER TABLE stocks ALTER COLUMN created_at SET DEFAULT now();
UPDATE stocks SET created_at = now() WHERE created_at IS NULL;
ALTER TABLE stocks ALTER COLUMN created_at SET NOT NULL;

ALTER TABLE stocks ALTER COLUMN updated_at SET DEFAULT now();
UPDATE stocks SET updated_at = now() WHERE updated_at IS NULL;
ALTER TABLE stocks ALTER COLUMN updated_at SET NOT NULL;

-- 4. symbol: ORM is NOT NULL (stub left it nullable); 0 rows so safe to enforce
ALTER TABLE stocks ALTER COLUMN symbol SET NOT NULL;

-- 4b. id autoincrement default. The ORM declares id SERIAL (autoincrement) and
--     all insert paths (scraper, Stock(...)) OMIT id, relying on a server
--     default. The observed stub RETAINED nextval('stocks_id_seq'), but a
--     variant stub created as a plain `id INTEGER PRIMARY KEY` would have no
--     default and every insert would then fail on a NULL id. Guard
--     idempotently: only act if the default is missing; harmless otherwise.
DO $$
BEGIN
    IF (SELECT column_default
          FROM information_schema.columns
         WHERE table_name = 'stocks' AND column_name = 'id') IS NULL THEN
        CREATE SEQUENCE IF NOT EXISTS stocks_id_seq OWNED BY stocks.id;
        ALTER TABLE stocks ALTER COLUMN id SET DEFAULT nextval('stocks_id_seq');
        -- advance past any existing ids (table is empty here -> next id = 1)
        PERFORM setval('stocks_id_seq', COALESCE((SELECT MAX(id) FROM stocks), 0) + 1, false);
    END IF;
END
$$;

-- 5. ORM declares index=True on symbol (separate from the unique constraint
--    stocks_symbol_key, which already exists). Add the plain lookup index for
--    parity; harmless if a unique index already covers it.
CREATE INDEX IF NOT EXISTS ix_stocks_symbol ON stocks (symbol);

COMMIT;

-- Post-repair verification (run separately, read-only):
--   \d stocks            -> 8 columns, symbol NOT NULL, is_active/created_at/updated_at NOT NULL
--   then: uv run rainier db init        (recreates the dropped stock_prices hypertable; idempotent)
--   then: uv run rainier scrape qu --session morning --cdp http://127.0.0.1:9222
--         -> expect "records_created: 200" (top100) + 200 (bottom100), 0 errors
