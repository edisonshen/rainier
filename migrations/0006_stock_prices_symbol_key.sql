-- Migration 0006 — realign `stock_prices` to symbol-keyed (in-place ALTER).
--
-- The `StockPrice` ORM (core/models.py) is symbol-keyed:
--   symbol VARCHAR(10) NOT NULL FK->stocks.symbol, index on symbol,
--   UNIQUE(symbol, date) = uq_stock_price_symbol_date, PK (id, date).
-- The live `stock_prices` table is still the OLD stock_id-keyed shape:
--   stock_id INTEGER FK->stocks.id, UNIQUE(stock_id, date) = uq_stock_price_date,
--   index ix_stock_prices_stock_id, PK (id, date). `create_all` never ALTERs an
--   existing table, so this drift was never migrated.
--
-- This realigns the table IN PLACE — NOT drop+recreate. Plain `ALTER TABLE`
-- works on a Timescale hypertable; recreating risks chunk/sequence/dependency
-- drift + catalog churn, and `create_hypertable` does NOT need re-running for an
-- in-place ALTER (the hypertable identity is preserved across ALTERs).
--
-- Steps (each guarded by a catalog check so a re-apply — including AFTER stock_id
-- is already gone — is a clean no-op):
--   1. ADD COLUMN symbol (nullable first).
--   2. Backfill symbol from stocks via stock_id (guarded: only if stock_id still
--      exists; after it's dropped, the backfill is skipped).
--   3. SET symbol NOT NULL.
--   4. Add FK symbol->stocks(symbol).
--   5. Add UNIQUE(symbol, date) uq_stock_price_symbol_date + index on symbol.
--   6. Drop the old stock_id FK / uq_stock_price_date / stock_id index / stock_id
--      column.
--
-- Timescale uniqueness rule: every unique index / PK on the hypertable MUST
-- include the partition column `date`. UNIQUE(symbol, date) ✓ and PK(id, date) ✓
-- are preserved; NEVER UNIQUE(symbol) or PK(id) alone.
--
-- `date TIMESTAMPTZ NOT NULL`, the `id` BIGINT sequence/default, and PK(id, date)
-- are all preserved unchanged.
--
-- Apply with (legacy local-TimescaleDB engine — NOT Neon, see memory
-- project_two_database_url_engines):
--   psql "$LEGACY_DATABASE_URL" -f migrations/0006_stock_prices_symbol_key.sql
--
-- Idempotent: re-applying is safe. market.* and all other public tables are
-- untouched.

BEGIN;

-- 1. ADD COLUMN symbol (nullable while we backfill).
ALTER TABLE stock_prices ADD COLUMN IF NOT EXISTS symbol VARCHAR(10);

-- 2. Backfill symbol from stocks via the old stock_id join — ONLY if stock_id
--    still exists (a re-run after stock_id was dropped skips this cleanly).
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'stock_prices'
          AND column_name = 'stock_id'
    ) THEN
        UPDATE stock_prices sp
           SET symbol = s.symbol
          FROM stocks s
         WHERE sp.stock_id = s.id
           AND sp.symbol IS NULL;
    END IF;
END
$$;

-- 3. SET symbol NOT NULL (idempotent: SET NOT NULL on an already-NOT-NULL column
--    is a no-op; guard avoids re-issuing the catalog rewrite needlessly and
--    keeps the intent explicit). Any NULL here means a stock_prices row whose
--    stock_id had no matching stocks row — that surfaces as an error, which is
--    correct (the FK below would reject it anyway).
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'stock_prices'
          AND column_name = 'symbol'
          AND is_nullable = 'YES'
    ) THEN
        ALTER TABLE stock_prices ALTER COLUMN symbol SET NOT NULL;
    END IF;
END
$$;

-- 4. Add FK symbol -> stocks(symbol) (discover by referenced column, not name,
--    so a pre-existing equivalent FK under any name is treated as present).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_constraint con
          JOIN pg_class rel ON rel.oid = con.conrelid
          JOIN pg_namespace ns ON ns.oid = rel.relnamespace
         WHERE rel.relname = 'stock_prices'
           AND ns.nspname = current_schema()
           AND con.contype = 'f'
           AND con.conkey = (
               SELECT array_agg(attnum)
                 FROM pg_attribute
                WHERE attrelid = rel.oid AND attname = 'symbol'
           )
    ) THEN
        ALTER TABLE stock_prices
            ADD CONSTRAINT fk_stock_prices_symbol
            FOREIGN KEY (symbol) REFERENCES stocks (symbol);
    END IF;
END
$$;

-- 5. Add UNIQUE(symbol, date) uq_stock_price_symbol_date + index on symbol.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_constraint con
          JOIN pg_class rel ON rel.oid = con.conrelid
          JOIN pg_namespace ns ON ns.oid = rel.relnamespace
         WHERE rel.relname = 'stock_prices'
           AND ns.nspname = current_schema()
           AND con.conname = 'uq_stock_price_symbol_date'
    ) THEN
        ALTER TABLE stock_prices
            ADD CONSTRAINT uq_stock_price_symbol_date UNIQUE (symbol, date);
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS ix_stock_prices_symbol ON stock_prices (symbol);

-- 6a. Drop the old stock_id FK (discover its name from the catalog by referenced
--     column — auto-name was stock_prices_stock_id_fkey).
DO $$
DECLARE
    fk_name text;
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'stock_prices'
          AND column_name = 'stock_id'
    ) THEN
        FOR fk_name IN
            SELECT con.conname
              FROM pg_constraint con
              JOIN pg_class rel ON rel.oid = con.conrelid
              JOIN pg_namespace ns ON ns.oid = rel.relnamespace
             WHERE rel.relname = 'stock_prices'
               AND ns.nspname = current_schema()
               AND con.contype = 'f'
               AND con.conkey = (
                   SELECT array_agg(attnum)
                     FROM pg_attribute
                    WHERE attrelid = rel.oid AND attname = 'stock_id'
               )
        LOOP
            EXECUTE format('ALTER TABLE stock_prices DROP CONSTRAINT %I', fk_name);
        END LOOP;
    END IF;
END
$$;

-- 6b. Drop the old uq_stock_price_date unique (on (stock_id, date)).
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
          FROM pg_constraint con
          JOIN pg_class rel ON rel.oid = con.conrelid
          JOIN pg_namespace ns ON ns.oid = rel.relnamespace
         WHERE rel.relname = 'stock_prices'
           AND ns.nspname = current_schema()
           AND con.conname = 'uq_stock_price_date'
    ) THEN
        ALTER TABLE stock_prices DROP CONSTRAINT uq_stock_price_date;
    END IF;
END
$$;

-- 6c. Drop the standalone stock_id index (auto-name ix_stock_prices_stock_id).
DROP INDEX IF EXISTS ix_stock_prices_stock_id;

-- 6d. Drop the stock_id column (CASCADE on the column itself is unnecessary —
--     its FK/unique/index are already gone above; any leftover dependent object
--     would be the unique we dropped).
ALTER TABLE stock_prices DROP COLUMN IF EXISTS stock_id;

COMMIT;

-- Downgrade lives in migrations/0006_stock_prices_symbol_key_downgrade.sql.
