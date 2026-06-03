-- Migration 0006 DOWNGRADE — reverses 0006_stock_prices_symbol_key.sql.
--
-- Restores the OLD stock_id-keyed `stock_prices` shape IN PLACE (NOT
-- drop+recreate), data-preserving and symmetric to the up migration:
--   1. ADD COLUMN stock_id (nullable while we backfill).
--   2. Backfill stock_id from stocks via the symbol join (guarded: only if
--      symbol still exists).
--   3. Assert no NULL stock_id (a NULL means a row whose symbol had no stocks
--      match — refuse to drop symbol and lose the key).
--   4. SET stock_id NOT NULL (restore the old non-null FK shape).
--   5. Drop the symbol FK / uq_stock_price_symbol_date / symbol index.
--   6. Add stock_id FK -> stocks(id) + UNIQUE(stock_id, date) uq_stock_price_date
--      + the standalone ix_stock_prices_stock_id index.
--   7. Drop the symbol column.
--
-- PK(id, date), `date TIMESTAMPTZ NOT NULL`, the `id` sequence/default, and the
-- hypertable identity are all preserved. market.* and other public tables are
-- untouched.
--
-- Apply with:
--   psql "$LEGACY_DATABASE_URL" -f migrations/0006_stock_prices_symbol_key_downgrade.sql
--
-- Idempotent: re-applying is safe (guards check catalog state throughout).

BEGIN;

-- 1. ADD COLUMN stock_id (nullable while we backfill).
ALTER TABLE stock_prices ADD COLUMN IF NOT EXISTS stock_id INTEGER;

-- 2. Backfill stock_id from stocks via symbol — ONLY if symbol still exists.
DO $$
BEGIN
    IF EXISTS (
        -- Resolve the schema of the table the UNQUALIFIED `stock_prices` binds
        -- to (mirror of the up migration; `current_schema()` is wrong when the
        -- search_path's first schema doesn't contain the table).
        SELECT 1 FROM information_schema.columns
        WHERE (table_schema, table_name) = (
                  SELECT n.nspname, c.relname
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                   WHERE c.oid = to_regclass('stock_prices')
              )
          AND column_name = 'symbol'
    ) THEN
        UPDATE stock_prices sp
           SET stock_id = s.id
          FROM stocks s
         WHERE sp.symbol = s.symbol
           AND sp.stock_id IS NULL;
    END IF;
END
$$;

-- 3. Assert no NULL stock_id remains (only meaningful when rows exist + a symbol
--    failed to resolve). Refuse the downgrade rather than silently lose the key.
DO $$
DECLARE
    n_null bigint;
BEGIN
    SELECT count(*) INTO n_null FROM stock_prices WHERE stock_id IS NULL;
    IF n_null > 0 THEN
        RAISE EXCEPTION
            'downgrade 0006: % stock_prices rows have NULL stock_id (symbol did not resolve to a stocks row); aborting to avoid data loss',
            n_null;
    END IF;
END
$$;

-- 4. SET stock_id NOT NULL (restore the old non-null FK shape).
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE (table_schema, table_name) = (
                  SELECT n.nspname, c.relname
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                   WHERE c.oid = to_regclass('stock_prices')
              )
          AND column_name = 'stock_id'
          AND is_nullable = 'YES'
    ) THEN
        ALTER TABLE stock_prices ALTER COLUMN stock_id SET NOT NULL;
    END IF;
END
$$;

-- 5a. Drop the symbol FK (discover by referenced column — up-migration named it
--     fk_stock_prices_symbol, but match by conkey for robustness).
DO $$
DECLARE
    fk_name text;
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE (table_schema, table_name) = (
                  SELECT n.nspname, c.relname
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                   WHERE c.oid = to_regclass('stock_prices')
              )
          AND column_name = 'symbol'
    ) THEN
        FOR fk_name IN
            SELECT con.conname
              FROM pg_constraint con
              JOIN pg_class rel ON rel.oid = con.conrelid
             WHERE rel.oid = to_regclass('stock_prices')
               AND con.contype = 'f'
               AND con.conkey = (
                   SELECT array_agg(attnum)
                     FROM pg_attribute
                    WHERE attrelid = rel.oid AND attname = 'symbol'
               )
        LOOP
            EXECUTE format('ALTER TABLE stock_prices DROP CONSTRAINT %I', fk_name);
        END LOOP;
    END IF;
END
$$;

-- 5b. Drop the uq_stock_price_symbol_date unique.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
          FROM pg_constraint con
         WHERE con.conrelid = to_regclass('stock_prices')
           AND con.conname = 'uq_stock_price_symbol_date'
    ) THEN
        ALTER TABLE stock_prices DROP CONSTRAINT uq_stock_price_symbol_date;
    END IF;
END
$$;

-- 5c. Drop the symbol index.
DROP INDEX IF EXISTS ix_stock_prices_symbol;

-- 6a. Add stock_id FK -> stocks(id) (discover by referenced column so a
--     pre-existing equivalent FK is treated as present).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_constraint con
          JOIN pg_class rel ON rel.oid = con.conrelid
         WHERE rel.oid = to_regclass('stock_prices')
           AND con.contype = 'f'
           AND con.conkey = (
               SELECT array_agg(attnum)
                 FROM pg_attribute
                WHERE attrelid = rel.oid AND attname = 'stock_id'
           )
    ) THEN
        ALTER TABLE stock_prices
            ADD CONSTRAINT stock_prices_stock_id_fkey
            FOREIGN KEY (stock_id) REFERENCES stocks (id);
    END IF;
END
$$;

-- 6b. Add UNIQUE(stock_id, date) uq_stock_price_date.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_constraint con
         WHERE con.conrelid = to_regclass('stock_prices')
           AND con.conname = 'uq_stock_price_date'
    ) THEN
        ALTER TABLE stock_prices
            ADD CONSTRAINT uq_stock_price_date UNIQUE (stock_id, date);
    END IF;
END
$$;

-- 6c. Restore the standalone stock_id index for true symmetry.
CREATE INDEX IF NOT EXISTS ix_stock_prices_stock_id ON stock_prices (stock_id);

-- 7. Drop the symbol column.
ALTER TABLE stock_prices DROP COLUMN IF EXISTS symbol;

COMMIT;
