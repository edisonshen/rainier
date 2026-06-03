-- Migration 0005 DOWNGRADE — reverses 0005_paper_tracker.sql.
--
-- Drops EXACTLY: paper_report_snapshot, paper_skip, paper_trade (+ its partial
-- unique index), and the four additive screened_stocks columns (entry_price,
-- stop_loss, target_price, rr_ratio) + their checks. `pattern_type` is left
-- intact (it predates this migration). market.* and every other public table
-- are untouched.
--
-- Apply with:
--   psql "$LEGACY_DATABASE_URL" -f migrations/0005_paper_tracker_downgrade.sql
--
-- Idempotent: re-applying is safe (IF EXISTS throughout).

BEGIN;

DROP TABLE IF EXISTS paper_report_snapshot;
DROP TABLE IF EXISTS paper_skip;
-- Dropping the table removes its constraints + indexes; the explicit index
-- drop is belt-and-suspenders for a partial-index left orphaned by a failed
-- partial apply.
DROP INDEX IF EXISTS idx_paper_trade_active_symbol;
DROP TABLE IF EXISTS paper_trade;

ALTER TABLE screened_stocks DROP COLUMN IF EXISTS rr_ratio;
ALTER TABLE screened_stocks DROP COLUMN IF EXISTS target_price;
ALTER TABLE screened_stocks DROP COLUMN IF EXISTS stop_loss;
ALTER TABLE screened_stocks DROP COLUMN IF EXISTS entry_price;

COMMIT;
