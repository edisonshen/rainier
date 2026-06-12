-- Migration 0008 DOWNGRADE — reverses 0008_paper_skip_zero_share.sql.
--
-- Restores the original 5-reason ck_paper_skip_reason CHECK. Rows carrying
-- the retired 'zero_share_price' reason are DELETED first (they would violate
-- the restored constraint); the skipped theses simply leave the skip ledger —
-- their paper_trade rows (status=expired) are untouched.
--
-- Apply with:
--   psql "$LEGACY_DATABASE_URL" -f migrations/0008_paper_skip_zero_share_downgrade.sql
--
-- Idempotent: re-applying is safe.

BEGIN;

DELETE FROM paper_skip WHERE reason = 'zero_share_price';

ALTER TABLE paper_skip DROP CONSTRAINT IF EXISTS ck_paper_skip_reason;
ALTER TABLE paper_skip ADD CONSTRAINT ck_paper_skip_reason
    CHECK (reason IN ('symbol_already_active','missing_levels',
                      'missing_screened_record','gap_invalidated',
                      'same_symbol_lower_conviction'));

COMMIT;
