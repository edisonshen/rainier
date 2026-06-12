-- Migration 0008 — paper_skip reason CHECK gains 'zero_share_price' (Phase 3).
--
-- The zero-share fill guard (TASK-PLAN qu100-miss-sweep-6b63 acceptance 9):
-- a T+1 open above the $10k notional floors shares = floor(10000/open) to 0;
-- the fill step now expires the pending row and records
-- paper_skip(reason='zero_share_price') instead of booking a 0-share
-- position. The 0005 CHECK (ck_paper_skip_reason) doesn't include the new
-- reason, so it is rebuilt here; the ORM CHECK in core/models.py matches.
--
-- Apply with (POST-#115 the legacy public-schema tables live on
-- LEGACY_DATABASE_URL, NOT DATABASE_URL which now points at Neon — see memory
-- project_two_database_url_engines):
--   psql "$LEGACY_DATABASE_URL" -f migrations/0008_paper_skip_zero_share.sql
--
-- ORDER: apply BEFORE deploying the Phase 3 code — the new fill guard inserts
-- reason='zero_share_price', which the old 5-reason CHECK rejects
-- (IntegrityError aborts that fill run). Old code + new schema is safe.
--
-- Idempotent: re-applying is safe (DROP IF EXISTS + ADD). market.* and every
-- other public table are untouched.

BEGIN;

ALTER TABLE paper_skip DROP CONSTRAINT IF EXISTS ck_paper_skip_reason;
ALTER TABLE paper_skip ADD CONSTRAINT ck_paper_skip_reason
    CHECK (reason IN ('symbol_already_active','missing_levels',
                      'missing_screened_record','gap_invalidated',
                      'same_symbol_lower_conviction','zero_share_price'));

COMMIT;

-- Downgrade lives in migrations/0008_paper_skip_zero_share_downgrade.sql.
