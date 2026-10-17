-- Migration 0015 DOWNGRADE — reverses 0015_strategy_signals.sql.
--
-- Drops EXACTLY: strategy_signals.

BEGIN;

DROP TABLE IF EXISTS strategy_signals;

COMMIT;
