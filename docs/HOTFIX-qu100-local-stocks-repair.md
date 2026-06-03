# HOTFIX — QU100 0-record P0: local `stocks` stub + drift detector

- **Severity:** P0 — QU100 morning scrape (2026-06-03 08:45 PT) persisted **0 records**; pipeline silently broken.
- **Scope:** local TimescaleDB (`$LEGACY_DATABASE_URL`, default `localhost:5432/rainier`). Neon (`$DATABASE_URL`) unaffected.
- **Branch:** `hotfix/repair-local-stocks-schema`.

## Diagnosis (evidence)

`data/qu-scrape.log` for the 08:45 run: Chrome connected, login OK, data fetched
(`qu100_data_date=2026-06-03`), then **persist failed** —
`(psycopg2.errors.UndefinedColumn) column "name" of relation "stocks" does not
exist` on `INSERT INTO stocks (symbol, name, sector, industry, is_active)` →
`records_created=0`. **The scrape itself worked; only the DB write failed.**

Live local schema (introspected 2026-06-03):

| Table | State | ORM expects |
|---|---|---|
| `stocks` | **stub: `id`, `symbol` only, 0 rows** | 8 cols (`name`,`sector`,`industry`,`is_active`,`created_at`,`updated_at`) |
| `stock_prices` | **dropped (does not exist)** | symbol-keyed hypertable |
| `money_flow_snapshots` | intact, **8,300 rows** | — (QU100 history safe) |

**Root cause:** the stock-prices migration work's *pre-fix* test fixtures ran
against the shared local DB and clobbered `public` — codex iter-2 on that PR
(#119) was exactly *"teardown used `DROP TABLE … CASCADE` on shared public
tables"*. That left `stocks` a 2-col stub and `stock_prices` gone. The isolation
fix shipped in #119 (throwaway schema), so it **won't recur** — but the local DB
is already damaged. `db init` (`create_all`) is additive only and cannot repair
an existing stubbed table, so the drift was invisible until the scrape failed.

Why it was silent: the failure Discord alert **400'd** (`Failed to send Discord
notification: 400`) — secondary alerting bug (see Follow-ups).

## Fix

### 1. Restore `stocks` — `scripts/repair_local_stocks_schema.sql`
Idempotent, non-destructive (`stocks` is empty): `ADD COLUMN IF NOT EXISTS` the
6 missing columns matching `core/models.py` Stock, set defaults/`NOT NULL` with
guarded backfills, `symbol SET NOT NULL`, add the ORM's `ix_stocks_symbol`
lookup index (the unique `stocks_symbol_key` already exists). Run inside one
transaction. **Run only against `$LEGACY_DATABASE_URL`, never Neon.**

### 2. Recreate the dropped `stock_prices`
`uv run rainier db init` — `create_all` recreates the missing table (additive;
won't touch the now-repaired `stocks` or the intact tables), `_create_hypertables`
restores the hypertable. Yields the current symbol-keyed shape.

### 3. Backfill today's data
`uv run rainier scrape qu --session morning --cdp http://127.0.0.1:9222` →
expect `records_created: 200` ×2, 0 errors. Re-runnable (snapshot table keyed on
`captured_at`).

## Regression guard — `core/schema_check.py` + `tests/test_schema_check.py`

`check_schema_drift(engine)` compares `Base.metadata` to the live schema and
returns `missing table:` / `missing column:` findings — drift `db init` can't
see. Tests (`requires_postgres`) prove it flags the exact incident: fresh
`create_all` → no drift; `stocks` missing `name` → flagged; `stock_prices`
dropped → flagged; full 6-col stub → all reported. Verified: 4 passed against an
isolated DB.

## Rollback

The repair only ADDs columns to an empty table; to undo, `ALTER TABLE stocks
DROP COLUMN …` the 6 added columns (no data loss — table was empty). `db init`
and the re-scrape are additive/idempotent.

## Follow-ups (not in this hotfix)

- **P1:** wire `check_schema_drift` into a `rainier db check` command + the scrape
  preflight so drift fails loudly (would have caught this at 08:45).
- **P1/P2:** fix the Discord failure-webhook (returning 400 → alerts silently lost).
- Confirm no other shared-DB tables were stubbed beyond `stocks`/`stock_prices`
  (introspection showed the rest intact).
