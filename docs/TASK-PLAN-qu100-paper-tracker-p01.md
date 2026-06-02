# TASK-PLAN — qu100-paper-tracker-p01-9de1 (Phase 0+1: "track it")

- **Parent design:** [docs/DESIGN-qu100-llm-feedback-loop.md](DESIGN-qu100-llm-feedback-loop.md) (codex-clean, 14 rounds) — §6 **combined Phase 0+1**.
- **Priority:** P2 · **Approved:** operator 2026-06-02 (defaults OK; combined Phase 0+1 first).
- **Base:** `main` (PR #115 merged — legacy engine → local) · **Branch:** `worker/qu100-paper-tracker-p01-9de1`
- **Scope = ONE PR.** Phase 0 (price ingest) and Phase 1 (paper tracker) are co-dependent (`--universe active` needs `paper_trade`) → ship together.
- **Engine:** everything on the **legacy local-TimescaleDB** engine (`core.database.get_session`), joining `thesis_evaluations`/`stock_prices`. NOT Neon.
- **Parallel-review (2026-06-02) deltas folded in:** (a) **fill-validity guard** — if T+1 open gapped past stop/target, skip `gap_invalidated` (source engine `qu100_portfolio.py:505-507`); (b) **freeze booked fills** + `price_basis` field (splits); (c) **same-symbol/day tiebreak** = highest `llm_confidence`, tie→earliest session (`same_symbol_lower_conviction` skip); (d) migration applies to **`$LEGACY_DATABASE_URL`** (NOT `$DATABASE_URL`); (e) **do NOT reuse `_save_prices_to_db`** (DO-NOTHING + symbol-level); (f) creation seam `_compute_theses_async` (`service.py:567`), **each insert its own `get_session()`** scope (clean-txn on conflict); (g) handle **NULL OHLC**; (h) daily report shows **realized AND MTM-including-open** + `same_bar_ambiguous_exits` (realization-asymmetry disclosure). Detailed cases: [TEST-SPEC-qu100-paper-tracker-p01.md](TEST-SPEC-qu100-paper-tracker-p01.md).

## Suggested build order (TDD; checkpoint WIP between steps)

1. **Migrations** (applied to **`$LEGACY_DATABASE_URL`**, + ORM models in `core/models.py` for fresh `db init`) — `paper_trade` (plain table; PK `id`, `UNIQUE(thesis_id)`, partial-unique `UNIQUE(symbol) WHERE status IN ('pending','open')`; fill fields incl. `price_basis` NULL while pending; `time_stop_days` nullable); `screened_stocks` level columns (`ADD COLUMN IF NOT EXISTS`, not `pattern_type`); `paper_report_snapshot`; `paper_skip` `(id PK, thesis_id UNIQUE, symbol, scan_date, reason check ∈ {symbol_already_active, missing_levels, missing_screened_record, gap_invalidated, same_symbol_lower_conviction}, created_at)`. See design §4.
2. **Price ingest** — `rainier db ingest-prices [--universe qu100|active|screened]`; `active` = `paper_trade` rows `status IN ('pending','open')`. Per-`(symbol,date)` gap detection over a recent window + re-fetch + `(symbol,date)` upsert; **canonical daily-bar tz normalization** (trading-date 00:00 UTC) applied identically on gap-check and upsert. Idempotent.
3. **Screened persistence** — populate the new level cols in `persist_screened_stocks`; switch its `ON CONFLICT DO NOTHING` → `DO UPDATE` of the level cols (when null).
4. **Thesis-time position creation** — in `service.compute_theses_and_persist`, for each `setup_long` passing D1 (`llm_confidence ≥ 6`), idempotent insert `ON CONFLICT(thesis_id) DO NOTHING`, **levels read from the persisted `screened_stocks` row** (not the in-memory candidate — Tier-1 cache replay safety); `time_stop_days` snapshot from `learned_time_stop_days` config (NULL until Phase 2 learns one). Record skipped duplicate picks.
5. **Fill** — `fill_pending_positions`: fill each pending at its **T+1 open** (sizing `shares = floor(10000/entry)` + residual cash) → status=open; pending with no T+1 open after 2 sessions → status=expired.
6. **Exit** — pure `evaluate_exit(position, ordered stock_prices from entry_date inclusive)`: SL via `day_low`, TP via `day_high`, same-day SL+TP → SL, gap-through → exit at open, **no time-exit when `time_stop_days` is NULL** (D6 baseline), apply the snapshotted `time_stop_days` as a `time_stop` exit when set. `update_open_positions` closes triggered positions with `exit_*`/`return_pct`/$`pnl`; idempotent (no double-close).
7. **Daily wiring** — extend `run_daily_eval` (17:00) to the authoritative order: (i) ingest `active ∪ screened` → (ii) fill → (iii) update → (iv) existing horizon eval → (v) render+persist+send the daily report.
8. **Report + CLI** — `paper_report_snapshot` daily payload; `rainier paper {open,update,report}` (`report` plain = re-render from snapshot; `--regenerate` = recompute from `paper_trade`/`stock_prices`/`screened_stocks`/`money_flow_snapshots`/`analysis_results` then upsert). Daily Discord push.

## Acceptance criteria

1. Migrations create the three objects with the exact constraints (incl. partial-unique); `downgrade` reverses them; `market.*` untouched.
2. Ingest: idempotent; fills per-`(symbol,date)` gaps (stale-row symbol DOES get today); normalization yields one canonical row regardless of yfinance time-of-day; `--universe active` = pending∪open.
3. A `setup_long` (conf≥6) creates exactly one `paper_trade` (idempotent on `thesis_id`); a re-picked open symbol is skipped + recorded; levels come from the persisted screened row.
4. Fill at T+1 open with correct sizing; missing T+1 open → expire after 2 sessions.
5. `evaluate_exit`: SL/TP/gap-through/same-day-SL+TP→SL correct; **no forced exit when `time_stop_days` NULL**; honors a snapshotted `time_stop_days`. $ P&L correct. No look-ahead. Idempotent day re-run.
6. Daily job runs ingest→fill→update→eval→report in order; daily report persists to `paper_report_snapshot` and pushes to Discord.
7. `rainier paper report` re-renders from snapshot; `--regenerate` reproduces from raw inputs (incl. `analysis_results`).
8. Full suite green; ruff clean.

## Tests (design §7 — Phase 0+1 set)

ingest idempotency/gap/normalization/`--universe active`; tracker: setup_long-only + conf gate, partial-unique dedupe + skipped recorded, `ON CONFLICT(thesis_id)`, T+1 fill + expire, sizing, `evaluate_exit` (SL/TP/same-day→SL/gap-through/no-time-exit-when-NULL/applies-snapshot), $ P&L, idempotent re-run, no look-ahead; daily snapshot + both re-render paths; migrations up/down + `DO UPDATE` backfill. Deterministic (fixture prices/theses, no network).

```bash
uv run pytest tests/ -q
uv run ruff check src/ tests/
```

## Non-goals (this PR)

- **No** calibration block, `discover_time_stop` (the learning/recommendation), or weight-tuning (Phase 2). **But the `time_stop` mechanism IS in scope** — the `time_stop_days` field, its fill-time snapshot from config, and `evaluate_exit`'s `time_stop` branch ship here (value NULL = no time-exit until Phase 2 learns one).
- **No** weekly miss-sweep (Phase 3). **No** Neon mirror / website (Phase 4).
- **No** changes to `db/engine.py` / `dualwrite.py` / `market.*` / legacy-engine wiring.
- Don't run `rainier jobs sync` (operator syncs crontab if a new cron is added; daily wiring extends the existing `run_daily_eval`, so likely no new crontab entry).

## Dependencies

PR #115 merged ✓ (legacy engine → local). No other.

## Review gates (Subagent Dispatch Contract)

- Worker: implement in the build order + tests, local commits, stop at `review-pending`. No `/review`, no push.
- Reviewer: `codex review` + `/review` until clean (two consecutive clean codex). Scrutinize: no-look-ahead in `evaluate_exit`, the tz-normalization in ingest, the `ON CONFLICT(thesis_id)` + partial-unique race path, snapshot vs `--regenerate` paths.
- Finisher: push, open PR against `main`, report URL.
