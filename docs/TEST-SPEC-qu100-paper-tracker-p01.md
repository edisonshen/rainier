# TEST-SPEC — qu100-paper-tracker-p01 (Phase 0+1)

Concrete, TDD-first test cases for the combined Phase 0+1 PR. Each case: **fixture →
action → expected**. Parent: [TASK-PLAN-qu100-paper-tracker-p01.md](TASK-PLAN-qu100-paper-tracker-p01.md)
· [DESIGN-qu100-llm-feedback-loop.md](DESIGN-qu100-llm-feedback-loop.md).

- **Status:** codex-reviewed (5 rounds) + 3-agent parallel review applied & re-confirmed (SHIP, no P0/P1/P2 — 2026-06-02). Decisions resolved (splits=freeze / tiebreak=highest-conf / missed-winner=coverage-only). Operator review of the cases welcome before dispatch.
- **Determinism (load-bearing):** all tests use in-memory/fixture data — **no network, no yfinance, no real LLM**. yfinance is injected/mocked; the thesis path is seeded with fixture `analysis_results`/`screened_stocks` rows. **Inject `as_of`/clock everywhere** (never real `date.today()`). **Assert persisted DB state, not log text.** Exit-evaluator input rows are explicitly ordered (or the evaluator must `ORDER BY date`).
- **Conventions:** `return_pct` is a **fraction** (`0.20`, `-0.10`), not a percentage number; dollar `pnl` excludes `residual_cash`.
- **Engine:** legacy local-TimescaleDB ORM; sqlite/in-memory pattern for logic tests + the PR #115 singleton-reset fixture; **Postgres-specific schema assertions run under `requires_postgres`** (CI), since sqlite can't model partial-unique/JSONB/checks/hypertable.
- **Shared fixtures:** `mk_prices(symbol, [(date,o,h,l,c,v)...])`; `mk_thesis(symbol, verdict, conf, entry, stop, target, scan_date, session)`; `mk_screened(...)`; injected trading-calendar so "session" counting skips weekends/holidays.

---

## A. Migrations

- **A1 paper_trade shape** (`requires_postgres`) — `id` PK; `thesis_id` UNIQUE; FKs → `analysis_results`/`screened_stocks`; `status` check ∈ {pending,open,closed,expired}; `exit_reason` check ∈ {stop_loss,target,time_stop,manual}; the fill columns incl. `time_stop_days` and **`price_basis`** exist (nullable); partial unique index predicate is **exactly** `status IN ('pending','open')`; **NOT a TimescaleDB hypertable** (catalog introspection).
- **A2 screened_stocks additive** — `entry_price/stop_loss/target_price/rr_ratio` added (nullable); `pattern_type` unchanged/not duplicated; existing rows preserved.
- **A3 paper_report_snapshot** (`requires_postgres`) — `id` PK; unique `(report_type, as_of_date)`; `report_type` check ∈ {daily,weekly}; `payload` is JSONB.
- **A4 paper_skip** (`requires_postgres` for the constraints) — `(id PK, thesis_id, symbol, scan_date, reason, created_at)` with **`UNIQUE(thesis_id)`** (idempotency) and a **`reason` check ∈ {symbol_already_active, missing_levels, missing_screened_record, gap_invalidated, same_symbol_lower_conviction}** (all five — must match D3/D1; a migration omitting any rejects a valid skip).
- **A5 downgrade** — drops exactly `paper_trade`, `paper_report_snapshot`, `paper_skip`, and the four `screened_stocks` columns (+ their indexes/checks); `market.*` and all other `public` tables untouched.
- **A6 partial-unique semantics (pending∪open)** (`requires_postgres`) — insert pairs same symbol: pending+pending → IntegrityError; pending+open → IntegrityError; open+open → IntegrityError; closed+pending → OK; expired+open → OK.

## B. Price ingest

- **B1 bootstrap** — empty `stock_prices`; ingest `['AAA']`, mock 5 bars → 5 rows.
- **B2 per-(symbol,date) gap fill** — AAA present through day 3 only; ingest recent window → days 4–5 added, 1–3 untouched (stale-row symbol DOES get new dates — regression for missing-symbol-only bug).
- **B3 idempotent re-run** — run B2 twice → 0 added on the second, no dup, no error.
- **B4 date normalization + tz/DST boundary** — bars for one trading day with differing intraday/tz timestamps (incl. one whose raw UTC instant lands on the **adjacent** calendar date, and a **US DST-transition** day in Mar/Nov) all map to **one** row keyed at the intended trading-date `00:00 UTC`; gap-check and upsert use the same key → no dup/miss (a naive `tz_localize/tz_convert` mix would fail the DST case).
- **B5 upsert overwrites adjusted** — existing day re-fetched with split-adjusted values → **all** of o/h/l/c/v update **together** (assert volume too); a partial/NaN re-fetch must **not** null-out a previously-good value; still exactly one row per `(symbol,date)`.
- **B6 `--universe active`** — fixture `paper_trade`: pending AAA, open BBB, closed CCC, expired DDD → `active` = `{AAA,BBB}` only; `screened` = today's top-50; `qu100` = full universe.
- **B7 yfinance-missing symbol** — empty result for one symbol → logged + skipped, others ingested, exit 0; assert the skip via **persisted absence**, not log text.

## C. Screened persistence

- **C1 levels populated** — `persist_screened_stocks(top50)` writes `entry_price/stop_loss/target_price/rr_ratio` per candidate.
- **C2 DO UPDATE backfill** — pre-existing row with NULL levels, re-persisted → levels filled (was `DO NOTHING`).
- **C3 no clobber of non-null** — re-persist does not overwrite already-populated levels (predicate backfills NULLs only).

## D. Thesis-time position creation

- **D1 setup_long → position** — `setup_long` conf 7 + persisted screened row (levels) → one `paper_trade` (pending), levels copied **from the screened row**, `planned_entry_price`=pattern entry, `time_stop_days`=NULL, **and all fill fields NULL while pending**: `entry_date`, `entry_price`, `shares`, `allocated_amount`, `residual_cash` all NULL (prevents using `planned_entry_price` as the actual entry before E1 fills).
- **D2 non-setup_long skipped** — `watch`/`no_setup` → no position.
- **D3 confidence gate** — conf 5 → no position; conf 6 → position (★ gate value).
- **D4 session gate** — `setup_long` from `morning`/`midday` → **no position**; only `afternoon`/`close` top-5 qualify (matches the thesis allowlist).
- **D5 idempotent on thesis_id** — run creation twice → one row (`ON CONFLICT(thesis_id) DO NOTHING`).
- **D6 levels from persisted row, not in-memory candidate** — simulate a Tier-1 cache hit (thesis returned without re-running evidence assembly; in-memory candidate differs) → the position's levels match the **persisted screened row** (cache-replay regression).
- **D7 one-active-per-symbol dedupe + skip ledger** — two fixtures: (a) AAA already **open**; (b) AAA already **pending**. In both, a new `setup_long` AAA → no second position **and** a `paper_skip` row `(thesis_id, AAA, scan_date, reason='symbol_already_active')` (concrete artifact, not a log). App catches the partial-unique conflict in a **clean transaction** (not just a pre-check on `open`) and still processes later symbols in the same batch.
- **D7b skip-ledger idempotency** — re-run creation for the same skipped thesis → **exactly one** `paper_skip` row (no duplicate; protects the feedback denominator).
- **D7c forced DB-conflict path** (`requires_postgres`) — exercise the **partial-unique `IntegrityError` directly** (not just the app pre-check): arrange an `open` AAA row to exist at insert time so the creation insert raises, in a batch `[AAA, BBB]`. Assert: the failed AAA insert **rolls back cleanly** (no aborted/poisoned transaction), a `paper_skip` row for AAA is written, **and BBB still inserts** in the same batch. This proves the DB index — not a non-race-safe pre-check — is the guard.
- **D8 null persisted levels → no position (zero rows + retry pinned)** — a `setup_long` whose screened row exists but has **any** required level NULL (parametrize `entry_price`/`stop_loss`/`target_price`/`rr_ratio`) → **zero `paper_trade` rows** for that thesis (assert count==0, not just "no open" — guard runs BEFORE the idempotent insert) + `paper_skip` `reason='missing_levels'`. **Retry:** after C2 backfills the levels, a later creation pass for the same thesis **does** create the position (the earlier skip must NOT permanently poison it via `ON CONFLICT(thesis_id)`). Pin this resolution explicitly.
- **D10 same-symbol multiple theses in one day (operator tiebreak)** — two `setup_long` AAA theses same day (afternoon conf 6, close conf 8) → exactly one position for the **conf-8** thesis; the conf-6 one → `paper_skip` `reason='same_symbol_lower_conviction'`. Tie on confidence → the **earlier session** wins. Deterministic regardless of processing order.
- **D11 session/confidence reject is NOT a skip** — a morning `setup_long` (session-gated) and an afternoon conf-4 `setup_long` (gate) → no position **and no `paper_skip` row** (they're "not a buy signal", excluded from the acted-vs-skipped denominator).
- **D9 absent screened row → no position** — a `setup_long` thesis with **no matching `screened_stocks` row** → **no position** + `paper_skip` `reason='missing_screened_record'` (must NOT fall back to in-memory candidate data — the persisted row is the source of truth).

## E. Fill

- **E1 T+1 open fill** — pending from day T; T+1 open=100 → open, `entry_price=100`, `shares=100`, `allocated=10000`, `residual_cash=0`, `entry_date`=T+1, **`price_basis` set** to the basis the fill was booked under (persisted, not in-memory-only).
- **E1b weekend/holiday T+1 = next TRADING session** — thesis on a **Friday** → fills at **Monday**'s open; assert `entry_date==Monday` (a `+1 calendar day` impl that fills Saturday/expires would fail). Exercises the injected trading calendar.
- **E1c fill-validity guard (D3, parallel-review P1)** — the pending row exists (created at thesis-time); T+1 open has **gapped past** the level (open=125, target=120; or open=88, stop=90) → the pending row transitions to **`status=expired`** (NOT left pending — assert it doesn't hold the symbol's partial-unique slot) and a `paper_skip` `reason='gap_invalidated'` is written. No `open` position is created. Prevents degenerate same-day round-trips.
- **E2 sizing residual** — T+1 open=33.0 → `shares=303`, `allocated=9999.0`, `residual_cash=1.0`.
- **E3 pending-until-priced then expire (trading sessions, boundary pinned)** — no T+1 open → stays pending; T+2 open appears → fills at T+2; if no open through **2 trading sessions** → status=expired. Pin the exact session it flips: it does **not** expire one session early, and a fixture spanning a weekend proves sessions≠calendar days.
- **E4 no double-fill** — re-run fill on an open position → no change.
- **E6 expired never resurrects** — a position that expired (E3); later the original T+1 open is back-filled by the idempotent ingest → it **stays `expired`**, no fill (symmetric partner to E4).
- **E5 fill-time time_stop snapshot (both directions)** — (a) config `learned_time_stop_days=10` at fill → position stores `10`; later config change/null → already-filled position **keeps 10**. (b) **NULL→10 (the dangerous one):** a position filled while config is NULL stores `time_stop_days=NULL`; config later adopts `10` → that already-open position **stays NULL** and is **never retroactively time-stopped** (D6 future-fills-only invariant).

## F. Exit evaluator `evaluate_exit` (pure fn, fixture OHLC)

- **F1 target hit** — entry 100, target 120, stop 90; day2 high=121 → exit `target` @120, `return_pct=0.20`, `pnl=+2000` (100 sh).
- **F2 stop hit** — day2 low=89 → `stop_loss` @90, `return_pct=-0.10`, `pnl=-1000`.
- **F3 same-day SL+TP → SL** — a day with low=89 AND high=121 → `stop_loss` (conservative), not target.
- **F4 gap-through stop** — day opens 85 (<90) → exit @ **open 85**; pnl reflects 85.
- **F5 gap-through target** — day opens 125 (>120) → exit @ **open 125**.
- **F6 same-day entry exit** — entry-day (T+1) low ≤ stop → exit same day (walks from `entry_date` inclusive).
- **F7 equality boundaries** — `low == stop` → stop triggers; `high == target` → target triggers; `open == stop` / `open == target` → exit at the level (define ≤/≥ inclusivity explicitly).
- **F7b gap-through + straddle corner** — a bar that **opens == stop** AND `high ≥ target` (gaps to stop, then rallies through target) → `stop_loss` @ open (SL-first AND gap-through both apply; catches an impl that checks the target-gap first).
- **F8 unordered input** — pass price rows out of date order → evaluator sorts (or `ORDER BY date`); result identical to ordered input.
- **F9 no time-exit when time_stop_days NULL** — 50-day flat path, never hits SL/TP, NULL → stays open (D6 baseline).
- **F10 honors snapshotted time_stop_days (off-by-one)** — flat path, `time_stop_days=10`, explicit trading-session dates → force-close `time_stop` at the **10th session's close** (entry day = session 1; not entry+10, not calendar day 10).
- **F11 no look-ahead BEFORE entry** — rows include days before `entry_date` → ignored.
- **F12 no look-ahead AFTER as_of (P0, strengthened)** — make T+4 a **target** bar and **T+5 a stop** bar. `update_open_positions(as_of=T+2)` with prices through T+5 present → **stays open**. `as_of=T+4` → closes with **`exit_reason==target` AND `exit_date==T+4`** (an impl that peeked to T+5 would instead book a stop — so the wrong exit_reason fails the test, proving the ceiling, not just ordering).
- **F13 idempotent + immutable close** — `update_open_positions` twice → closed once, no double `pnl`; then **advance `as_of` and add a new bar that would produce a different exit** → the closed row's `exit_date/exit_price/pnl` are **unchanged** (catches a re-close that mutates an already-closed position).
- **F14 $P&L precision** — fixed numeric cases asserting exact `return_pct` (fraction) and dollar `pnl`; residual cash not counted as P&L. Include a **fractional-entry** case (entry 33.0, 303 shares): assert `pnl == shares*(exit-entry)` (303-based), **not** `return_pct * 10000` notional (catches the source engine's ×100 / notional confusion).
- **F15 NULL-OHLC day** (parallel-review P2) — `stock_prices` rows are nullable; a day with NULL high/low/open → `evaluate_exit` treats it as a no-data day (skips, does not crash, does not phantom-trigger); a NULL T+1 open → fill treats it as "not priced yet" (E3 path), not a 0 fill.
- **F16 split-basis freeze** (operator decision) — a mid-hold split auto-adjusts `stock_prices`; assert a closed position's booked `entry_price/exit_price/pnl/shares` are **unchanged**, and an open position's `evaluate_exit` walks the OHLC on the position's `price_basis` so no phantom stop/target hit appears from the basis mismatch.

## G. Daily wiring order

- **G1 order** — `run_daily_eval` calls **ingest → fill → update → horizon-eval → report** (spy each); ingest target = `active ∪ screened`.
- **G2 ingest precedes fill** — a pending whose T+1 open is only produced by the ingest step → it fills in the same run.
- **G3 horizon eval unaffected** — existing `ThesisEvaluation` path still runs (after update).

## H. Report snapshot + CLI

- **H1 daily snapshot payload (pinned fields)** — over fixture `paper_trade`, the daily `paper_report_snapshot.payload` contains, with asserted values: counts by status; **realized $P&L (closed only) AND mark-to-market-including-open** (both — never realized-only, per the realization-asymmetry disclosure); open MTM from the as-of close; today's exits keyed by `exit_date==as_of`; win-rate over closed trades only; `same_bar_ambiguous_exits` count; total residual cash. (Pin the schema so a skeleton payload fails.)
- **H1b MTM when as-of close absent** — an open position whose symbol has **no price row on `as_of`** (holiday/halt/gap): pin the MTM source (last-available close, with the as-of staleness flagged) — not NULL-that-crashes, not silently 0. Common under the indefinite-hold baseline.
- **H2 idempotent upsert** — re-run same `as_of` → one row, upserted.
- **H3 plain re-render from snapshot** — `rainier paper report --date D` reads the **snapshot only**: mutate underlying `paper_trade` after the snapshot, re-render → matches the **snapshot**, not the mutated data.
- **H4 `--regenerate`** — recomputes from raw inputs (`paper_trade/stock_prices/screened_stocks/money_flow_snapshots/analysis_results`) and **upserts** → assert **still one row** for `(daily, as_of)` (no dup), and a subsequent plain `paper report --date D` returns the **regenerated** payload (proves the upsert actually landed in the snapshot the plain path reads — not just an in-memory render).
- **H5 weekly scope** — `--week` (plain) renders an existing weekly snapshot from snapshot-only; **`--week --regenerate` is out of scope (Phase 3)** → asserts a clear "not-yet-implemented/Phase 3" error (not a silent wrong result).
- **H6 Discord non-fatal** — send failure → logged, report still persisted; **no webhook configured** → also non-fatal (skip + persist).

## I. End-to-end

- **I1 happy path** — seed `setup_long` (afternoon, conf 7) + screened row + price path hitting target day 4; run the daily job across T+1..T+5 with `as_of` advancing; assert: pending→open (T+1), stays open T+2/T+3 (no look-ahead), closes `target` day 4 with exact $pnl, a `paper_report_snapshot` each day. Deterministic.

---

## Decisions (resolved with operator 2026-06-02)

- **Splits:** freeze booked fills + `price_basis` (F16). **Same-symbol/day:** highest `llm_confidence`, tie→earliest session (D10). **Missed-winner:** coverage diagnostic only.
- Confirmed defaults: confidence gate `llm_confidence ≥ 6` (D3); `paper_skip` table kept (A4); floor-shares + residual cash (E2); gap-through→exit-at-open (F4/F5/F7b); equality inclusive `low ≤ stop` / `high ≥ target` (F7).

## Still-open ★ (non-blocking; can settle in review)

- Dividends/fees: omitted, disclosed as a downward-bias caveat — OK to leave out of Phase 0+1?
- Halted/no-trade or symbol-leaves-QU100 mid-hold: F15 covers NULL-OHLC; confirm a multi-day price gap doesn't need a forced `manual`/stale exit in Phase 0+1.
