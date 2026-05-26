# INVESTIGATION — QU historical money-flow endpoint access

- **Status:** FINDINGS (no implementation in this PR)
- **Owner:** rainier coord (`qu-money-flow-backfill-ea9d`)
- **Date:** 2026-05-25
- **Parent task:** [`TASK-PLAN-qu-money-flow-backfill-ea9d.md`](TASK-PLAN-qu-money-flow-backfill-ea9d.md) (deliverable §Acceptance, item 3)
- **Parent design:** [`DESIGN-rainier-data-backfill-for-maunakea.md`](DESIGN-rainier-data-backfill-for-maunakea.md) v2 §3.1

## TL;DR

**Yes — QU exposes a single JSON endpoint that already accepts an arbitrary historical date.** The same endpoint the daily scrape hits (`/api/v3/mf100`) takes `?date=YYYY-MM-DD&top=true|false&frequency=daily` and returns the QU100 ranking that was in effect on that calendar day, going back at least to the QU product launch. Rainier's existing `rainier scrape qu --session close --start-date 2025-01-01` CLI path already drives a multi-date backfill against this endpoint with rate-limiting; **no new endpoint discovery or scraper code is required**.

A retroactive backfill of 2025-01-01 → 2026-05-13 is therefore feasible. It is NOT delivered in this PR per the plan's non-goal; this doc documents the discovery so the operator can promote a follow-up task.

## 1. What the live scraper already does

`src/rainier/scrapers/qu/scraper.py:_scrape_qu100` makes exactly two HTTP calls per target date:

```
GET /api/v3/mf100?date=YYYY-MM-DD&top=true&frequency=daily   → top100 ranking
GET /api/v3/mf100?date=YYYY-MM-DD&top=false&frequency=daily  → bottom100 ranking
```

The `date=` query param is the only date input — there is no separate "historical" endpoint. The same URL serves "today" and "any past day" alike; the SPA's date-picker UX just controls which `date=` string the React app sends.

Discovery trace (relevant source spots, all already in tree):

- `src/rainier/scrapers/qu/selectors.py:54` — `QU100_API_URL = "/api/v3/mf100"`. The single API surface used by the SPA.
- `src/rainier/scrapers/qu/scraper.py:494–501` — fetch construction:
  ```python
  url = (
      f"{sel.QU100_API_URL}?date={data_date.isoformat()}"
      f"&top={top_flag}&frequency=daily"
  )
  ```
- `src/rainier/cli.py:1206–1241` — `_run_qu_scrape` already accepts `--start-date YYYY-MM-DD`, expands to an NYSE-session list via `exchange_calendars`, and loops the scraper over each date with `backfill_delay_seconds` jitter.

There is no parallel "historical-only" URL on QU. There is no per-account quota wall the SPA exposes. There are no auth differences between today's-data and historical-data queries — the same `session` + `cf_clearance` cookies authorize both.

## 2. How far back does it go?

Confirming the depth requires hitting the live API, which this PR does NOT do (scope: investigation doc only, no DB writes). The signals that the endpoint supports the operator's target window (2025-01-01 → 2026-05-13) are:

- **CLI surface is already wired for it.** The `_run_qu_scrape` path accepts arbitrary `--start-date` strings as far back as `exchange_calendars` recognizes (XNYS calendar goes back to 1970). No code-level guard caps the lookback.
- **No date-validity error path is documented.** The scraper's failure modes (`_fetch_qu_api` raises `RuntimeError` on 4xx/5xx) treat "this date returned no data" as a soft fail per-date, not a fatal stop. So a backfill run would attempt every requested date; any "too far back" responses would just produce an empty payload (`{"data": null}`, which `_scrape_qu100` already coalesces to `[]`) and log a warning, not crash the run.
- **Operator-confirmed live history starts 2025-01-01** (QU product was already publishing money-flow rankings then; rainier's `money_flow_snapshots` table just doesn't have those rows because the scraper started on ~2026-05-13).

**Action required before a backfill commits:** smoke-test a single old date (e.g., `--start-date 2025-01-02 --days-back 1`) and inspect the returned payload. If the response is empty or 4xx, narrow the date range until a known-good earliest date is found. This smoke-test is the natural first step of the follow-up task, not blocking this investigation doc.

## 3. Rate-limiting + politeness budget

QU does not publish a rate limit. The SPA's normal usage pattern is ~2–4 calls/day (one per scrape session × 2 for top/bottom). Rainier's existing knobs:

- `QuantUnicornConfig.backfill_delay_seconds = 2.0` (default; `src/rainier/core/config.py:238`) — sleep between dates. `_run_qu_scrape` adds 0–25% jitter on top.
- 2 HTTP calls per date × 2.0s default delay → roughly **4s per trading day**.
- A 2025-01-02 → 2026-05-13 backfill is ~340 NYSE trading days. At 4s/date that's ~22.6 minutes wall-clock if zero retries.

That's well within "polite" range for a daily endpoint that's normally hit 4x/day, but the operator should still run the backfill off-peak (e.g., a weekend morning) to keep the user-visible site quiet.

## 4. What rows look like once persisted

The scraper already maps `/api/v3/mf100` responses into the production schema. A historical backfill writes the same row shape as today's daily scrape, with these caveats specific to the backfill path:

- `captured_at` — current `_scrape_qu100` uses `datetime.now(timezone.utc)` as `captured_at`, even when scraping a historical `data_date`. That means a 2025-03-15 row backfilled in 2026-05-26 carries `captured_at = 2026-05-26T<wall-clock>Z`. Per the existing schema (`MoneyFlowSnapshot.captured_at`) this is correct: `captured_at` is "when we ingested this row" and `data_date` is "what trading day the row describes". Backfill rows will be distinguishable from live rows by `captured_at - data_date` >> 0.
- `capture_session` — `_scrape_qu100` writes whatever `--session` flag was passed (e.g., `close`). For a backfill we don't really know which session-of-day the data corresponds to historically. Operator preference TBD; recommend `capture_session="backfill"` (a new value), or `capture_session="close"` (reusing the closest session-of-day semantics). The current `String(20)` column accepts either.
- `raw_data` JSONB — populated from the API payload, same as live rows.

Recommend the follow-up task introduce a `--capture-session backfill` flag on the scrape CLI so historical rows are queryable by `capture_session = 'backfill'` for any later "is this row from a backfill or a live scrape?" question.

## 5. Schema review per plan §Acceptance item 5

> Schema audit: confirm `money_flow_snapshots` carries per-day raw snapshots (not pre-aggregated 5d windows).

**Confirmed: the table is per-day raw.** Each row carries:

- `data_date` (Date) — the calendar trading day the row applies to.
- `ranking_type` (top100/bottom100) — partition by which leaderboard.
- `rank` (int) — 1..100 ordinal within (data_date, ranking_type).
- `raw_data` (JSONB) — the full API payload row, preserved for re-parsing.

Two snapshot rows per day per ranking-type × 100 ranks = **200 rows / day** (top100 + bottom100). No 5d aggregation is performed at the DB layer; any window roll-up is the consumer's responsibility (e.g., a maunakea-side feature transform). No schema change required for the v2 reframe.

## 6. Recommended follow-up tasks (NOT delivered here)

If the operator approves the backfill:

1. **`qu-money-flow-historical-backfill-XXXX`** (P2, deferred until operator approval).
   - Smoke-test depth: one-day fetch against `--start-date 2025-01-02 --days-back 1`. Confirm payload is non-empty.
   - Add `--capture-session backfill` flag (so backfill rows are queryable separately from live `close`/`morning`/etc. rows).
   - Backfill 2025-01-02 → 2026-05-12 in one run, off-peak. ~23 minutes wall-clock at the 2.0s default delay.
   - Re-run `rainier qu money-flow-coverage --lookback-days 500` to confirm the table is now contiguous.
   - Acceptance: zero missing trading days between 2025-01-02 and today.

2. **`qu-historical-bottom-coverage-XXXX`** (P3, file-and-defer).
   - If the smoke-test shows historical `top=false` (bottom100) returns thinner data than top100 (a known QU quirk on some legacy dates), document the rule and adjust the coverage CLI's `expected_trading_days` to track top100 only for that window.

## 7. If the answer had been NO

(Doc'd here per the parent task's instructions, even though the answer was YES.)

If QU had no historical endpoint we could hit, the gap would be documented per [`DESIGN-rainier-data-backfill-for-maunakea`](DESIGN-rainier-data-backfill-for-maunakea.md) D-024 non-goal: rainier does NOT retroactively synthesize money-flow data from any other source. Maunakea's backtests would be capped at `data_date >= 2026-05-13` (rainier's live-forward start), and the operator would accept that bound when designing maunakea windows. The fact that QU does expose the endpoint is the "luckier than expected" outcome that changes the math.

## 8. Source links

- `src/rainier/scrapers/qu/scraper.py:_scrape_qu100` — fetch construction + per-date loop
- `src/rainier/scrapers/qu/selectors.py:QU100_API_URL` — the single API endpoint
- `src/rainier/cli.py:_run_qu_scrape` — multi-date backfill driver (`--start-date`, `--days-back`, `--dates`)
- `src/rainier/core/config.py:QuantUnicornConfig.backfill_delay_seconds` — rate-limit knob
- `src/rainier/core/models.py:MoneyFlowSnapshot` — DB schema (per-day raw snapshots; no aggregation)

## 9. Open questions for operator

1. **Backfill `capture_session` value.** Use `"backfill"` (new value) or reuse `"close"` (closest session-of-day)? Recommend `"backfill"` for queryability.
2. **Off-peak window for the backfill run.** Pick a weekend morning slot? Or run it inline with the next maintenance window?
3. **Bottom100 historical coverage.** Smoke-test required to confirm `top=false` returns non-empty for old dates. If it's spotty, do we backfill top100 only or accept top+bottom together?
