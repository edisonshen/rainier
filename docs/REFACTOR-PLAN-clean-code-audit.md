# Rainier Refactor Plan — Test Recheck, Mutation Testing & Clean Code Audit

Date: 2026-09-01. Baseline: `main` @ current HEAD.

This document is the output of a full audit: (1) re-running every test, (2) mutation
testing the core price-action pipeline, (3) a module/dependency analysis, and (4) a
Clean Code (Robert C. Martin) conformance review. It ends with a phased refactor
roadmap ordered by risk-adjusted payoff.

---

## 1. Test recheck — current state

| Run | Result |
|---|---|
| `uv run pytest tests/` (no Postgres binaries) | 1793 passed, **352 skipped** |
| Same, with `postgresql-14` on PATH | **2129 passed, 16 skipped, 0 failed** |
| `uv run ruff check src/ tests/` | clean |

Findings:

- **The suite is green.** No broken tests anywhere, including all
  `requires_postgres` tests once `pg_config`/`initdb` are available
  (pytest-postgresql spins up throwaway clusters).
- **352 tests (17%) silently skip without Postgres.** Any dev box or CI runner
  without postgres server binaries quietly loses a sixth of the suite — including
  all migration, dual-write, backup/reconcile and schema-drift tests. Action:
  install `postgresql` in the dev/CI environment (done in the Devin blueprint) and
  make CI fail if the skip count exceeds a threshold.
- Line coverage is **69%** overall, but very unevenly distributed:
  - 0%: `backtest/qu100_backtest.py` (587 stmts!), `dashboard/app.py`,
    `reports/daily.py`, `data/yfinance_provider.py`, `data/persistence.py`,
    `signals/journal.py`, `research/schemas.py`, `backtest/report.py`
  - <35%: `viz/charts.py` (13%), `scheduler/jobs.py` (22%),
    `backtest/qu100_portfolio.py` (23%), `paper/ingest.py` (30%),
    `paper/positions.py` (32%), `paper/reclaim.py` (32%)
  - The *live money paths* (paper positions/ingest/reclaim, scheduler jobs) are
    among the least tested code in the repo. That is inverted risk.
- 16 remaining skips are environmental (TimescaleDB extension, live LLM), fine.
- 28 deprecation warnings, dominated by **kaleido < 1.0** (support removed after
  Sept 2025 — i.e. already past). Upgrade `kaleido`/plotly image export soon.

## 2. Mutation testing (mutmut 3, core pipeline)

Scope: `analysis/{pivots,pinbar,sr_horizontal,sr_diagonal,bias,inside_bar,analyzer}`,
`signals/{generator,scorer,emitter}`, `backtest/engine`, `features/{extractor,labels}`
— the deterministic heart of the system. 2 846 mutants generated.

**Result: 1 397 killed, 1 019 survived, 430 had no covering tests.**
Kill rate on covered code: **58%** — a serious gap for numeric trading logic
(healthy suites kill 80–90%+). Highlights:

| Hotspot | Survived | What it means |
|---|---|---|
| `backtest/engine.py compute_metrics` | 103 | Metrics fields largely unasserted — e.g. deleting `largest_win=` entirely survives. Tests only check a few fields. |
| `backtest/engine.py _close_trade` | 68 | PnL/slippage/commission arithmetic mutable without failure. |
| `analysis/pinbar.py detect_pin_bars_raw` | 55 | Threshold comparisons (`>=` → `>`, constant tweaks) survive; fixtures don't sit on boundaries. |
| `features/extractor.py _add_sr_features / _add_trend_features` | 100 | Feature values not asserted numerically — only shape/NaN checks. |
| `analysis/sr_diagonal.py _fit_lines` | 45 | Line-fit math undertested. |
| `signals/generator.py _compute_levels` | 36 | Entry/SL/TP arithmetic mutable. |
| **`analysis/analyzer.py` — 317 mutants, zero covering tests** | n/a | The orchestrator (`analyze`, `analyze_multi_tf`, `_merge_multi_tf_levels`) has **no direct unit tests at all**. |
| **`analysis/bias.py` — 94 mutants, zero covering tests** | n/a | `determine_bias` / `_sma_bias` untested. |
| `signals/emitter.py PinBarSignalEmitter` | 19 no-tests | The *protocol implementation the backtest depends on* is untested. |

The mutmut config used for this run is committed in `pyproject.toml`
(`[tool.mutmut]`); re-run with `uv run mutmut run && uv run mutmut results`.

**Testing actions (highest ROI first):**
1. Add unit tests for `analysis/analyzer.py`, `analysis/bias.py`,
   `signals/emitter.py` — currently 0 tests for ~430 mutants.
2. `compute_metrics`: one golden test asserting *every* field of
   `BacktestMetrics` from a small hand-computed trade list.
3. `_close_trade` / `_compute_levels`: hand-computed PnL and level fixtures
   (long+short, with slippage/commission, boundary R:R).
4. Pin bar / S/R detectors: boundary fixtures that sit exactly on each
   configured threshold (ratio == min, touches == min_touches, etc.).
5. Feature extractor: assert exact numeric values for a tiny deterministic
   OHLCV frame, not just column presence.
6. Ratchet: run mutmut on changed files in CI (or weekly), track kill rate.

## 3. Module & dependency analysis

CLAUDE.md declares clean layering ("modules depend on protocols, not each other",
`cli.py` is the composition root). Reality has drifted:

### 3.1 Dependency-rule violations
- **`backtest/qu100_portfolio.py` imports `paper/`** (`ingest`, `calendar`) —
  breaks the hard rule "backtest imports ONLY from core". Fix: move
  `TradingCalendar`, `canonical_instant`, `normalize_to_trading_date`,
  cohort-selection into `core/` (they are domain-neutral time/calendar utilities),
  or extract a `market_calendar/` module both may use.
- **`paper/` ↔ `llm_thesis/` import cycle** (5 imports each direction). Break it
  by extracting the shared types into `core/types.py` (per the stated convention)
  or an interface in `core/protocols.py`.
- **`scheduler/` imports `paper`, `llm_thesis`, `scrapers`, `alerts` directly**
  (19 imports) — the scheduler has become a second composition root. It should
  receive callables/jobs wired in `cli.py`, per the declared architecture.

### 3.2 God module: `cli.py` — 6 722 lines
49 imports from core, 22 from db, 21 from backtest… it contains not just wiring
but real logic: `_recover` (305 lines, cyclomatic complexity **E/37**),
`db_backfill_prices` (179 lines), a 20-parameter `backtest` command.
Fix: split into a `cli/` package (`cli/futures.py`, `cli/qu100.py`, `cli/db.py`,
`cli/jobs.py`, `cli/ml.py`, `cli/dashboard.py`…) where each file only *parses
options and delegates* to a module-level function that takes a config object.
Recovery logic belongs in `scheduler/recovery.py`, backfill logic in `db/`.

### 3.3 Duplicate/confusing modules
- **`breadth/` vs `market_breadth/`** — two sibling packages with overlapping
  names and both containing a `universe_loader.py`. Merge or rename
  (`breadth/` is thematic-rank ML features; `market_breadth/` is indicator
  computation — the names don't say that).
- **`alerts/` vs `notifications/`** — two notification stacks (Discord webhooks
  vs Apprise). Apprise already supports Discord; consolidate behind one
  `Notifier` protocol in `core/protocols.py`.
- `core/test_schema_gc.py` — despite the docstring, a `test_`-prefixed module in
  `src/` is a landmine (pytest may collect it; readers assume it's a test).
  Rename to `core/schema_gc.py`.
- `signals/journal.py` — **dead code** (no importers, 0% coverage). Delete.
- `reports/daily.py` — 0% coverage, only imported by `cli.py`; verify it still
  works or delete.

### 3.4 Size/complexity hotspots (Clean Code: "functions should be small")
230 of 1 123 functions exceed 50 lines; 79 take >5 parameters. Worst:

| Function | Lines | CC |
|---|---|---|
| `backtest/qu100_portfolio.py run_qu100_portfolio_backtest` | 394 | **F (75)** |
| `backtest/tqqq_sma_report.py render_report` | 377 | D (22) |
| `viz/charts.py create_tabbed_chart` | 345 | — |
| `cli.py _recover` | 305 | E (37) |
| `breadth/ranks.py compute_thematic_features` | 289 | E (40) |
| `llm_thesis/service.py _compute_theses_async` | 233 | D (24) |

`run_qu100_portfolio_backtest` (CC 75, 23% file coverage) is the single most
dangerous function in the repo: it computes money-relevant results and is
effectively untestable in its current shape. Decompose into
`build_cohorts → simulate_day → apply_exits → compute_result` steps, each pure
and unit-tested (mirror the clean `backtest/engine.py` design, which already
does this well).

## 4. Clean Code conformance summary

What's already good: protocol-based boundaries in `core/protocols.py`; shared
types centralized; pure-function analysis modules; meaningful names; ruff-clean;
strong docstring culture; tests use synthetic fixtures.

Gaps, mapped to the book:

- **Small functions / one level of abstraction** — 230 functions >50 lines, six
  200–400-line monsters (see 3.4).
- **Function arguments** — 79 functions with >5 params; `cli.py backtest` takes
  20. Introduce parameter objects (`BacktestConfig` already exists — use it as
  the CLI boundary object).
- **No duplication (DRY)** — `breadth`/`market_breadth`, `alerts`/`notifications`
  duplicate concerns; two DB engines (`core/database.py` psycopg2 legacy vs
  `db/` psycopg3 canonical) doubles every persistence code path. Finish the
  canonical-store migration and delete the legacy engine.
- **Boundaries** — the declared dependency rules are violated in 3 places (3.1);
  add an automated import-linter contract (`import-linter` or a unit test that
  walks the AST) so violations fail CI instead of accumulating.
- **Error handling** — several broad `except Exception` blocks in scrapers/
  scheduler swallow context; prefer narrow exceptions + structlog with cause.
- **Tests (FIRST / one assert concept)** — the suite is fast (90 s) and
  independent, but assertion depth is shallow where it matters most (58% mutation
  kill rate, §2). Comments are heavily used to encode PR history ("codex iter-2
  [P1]…") — move that provenance to commit/PR descriptions; keep comments
  describing the code, not the diff.

## 5. Phased roadmap

Each phase is independently shippable and keeps the suite green.

**Phase 1 — Safety net first (do before any refactor)**
1. Postgres in dev/CI env so all 2 129 tests always run (env/blueprint change).
2. Kill the top mutation survivors: tests for `analyzer.py`, `bias.py`,
   `emitter.py`, `compute_metrics`, `_close_trade`, `_compute_levels` (§2).
3. Add an import-contract test enforcing CLAUDE.md's dependency rules —
   with today's 3 violations temporarily allowlisted.

**Phase 2 — Untangle dependencies**
4. Extract calendar/time utilities from `paper/` into `core/` (fixes
   backtest→paper violation), break the paper↔llm_thesis cycle, then shrink the
   allowlist to zero.
5. Delete dead code: `signals/journal.py`; audit `reports/daily.py`; rename
   `core/test_schema_gc.py`.

**Phase 3 — Split the god module**
6. `cli.py` → `cli/` package of thin command groups; move `_recover` into
   `scheduler/recovery.py` and backfill logic into `db/`; replace >8-param
   commands with config objects. Pure mechanical moves, protected by existing
   CLI tests.

**Phase 4 — Decompose the hotspots**
7. `run_qu100_portfolio_backtest` → pure steps + unit tests (biggest win).
8. `render_report`, `create_tabbed_chart`, `compute_thematic_features`,
   `_compute_theses_async` — same treatment, one PR each.

**Phase 5 — Consolidate duplicates**
9. Merge/rename `breadth` vs `market_breadth`; unify `alerts` + `notifications`
   behind one protocol; finish legacy→canonical DB engine migration and delete
   `core/database.py`.

Estimated effort: Phases 1–2 ≈ one session; Phase 3 ≈ one session;
Phases 4–5 ≈ one to two sessions, one hotspot per PR.
