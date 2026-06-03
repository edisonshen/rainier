# DESIGN — Rainier "Slim" Audit

- **Task:** `rainier-slim-audit-2732` (P1, operator-requested). **Type:** read-only audit → plan.
- **Base:** `main`. **Branch:** `worker/rainier-slim-audit-2732`. **Deliverable:** this doc (+ `.html` companion).
- **Date:** 2026-06-03. **Scope:** the 7 audit dimensions in `TASK-PLAN-rainier-slim-audit.md`. No cuts landed — this is the plan; follow-up cut tasks are listed but **not** filed.

> **Measurement note.** Disk/git facts (`data/`, `.git`, `reports/`) were measured against the **main repo** at `/Users/pinkbear/projects/rainier` — the worktree is sparse (`.git` is a link file, `data/` absent), so its `du` does not reflect real footprint. Code/test/dep facts (tracked content) were measured in the worktree; tracked content is identical, so this is sound.

---

## 0. Ranked summary — what to cut, by (savings ÷ risk)

Headline: the **real disk pressure is local-only generated data, not git**, and the **real git bloat is committed rendered HTML reports**. Source code is lean and ruff-clean; dead code is small and surgical. Dependencies are the second-biggest lever (install footprint), gated behind one decision.

| # | Recommendation | Saves | Risk | Split | Dim |
|---|---|---|---|---|---|
| R1 | Stop committing rendered backtest HTML/xlsx/PDF; gitignore `docs/*.html`, `reports/*.html`, `*.xlsx` artifacts already-tracked | ~28 MB working tree; halts future `.git` growth | **low** — regenerable from `backtest`/`report` CLI | **safe-now** | 1 |
| R2 | `git rm --cached` the 8 large committed artifacts (keep on disk), so HEAD stops shipping them | ~28 MB out of HEAD checkout | **low** | **safe-now** | 1 |
| R3 | Document/automate `data/cache/tqqq_sma/results.parquet` (570 MB) as a regenerable sweep artifact; add a `clean` make-target or `--no-cache-results` flag | 544 MB local disk | **low** — already gitignored; pure local hygiene | **safe-now** | 1 |
| R4 | Delete dead code: `apis/x/` tree (193 LOC), `backtest/export.py::export_equity_curve`+`export_summary`, `qu100_backtest.py::run_qu100_backtest_skip_entry`, `stock_screener.py::_best_pattern`, `cli.py::_make_sample_candidates` | ~280 LOC | **low** — zero callers (evidence below) | **safe-now** | 3 |
| R5 | Consolidate Playwright mock helpers (`fake_goto`, `fake_fetch`, `_make_scraper`, `_make_mock_page`) into `tests/test_scrapers/conftest.py` | ~150–250 test LOC | **low** | **safe-now** | 4 |
| R6 | Move heavy ML/sci/scrape/viz deps to optional extras (`[ml]`, `[scrape]`, `[viz]`); make core install lean | ~600–800 MB install footprint | **medium** — changes install contract; cron/runtime paths must pull the right extras | **needs-approval** | 2 |
| R7 | `git filter-repo` to purge the ~88 MB of historical HTML/PDF report blobs from `.git` | ~85–90 MB `.git` | **high** — rewrites history, invalidates clones/PRs, operator-gated | **needs-approval** | 1 |
| R8 | Fix doc rot: CLAUDE.md says "10 ORM tables" (actually 18); module map lists `trader/`/`dashboard/` as placeholders (`trader/` is empty, `dashboard/` is live). Triage 57 untracked `docs/*.md`. | 0 MB, correctness | **low** | **safe-now** | 7 |
| — | `beautifulsoup4` is **not** in `src/` (only `scripts/seed_sp500_universe.py`). Keep declared (script is real) but note it's not a runtime dep. | — | — | **do-not (yet)** | 2 |

**Do-not list (explicitly):** do not remove `ib_insync` (wired into `cli.py` + `config.py` + tested), `kaleido`/`psycopg`/`psycopg2` (runtime-only deps used via plotly export + DB driver URLs, no Python import is expected), the two `get_engine` factories (intentional two-engine split), or the parallel `migrations/`+`db/alembic/` systems (intentional public-vs-`market.*` split). None are bloat.

---

## 1. Repo disk footprint

### data/ — 620 MB, entirely untracked (gitignored)

`du -sh data/*` (main repo):

| Path | Size | Tracked? | Class |
|---|---|---|---|
| `data/cache/tqqq_sma/results.parquet` | **570 MB** | no (`data/` gitignored) | generated — TQQQ/SQQQ SMA sweep output (~3.35M backtests) |
| `data/cache/sp500_universe.parquet` | 13 MB | no | regenerable (seed script) |
| `data/fetch.log` | 52 MB | no | log — rotate/truncate |
| `data/cache/*.parquet` (thematic/macro/breadth) | ~6 MB | no | regenerable feature caches |
| `data/*.log` (8 logs) | ~1 MB | no | logs |

**Evidence:** `git ls-files data/ | wc -l` → `0` (nothing tracked). `git check-ignore data/cache/` → `data/cache/`. So the 620 MB is **local working state, not in git** — it never bloats clones. The single 570 MB `results.parquet` is the entire disk story.

**Recommendation R3 (safe-now):** this is pure local hygiene. The sweep writes a fingerprinted results cache; document it as regenerable and add a `clean-cache` target (or a `--results-cache none` flag on the sweep CLI). No git impact. Logs should rotate (`data/fetch.log` at 52 MB).

### Tracked artifacts — 28 MB of rendered reports in HEAD

`git ls-files | grep -E '\.(html|xlsx|pdf)$' | xargs du -ch` → **28 MB total**. The offenders (all regenerable):

| File | Size | Regenerable by |
|---|---|---|
| `reports/tqqq_strategy_report.html` | 7.0 MB | `scripts/backtest_tqqq_full_report.py` |
| `reports/tqqq_sma_sma_report.html` | 6.3 MB | `backtest/tqqq_sma_report.py` |
| `docs/tqqq-sma-backtest-report.html` | 6.2 MB | sweep report |
| `docs/tqqq-sma-backtest-phase2-report.html` | 6.2 MB | sweep report |
| `docs/time_series_momentum.pdf` | 956 KB | external paper (keep — source-of-truth reference) |
| `docs/tqqq_top20_all_combos.html` | 676 KB | `scripts/backtest_tqqq_top20.py` |
| `reports/tqqq_strategy_demo.html` + `docs/tqqq_strategy_demo.html` | 656 KB | demo script (duplicated across two dirs) |
| `reports/tqqq_strategy_report.xlsx` | 84 KB | `scripts/backtest_tqqq_excel.py` |

`.gitignore` **already** matches `reports/tqqq_*` and `docs/DESIGN-*.html` etc., but these specific files were committed **before** the ignore rules and remain tracked (git keeps tracking files already in the index regardless of `.gitignore`).

**R1 (safe-now):** extend `.gitignore` to cover `docs/*-backtest-*.html`, `docs/tqqq_*.html`, `reports/*.html`, `reports/*.xlsx` (keep `time_series_momentum.pdf` and any genuinely-authored docs). **R2 (safe-now):** `git rm --cached` the 8 artifacts (files stay on disk) so HEAD stops shipping ~28 MB. Together these stop the bleeding without touching history.

### .git history — 97 MB objects, ~88 MB is report blobs

`du -sh .git/objects` → **97 MB**. Sum of blob sizes per path across all history (`git rev-list --objects --all | git cat-file --batch-check | awk`):

| Path (summed across all versions in history) | Total |
|---|---|
| `docs/tqqq-sma-backtest-report.html` | 25.9 MB |
| `docs/tqqq-sma-backtest-phase2-report.html` | 19.4 MB |
| `reports/tqqq_strategy_report.html` | 13.8 MB |
| `src/rainier/cli.py` | 11.5 MB (across ~hundreds of commits — legit churn) |
| `uv.lock` | 7.5 MB (churn — legit) |
| `reports/tqqq_sma_sma_report.html` | 6.6 MB |
| `docs/sqqq-sma-backtest-report.html` | 6.4 MB |

The HTML/PDF report blobs sum to **~85–90 MB** of the 97 MB — i.e. **~90% of `.git` is historical rendered reports**, many near-identical 6 MB versions.

**R7 (needs-approval, high-risk):** a `git filter-repo --path-glob '*backtest*report*.html' --invert-paths` (plus the demo/top20 HTML) would reclaim ~85 MB. This **rewrites history** → invalidates every clone/open-PR SHA, must be force-pushed, and is operator-gated per the dispatch contract (hard prohibition on history rewrites in *this* task; a separate operator-approved task only). Not blocking — R1+R2 prevent further growth; R7 is a one-time reclaim the operator may schedule when no PRs are in flight.

---

## 2. Dependency bloat

Method: for each declared dep, `grep -rEl "^\s*(import|from)\s+<pkg>" src/` (runtime imports, excluding tests/scripts), then verify zeros by hand.

### Direct-import counts (src/ only)

| Dep | src import sites | Verdict |
|---|---|---|
| pandas | 67 | core |
| sqlalchemy | 31 | core |
| numpy | 24 | core |
| structlog | 15 | core |
| yfinance | 10 | core |
| yaml, pydantic, scipy, plotly, playwright, httpx, exchange_calendars | 3–7 | live |
| pyarrow, sklearn, xgboost, shap, jinja2, ruamel | 2–5 | live |
| alembic, apprise, apscheduler, dotenv, hmmlearn, litellm, numba, pydantic_settings | 1 | live (single owner) |
| ib_insync | 1 (`data/ibkr_provider.py`) | **live** — also referenced in `cli.py`, `core/config.py`, `data/__init__.py`, `tests/test_ibkr_provider.py` |
| **beautifulsoup4 (bs4)** | **0 in src/** | only `scripts/seed_sp500_universe.py` |
| kaleido | 0 (import) | runtime via `plotly fig.to_image(engine="kaleido")` at `llm_thesis/chart_export.py:141` — **keep** |
| psycopg / psycopg2 | 0 (import) | DB driver via connection-string URLs (`postgresql+psycopg://…`) — **keep both** (psycopg2 = legacy engine, psycopg3 = canonical) |
| streamlit | 1 (`dashboard/app.py`) | already an optional extra `[dashboard]` ✓ |
| gymnasium | 1 (`breadth/ml_builders.py`) | already an optional extra `[rl]` ✓ |

**Evidence for zeros:** `grep -rln "bs4\|BeautifulSoup" --include="*.py"` → only `scripts/seed_sp500_universe.py`. `grep -rn "import ib_insync"` → 4 hits in `data/ibkr_provider.py`, and `ibkr_provider`/`IBKR` referenced in `cli.py` + `core/config.py` (so **not** dead — do not remove).

### Heavy install hitters → optional extras (R6, needs-approval)

The TASK-PLAN flagged kaleido (271 M), playwright (129 M), pyarrow (118 M), numba+llvmlite (131 M), scipy (82 M), litellm (55 M). Where each is used in `src/`:

| Dep cluster | Used only in | Proposed extra |
|---|---|---|
| numba (+llvmlite) | `backtest/tqqq_sma_sweep.py` (1 file) | `[ml]` or `[sweep]` |
| xgboost, shap, sklearn, hmmlearn | `ml/*` + `cli.py` ml subcommands | `[ml]` |
| scipy | `llm_thesis/research.py`, `llm_thesis/eval.py`, `breadth/ranks.py` | `[ml]` (or keep — only 3 sites) |
| playwright | `scrapers/*` (3 sites) | `[scrape]` |
| kaleido, plotly | `viz/charts.py`, `llm_thesis/chart_export.py` | `[viz]` |
| litellm | `llm_thesis/service.py` (1 site) | `[llm]` |

**Trade-off:** moving these to extras shrinks the *core* install by ~600–800 MB, but every runtime entrypoint (cron jobs, scheduler, scrapers, dashboards) would need the right extras installed, and lazy-import guards added where a module is imported eagerly. This is **medium-risk** because it changes the install contract — exactly the kind of thing that caused the two-engine P0 if a cron path silently loses a dep. **Recommend a single follow-up task** that (a) audits each entrypoint's transitive dep set, (b) introduces extras, (c) adds `pip install rainier[scrape,ml,viz,llm]` to the cron/deploy docs. Gate on operator approval since it touches how the thing is installed and run.

`uv.lock` cross-check: lock is consistent with `pyproject.toml`; no orphan top-level entries flagged. The `psycopg[binary]` + `psycopg2-binary` dual is intentional (documented inline in `pyproject.toml:20–24`).

---

## 3. Dead code & LOC

`src/` = **42,281 LOC / 164 files**. `ruff check src/ --select F401` → **clean** (no unused imports). Dead-code candidates from `uvx vulture src/rainier --min-confidence 60`, then **each verified by grep** (vulture cannot see `@click` decorator registration, so all `cli.py` "unused function" hits are false positives — confirmed by spot-checking; they are command handlers).

**Verified dead (zero callers excluding the definition; safe to delete — R4):**

| Symbol | Location | Evidence |
|---|---|---|
| `apis/x/` entire tree | `apis/x/client.py` (144 LOC), `types.py` (43), `__init__.py` (6) = **193 LOC** | `grep -rln "apis.x\|XClient"` → only `tests/test_x_client.py`; never wired into any pipeline. Twitter/X client built but unused. |
| `export_equity_curve`, `export_summary` | `backtest/export.py:36,47` | `grep -rn` outside their defs → **0 callers** in src/tests/scripts |
| `run_qu100_backtest_skip_entry` | `backtest/qu100_backtest.py:909` | **0 callers** |
| `_best_pattern` | `analysis/stock_screener.py:413` | **0 callers** (only the def) |
| `_make_sample_candidates` | `cli.py:1095` | **0 callers** (only the def) |

**Vulture false positives (do NOT delete — have real callers):** `load_thematic_panel` (5), `load_supervised` (10), `build_panel_tensor` (7), `ThematicTradingEnv` (8), `find_neckline` (5), `_format_thesis_message` (2), and all `cli.py` `@click` handlers. Counts from `grep -rn "\b<name>\b" --include=*.py src/ tests/ scripts/ | grep -v "def <name>"`.

**Empty placeholder:** `src/rainier/trader/` contains only an empty `__init__.py` (0 bytes) — Phase 3 IB-execution placeholder. Keep (roadmap), but CLAUDE.md module map describes a richer `trader/` than exists (doc rot — see R8). `dashboard/` is NOT a placeholder (has `app.py`, `render_etf.py` 949 LOC, live).

**Scripts:** `scripts/` = 5,872 LOC / 13 `.py` + 5 `.sh`. Several are clearly one-off fixture/report generators (`_make_*_fixture.py`, `backtest_tqqq_*.py`). They are tooling, not runtime; low priority. The fixture-makers (`_make_etf_fixture.py` etc.) are referenced by the tests they seed; keep.

---

## 4. Test sprawl

`tests/` = **38,259 LOC** (~0.9:1 with src — appropriate, not bloat). `ruff`-clean. Largest files:

| File | LOC |
|---|---|
| `test_db_dual_write.py` | 1329 |
| `test_tqqq_sma_sweep.py` | 1157 |
| `test_market_breadth/test_render.py` | 1114 |
| `research/test_backfill_thematic_universe.py` | 1003 |
| `test_dashboard/test_render_etf.py` | 943 |
| `test_db_migrations.py` | 941 |

These are large but cover large modules (e.g. `market_breadth/render.py` is 1675 LOC) — proportionate, not redundant.

**Real duplication (R5, safe-now):** Playwright mock helpers are copy-pasted across scraper tests. `grep -rhn "def fake_\|def _make_" tests/ | sort | uniq -c`:

- `fake_goto` defined **7×**, `fake_fetch` **7×**, `_make_scraper` **5×**, `_make_mock_page` **4×**, `fake_persist` **3×**, `fake_exec` **3×**, `_make_settings` **3×**.

Defined across `test_db_dual_write.py`, `test_db_breadth_write.py`, and 5 files in `test_scrapers/` (`test_cdp_cf_recovery.py`, `test_scraper_in_page_fetch.py`, `test_verify_session.py`, `test_cdp_auth.py`, `test_qu_api_retry.py`). Consolidating `fake_goto`/`fake_fetch`/`_make_mock_page`/`_make_scraper` into `tests/test_scrapers/conftest.py` (and a shared `_make_settings` fixture in `tests/conftest.py`) removes ~150–250 LOC of copy-paste. Meets the house-style "3+ real duplications" bar.

> Note: per the TASK-PLAN, the full suite was **not** run for timing (`pytest --collect-only` was used for inventory only). No slow-test profiling claimed here.

---

## 5. DB schema audit

**18 ORM tables** in `core/models.py` (not 10 — CLAUDE.md doc rot, see R8): candles, signals, trades, stocks, money_flow_snapshots, stock_capital_flow, capital_flow_bars, stock_prices, chart_images, monitor_readings, monitor_alerts, backtest_trading_log, analysis_results, screened_stocks, thesis_evaluations, research_insights, paper_trade, paper_skip, paper_report_snapshot. **5 hypertables** (`HYPERTABLES` dict at `core/models.py:734`): money_flow_snapshots, stock_capital_flow, capital_flow_bars, stock_prices, monitor_readings — all use composite PKs including the partition column (e.g. `capital_flow_bars` PK `(id, bar_time)`), consistent with the TimescaleDB rule in memory.

### Two-engine split — intentional, not duplication

Per memory `project_two_database_url_engines`: legacy `core.database.get_engine` (`public.*` ORM tables, local TimescaleDB, `LEGACY_DATABASE_URL`) vs canonical `db.engine.get_engine` (`market.*`, Neon, `DATABASE_URL`). Two `get_engine` definitions (`core/database.py:22`, `db/engine.py:26`) and two migration systems are **by design**:

- `migrations/*.sql` (raw SQL, run manually) → `public.*` tables (e.g. `0005_paper_tracker.sql` creates `paper_trade`/`paper_skip`/`paper_report_snapshot`; explicitly notes "market.* and all other public tables are untouched").
- `db/alembic/versions/*.py` (alembic) → `market.*` only (`0001_initial_market_schema` … `0004_backup_money_flow_snapshots`, all `schema=SCHEMA`).

No drifted/duplicate tables across the two engines. **No schema cut recommended.**

### `stock_id → symbol` drift — already resolved (TASK-PLAN note is STALE)

The TASK-PLAN claims "`capital_flow_bars` still carries `stock_id`." **It does not.** `grep -rn "stock_id" src/rainier/core/models.py` → **zero hits**; `CapitalFlowBar` uses `symbol: Mapped[str] = mapped_column(String(10), ForeignKey("stocks.symbol"))` (`models.py:185–187`) with `primaryjoin="CapitalFlowBar.symbol == Stock.symbol"`. `grep -rln "stock_id" --include=*.py src/` → **none**. The `capital-flow-bars-symbol-5ea4` migration appears complete. **Recommendation:** verify the tracking task is closed; no schema work needed.

---

## 6. Duplicate functions / code duplication

DB-session/engine boilerplate is **well-centralized**, not duplicated:

- `with get_session()` / `get_engine()` used **87×** across src, all routing through the two canonical factories.
- `get_settings()` called **15×** — single source.
- Inline `create_engine` outside the factories: only **2 real cases** (`research/cli.py:280` sqlite for offline research; `scrapers/qu/coverage.py:540`) — below the "3+" bar, not worth extracting.

**Config loading:** single `get_settings()` singleton (`core/config.py`); no scattered re-parsing. **Conclusion:** no production-code extraction recommended (house-style: no premature abstraction). The only 3+-duplication is in tests (covered by R5).

---

## 7. General improvement opportunities

- **CLAUDE.md doc rot (R8, safe-now):** "10 ORM tables" → actually **18**; module map presents `trader/` and `dashboard/` as placeholders, but `trader/` is empty (0-byte `__init__.py`) and `dashboard/` is fully built (`app.py`, `render_etf.py`, `render_combined.py`). Update the module map + table count.
- **docs/ sprawl (R8):** **57 untracked `docs/*.md`** (DESIGN-*/TASK-PLAN-*/SPIKE-*/RESEARCH-*) vs 17 tracked. These are the design-doc-companion workflow output; many are completed task plans. Triage: commit the ones that are durable design records, delete superseded ones. The `.html` companions are correctly gitignored (`git check-ignore docs/DESIGN-rainier-slim-audit.html` → matched). Low effort, improves discoverability.
- **`out/` not gitignored:** `out/dashboards` (640 KB) is untracked but **not** in `.gitignore` — one accidental `git add out/` would commit generated dashboards. Add `out/` to `.gitignore` (safe-now, prevents a future R1-style mistake).
- **Config:** `config/` is lean (1,145 YAML LOC across 6 files); no sprawl. No redundant settings flagged.
- **Log rotation:** `data/fetch.log` (52 MB) and friends grow unbounded; add rotation to the scheduler/cron wrapper (low priority, local-only).

---

## Proposed follow-up cut tasks (NOT filed — operator promotes after review)

```
# safe-now bucket (P1 cleanup per global rule: cleanup is always P0/P1)
fleet tasks add --project projects-rainier --priority P1 --slug slim-untrack-reports \
  "Untrack + gitignore committed backtest HTML/xlsx artifacts (R1+R2): git rm --cached the 8 large rendered reports, extend .gitignore for docs/*-backtest-*.html + reports/*.html + reports/*.xlsx; ~28MB out of HEAD. Verify regen scripts still produce them."

fleet tasks add --project projects-rainier --priority P1 --slug slim-dead-code \
  "Delete verified-dead code (R4): apis/x/ tree (193 LOC, only test_x_client refs it), backtest/export.py export_equity_curve+export_summary, qu100_backtest.run_qu100_backtest_skip_entry, stock_screener._best_pattern, cli._make_sample_candidates. Remove test_x_client.py. ~280 LOC + bs4 stays (scripts only)."

fleet tasks add --project projects-rainier --priority P2 --slug slim-test-mock-conftest \
  "Consolidate Playwright test mocks (R5): hoist fake_goto/fake_fetch/_make_scraper/_make_mock_page (dup x4-7) into tests/test_scrapers/conftest.py; shared _make_settings into tests/conftest.py. ~150-250 LOC."

fleet tasks add --project projects-rainier --priority P2 --slug slim-doc-rot \
  "Fix CLAUDE.md doc rot (R8): 10->18 ORM tables, correct trader/ (empty) vs dashboard/ (live) in module map; gitignore out/; triage 57 untracked docs/*.md."

fleet tasks add --project projects-rainier --priority P2 --slug slim-cache-hygiene \
  "Local-disk hygiene (R3): document data/cache/tqqq_sma/results.parquet (570MB) as regenerable; add clean-cache target / --no-results-cache flag; rotate data/*.log."

# needs-approval bucket (operator decision required)
fleet tasks add --project projects-rainier --priority P2 --slug slim-optional-extras \
  "Move heavy deps to optional extras (R6, NEEDS-APPROVAL): [ml] xgboost/shap/sklearn/hmmlearn/numba, [scrape] playwright, [viz] plotly/kaleido, [llm] litellm. Audit each entrypoint's transitive set + add lazy-import guards + update deploy/cron install docs. ~600-800MB core-install reduction. Changes install contract."

fleet tasks add --project projects-rainier --priority P3 --slug slim-git-filter-history \
  "Purge historical report blobs from .git (R7, NEEDS-APPROVAL, HISTORY REWRITE): git filter-repo to drop ~85-90MB of *backtest*report*.html + demo/top20 HTML across history. Operator-gated; schedule when no PRs in flight; force-push + reclone required."
```

---

## Evidence appendix (commands run, all read-only)

- Disk: `du -sh */ .git` (main repo); `du -sh data/* data/cache/*`.
- Git tracking: `git ls-files <path> | wc -l`; `git check-ignore <path>`.
- Git history blob sizing: `git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' | awk '/^blob/{sum[$3]+=$2}END{...}' | sort -rn`.
- Dep imports: `grep -rEl "^\s*(import|from)\s+<pkg>([. ]|$)" src/`.
- Dead code: `uv run ruff check src/ --select F401` (clean); `uvx vulture src/rainier --min-confidence 60`; per-symbol `grep -rn "\b<name>\b" --include=*.py src/ tests/ scripts/ | grep -v "def <name>"`.
- Tests: `find tests -name '*.py' | xargs wc -l`; `grep -rhn "def fake_\|def _make_" tests/ | sort | uniq -c | sort -rn`.
- Schema: `grep -n "__tablename__" src/rainier/core/models.py`; `grep -rn "stock_id" src/rainier/core/models.py` (none); `ls db/alembic/versions/ migrations/`.

## Change log

- 2026-06-03 — initial audit (worker `rainier-slim-audit-2732`).
