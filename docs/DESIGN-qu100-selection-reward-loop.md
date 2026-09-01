# DESIGN — QU100-LLM selection reward loop (selection → track → reward → evolve)

- **Status:** draft for operator review.
- **Priority:** P2 — closes the last gap in the qu100-llm feedback loop: a *reward system* that scores every selection decision and drives systematic evolution of the selection levers.
- **Surface:** `src/rainier/paper/` (reward compute), `src/rainier/research/rewards/` (registry — currently a Slice-0 stub), `src/rainier/research/bandit/` (currently empty), `src/rainier/llm_thesis/research.py` (weekly job), `core/models.py` (+1 migration), `config/model/` (champion mechanism reuse).
- **Related docs:** `DESIGN-qu100-llm-feedback-loop.md` (tracking loop — mostly shipped), `stock_screener_design.md` (champion.yaml WS), `TASK-PLAN-qu100-paper-tracker-p01.md`.

## 1. Problem

The operator's ask: *"qu-100-llm selection → performance tracking → back-tracking → adjust the selection loop."* Three of the four stages already exist in some form; the missing stage is a **reward system** — a single, honest scalar per decision that lets the loop *rank its own levers* and evolve them, instead of relying on ad-hoc human reading of reports.

### What already exists (do NOT rebuild)

| Loop stage | Asset | Where |
|---|---|---|
| **Selection** | QU100 scrape → top-50 screen → top-5 LLM thesis (`setup_long`/`watch`/`no_setup`, conf 1–10) | `scrapers/qu/`, `analysis/stock_screener.py`, `llm_thesis/service.py` |
| **Tracking** | $10k paper book: pending→fill(T+1 open)→SL/TP/time-stop exit, $P&L; skip ledger; daily/weekly snapshots + Discord | `paper/positions.py`, `paper/exit.py`, `paper/report.py`, migrations 0005–0013 |
| **Tracking** | Fixed-horizon unbiased eval (1d/5d/10d) per thesis | `llm_thesis/eval.py` → `thesis_evaluations` |
| **Back-tracking** | Weekly missed-winner sweep (coverage diagnostic, D8); pattern-layer replay; shadow WATCH-buy book through the real fill/exit engine; reclaim queue | `paper/sweep.py`, `paper/pattern_replay.py`, `paper/replay.py`, `paper/reclaim.py` |
| **Feedback (prompt-level)** | Daily calibration block (unbiased headline + realized supplement) + per-trade LLM reflections injected into the next prompt | `paper/calibration.py`, `paper/reflection.py` |
| **Feedback (config-level)** | Weekly research checks (`check_signal_underperform`, `check_verdict_drift`, `check_calibration_off`, `check_prompt_regression`, `discover_time_stop`) emitting gated `ResearchInsight` actions | `llm_thesis/research.py` |
| **Promotion mechanism** | champion.yaml versioned model config (version/parent/score, history dir, parquet registry) — **screener only today** | `core/champion.py`, `config/model/` |
| **Stubs awaiting this design** | Reward registry (empty dict), Thompson bandit package (empty), trade simulator | `research/rewards/`, `research/bandit/`, `research/evaluator/` |

### The gap

1. **No reward function.** Outcomes exist (`paper_trade.pnl`, `thesis_evaluations.return_pct`) but there is no normalized per-decision score. $P&L is not comparable across prices/volatility; win-rate ignores magnitude; fixed-horizon return ignores the plan (stop/target) the LLM actually committed to.
2. **No decision-level reward ledger.** Rewards must attach to *every* decision — including `watch`/`no_setup` declines and skips — or the loop only ever learns from what it bought (the miss-sweep showed the book can go weeks with 0 `setup_long`).
3. **Feedback is advisory, not evolutionary.** Research checks emit human-readable insights; nothing systematically proposes → shadow-tests → promotes/rejects changes to the selection levers. `ThesisSignalConfig.weight` is still inert; champion.yaml covers only screener params, not the LLM layer (prompt_version, confidence gate, session mix).

## 2. Goals / non-goals

**Goals:** (a) a pinned, versioned reward function per selection decision; (b) a durable reward ledger joining decision → outcome → reward; (c) a lever scorecard (which knob is helping/hurting, with sample sizes); (d) a champion/challenger evolution loop that shadow-tests proposed lever changes on the real engine and promotes only on gated evidence.

**Non-goals:** no real-money execution; no model fine-tuning / RLMF (feedback stays prompt-context + config); no tuning on the missed-winner coverage diagnostic (D8 framing stands — it is not a reward); no new dashboards (reuse `paper_report_snapshot` + Discord); no Neon writes.

## 3. Design

### 3.1 Reward function (the scalar)

**Primary reward = R-multiple against the committed plan.** For a decision with pattern levels (entry `E`, stop `S`, target `T`) and realized exit `X`:

```
risk_per_share = E - S                    (must be > 0; else no reward, ledger reason=invalid_levels)
R              = (X - E) / risk_per_share
```

- Closed paper trades: `X` = booked `exit_price` → **realized R**.
- Open trades: `X` = latest close on the trade's `price_basis` → **MTM R**, labeled `provisional=true` (recomputed daily until close; never used in promotion gates).
- Declined decisions (`watch`, `no_setup`, confidence-gated, `paper_skip`): scored via the **shadow/counterfactual path** — the same plan levels walked through the pure `evaluate_exit` over `stock_prices` (this is exactly what `paper/replay.py` + the shadow book already do for WATCH). Reward stored with `counterfactual=true`.
  - `no_setup` on a symbol with **no valid long-shape levels** gets **no R** — there is no plan to score. It is scored only in the coverage diagnostic (unchanged).
- **Why R and not return%:** R normalizes by the risk the plan took, so a 2% gain on a tight-stop setup and a 10% gain on a wide-stop setup compare honestly; expectancy in R (`mean(R)`) is the standard objective for per-decision quality, which is what this book measures (independent $10k sleeves, no portfolio cap — D2).

**Aggregate objectives — the operator's three axes (2026-08-31: total return, minimize risk, high win rate — "that is it"):**

| Axis | Metric | Definition |
|---|---|---|
| **Total return** | `total_R = sum(R)` and `total_pnl_$` | how much the cohort made in total (sum of independent $10k sleeves, per D2/D11 framing) |
| **Risk** | `max_drawdown_R` (path-level, from the by-holding-day curves already built for `discover_time_stop`) + `std(R)` | worst peak-to-trough of the cohort's cumulative-R curve; dispersion |
| **Win rate** | `win_rate = wins / matured decisions` (win = `R > 0`) | over the honest denominator incl. counterfactually-scored declines |

Cohort comparisons (scorecard + promotion) use one composite that encodes all three, with the axes also reported separately so trade-offs stay visible:

```
score = total_R − λ · max_drawdown_R          (λ = 1.0 default ★)
subject to: win_rate ≥ champion's win_rate − 5pp   (win-rate floor — a challenger may not
                                                    buy total return with materially more losers)
```

Supporting stats always attached: `n`, `expectancy_R = mean(R)`, `t_stat_R = mean(R)/(std(R)/√n)` — the promotion gate keeps the t-stat requirement (small-sample discipline, same spirit as D6), applied to the composite's return leg.

**Known tension (stated, not hidden):** win rate and total return fight each other — tight targets raise win rate but cap total return; letting winners run does the opposite. The composite + floor resolves it in favor of *total return at no-worse risk and no-materially-worse win rate*; the scorecard reports all three axes per lever bucket so the operator sees which axis a change trades away.

All reward bodies are registered in `research/rewards.REGISTRY` (the Slice-0 stub becomes real): `r_multiple_realized`, `r_multiple_mtm`, `r_multiple_counterfactual`, `total_R`, `win_rate`, `max_drawdown_R`, `composite_score`, `expectancy_R`, `t_stat_R`, plus the fixed-horizon `h5_return` / `h10_return` (thin wrappers over `thesis_evaluations`) for cross-checking. **`reward_version: 1`** is pinned; any formula change bumps it (same convention as `feature_version` in `paper/features.py`).

### 3.2 Reward ledger (data model)

New table `selection_reward` (plain Postgres, LEGACY local engine — joins `analysis_results`, `paper_trade`, `stock_prices`):

```
id            BIGINT PK
thesis_id     FK→analysis_results, part of UNIQUE(thesis_id, reward_name, as_of_date)
symbol, scan_date, session_name
decision      ∈ {setup_long_filled, setup_long_skipped, watch, no_setup, gap_invalidated, ...}
              (denormalized from paper_trade/paper_skip/thesis verdict — the honest denominator)
lever_context JSONB — the lever values LIVE AT DECISION TIME:
              {prompt_version, llm_confidence, confidence_gate, session, model,
               pattern_type, signal_set_hash, screener_champion_version, time_stop_days}
reward_name   TEXT (registry key), reward_version INT
value         DOUBLE PRECISION,  provisional BOOL, counterfactual BOOL
as_of_date    DATE, created_at
```

- Written by a new daily step **(vi)** in the authoritative daily-job order (after reports/calibration): compute/refresh rewards for decisions whose outcome state changed (new fill, new close, matured horizon). Idempotent upsert on `(thesis_id, reward_name, as_of_date)`.
- **`lever_context` snapshot at decision time is the core trick** — it makes every later group-by ("expectancy by prompt_version", "by confidence bucket", "by pattern_type", "by champion version") a plain SQL aggregation with no reconstruction, and keeps attribution honest when a lever changes mid-stream (same rationale as the D6 fill-time `time_stop_days` snapshot).
- Provisional MTM rows are upserted in place daily; the realized row (on close) is final/immutable.

### 3.3 Lever scorecard (weekly)

Extend the Friday research job with `compute_lever_scorecard()`: for each lever in `lever_context`, group matured, **non-provisional** rewards by lever value and emit the three operator axes per bucket — `total_R / max_drawdown_R / win_rate` — plus `composite_score`, `expectancy_R`, `t_stat`, `n`, plus the same split for counterfactual cohorts (e.g. "watch conf≥7 counterfactual expectancy vs live setup_long expectancy" — exactly the WATCH-flip question WS A measures, now standing infrastructure). Output: a `lever_scorecard` section in the weekly `paper_report_snapshot` payload + Discord table. This is the human-readable "back-tracking" view the operator asked for.

### 3.4 Evolution loop (champion / challenger, gated)

Generalize the existing screener `champion.yaml` mechanism into a **selection champion** covering the LLM-layer levers too. One config unit, versioned/parented/scored exactly like `core/champion.py` does today:

```
config/model/selection_champion.yaml
  version / parent / created / note / score      (metadata, stripped)
  confidence_gate: 6
  act_on_watch: {enabled: false, min_conf: null}   # WS A flip, gated here
  prompt_version: "v..."
  signal_weights: {rank_trajectory: 1.0, capital_flow_streak: 1.0, ...}
  learned_time_stop_days: null
```

**Loop cadence (weekly, Fri, after the scorecard):**
1. **Propose.** Challenger generator produces ≤2 candidate configs per week, each a *single-lever* delta from champion, sourced from: (a) scorecard buckets with `t_stat` beyond threshold (e.g. "conf gate 7 beats 6"), (b) existing research insights (`discover_time_stop`, `check_signal_underperform` → weight/disable deltas — this finally makes D7b real, and wiring `signal_weights` into prompt rendering is a prerequisite subtask, per D7b), (c) operator-suggested. Single-lever deltas keep attribution clean.
2. **Shadow-test.** Each challenger runs as a **shadow book** through the real engine — reuse the `paper_trade.shadow` mechanism (migration 0013) with a new `challenger_id` tag, and `paper/replay.py` for historical warm-start where the lever permits it (confidence gate and act-on-watch replay cleanly from stored theses; a `prompt_version` change **cannot** be replayed and must accrue live shadow decisions only — the ledger's `counterfactual` flag already distinguishes these).
3. **Promote/reject (gated).** Promotion requires ALL of: `n ≥ 30` matured shadow decisions; challenger `composite_score` (total_R − λ·drawdown) beats champion's on the same window with `t_stat ≥ 1.5` on the return leg; challenger passes the **win-rate floor** (≥ champion − 5pp); direction stable across **≥2 consecutive weekly runs** (D6 discipline); and **operator approval** via the existing `ResearchInsight` action gate (new kind `challenger_promotion`, executor writes the new `selection_champion.yaml` version++, parent=old, score=composite_score, history retained). No auto-apply in v1.
4. **Bandit allocation (Phase R3, `research/bandit/`).** When >1 challenger competes for live shadow slots, allocate symbols/sessions via Thompson sampling over `composite_score` posteriors instead of running all challengers on everything (LLM cost control). This fills the empty L5 bandit package; it allocates *measurement budget*, never live capital.

**Guardrails (hard rules):**
- Coverage diagnostics (missed-winner sweep) never feed rewards or promotion (D8 stands).
- Rewards used in gates: realized + matured counterfactual only — never provisional MTM.
- All counterfactual walks respect the as-of discipline (prices ≤ analysis date, entry-day-inclusive, same-bar SL-first, gap-through-at-open — the pure `evaluate_exit` is the single evaluator for live, shadow, and counterfactual paths).
- One champion change at a time; every promotion is reversible (parent pointer; `rainier selection-champion rollback`).

### 3.5 CLI

```
uv run rainier reward compute [--as-of DATE]         # manual/backfill of daily step (vi)
uv run rainier reward scorecard [--week]             # render lever scorecard from ledger
uv run rainier selection-champion {show,history,rollback}
uv run rainier challenger {list,status,promote,reject}
```

## 4. Phasing (each phase = its own PR)

- **R1 — reward ledger + registry.** `selection_reward` migration; reward bodies in `research/rewards` (registry stub → real); daily step (vi); counterfactual scoring for declined decisions (reusing `evaluate_exit`); `rainier reward compute`; tests (R math incl. invalid levels / basis mismatch, idempotent upsert, provisional→realized transition, counterfactual as-of discipline).
- **R2 — scorecard + selection champion.** Weekly `compute_lever_scorecard` + snapshot/Discord; `selection_champion.yaml` + loader (generalizing `core/champion.py`); wire `signal_weights` into prompt rendering (D7b prerequisite); `challenger_promotion` insight kind + gated executor; CLI; tests.
- **R3 — challenger automation + bandit.** Challenger generator; shadow-book-per-challenger (`challenger_id`), replay warm-start; promotion gate evaluation; Thompson allocation in `research/bandit`; tests.

R1 is independently valuable (the ledger + scorecard answer "is the selection working, and which knob is the problem" even with zero automation).

## 5. Open questions (★ operator)

- ★ Reward horizon for counterfactual declines with no learned time-stop: cap the walk at 20 trading days (proposal) or walk until SL/TP indefinitely?
- ★ Promotion thresholds: `n ≥ 30`, `t_stat ≥ 1.5`, 2-week stability — confirm or tighten.
- ★ v1 keeps operator approval on every promotion. Appetite for auto-apply later (e.g. confidence-gate only, bounded ±1)?
- ★ LLM cost ceiling for challenger shadow runs (each live-shadow prompt_version challenger doubles thesis calls for its slice).
