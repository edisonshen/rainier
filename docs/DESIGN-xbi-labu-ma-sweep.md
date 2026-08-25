# DESIGN — XBI → LABU/LABD moving-average rotation sweep

- **Status:** draft — awaiting operator approval (★ marks the decisions to confirm).
- **Priority:** P2 (operator-requested research; phased, each phase ships independently).
- **Surface:** `src/rainier/backtest/ma_sweep.py` (generalized from `tqqq_sma_sweep.py`), `src/rainier/backtest/ma_sweep_report.py`, `src/rainier/cli.py` (`rainier ma-sweep`), `tests/test_ma_sweep.py`, `docs/xbi-labu-ma-backtest-report.html`.
- **Prior art in-repo:** `backtest/tqqq_sma_sweep.py` + `tqqq_sma_report.py` (PRs #75/#76 — QQQ-signal → TQQQ/SQQQ SMA grid, njit inner loop, resumable parquet cache, walk-forward, HTML report), the one-off EMA scripts `scripts/backtest_tqqq_ema.py` / `_sweep.py` / `_top20.py`, and `docs/stability_score.md` (composite ranking metric).

## 1. Problem

Operator wants the QQQ→TQQQ study repeated on biotech: **XBI is the signal, LABU is the position**.

> Buy LABU when XBI crosses **up** through a moving average; sell LABU when XBI drops **below** a moving average. Find the **best combination of entry signal and exit signal**, where each may be an **SMA or an EMA** of any window.

Two things this adds over the existing TQQQ sweep:

1. **MA type becomes a searched dimension** (per leg, not per run) — entry can be `EMA(13)` while exit is `SMA(20)`, and the grid decides.
2. **A different, harsher instrument** — LABU is 3× a 130-name equal-weight biotech basket, i.e. far more volatile and decay-prone than TQQQ, with ~5 fewer years of history.

## 2. Goals / non-goals

**Goals**

- Exhaustively sweep `(entry_ma_type, entry_window, exit_ma_type, exit_window)` for the long-LABU/cash strategy on XBI signals, and rank by a robustness-aware score (not raw final value).
- Report an **honest** answer: in-sample winners *and* their out-of-sample behavior, the parameter plateau around them, and the baselines they must beat (buy-and-hold XBI, buy-and-hold LABU, XBI 200-day rotation).
- Generalize the existing TQQQ machinery instead of forking it, so the next pair (SOXX→SOXL, IWM→TNA, …) is a config row rather than a new module.

**Non-goals**

- No live trading, no order routing, no scheduler wiring. This is offline research producing a report.
- No intraday signals — daily closes only, signal evaluated on the close, position taken at the same close (see D6 for the execution-lag decision).
- No optimization of stop-losses, position sizing, or partial allocation in Phase 1 — the strategy is binary (100% LABU or 100% cash).
- No new database tables. Everything lands in the regenerable `data/cache/` parquet + a committed HTML report.

## 3. Decisions

- **D1 — pair registry, not a copy-paste module.** `tqqq_sma_sweep.py` hardcodes `qqq`/`tqqq`/`sqqq` in its column names, cache paths, and fingerprint. Introduce a frozen `PairSpec(name, signal, bull, bear|None, inception, cache_dir)` and rename the module to `backtest/ma_sweep.py` with `PAIRS = {"qqq-tqqq": …, "xbi-labu": …}`. The existing `rainier sma-sweep` command stays as a thin alias to `--pair qqq-tqqq --ma-type sma`, so PR #75/#76 output remains reproducible. ★ *Confirm the rename is acceptable vs. adding a parallel module.*

- **D2 — the searched grid (Phase 1, long-only).** Four dimensions:

  | Dim | Values | Count |
  |---|---|---|
  | `entry_ma_type` | `SMA`, `EMA` | 2 |
  | `entry_window` | 2 … 200 | 199 |
  | `exit_ma_type` | `SMA`, `EMA` | 2 |
  | `exit_window` | 2 … 200 | 199 |

  = **158,404 combos**. Unlike the TQQQ Phase 1 sweep we do **not** constrain `exit_window ≥ entry_window`: an exit faster than the entry ("get out quick, get back in slow") is a legitimate and common configuration for a 3× instrument, and the constraint exists in the TQQQ code only to bound a 4-leg grid. Window ceiling is 200 (not 60) because the 200-day MA is the canonical regime filter and biotech's cycles are slow. ★ *Confirm 2–200; a 2–250 ceiling costs ~2.5× and is still cheap.*

- **D3 — EMA definition and warmup.** `EMA_w` uses `alpha = 2/(w+1)`, `adjust=False`, **seeded with `SMA_w` at index `w-1`** (matching `pandas.ewm` convention used in `scripts/backtest_tqqq_ema.py`, but with an SMA seed rather than the first price so short and long windows start on comparable footing). The signal is marked `valid` only from index `w-1 + burn_in` where **`burn_in = 2w`** (EMA never fully "fills", so a burn-in replaces the SMA's exact validity boundary; `2w` puts >90% of the kernel weight inside observed data). The existing `valid`-mask contract from `precompute_sma_signals` is preserved: `valid=False` means *no signal*, never *signal-false* — this is the bug class that TQQQ review already caught for the short leg.

- **D4 — state semantics: level, not crossing event.** Entry fires when `XBI_close > MA_entry` and we are in cash; exit fires when `XBI_close < MA_exit` and we hold LABU. Since a position can only be opened from cash, the first bar where the level condition turns true **is** the cross-up, so "cross" and "level" are equivalent here — but the level form is stateless per bar and cheaper to precompute (bool matrix), and it self-heals if a signal is missed. Documented explicitly so the report isn't misread.

- **D5 — the short leg is Phase 2, and optional.** The operator's request is long-LABU-only, so Phase 1 is a 2-state machine (`CASH ↔ LONG_LABU`). Phase 2 adds `SHORT_LABD`… but note LABD has lost >99% since inception and reverse-splits repeatedly; a "short-biotech via LABD" leg is far more path-dependent than SQQQ. Phase 2 therefore also evaluates **cash-with-yield** as the defensive state (D8) as its comparison baseline. ★ *Confirm Phase 2 is wanted at all.*

- **D6 — execution timing: same-close fill, with a T+1 sensitivity run.** The TQQQ engine marks to market with the next day's return and transitions on the same bar's signal — i.e. it assumes you can transact at the close that generated the signal. That's realistic (MOC order on a signal computed from the last print) but optimistic. Phase 1 ships a `--exec-lag {0,1}` flag and the report shows **both**; if the edge survives only at lag 0, that is a finding, not a footnote.

- **D7 — costs.** Default `--slippage-bp 10` per state transition (vs. 5 for TQQQ): LABU's spread and impact are materially worse than TQQQ's. The report includes a **cost-sensitivity strip** at 0/5/10/25/50 bp, because a high-turnover fast-MA winner that dies at 25 bp is not a strategy. The 0.96% expense ratio and swap financing are already inside LABU's price series — do **not** subtract them again.

- **D8 — cash yield.** The TQQQ sweep engine assumes 0% in cash; the one-off scripts assumed a flat 4.5%. Both are wrong across 2015–2025 (ZIRP then 5%). Use the **actual 13-week T-bill series (`^IRX`) as a daily accrual** when flat, with `--cash-yield {none,irx,<const>}` and `irx` as the default. This matters a lot here: a good XBI-MA strategy sits in cash ~40% of the time.

- **D9 — ranking metric = the existing composite, not `final_value`.** Reuse `docs/stability_score.md`: `Rank Score = 0.4·Stability + 0.3·CAGR_pct + 0.3·(1 − |MaxDD|_pct)`, with `Stability = 0.5·(1/(1+std(rolling-12m Sharpe))) + 0.3·pct_months_positive + 0.2·(1/(1+ulcer))`. The TQQQ sweep's walk-forward selects on `final_value`, which is exactly the metric most prone to picking a single lucky path; we select on Rank Score and report `final_value` alongside. ★ *Confirm reusing the existing weights unchanged.*

- **D10 — dedup by equity-curve identity.** Keep the `strategy_id` FNV-1a hash over the daily `(state, cumulative n_trades)` sequence from `run_backtest`, and dedup before building any leaderboard. With MA type in the grid this matters more, not less: `SMA(2)` and `EMA(2)` produce near-identical (often identical) signal paths, so a naive top-50 would be 50 aliases of one strategy.

- **D11 — cache layout.** `data/cache/ma_sweep/<pair>/{prices,results,top_walkforward}.parquet` + the existing `.sha256`/`.fingerprint.txt` companions; regenerable, gitignored, listed in `rainier cache clean` and in CLAUDE.md's disk-hygiene section. The sweep fingerprint gains `pair`, `ma_types`, `exec_lag`, `cash_yield` so a rerun with different assumptions cannot silently mix rows (the `SweepInputMismatchError` path already exists).

## 4. Data & instrument caveats (verified 2026-08-25)

| Fact | Value | Consequence for the study |
|---|---|---|
| LABU / LABD inception | 2015-05-28 (Direxion) | **~2,580 trading days.** Half the TQQQ sample. Overfitting risk is materially higher → §6 guards are mandatory, not optional. |
| Daily target | LABU +300%, LABD −300% (still 3×; LABU/LABD were **not** in Direxion's 2020 3×→2× reductions) | No leverage structural break inside the sample. |
| Expense ratio | LABU 0.96%, LABD 1.08% gross | Already in the price series. Do not double-count. |
| Underlying index | S&P Biotech Select Industry (SPSIBI) — **equal-weight**, ~130 names, XBI tracks it | XBI is a valid signal proxy for LABU's exposure, unlike a cap-weighted biotech ETF (IBB). Say so in the report. |
| Regime composition | 2015–2018 chop, 2019–Feb 2021 melt-up, Feb 2021–Oct 2023 −60%+ bear, 2024–2025 recovery | One giant bull and one giant bear. A parameter set can win the whole sample by getting *one* regime right → require per-regime attribution in the report. |
| LABU price history | multiple reverse splits | Use `auto_adjust=True` (already the convention in `fetch_prices`); assert monotonic date index and no zero/NaN closes before sweeping. |
| Extension | XBI history starts 2006 | Optional **synthetic-LABU** series (`3 × XBI daily return − financing − fee`) to back-test 2006–2015 as an *out-of-sample sanity check only*, never as the headline. ★ *Want this?* |

## 5. Code plan

```
backtest/ma_sweep.py
    PairSpec(name, signal, bull, bear, inception, cache_dir)
    PAIRS: dict[str, PairSpec]                     # qqq-tqqq, xbi-labu
    fetch_prices(pair, start, end, refresh)        # generalized; columns signal/bull/bear
    precompute_ma_signals(close, max_window, ma_type) -> (above, valid)   # SMA | EMA
    stack_signal_matrices(close, max_window)       # -> above[type][d, w-1], valid[...]
    run_backtest_long_only(above_e, valid_e, above_x, valid_x, bull_ret, cash_ret,
                           entry_w, exit_w, slippage_bp, exec_lag) -> metrics + curve_hash   # @njit
    iter_long_only_combos(max_window, ma_types)    # 158,404 at 2..200 x {sma,ema}^2
    run_sweep(...)                                 # unchanged skeleton: pool + resumable parquet
    walk_forward_top_n(...)                        # selection on rank_score (D9)
    rank_scores(results, curves)                   # stability_score.md composite
backtest/ma_sweep_report.py                        # ports tqqq_sma_report sections + §6 additions
cli.py: rainier ma-sweep --pair xbi-labu [--ma-types sma,ema] [--max-window 200]
                         [--exec-lag 0] [--slippage-bp 10] [--cash-yield irx] [--report]
        rainier sma-sweep  -> alias of --pair qqq-tqqq --ma-types sma (kept for reproducibility)
```

**Runtime.** 158k combos × ~2.6k bars is ~4×10⁸ inner steps — under a minute on the existing njit + multiprocessing path (the TQQQ sweep does 3.35M combos × 3.9k bars in <20 min). Precompute is `2 types × 199 windows` bool columns (~1 MB). The `results.parquet` is ~10 MB, not 570 MB, so the `--no-results-cache` dance is unnecessary here.

## 6. Validation — how we avoid fooling ourselves

158k combos over 2,580 days will *always* produce a spectacular in-sample winner. The report is only credible with all five:

1. **Walk-forward** — train `< 2021-02-08` (the XBI peak) / test after. Report train vs test Rank Score and the delta scatter, as in the TQQQ report. A winner that inverts sign out-of-sample is labeled as such.
2. **Parameter plateau, not peak** — for each leaderboard entry, report the mean Rank Score of its 8-neighborhood in `(entry_window, exit_window)` at fixed MA types. Prefer a broad plateau to an isolated spike; the report ranks by plateau-mean and shows the peak.
3. **Regime attribution** — per-year and per-regime (bull/bear/chop) returns for every leaderboard entry, so "won 2020 only" is visible.
4. **Baselines it must beat** — buy-and-hold XBI, buy-and-hold LABU, XBI>SMA(200)→LABU, and the **best QQQ→TQQQ parameters transplanted onto XBI/LABU** (a cross-asset transfer test: if the TQQQ optimum works here too, the signal is structural rather than fitted).
5. **Cost + lag sensitivity** — the D6/D7 grids, shown as a strip, not buried.

Optional if the operator wants a hard statistical statement: a stationary block-bootstrap (~10-day blocks, 1,000 resamples) of the daily XBI/LABU return pair, re-running the top-20, to get a distribution of Rank Score under resampled histories. ★

## 7. Report (`docs/xbi-labu-ma-backtest-report.html`)

Same self-contained hub style as `tqqq_sma_report.py` (inline Plotly, no CDN, no JS deps): 1. framing + data provenance · 2. top-50 leaderboard (deduped, plateau-mean, train/test delta) · 3. baselines · 4. **MA-type heatmaps — a 199×199 window heatmap per `(entry_type, exit_type)` quadrant**, the visual answer to "SMA or EMA?" · 5. outcome distribution · 6. equity curves (top-5 + baselines) · 7. per-regime/per-year attribution · 8. cost + lag sensitivity · 9. honest discussion (sample size, one-bull-one-bear, decay) · 10. reproducibility footer (parquet SHA-256, git rev, wall time).

## 8. Phasing

| Phase | Scope | Ships |
|---|---|---|
| **P1** | Pair registry + EMA support + long-only 2-leg sweep + rank score + walk-forward + report; `sma-sweep` alias preserved | one PR, the operator's actual question answered |
| **P2** | Defensive-state extension: `SHORT_LABD` vs `CASH+yield` 4-leg grid (D5) | separate PR, reuses the parquet schema |
| **P3** | Optional: synthetic-LABU 2006–2015 out-of-sample, block bootstrap, additional pairs (SOXX→SOXL, IWM→TNA) | separate PRs, config rows |

**P1 acceptance criteria**

1. `rainier ma-sweep --pair xbi-labu` completes the 158,404-combo grid and writes results + walk-forward parquets with a fingerprint; a rerun with different `--slippage-bp`/`--exec-lag`/`--cash-yield` raises `SweepInputMismatchError` rather than mixing rows.
2. `rainier sma-sweep` reproduces the PR #75 Phase-1 numbers bit-for-bit (regression test on a fixture price frame).
3. EMA signals match `pandas.ewm(span=w, adjust=False)` seeded with `SMA_w`, to 1e-12, and are `valid=False` through the burn-in.
4. Long-only backtest agrees with a slow pure-pandas reference implementation on a fixture (final value, n_trades, max_dd) to 1e-9.
5. Dedup collapses `strategy_id` aliases; the top-50 contains 50 distinct equity curves.
6. Report renders offline with all 10 sections, including the four MA-type heatmap quadrants and the cost/lag strips.
7. `uv run pytest tests/ -q` green, `uv run ruff check src/ tests/` clean.

**Tests** (`tests/test_ma_sweep.py`, deterministic, no network): EMA correctness + burn-in mask; valid-mask never fires a signal during warmup; long-only state machine (entry/exit/same-bar re-entry/slippage accounting/exec-lag 0 vs 1); cash-yield accrual; combo iterator count = 158,404; `strategy_id` equality for identical paths and inequality for same-final-value-different-path; resumability skip-set; fingerprint mismatch raises; rank-score components against hand-computed fixtures; `PairSpec` registry wiring; the `sma-sweep` back-compat regression.

## 9. Open questions for the operator (★)

1. Window range 2–200 as proposed, or 2–250?
2. Rank Score weights from `stability_score.md` unchanged, or re-weight for a 3× instrument (e.g. heavier drawdown penalty)?
3. Is Phase 2 (LABD short leg) wanted, or is cash the only defensive state you'd actually trade?
4. Synthetic-LABU 2006–2015 extension — useful sanity check, or noise you'd rather not see?
5. Module rename `tqqq_sma_sweep.py` → `ma_sweep.py` with a back-compat CLI alias: OK?
