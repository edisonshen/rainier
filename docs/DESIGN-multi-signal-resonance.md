# Design Plan — Multi-Signal Resonance Gate for TQQQ

**Status:** decisions locked, for build review · **Scope:** a weighted-signal entry/exit gate for TQQQ · **Window:** 2020-10 → now (real TQQQ) · **Constraint:** no moving-average / lookback > 66 bars · **Depends on:** `src/rainier/signals/`, `src/rainier/backtest/` · **PR base:** main

---

## 1. The problem, in plain English

We want to buy and sell TQQQ (3× Nasdaq-100) on the **combined verdict of many signals**, not one. A single indicator — "price is above its 44-day average" — flips on every wiggle and gives false buys. We have *dozens* of signals (trend, momentum, volatility, structure). The model you want:

> **Give each signal a power weight. Add them into one confidence score. When the score crosses a BUY line → buy TQQQ. When it drops below a SELL line → sell everything to cash.** The gap between the two lines stops the whipsaw.

We call the weighted agreement of many signals **resonance**: when lots of independent indicators line up, the score is high and the buy is more trustworthy than any one signal alone.

### What the data *suggests* (a lead, not yet proof)

On 2020-10 → now we bucketed every day by how many signals agreed, then looked at TQQQ's *next 20 days*:

```
 signals agreeing →  win-rate of the next 20 days
   0–20%               48%   (coin flip)
  20–40%               59%
  40–60%               60%
  60–80%               61%
  80–100%              62%   ← most agreement = highest win-rate
```

More agreement → higher win-rate. That is the premise of a "buy when the score is high" gate. **But treat it as a lead, not proof, until §6.** Two reasons to doubt it: (1) the forward-20-day windows **overlap** (adjacent days share ~19/20 of their data), so the *effective* sample is ~N/20 ≈ a few dozen independent points over 5 buckets — too few to be significant without confidence intervals; (2) it's one window dominated by a single 2022 bear + the 2020–21 recovery. **§6 gates it:** block-bootstrap CIs + effective N per bucket; only call it real once a CI excludes the null.

---

## 2. How it works today

No integrated system — just exploratory scripts. Two pieces exist:

- **An exploratory signal panel** (`regime_signal_search.build_signals`), each member a daily risk-on (1) / risk-off (0) series. It's *unfiltered* — most members use >66 lookback and two use expanding medians; v1 takes only its **≤66 subset** and rewrites the expanding members (§5.2). There is **no weighting and no combined score** today.
- **A timing gate** — the asymmetric SMA gate (enter QQQ > SMA22, exit QQQ < SMA44). In-window the best single-signal design (Calmar ≈ 1.37, #2 of 270, on a stable plateau). This is the **bar to beat**.

```
TODAY:  one signal  ──►  in/out of TQQQ
        (e.g. price>SMA44)        (binary, one indicator, no weighting)
```

---

## 3. What we learned, and how v1 differs

We already tested the **naive** version — equal-weight consensus, "be in when ≥70% agree." **It lagged and lost** to the simple SMA gate (Calmar 0.20 vs 0.34 on the synthetic full-cycle). Why consensus lags:

```
THE LAG TRAP:
   price ────╲                      ╱──── recovery
              ╲                    ╱
   exits late  ╲__________________╱   re-enters late
                  ▲              ▲
   score drops  LATE           LATE   ← it takes many signals to flip the score;
   below sell (already down)  (misses the bounce)
```

v1 is **not** that naive version. Two differences, both of which we **A/B test** rather than assume:
- **Power weights** per signal (not equal) — a strong signal moves the score more.
- **Two tunable lines** (buy high, sell lower) instead of one 70% line — the gap is a hysteresis buffer that can be tuned to cut whipsaw.

**Honesty rail:** this is still the *vote-gate family* that lost to the SMA gate once already. v1 earns its place only if the A/B test (§6) shows the weighted dual-threshold gate beats the SMA22/44 gate **and** buy-hold, out-of-sample. If it doesn't, we ship the SMA gate and stop. Eyes open.

---

## 4. The design — a weighted-score dual-threshold gate

```
   ≤66 signals, each with a power weight
            │
            ▼
   resonance score  r[t] = Σ wᵢ·signalᵢ[t] / Σ wᵢ   ∈ [0,1]
            │
            ▼
   ┌─ if flat AND r ≥ BUY line   → buy TQQQ (full)
   ├─ if held AND r ≤ SELL line  → sell all → cash
   └─ else                       → hold current state
            │
            ▼
   position is BINARY: 100% TQQQ or 100% cash. No sizing.
```

- **Binary, no sizing** — you were right that "size by conviction" was likely noise; v1 drops it. The score's only job is to cross the buy/sell lines.
- **Risk-off = 100% cash/T-bills** (Q1). When out, we hold nothing but cash.
- **BUY line > SELL line** (Q2) — the gap is the hysteresis band; widening it trades fewer trades for later entries (a swept config).
- The whole gate is **A/B tested against the SMA22/44 gate** and against combinations (resonance-gate alone, SMA alone, resonance AND SMA, resonance OR SMA) — data picks the winner (Q3).

---

## 5. Implementation detail (for engineers)

### 5.1 Components

| Component | Role | Where |
|---|---|---|
| `SignalPanel` | registry of ≤66-lookback signals; each `(df) → risk-on series ∈ {0,1}` | new `src/rainier/signals/panel.py` |
| `ResonanceScorer` | panel + per-signal weights → daily score ∈ [0,1] | new `src/rainier/signals/resonance.py` |
| `ResonanceGate` | dual-threshold state machine → daily **per-asset target weights** {TQQQ} | new `src/rainier/signals/resonance_gate.py` |
| Daily-MTM sim | per-asset weights → equity, no lookahead, costs | `src/rainier/backtest/` (productionize `scripts/leveraged_common.py`) |

**New boundary — NOT `SignalEmitter`.** `SignalEmitter` returns `list[Signal]` (discrete entry/SL/TP trades); the current engine consumes discrete trades and can't represent a held daily weight. v1 adds a **`WeightStrategy`** protocol — `(df, symbol, timeframe) → daily per-asset weight series` (a dict like `{TQQQ: 0 or 1}`, weights ≥0 summing ≤1, remainder cash). Per-asset (not a bare boolean) so a future non-cash risk-off is representable without a redesign; matches the existing `run_portfolio` sim, which already takes a per-asset weights dict. Additive: it doesn't touch `SignalEmitter` or the discrete-trade engine.

*Reference artifacts (local, NOT committed in this PR): the panel + sim live only as untracked exploratory scripts (`scripts/regime_signal_search.py`, `scripts/leveraged_common.py`, `scripts/full_combo_search.py`, `scripts/resonance_study.py`). Numbers in this doc are exploratory, reproduced via those scripts. v1's **first step** is to promote them into `src/rainier/` (committed) per the `CLAUDE.md` module map.*

### 5.2 The panel (≤66 lookback, trend-only for v1)

v1 is **trend-only** (Q7) — every member is a *risk-on / uptrend-confirming* signal (price above MAs, momentum up, vol calm). Pullback/oversold signals (price *below* short MAs) are deferred to v2.

**The enumerated list below IS the single source of truth** (the score denominator and category counts derive from it). It was *seeded* from the ≤66 subset of `build_signals` (complement of the `BLOCK` set in `scripts/full_combo_search.py`), then **deduplicated** — one threshold per indicator (RSI>50, drop RSI>55; VIX<25, drop VIX<20/<30; one of ADX/+DI) so no indicator is triple-counted — with the two expanding-median members **rewritten** to bounded form and `price>SMA66` **added**. Where prose and the list disagree, **the list wins**.

- **Trend:** price > SMA{20,50,66}, price > EMA{20,50}, SMA22>SMA44, SMA50-rising.
- **Momentum:** RSI14>50, MACD-hist>0, ROC{20,60}>0, price>Donchian50-mid.
- **Volatility:** realizedVol(20, rolling std) < its **rolling-40 median** (20+40 = 60, finite), ATR%(14, **simple rolling mean** of true range, *not* Wilder/EWMA) < its rolling-46 median (14+46 = 60), VIX<25, VIX<SMA20, VIX-falling. *(Replaces the prototype's `expanding(60).median()` and EWMA `atr_pct` — both unbounded.)*
- **Structure:** within 5% of 60-day high, ADX>20 & +DI>−DI, higher-high+higher-low.
- **Cross-asset:** SPY>SMA22, SPY>SMA44 (NEW code — `build_signals(qqq, vix)` takes no SPY today; v1 adds an SPY input).
- **Breadth (NEW, in v1 — Q5):** **% of top-N QQQ holdings above their SMA** — the closest proxy to "market breadth" (yfinance has no breadth index). Built in v1, and **also surfaced on the QQQ market-breadth dashboard** (separate deliverable, §7).

*Every member's **composed/total** lookback ≤ 66 (a 20-day measure vs a 40-day median = 60, not "66 because the outer window is 66"). Finite-window members have exact composed support ≤66; EMA-family members (EMA, RSI, MACD, ADX) have bounded effective memory (invariant in §5.5).*

### 5.3 Scoring (power weights)

`resonance[t] = Σ wᵢ · signalᵢ[t] / Σ wᵢ` ∈ [0,1] — the **power-weighted** fraction risk-on.

- **Per-signal power weights `wᵢ`** (Q2) — a strong signal moves the score more. Default prior = **category-balanced** (Q4): the 5 categories weighted equally, split within, so the many trend signals don't make the score really mean "trend ×4." Beyond that prior, weights are a **swept/tested** config (capped per §6.4 — every free weight is overfit risk on a tiny sample).
- A/B includes **equal-weight vs category-balanced vs tuned** weights to confirm weighting actually helps (Q4: "testing can tell us more").

### 5.4 The dual-threshold gate (state machine)

```
score r[t]  ──►  if state==CASH  and r[t] ≥ BUY  → state = TQQQ (enter at close)
            ──►  if state==TQQQ  and r[t] ≤ SELL → state = CASH (sell all)
            ──►  else                            → hold state
       BUY > SELL ; the gap is the hysteresis band.
```

Boot: at the first valid bar `t0` (after the §5.5c per-member warmup), state = TQQQ if `r[t0] ≥ BUY` else CASH. Deterministic and unit-testable — a fixed score sequence yields a fixed state sequence; a test asserts it. No second hysteresis layer, no sizing curve.

### 5.5 No-lookahead + costs

Every input (TQQQ, QQQ, SPY, VIX, breadth) uses its **close[t]** value; the score and the gate share the *same* timestamp alignment and the *same* +1 shift — state decided on close[t] applies to the t→t+1 return. Warmup buffer from 2019-06; **measure only over 2020-10 → now** (real TQQQ).

**Costs — no double-counting:** real adjusted TQQQ prices *already embed* the 3× daily-reset financing and decay. On real TQQQ we charge **only** (a) turnover/slippage on each buy/sell and (b) the T-bill rate the cash sleeve earns — **NOT** a synthetic 3× financing charge. The synthetic formula (`3·r − 2·rate − fee`) is reserved for the §6.2 pre-2010 OOS test.

**Leakage test:** inject a known future spike into bar *t+1*; assert state[t] is unchanged across all inputs.

**Lookback invariant:** the **composed/total** lookback of every member ≤ 66, no expanding windows. "Composed" matters for *nested* indicators (realizedVol(20) into a rolling-k median has span 20+k, so k ≤ 46). *Finite-window* members (SMA, rolling-median, Donchian, ROC, rewritten nested-vol) have exact composed support ≤66; *EMA-family* (EMA, RSI, MACD, ADX) have bounded effective memory. Tests: (a) every composed span ≤66; (b) finite-window signals bit-identical when history before *t−66* dropped; (c) **per-member** warmup — empirically measure each EMA-family signal's convergence (ADX double-smoothed, MACD-hist EMAs-of-EMAs) and set the buffer from the *slowest* member. The two prototype `expanding(60).median()` members fail test (b) and **must** be rewritten — required, not optional.

### 5.6 Config (and what is NOT swept)

Swept: `panel membership · per-signal weights · BUY threshold · SELL threshold · gate-combination mode (resonance-only / SMA-only / AND / OR)`. Total free parameters tuned on data are **capped and pre-registered** before any OOS run (§6.4) — the effective sample is tiny, every extra knob is overfit risk.

---

## 6. Test plan — A/B against the SMA gate, built to *reject*

The window has **one** bear (2022). The plan tries to kill the idea, not flatter it.

### 6.1 Significance, not just point estimates
| Test | Input | Pass criteria |
|---|---|---|
| Thesis CI | resonance buckets × fwd-20d | block-bootstrap CIs + effective non-overlapping N per bucket; the win-rate trend's CI must exclude the null. If not, the premise fails — stop. |

### 6.2 No data-snooping — frozen train/test + genuine OOS
The gate, panel, weights, and thresholds were all discovered on 2020-10→now. Freezing that config and reporting 2023→now still leaks.

| Test | Protocol | Pass criteria |
|---|---|---|
| Re-derived split | Re-run the **entire** selection (panel, weights, BUY/SELL thresholds, combination mode) on **only ≤2022-12** data — discard prior picks — report **once** on 2023→now | edge persists. If impractical, 2023→now is **descriptive only** and the row below is load-bearing. |
| **True OOS** (load-bearing) | run the frozen config on **synthetic 3× QQQ pre-2010** (dot-com + GFC) and a **different leveraged underlying** | edge survives on data that informed no choice |

For a different underlying, the signal source **switches** to that underlying's own inputs. **SPXL/UPRO are near-clones** of the QQQ trade → weak evidence; **SOXL (semis)** is genuinely different → stronger. For pre-2010 synthetic, **pre-register** the cost params (`rate` = historical 3-month T-bill *series*; `fee` fixed) and report drawdown sensitivity to ±50 bps fee. In-window per-regime numbers are **descriptive only**.

### 6.3 The A/B — does the weighted gate actually beat the simple one?
| Test | Comparators (all on the §6.2 OOS slices) | Pass criteria |
|---|---|---|
| Gate A/B | resonance-gate · SMA22/44-gate · (resonance AND SMA) · (resonance OR SMA) · buy-hold TQQQ · buy-hold QQQ | the resonance-gate (or a combination) must beat the **SMA22/44 gate AND buy-hold** on Calmar **and** drawdown, on **every** OOS slice. If the plain SMA gate wins, **ship the SMA gate** and stop. |
| Weighting A/B | equal-weight vs category-balanced vs tuned weights | weighting must measurably help, else use equal weights (simpler) |

### 6.4 Overfit guard (hard, not vibes)
- **Cap free parameters tuned on data**; pre-register a single config before OOS.
- Penalize multiple testing: report a **deflated** Sharpe/Calmar (Bonferroni or López de Prado) over the number of configs tried.
- Cost/turnover: net edge must survive realistic cost; report switch count (the dual-threshold gap is the lever).

**Honesty rails:** sub-20% drawdown in this window is optimistic — the bare gate's *full-cycle* (1999–2026 synthetic) drawdown was −76%. Report in-window as in-window. Prefer plateaus over single best cells. **If §6.1 or §6.3 fails, the design is rejected, not patched.**

---

## 7. Decisions (locked)

| # | Decision | Choice |
|---|---|---|
| Q1 | Risk-off floor | **Sell all → 100% cash/T-bills.** No QQQ floor. |
| Q2/Q3 | Gate model | **Power-weighted score + dual BUY/SELL thresholds, binary in/out, no sizing.** A/B tested vs the SMA gate and combinations. |
| Q4 | Weighting | **Category-balanced** prior + per-signal power weights; equal-vs-balanced-vs-tuned A/B'd. |
| Q5 | Breadth | **Build "% of top-N QQQ holdings above MA" in v1**, and **add it to the QQQ market-breadth dashboard** (separate deliverable). |
| Q6 | Conviction cap | **N/A** — no sizing, so no cap. |
| Q7 | Pullback axis | **v1 trend-only.** Pullback/oversold entries deferred to v2. |

---

## 8. Why this shape

- It's *your* model: weighted signals, one score, a buy line and a sell line, binary in/out.
- It keeps the proven SMA gate as the explicit **bar to beat**, and A/B tests honestly instead of assuming the weighted gate wins.
- Trend-only v1 is the simplest version that can be tested; pullback (a contrarian, opposite-direction setup) is cleanly deferred to v2.
- Respects every constraint (≤66 MAs, 2020-10 window, real TQQQ, costs, no-lookahead) and the rigor from review (CIs, true OOS, overfit guard).
- **Falsifiable:** if the weighted gate doesn't beat the SMA gate + buy-hold out-of-sample (§6.3), we don't ship it — simpler wins.
