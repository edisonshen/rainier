# Design Plan — Multi-Signal Resonance for TQQQ Conviction

**Status:** draft for review · **Scope:** signal-resonance layer for the TQQQ timing strategy · **Window:** 2020-10 → now (real TQQQ) · **Constraint:** no moving-average / lookback > 66 bars · **Depends on:** `src/rainier/signals/`, `src/rainier/backtest/` engine · **PR base:** main

---

## 1. The problem, in plain English

We want to buy TQQQ (3× Nasdaq-100) **only when we're confident**, and size the bet to that confidence. A single indicator — "price is above its 44-day average" — flips in and out on every wiggle and gives plenty of false buys. We have *dozens* of signals (trend, momentum, volatility, structure). The question this plan answers:

> **How do we combine many signals into one "conviction score" so that more agreement → more confidence → a bigger, higher-win-rate bet — without the lag that killed naive consensus voting?**

We call the agreement of many signals **resonance**: when lots of independent indicators all say "risk-on" at once, that's a stronger, more trustworthy signal than any one of them alone.

### What the data *suggests* (a lead, not yet proof)

Across 2020-10 → now we bucketed every day by how many of a 26-signal panel agreed, then looked at TQQQ's *next 20 days*:

```
 signals agreeing →  win-rate of the next 20 days
   0–20%               48%   (coin flip)
  20–40%               59%
  40–60%               60%
  60–80%               61%
  80–100%              62%   ← most agreement = highest win-rate
```

The win-rate rises with agreement — a promising, monotone signal. **But treat it as a lead, not proof, until it clears the bar in §6.** Two reasons to be skeptical: (1) the forward-20-day windows **overlap** (adjacent days share ~19/20 of their data), so the *effective* sample is ~N/20 ≈ a few dozen independent points spread over 5 buckets — too few to call significant without confidence intervals; (2) it's one window dominated by a single 2022 bear and the 2020–21 recovery. The "biggest returns at 60–80% agreement" remark is likewise an unverified in-sample observation (and win-rate ≠ mean return — different statistics). **§6 gates this:** report block-bootstrap CIs and the effective non-overlapping N per bucket; only call it real once a CI excludes the null. The rest of this plan is conditional on that.

---

## 2. How it works today

Right now there is no integrated system — just a pile of exploratory scripts. Two things exist:

- **A 26-signal panel** (`regime_signal_search.build_signals`), each emitting a daily risk-on (1) / risk-off (0) series, all with lookback ≤ 66.
- **A timing gate** — the asymmetric SMA gate (enter when QQQ > SMA22, exit when QQQ < SMA44). In-window it was the single best risk-adjusted design (Calmar ≈ 1.37, #2 of 270 in a full sweep, on a stable plateau).

```
TODAY:  one signal  ──►  in/out of TQQQ at full size
        (e.g. price>SMA44)         (binary, no notion of confidence)
```

There is **no concept of conviction or position size** — every trade is all-or-nothing, and the dozens of other signals are unused.

---

## 3. What goes wrong (why we don't just "vote")

The obvious idea — *be in TQQQ when ≥70% of signals agree, else cash* — we tested. It underperformed the simple gate (Calmar 0.20 vs 0.34 on the synthetic full-cycle; worse in-window too). *This negative result is **motivating, not load-bearing** — same one-window caveats apply; it's enough to steer away from voting-as-a-gate, not a proven fact.* The mechanism is clear regardless:

```
THE LAG TRAP:
   price ────╲                      ╱──── recovery
              ╲                    ╱
   gate exits  ╲__________________╱   gate re-enters (fast)
                  ▲              ▲
   votes flip   LATE           LATE   ← consensus needs many signals to flip;
   risk-off  (already down)  (misses the bounce)
```

- **Entering on consensus lags the bottom:** by the time 70% of signals turn risk-on, the rebound is half over.
- **Exiting on consensus lags the top:** the drawdown is already eaten before enough signals flip.
- Consensus voting flips *less* (less whipsaw) but each move is *late*, so it has higher win-rate yet **worse risk-adjusted return**.

**Lesson: resonance is good at measuring *confidence*, bad at deciding *timing*.** The fast SMA gate is the opposite — good timing, no confidence. The design uses each for what it's good at.

---

## 4. The fix — a two-layer design

Separate **timing** from **conviction**:

```
LAYER 1 — TIMING (fast):        LAYER 2 — CONVICTION (resonance):
  asymmetric SMA22/44 gate         consensus of the 26-signal panel
  decides IN vs OUT of TQQQ        decides HOW BIG when in
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
            position size = gate_on × size(resonance)
```

- **Layer 1 (gate)** answers *"should I be in TQQQ at all right now?"* — fast, responsive, the proven timing engine. When the gate is OFF → cash (T-bills). No lag penalty on timing.
- **Layer 2 (resonance)** answers *"how convinced am I?"* — when the gate is ON, scale the position by the conviction score. High resonance (many signals agree) → full 3× exposure. Weak resonance (signals split) → reduced exposure (e.g. partial TQQQ + cash, or 1× QQQ).

This directly uses the confirmed 48%→62% win-rate edge as a **sizing** signal, where lag doesn't hurt, instead of a **timing** signal, where it does.

### Your "confirmed buy" is a *second* axis — pullback resonance

A distinction the design must respect: your example — *QQQ<SMA10 AND SPY<SMA10 AND breadth<30% AND VIX<23 AND fractal buy* — is built from **oversold / pullback** conditions (price *below* short MAs, washed-out breadth). The trend-resonance score above counts *risk-on* signals (price *above* MAs), so this exact cluster would score **low**, not high. They are two different axes:

- **Trend resonance** — "the uptrend is broadly confirmed" (price above MAs, momentum positive, vol calm). Drives Layer-2 **sizing**.
- **Pullback resonance** — "an oversold dip is bottoming *inside* an uptrend" (price below SMA10, breadth washed out, fractal turn, VIX not panicked). A separate **entry** trigger for buying the dip.

v1 models your confirmed-buy as an **optional pullback sub-score / alternate entry**, regime-gated (only buy dips while a slower trend filter is still up — don't catch knives in a real bear), **not** as the high end of trend resonance. Richer, and an open decision (Q7).

### What it looks like day to day

```
  gate OFF                         → 0% TQQQ (cash, earning T-bills)
  gate ON, resonance 50–65%        → reduced size (e.g. 40–60% TQQQ)
  gate ON, resonance 65–80%        → full size      (peak-return zone)
  gate ON, resonance > 80%         → full size, but capped (don't chase unanimity)
```

---

## 5. Implementation detail (for engineers)

### 5.1 Components

| Component | Role | Where |
|---|---|---|
| `SignalPanel` | registry of ≤66-lookback signals; each `(df) → risk-on series ∈ {0,1}` | new `src/rainier/signals/panel.py` |
| `ResonanceScorer` | panel → daily conviction score ∈ [0,1] (weighted consensus) | new `src/rainier/signals/resonance.py` |
| `ResonanceStrategy` | Layer-1 gate × Layer-2 size → daily **target-weight series** ∈ [0,1] | new `src/rainier/signals/resonance_strategy.py` |
| Daily-MTM sim | weight series → equity, no lookahead, financing + costs | `src/rainier/backtest/` (productionize the prototype in `scripts/leveraged_common.py`) |

**New boundary — NOT `SignalEmitter`.** The existing `SignalEmitter` returns `list[Signal]` (discrete entry/SL/TP trades) and the current engine consumes discrete trades — it *cannot* represent a continuously-sized daily weight. v1 introduces a distinct **`WeightStrategy`** protocol — `(df, symbol, timeframe) → daily target-weight series ∈ [0,1]` — plus a weight-consuming backtest path (the daily-MTM sim). This is **additive**: it does not change `SignalEmitter` or the discrete-trade engine, which keep serving the pin-bar/fractal strategies.

*Note: the panel signals and the daily-MTM sim currently live only as untracked exploratory scripts (`scripts/regime_signal_search.py`, `scripts/leveraged_common.py`). v1 promotes them into `src/rainier/` per the module map in `CLAUDE.md`; the scripts are the reference implementation, not the shipping location.*

### 5.2 The panel (≤66 lookback)

*The exact membership and per-category counts are **pinned from `regime_signal_search.build_signals`** at build time (the prototype's real list), not the prose below — the "~26" figure is provisional and the breadth "(Gap)" / "(Optional)" members are excluded until added. Category-balanced weighting (§5.3) depends on the final per-category counts, so v1 records the frozen list explicitly.*

- **Trend:** price > SMA{20,50,66}, price > EMA{20,50}, SMA22>SMA44, SMA50-rising.
- **Momentum:** RSI14>50, MACD-hist>0, ROC{20,60}>0, price>Donchian50-mid.
- **Volatility:** realizedVol<rolling-66 median, ATR%<rolling-66 median, VIX<25, VIX<SMA20, VIX-falling. *(Use a bounded 66-bar rolling median, NOT an expanding median — an expanding window grows past the 66-bar cap and would violate the constraint.)*
- **Structure:** within 5% of 60-day high, ADX>20 & +DI>−DI, higher-high+higher-low.
- **Cross-asset:** SPY>SMA22, SPY>SMA44 (broad-market confirm).
- **(Optional)** fractal:simple_turn as a momentum/structure member.
- **(Gap)** market breadth (% NDX above its MA) — not in yfinance; needs a proxy (compute "% of top-N QQQ holdings above their SMA"). Tracked as a follow-up, not v1-blocking.

*Every member's controlling parameter (window / EMA-span / period) is ≤ 66. Finite-window members (SMA, rolling-median, Donchian, ROC) have exact support ≤66; EMA-family members (EMA, RSI, MACD, ADX) are recursively smoothed with span/period ≤66 — theoretically infinite tail but bounded **effective** memory (see the invariant in §5.5).*

### 5.3 Scoring

`resonance[t] = Σ wᵢ · signalᵢ[t] / Σ wᵢ` — weighted fraction risk-on.

- **Category-balanced weights** so one over-represented family (we have many trend signals) doesn't dominate: weight each of the 5 categories equally, split within. Avoids "resonance" really being "trend, four times."
- Anti-chatter hysteresis is applied **once**, at the stepped-value stage of the §5.4 state machine — *not* here on the raw score (avoids the double-hysteresis ambiguity).

### 5.4 Sizing curve + state machine (gate-ON only)

`size(r)` maps resonance → target TQQQ fraction, clamped [0, 1]: `0` below `r_lo`, ramp to `1` by `r_hi`, flat `1` above. To avoid ambiguity, the weight is computed by **one** ordered state machine — no second hysteresis layer:

```
raw resonance r[t]  ──►  (1) curve: size = clamp((r−r_lo)/(r_hi−r_lo), 0, 1)
                    ──►  (2) round to coarse step ∈ {0, ½, 1}
                    ──►  (3) hysteresis ON THE STEPPED VALUE: only change the
                              held step if the new step differs by ≥1 level AND
                              persists ≥ `dwell` bars  (one place, applied last)
                    ──►  (4) rebalance only when the held step changes
```

Hysteresis lives **only** at step (3), on the rounded step — not on the raw score (that was the §5.3 over-spec; remove it there). This makes the path deterministic and unit-testable.

### 5.5 No-lookahead + costs

Every input (TQQQ, QQQ, SPY, VIX) uses its **close[t]** value; the Layer-1 gate and the Layer-2 size both use the *same* timestamp alignment and the *same* +1 shift — weight decided on close[t] is applied to the t→t+1 return. Warmup buffer: load from 2019-06 so every ≤66 signal is valid by 2020-10; **measure only over 2020-10 → now** (real TQQQ).

**Costs — no double-counting:** real adjusted TQQQ prices *already embed* the 3× daily-reset financing and decay. So on real TQQQ we charge **only** (a) turnover/slippage on rebalances and (b) the T-bill rate the cash sleeve earns/forgoes — **NOT** a synthetic 3× financing charge (that would double-count). The synthetic-3× financing formula (`3·r − 2·rate − fee`) is reserved for the §6.2 pre-2010 OOS test, where no real ETF exists.

**Leakage test:** inject a known future spike into bar *t+1*; assert weight[t] is unchanged (zero leakage), across all four inputs and both layers.

**Lookback invariant:** no indicator parameter (window / EMA-span / period) exceeds 66 bars, and no expanding/all-history windows. *Finite-window* members (SMA, rolling-median, Donchian, ROC) have exact support ≤66. *EMA-family* members (EMA, RSI, MACD, ADX) are recursively smoothed with span/period ≤66 — infinite tail in theory, bounded *effective* memory. Tests: (a) every signal's max parameter ≤66; (b) finite-window signals bit-identical when history before *t−66* is dropped; (c) **per-member** warmup — empirically measure each EMA-family signal's convergence (ADX is double-smoothed, MACD-hist is EMAs-of-EMAs, so they need more warmup than a raw EMA) and set the buffer from the *slowest* member, not a global asserted constant.

### 5.6 Config (and what is NOT swept)

Swept: `panel membership · per-category weights · gate (entry/exit SMA) · rebalance granularity`. **Fixed a priori (not swept):** `r_lo`, `r_hi` (read off the §6.1 bucket curve), the cap-above-`r_hi` rule (kept only if §6.1 says the 60–80% return peak is significant). Total free parameters tuned on data are **capped and pre-registered** before any OOS run (see §6.4) — the effective sample is tiny, so every extra knob is overfit risk.

---

## 6. Test plan

The window has **one** bear (2022). A sizing overlay that simply de-levers *mechanically* cuts drawdown, so naive "beats the gate on drawdown" proves nothing. The plan below is built to *try to kill* the idea, not flatter it.

### 6.1 Significance, not just point estimates

| Test | Input | Pass criteria |
|---|---|---|
| Thesis CI | resonance buckets × fwd-20d | report **block-bootstrap CIs** + effective non-overlapping N per bucket; the win-rate trend must have a CI that excludes the null. If it doesn't, the premise fails — stop. |
| Return-by-bucket | same buckets, forward *return* (not win-rate) | show the return curve with CIs; only keep the "cap above 80%" rule if the 60–80% peak is significant — else drop the cap and the claim. |

### 6.2 No data-snooping — frozen train/test + genuine OOS

The gate (SMA22/44), the panel, and the sizing thresholds were all discovered on 2020-10→now. So an in-window 60/40 split is **not** out-of-sample — the "OOS" slice already shaped the choices.

| Test | Protocol | Pass criteria |
|---|---|---|
| Frozen split | **freeze** gate + panel + r_lo/r_hi + weights on a train slice (≤2022-12), report untouched on 2023→now | edge persists on the held-out slice |
| **True OOS** | run the *frozen* config on a **different leveraged underlying** (SPXL/UPRO/SOXL) and on **synthetic 3× QQQ pre-2010** | edge survives on data that did not inform any choice — this is the real test |

In-window per-regime numbers (2021/2022/2023-24/2025) are **descriptive only**, never validation — one path each, and 2025 is a partial year.

### 6.3 Is it the *signal*, or just less leverage?

| Test | Control | Pass criteria |
|---|---|---|
| Matched-exposure | a **constant** scaler and a **volatility-targeted** scaler, each tuned to the *same average exposure* as the resonance sizer | resonance must beat *both* matched-exposure controls on Calmar — otherwise the "edge" is just de-levering and we ship the simpler vol-target instead |

### 6.4 Overfit guard (hard, not vibes)

- **Cap free parameters tuned on data.** Fix `r_lo`/`r_hi` *a priori* from the §6.1 bucket curve — do **not** re-sweep them. Pre-register a single config before looking at OOS.
- Penalize multiple testing: report a **deflated** Sharpe/Calmar (Bonferroni or López de Prado deflated-Sharpe) over the number of configs tried.
- Cost/turnover: net edge must survive realistic cost; report switch count.

### 6.5 Baseline floor
Must beat buy-hold TQQQ **and** QQQ risk-adjusted, **and** the bare SMA22/44 gate, **and** the matched-exposure control. Failing any → don't ship; simpler wins.

**Honesty rails:** sub-20% drawdown in this window is optimistic — the bare gate's *full-cycle* (1999–2026 synthetic) drawdown was −76%. Report in-window numbers as in-window. No single-best-cell worship; prefer plateaus. If §6.1 or §6.3 fails, the design is **rejected**, not patched.

---

## 7. Open decisions (need your call)

- **Q1 — Risk-off floor:** when the gate is OFF, hold **cash/T-bills** (max safety) or **1× QQQ** (stay exposed)? Prior tests: cash was cleaner; QQQ-floor lifted return but raised drawdown.
- **Q2 — Sizing granularity:** continuous size = f(resonance), or coarse **steps** (0 / ½ / full) to cut turnover? Steps are simpler and cheaper; continuous is smoother.
- **Q3 — Does resonance also gate entry, or only size?** Pure design: gate=timing, resonance=size only. Your "confirmed buy" instinct suggests *also* requiring a minimum resonance to enter at all (veto low-conviction). Include the veto, or keep layers clean?
- **Q4 — Panel weighting:** equal-per-signal (simple) or **category-balanced** (so trend doesn't dominate)? I lean category-balanced.
- **Q5 — Breadth proxy now or later?** Build the "% of top QQQ holdings above MA" breadth signal for v1, or ship v1 without it and add later?
- **Q6 — Max conviction cap:** cap size at full 3× TQQQ, or allow a *defensive* sub-full mode (e.g. never exceed 1× in the first N days after a regime flip)?
- **Q7 — Pullback axis in v1?** Include the second (pullback/oversold) conviction axis — your confirmed-buy idea — in v1, or ship trend-resonance sizing first and add pullback entries in v2? Needs the breadth proxy (Q5) to be at its best.

---

## 8. Why this is the right shape

- Uses the **proven** edge (resonance → win-rate) where it works (sizing), not where it doesn't (timing).
- Keeps the **proven** timing engine (SMA gate) untouched and responsive.
- Generalizes your multi-signal "confirmed buy" into a tunable conviction dial.
- Respects the constraints we established (≤66 MAs, 2020-10 window, real TQQQ, costs, no-lookahead).
- Falsifiable: if resonance sizing doesn't beat the bare gate on Calmar **and** drawdown (§6), we don't ship it — simpler wins.
