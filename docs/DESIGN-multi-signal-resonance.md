# Design Plan — Multi-Signal Resonance for TQQQ Conviction

**Status:** draft for review · **Scope:** signal-resonance layer for the TQQQ timing strategy · **Window:** 2020-10 → now (real TQQQ) · **Constraint:** no moving-average / lookback > 66 bars · **Depends on:** `src/rainier/signals/`, `src/rainier/backtest/` engine · **PR base:** main

---

## 1. The problem, in plain English

We want to buy TQQQ (3× Nasdaq-100) **only when we're confident**, and size the bet to that confidence. A single indicator — "price is above its 44-day average" — flips in and out on every wiggle and gives plenty of false buys. We have *dozens* of signals (trend, momentum, volatility, structure). The question this plan answers:

> **How do we combine many signals into one "conviction score" so that more agreement → more confidence → a bigger, higher-win-rate bet — without the lag that killed naive consensus voting?**

We call the agreement of many signals **resonance**: when lots of independent indicators all say "risk-on" at once, that's a stronger, more trustworthy signal than any one of them alone.

### What we already proved (so this isn't a guess)

Across 2020-10 → now we bucketed every day by how many of a 26-signal panel agreed, then looked at TQQQ's *next 20 days*:

```
 signals agreeing →  win-rate of the next 20 days
   0–20%               48%   (coin flip)
  20–40%               59%
  40–60%               60%
  60–80%               61%
  80–100%              62%   ← most agreement = highest win-rate
```

**Resonance is real: more agreement genuinely raises the win-rate (48% → 62%).** That is the empirical foundation of this design. One nuance we also found: the *biggest forward returns* come at **60–80%** agreement (a building consensus, early in a move), not at 100% (which tends to be mid-trend, after the easy money). So conviction should reward *strong* consensus but not blindly chase *unanimous* consensus.

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

The obvious idea — *be in TQQQ when ≥70% of signals agree, else cash* — we tested. **It underperforms the simple gate** (Calmar 0.20 vs 0.34 full-cycle; worse in-window too). Why:

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

### 5.2 The panel (≤66 lookback, ~26 signals)

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
- **Hysteresis** on the score feeding any threshold, to avoid size chatter at the boundary.

### 5.4 Sizing curve (gate-ON only)

`size(r)` maps resonance → target TQQQ fraction, clamped [0, 1]:
- piecewise: `0` below `r_lo` (≈0.5), ramp to `1` by `r_hi` (≈0.7), flat `1` above (cap — don't over-size unanimity).
- turnover-aware: round size to coarse steps (e.g. 0/0.5/1.0) so it doesn't rebalance daily and bleed cost.

### 5.5 No-lookahead + costs

Signals computed on close[t]; target weight applied to t+1 return (shift +1). Warmup buffer: load from 2019-06 so every ≤66 signal is valid by 2020-10; **measure only over 2020-10 → now** (real TQQQ). Charge turnover cost on size changes; cash earns the 13-week T-bill.

**Lookback invariant:** no indicator parameter (window / EMA-span / period) exceeds 66 bars, and no expanding/all-history windows (medians, z-scores, percentiles). Two flavors:
- *Finite-window* members (SMA, rolling-median, Donchian, ROC) — exact support ≤66.
- *EMA-family* members (EMA, RSI, MACD, ADX) — recursively smoothed, span/period ≤66; infinite tail in theory but bounded *effective* memory once warmed up.

Tests: (a) assert every signal's max parameter ≤66; (b) finite-window signals are bit-identical when history before *t−66* is truncated; (c) EMA-family signals converge to <1e-6 relative given ≥150 warmup bars (the 2019-06 buffer supplies ~330). This is why the constraint says ≤66 *parameter*, not ≤66 *exact support*.

### 5.6 Config (sweepable)

`panel members · per-category weights · gate (entry/exit SMA) · sizing curve (r_lo, r_hi, steps) · hysteresis band · rebalance granularity`.

---

## 6. Test plan

| Test | Input | Expected / pass criteria |
|---|---|---|
| Thesis (sanity) | resonance buckets × fwd-20d TQQQ | win-rate rises with resonance (reproduce 48→62%) |
| Strategy backtest | 2020-10→now, real TQQQ | beats SMA22/44-gate baseline on Calmar **and** drawdown, or it's not worth the complexity |
| Sizing ablation | full-size gate vs resonance-sized | resonance sizing improves Calmar / cuts drawdown at similar return |
| Per-regime | 2021 bull / 2022 bear / 2023-24 / 2025 | smaller drawdown in 2022 & 2025 than the bare gate |
| Walk-forward | 60/40 split in-window | sized strategy's edge survives OOS split |
| Cost/turnover | sweep rebalance granularity | net edge survives realistic cost; switches stay sane |
| Baseline floor | vs buy-hold TQQQ & QQQ | must beat both risk-adjusted |

**Honesty rails (carried from prior work):** one real bear (2022) in this window — sub-20% drawdown here is optimistic; the bare gate's *full-cycle* drawdown was −76%. Report in-window numbers as in-window. No single-best-cell worship; prefer plateaus.

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
