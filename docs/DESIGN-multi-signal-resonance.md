# Design Plan — Multi-Signal Resonance for TQQQ Conviction

**Status:** draft for review · **Scope:** signal-resonance layer for the TQQQ timing strategy · **Window:** 2020-10 → now (real TQQQ) · **Constraint:** no moving-average / lookback > 66 bars · **Depends on:** `src/rainier/signals/`, `src/rainier/backtest/` engine · **PR base:** main

---

## 1. The problem, in plain English

We want to buy TQQQ (3× Nasdaq-100) **only when we're confident**, and size the bet to that confidence. A single indicator — "price is above its 44-day average" — flips in and out on every wiggle and gives plenty of false buys. We have *dozens* of signals (trend, momentum, volatility, structure). The question this plan answers:

> **How do we combine many signals into one "conviction score" so that more agreement → more confidence → a bigger, higher-win-rate bet — without the lag that killed naive consensus voting?**

We call the agreement of many signals **resonance**: when lots of independent indicators all say "risk-on" at once, that's a stronger, more trustworthy signal than any one of them alone.

### What the data *suggests* (a lead, not yet proof)

Across 2020-10 → now we bucketed every day by how many of a 26-signal *exploratory* panel agreed (the ≤66 subset of `build_signals` + SPY, as in `scripts/resonance_study.py`), then looked at TQQQ's *next 20 days*. (v1 freezes a slightly different ~22-member panel — §5.2 — so the resonance **denominator is the frozen count**, never a hardcoded 26.)

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

- **An exploratory signal panel** (`regime_signal_search.build_signals`), each emitting a daily risk-on (1) / risk-off (0) series. It is *unfiltered* — most members today use >66 lookback and two use expanding medians; v1 takes only its **≤66 subset** and rewrites the expanding members (see §5.2).
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
  asymmetric SMA22/44 gate         consensus of the ≤66-lookback panel (§5.2)
  decides IN vs OUT of TQQQ        decides HOW BIG when in
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
            position size = gate_on × size(resonance)
```

- **Layer 1 (gate)** answers *"should I be in TQQQ at all right now?"* — fast, responsive, the best-tested timing engine so far. When the gate is OFF → cash (T-bills). No lag penalty on timing.
- **Layer 2 (resonance)** answers *"how convinced am I?"* — when the gate is ON, scale the position by the conviction score. High resonance (many signals agree) → full 3× exposure. Weak resonance (signals split) → reduced exposure (e.g. partial TQQQ + cash, or 1× QQQ).

This uses the **candidate** 48%→62% win-rate edge (pending §6.1 significance) as a **sizing** signal, where lag doesn't hurt, instead of a **timing** signal, where it does.

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

**Provenance — important:** v1 does **NOT** pin the full `regime_signal_search.build_signals` list. That list is mostly **>66** lookback (SMA100/150/200, EMA100/200, ROC120, 12-1 momentum, 90/120-day highs, slow 50/150 & 50/200 crosses) and uses **expanding** medians — both forbidden here.

**The enumerated list below IS the single source of truth** for the v1 panel (the resonance denominator and category counts derive from it — nothing else). It was *seeded* from the ≤66 subset of `build_signals` (the complement of the `BLOCK` set in `scripts/full_combo_search.py`), then deliberately **deduplicated** — keep one threshold per indicator (RSI>50, drop RSI>55; VIX<25, drop VIX<20/<30; one of ADX/+DI) so no single indicator is triple-counted in the consensus — with two expanding-median members **rewritten** to bounded form and `price>SMA66` **added**. So the list is *not* identical to the raw BLOCK-complement; where prose and the list disagree, **the list wins**. New members (SPY cross-asset, optional fractal, breadth proxy) are fresh code and each count against the §6.4 parameter budget. The frozen list is recorded verbatim at build time.

- **Trend** (from ≤66 subset): price > SMA{20,50,66}, price > EMA{20,50}, SMA22>SMA44, SMA50-rising.
- **Momentum** (from ≤66 subset): RSI14>50, MACD-hist>0, ROC{20,60}>0, price>Donchian50-mid.
- **Volatility** (rewritten): realizedVol(20, rolling std) < its **rolling-40 median** (20+40 = 60, both finite), ATR%(14, **simple rolling mean** of true range — *not* Wilder/EWMA) < its rolling-46 median (14+46 = 60, both finite), VIX<25, VIX<SMA20, VIX-falling. *(Replaces the prototype's `expanding(60).median()` AND its Wilder-EWMA `atr_pct` — both have unbounded lookback; the finite-window invariant §5.5(b) requires simple rolling forms here.)*
- **Structure** (from ≤66 subset): within 5% of 60-day high, ADX>20 & +DI>−DI, higher-high+higher-low.
- **Cross-asset** (NEW code): SPY>SMA22, SPY>SMA44 — `build_signals(qqq, vix)` takes no SPY today; v1 adds an SPY input.
- **(NEW, optional)** fractal:simple_turn — not in `build_signals`; added if Q7 says so.
- **(Gap)** market breadth (% NDX above its MA) — not in yfinance; proxy = "% of top-N QQQ holdings above their SMA". Follow-up, not v1-blocking.

*Every member's **composed/total** lookback is ≤ 66 — a 20-day measure compared to a 40-day median counts as 60, not "66 because the outer window is 66." Finite-window members have exact composed support ≤66; EMA-family members (EMA, RSI, MACD, ADX) are recursively smoothed with span/period ≤66 — infinite tail in theory, bounded **effective** memory (invariant in §5.5).*

### 5.3 Scoring

`resonance[t] = Σ wᵢ · signalᵢ[t] / Σ wᵢ` — weighted fraction risk-on.

- **Category-balanced weights** so one over-represented family (we have many trend signals) doesn't dominate: weight each of the 5 categories equally, split within. Avoids "resonance" really being "trend, four times."
- Anti-chatter hysteresis is applied **once**, at the stepped-value stage of the §5.4 state machine — *not* here on the raw score (avoids the double-hysteresis ambiguity).

### 5.4 Sizing curve + state machine (gate-ON only)

`size(r)` maps resonance → target TQQQ fraction, clamped [0, 1]: `0` below `r_lo`, ramp to `1` by `r_hi`, flat `1` above. To avoid ambiguity, the weight is computed by **one** ordered state machine — no second hysteresis layer:

```
raw resonance r[t]  ──►  (1) curve: size = clamp((r−r_lo)/(r_hi−r_lo), 0, 1)
                    ──►  (2) round to coarse step ∈ {0, ½, 1}  → target_step[t]
                    ──►  (3) hysteresis: the HELD step moves to target_step[t]
                              only after target_step has held the SAME new value
                              for ≥ `dwell` consecutive bars; otherwise the held
                              step is carried forward unchanged
                    ──►  (4) rebalance only when the held step actually changes
```

Boot: at the first valid bar `t0` (after the §5.5c per-member warmup), held_step = target_step[t0] — provisional, taken without confirmation since there's no prior; every *subsequent* move still requires the full `dwell`. Moves are **direct** to the confirmed target (e.g. 0→1 in one transition once the new step holds `dwell` bars), not forced stepwise through ½. The dwell counter resets whenever target_step changes value. Hysteresis lives **only** here — not on the raw score (the §5.3 over-spec is removed). Deterministic and unit-testable; a test asserts a fixed input sequence yields a fixed held-step sequence.

### 5.5 No-lookahead + costs

Every input (TQQQ, QQQ, SPY, VIX) uses its **close[t]** value; the Layer-1 gate and the Layer-2 size both use the *same* timestamp alignment and the *same* +1 shift — weight decided on close[t] is applied to the t→t+1 return. Warmup buffer: load from 2019-06 so every ≤66 signal is valid by 2020-10; **measure only over 2020-10 → now** (real TQQQ).

**Costs — no double-counting:** real adjusted TQQQ prices *already embed* the 3× daily-reset financing and decay. So on real TQQQ we charge **only** (a) turnover/slippage on rebalances and (b) the T-bill rate the cash sleeve earns/forgoes — **NOT** a synthetic 3× financing charge (that would double-count). The synthetic-3× financing formula (`3·r − 2·rate − fee`) is reserved for the §6.2 pre-2010 OOS test, where no real ETF exists.

**Leakage test:** inject a known future spike into bar *t+1*; assert weight[t] is unchanged (zero leakage), across all four inputs and both layers.

**Lookback invariant:** the **composed/total** lookback of every member ≤ 66 bars, and no expanding/all-history windows. "Composed" matters for *nested* indicators: `realizedVol(20)` fed into a `rolling-k median` has total span `20+k`, so the median window must be ≤46, not 66. *Finite-window* members (SMA, rolling-median, Donchian, ROC, and the rewritten nested-vol members) have exact composed support ≤66. *EMA-family* members (EMA, RSI, MACD, ADX) are recursively smoothed with span/period ≤66 — infinite tail in theory, bounded *effective* memory. Tests: (a) every signal's composed span ≤66 (inner+outer for nested); (b) finite-window signals bit-identical when history before *t−66* is dropped; (c) **per-member** warmup — empirically measure each EMA-family signal's convergence (ADX is double-smoothed, MACD-hist is EMAs-of-EMAs, so they need more warmup than a raw EMA) and set the buffer from the *slowest* member, not a global asserted constant. The two prototype `expanding(60).median()` members fail test (b) and **must** be rewritten to the bounded form in §5.2 — this is a required panel change, not optional.

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

The gate (SMA22/44), the panel, and the sizing thresholds were all discovered on 2020-10→now. So merely *freezing* that config and reporting 2023→now still leaks — the held-out slice already shaped the choices.

| Test | Protocol | Pass criteria |
|---|---|---|
| Re-derived split | Re-run the **entire** selection pipeline (gate sweep, bucket thresholds, panel, weights) using **only ≤2022-12** data — discard the prior full-window picks — then report **once** on 2023→now, untouched | edge persists. If re-deriving is impractical, 2023→now is **descriptive only** and the load-bearing test is the row below. |
| **True OOS** (load-bearing) | run the frozen config on **synthetic 3× QQQ pre-2010** (dot-com + GFC — the regimes leverage decay punishes most) and on a **different leveraged underlying** | edge survives on data that informed no choice |

**Genuine-independence caveats:** for a different underlying, the signal source **switches** to that underlying's own inputs (SPX/QQQ-correlated), with gate/panel *structure* held fixed. **SPXL/UPRO are near-clones of the QQQ trade** → weak independent evidence; **SOXL (semis)** is a genuinely different regime → stronger. For pre-2010 synthetic, **pre-register** the cost params before running: `rate` = the historical 3-month T-bill *series* (not a constant — it swings 0–5%), `fee` = a fixed expense; report drawdown **sensitivity** to ±50 bps fee, since the dot-com/GFC verdict hinges on them.

In-window per-regime numbers (2021/2022/2023-24/2025) are **descriptive only**, never validation — one path each, 2025 partial.

### 6.3 Is it the *signal*, or just less leverage?

| Test | Control (precisely pinned) | Pass criteria |
|---|---|---|
| Matched-exposure | (a) **constant** scaler = the resonance sizer's *exact* mean weight; (b) **vol-target** scaler matched on *both* mean exposure AND realized-vol budget. **Per slice:** on each §6.2 OOS slice, both controls' mean is re-derived from *that slice's own* realized resonance-weight series (not carried across underlyings). | resonance must beat the **better** of (a)/(b) on Calmar **on every §6.2 OOS slice** (2023→now, pre-2010 synthetic, different-underlying) — not one cherry-picked slice. Else the "edge" is just de-levering, and we ship the simpler vol-target instead. |

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

- Uses the **candidate** edge (resonance → win-rate, pending §6.1) where it would work (sizing), not where it doesn't (timing).
- Keeps the **best-tested** timing engine (SMA gate) untouched and responsive.
- Generalizes your multi-signal "confirmed buy" into a tunable conviction dial.
- Respects the constraints we established (≤66 MAs, 2020-10 window, real TQQQ, costs, no-lookahead).
- Falsifiable: if resonance sizing doesn't beat the bare gate on Calmar **and** drawdown (§6), we don't ship it — simpler wins.
