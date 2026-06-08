# RESEARCH — MES 5-minute "sweet spot" study

- **Status:** exploratory research (one PR for the artifact; no production wiring)
- **Scope:** decode the operator's discretionary 5-min MES chart stack → backtest it → find the risk-adjusted-best config
- **Priority:** P2 (research; informs a future FractalSignalEmitter, builds nothing live)
- **Artifacts:** `scripts/mes_5m_sweetspot_study.py` (runnable), `docs/RESEARCH-mes-5m-sweetspot.html` (live render of the numbers)
- **Data:** `data/csv/MES_5m.csv` — ~28k 5-min bars, 2026-01-09 → 2026-06-08 (~5 months)

---

## The problem, in plain English

The operator day-trades **MES** (the Micro E-mini S&P 500 future — a small contract that
moves $5 per 1.00 index point) on a **5-minute chart**. The chart is covered in indicators:
green/red triangles, a fan of moving-average lines, a heavy white line, little "7" and "9"
numbers, and a volume histogram down the left edge. The operator reads all of this *by eye*
and decides when to buy.

"By eye" is not reproducible. We can't measure whether the read actually makes money, we
can't tune it, and we can't ever automate it. **The goal of this study: turn that
discretionary read into a small set of exact, testable rules, run them over real 5-minute
MES history, and find the "sweet spot" — the rule combination that is genuinely good on a
risk-adjusted basis, not just lucky.**

A blunt honesty constraint up front: we only have **~5 months** of 5-minute data. That is
**one market regime** (a broad grind higher with a couple of pullbacks). Any "winner" found
on 5 months is at high risk of being **overfit** — i.e. it fit the noise of this particular
window and will not repeat. So we do not just report the best number; we **walk-forward
validate** it (tune on the first 60% of the data, then test the frozen rule on the unseen
last 40%) and we compare everything to the dumbest possible benchmark: **buy and hold MES**.

---

## The chart setup, decoded

The operator's chart is five separate indicators stacked together. We decoded each one from
the operator's actual Pine (TradingView) source plus the chart screenshots, and implemented
each as a clean, unit-tested function.

```
   ┌─────────────────────── MES 5-min chart ───────────────────────┐
   │                                                                │
   │  (1) Fractal triangles  ▲ green = buy-the-bottom signal        │
   │  (2) MA ribbon          ≈≈≈ fan of 7 moving averages (trend)   │
   │  (3) "VWAP 55"          ── heavy white line (dynamic support)  │
   │  (4) 九轉序列 TD count   7 9  exhaustion counter at turns        │
   │  (5) VRVP profile       ▌▌▌ volume-by-price (support/resist.)  │
   │                                                                │
   └────────────────────────────────────────────────────────────────┘
```

**(1) Fractal — the green BUY triangle.** This is the actual entry signal. The Pine code
(`fractalUpTrend`) fires a green triangle when a *bottoming* pattern completes: the recent
lows carved a little V (a "down-fractal"), then three bars push up with a strong green body
that makes a new high, and the market had been in a downtrend just before (a fast EMA below
a slow EMA five bars back). In one sentence: **"a sharp green reversal off a fresh swing
low, after a dip."** It is long-only. Note the EMAs are 5 and 11 (not the usual 12/26) — we
honored the source exactly.

**(2) MA ribbon — the trend backdrop.** Seven moving averages (SMA 5, EMA 25, SMA 22, SMA
44, SMA 120, SMA 200, SMA 233) plus Bollinger Bands. By itself it generates no trade; it
tells you *which way the river is flowing*. When the short averages sit above the long ones
and they all slope up, the trend is up. We use it as a **filter**: "only take the green
triangle if the ribbon agrees."

**(3) "VWAP 55" — dynamic support.** Despite the name it is **not** a session VWAP. The Pine
source is a **55-bar rolling volume-weighted average** (a VWMA): the average price of the
last 55 bars, weighting heavier-volume bars more. Price tends to pull back to it and bounce.
We use it as a filter too: "is price holding above this line?"

**(4) 九轉序列【老貓】 — the 7/9 exhaustion counter.** This is the classic **DeMark TD
Sequential setup count**. It counts consecutive bars that each close lower than the bar four
bars earlier; when the count reaches 9 (the operator also watches 7), the down-move is
"exhausted" and a bounce is likely. We use it as a **timing context**: "did a 7+ exhaustion
just complete near here?" (We implemented the standard published DeMark rule, since no Pine
was provided for it.)

**(5) VRVP — volume-by-price.** The left-edge histogram shows how much volume traded at each
price. Fat bars ("high-volume nodes") act as support/resistance; thin bars get crossed fast.
This is the most discretionary of the five. We **approximate** it with a rolling
volume-by-price node-proximity flag and test it as an **optional add-on** layered onto the
bare-fractal and full-confluence anchor configs (so its marginal effect is genuinely
measured), rather than crossing it with every combo. We keep it low-priority and say so
plainly, because faking precision on a discretionary tool would be dishonest.

---

## What the screenshots show

We read ~7 of the operator's 26 chart screenshots (as analysis — there is **no** vision model
here). The recurring pattern is clear and consistent:

- **A winning green triangle** prints at a swing low **while price is at/reclaiming the
  VWMA55 and the ribbon is rising / stacked bullish** (short MAs above long MAs). Price then
  rides up the ribbon. Often a TD 7/9 exhaustion completed just before — the dip ran out of
  sellers right into the entry.
- **A failing green triangle** prints **below a falling ribbon and below the VWMA55** — a
  counter-trend "catch the falling knife." Greens and reds cluster with no follow-through.
- **VRVP nodes** stall and reverse price at the fat part of the histogram; thin nodes get
  traversed quickly.

This is the intuition the backtest is built to test: *the triangle is only worth taking with
trend + VWMA confluence.* The screenshots say the bare triangle should be worse than the
filtered one. **The data, interestingly, disagrees — see below.**

---

## What the backtest found

We backtested the green-triangle entry **alone** (the baseline) and in **confluence
combinations** (triangle + above-VWMA55 + ribbon-rising/stacked + recent-TD-buy + an
optional regular-trading-hours filter), each paired with realistic exits (ATR stop × reward
target, time-stops of 30 min / 1 h / 2 h / 4 h, and a "exit when price loses the VWMA55"
rule). That is **374 config combinations** (including the optional HVN/VRVP add-ons). We
charged a **1.0-index-point round-trip cost**
(MES is liquid; ~0.5–1.0 pt of slippage+commission is realistic). We ranked by **Ret/DD**
(total window return ÷ worst drawdown — a risk-adjusted score that, unlike annualized Calmar,
is **not** inflated by annualizing a 5-month window) with a **30-trade floor** so tiny lucky
samples can't win.

**The sweet spot:** the **bare green triangle with a 4-hour time-stop exit** —
**+10.6% return, Ret/DD 3.5, 57% win rate, 199 trades.** It beats **buy-and-hold MES**
(+7.1%, Ret/DD 0.74) on both return and drawdown.

**The honest, counter-intuitive finding:** **adding the confluence filters did NOT improve
the risk-adjusted result.** Every screenshot-inspired filter (require VWMA, require ribbon,
require TD) *reduced* the trade count and *did not raise* Ret/DD above the bare triangle.
Two readings, both worth stating:

1. In this **one up-trending regime**, the market drifted higher, so "buy any sharp dip and
   hold ~4 hours" caught the drift. The filters threw away winning dip-buys without removing
   proportionally more losers — they cost more than they saved.
2. The screenshots' confluence logic is a **trend/downtrend discriminator**. In a 5-month
   uptrend there were few sustained downtrends for it to filter out, so it had little to do.
   In a genuine bear regime the filters would likely matter — **we just don't have that data
   to prove it.**

**Walk-forward (the honest read):** we tuned the risk-adjusted-best config on the first 60%
(warming the rolling indicators from pre-split history, as a real forward test would) and ran
that frozen config on the unseen last 40%. The in-sample winner that the tuner picked was a
*different* config than the full-window leaderboard's sweet spot (it only saw 60% of the
data), and **out of sample it collapsed to roughly flat (~+0.0%, Ret/DD ≈ 0.05 over ~32
trades).** In other words: **the specific config the walk-forward tuner selected did NOT
generalize** — a textbook in-sample-fit warning. The full-window sweet spot (bare fractal +
4h-stop) is more robust *by construction* (it's the simplest, highest-trade-count config),
but the walk-forward result is a flashing caution light, not a green one. The exact numbers
are in the HTML render's walk-forward table.

### Bottom line

> **The operator's green-triangle entry is a real dip-buy signal on MES 5-min, best paired
> with a simple ~4-hour time-stop; on this window it beat buy-and-hold on both return and
> drawdown. BUT the walk-forward test is a caution light: the config the tuner picked on the
> first 60% went flat out-of-sample, and the screenshot-driven confluence filters did not add
> risk-adjusted value here. 5 months is one regime. Treat the sweet spot as a _hypothesis to
> forward-test_, not a deployable edge. One bear market could flip the verdict — and the
> filters that looked useless here are exactly what would matter then.**

This is a *credible* result, consistent with project memory: simple signals on short windows
look good in-sample and must be distrusted out-of-sample (`project_tqqq_regime_switching`).

---

## Where this fits vs the existing codebase (the gap)

The repo already has signal infrastructure, but **none of it does this job**:

- `src/rainier/signals/resonance.py` + `panel.py` + `resonance_gate.py` is a **daily** trend
  gate for **TQQQ/QQQ** — a power-weighted panel of ≤66-day indicators that flips in/out of a
  leveraged ETF. Different instrument, different timeframe (daily, not 5-min), different
  purpose (regime gate, not intraday entry).
- `scripts/fractal_backtest_study.py` runs the fractal signal on **daily** MES bars and found
  it too rare to trade there.

**Neither systematizes this 5-minute intraday stack.** That is the gap this study fills on the
*research* side. The natural next step — explicitly **out of scope here** — is to promote the
winning entry into a `FractalSignalEmitter` implementing the repo's `SignalEmitter` protocol
(`emit(df, symbol, timeframe) → list[Signal]`, per CLAUDE.md "Adding a New Signal Strategy").
The backtest engine, sweep runner, and export would then work on it unchanged. **We recommend
that as a follow-up, but do not build it now** — the edge is too fragile on 5 months to
deserve production wiring yet.

---

## Recommended next step

1. **Forward-test on paper** (the only honest validation on a 5-month sample): run the bare
   triangle + 4h-stop live-but-simulated for a quarter and compare to this study's stats.
2. **Get more data** spanning a real downtrend (e.g. a sustained bear stretch). Only then can
   we tell whether the confluence filters earn their keep — the screenshots strongly suggest
   they do exactly when this window can't show.
3. **If both hold up**, promote to a `FractalSignalEmitter` (separate PR) behind the existing
   `SignalEmitter` protocol. Not before.

---

## Implementation detail (for engineers)

Everything below is the precise mechanics; the accessible explanation is above.

### Data cleaning
`load_clean_5m`: drop exact-duplicate timestamps (keep last), drop zero-range bars
(`high-low<=0`), drop zero-volume edge bars (VWMA needs volume), sort by time. Per
`project_mes_csv_dirty`, MES CSVs carry phantom flat/zero-volume bars that wreck
volume-dependent signals. The 5-min file had 89 zero-range and 226 zero-volume bars,
0 duplicate timestamps; ~28k → ~27.9k bars survive.

### Signals (each a pure, unit-tested function)
- `fractal_up_trend(df)` — faithful Pine `fractalUpTrend`. Pine `[k]` = k bars ago; evaluated
  at `close[t]`. `fractalsDown` = the V-shape in lows over `t-5..t-1` AND the volume gate
  `sma6Volume[t-2] > fractalVolumeChange[t]` where `fractalVolumeChange = (vol-sma6vol)/sma6vol*100`.
  `strongFractal` = `close[t]>open[t-1]` AND `close[t-1]>open[t-1]` AND `close[t-2]>open[t-2]`
  AND `|close[t-1]-open[t-1]| > |close[t-3]-open[t-3]|` (body expansion) AND `high[t]>high[t-3]`
  AND `ema5[t-3] < ema11[t-3]`. Returns the SIGNAL bar `t`; the sim enters at `open[t+1]`
  (no same-close fill — honors the repo's r2 timing fix).
- `vwma(df, 55)` — `sum(hlc3·vol,55)/sum(vol,55)`, rolling (NOT session-reset). `above_vwma`
  = `close >= vwma`.
- `td_setup_buy(df, completed=9)` / `td_buy_context(df, lookback=6, min_count=7)` — canonical
  DeMark buy-setup: count increments while `close < close[4]`, resets otherwise; flags a
  completed `completed` run. `td_buy_context` = "a ≥`min_count` run completed within the last
  `lookback` bars." Rolling, not session-reset (mirrors the chart).
- `ribbon_bullish` = `close>sma44 AND sma44 rising`. `ribbon_stacked` = `sma22>sma44>sma120`.
- `hvn_proximity` — VRVP approximation: rolling (240-bar) volume-by-price histogram, flag the
  top-30%-volume bins as HVNs, True when the close sits in an HVN bin. Wired as an **optional
  add-on** on the bare-fractal and full-confluence anchor configs (`require_hvn`), so its
  marginal effect IS in the sweep — but it is not crossed with every combo.
- `in_rth` — approximate US cash session (13:30–20:00 UTC ≈ 09:30–16:00 ET during EDT). The
  sample spans an EST→EDT change so a fixed offset can't track DST exactly; documented as
  approximate. The TD count and VWMA are intentionally NOT session-reset (the Pine versions
  are rolling) — only end-of-session *flattening* and the time-of-day *filter* are
  session-aware.

### Simulation (`simulate`)
Long-only, one position at a time, full equity per trade. No lookahead: signal at `close[t]`
→ enter `open[t+1]`. Exits: ATR (`stop=entry-k·ATR(14)`, `target=entry+RR·(entry-stop)`,
intrabar fill at the level), time (`bars` later, at close), vwmaCross (close < VWMA55).
Always flatten at the last bar of a UTC session and at end-of-data. Cost = `ROUND_TRIP_PTS=1.0`
index points charged once per round-trip (converted to fractional on the entry price).

### Ranking & validation
Primary key = **`mar` = window total-return / max-drawdown** (`_mar_key` treats `inf`
zero-DD as a large-but-finite +1e6 so a real-DD winner isn't beaten by a no-DD fluke; a
negative-return zero-DD config maps to −1e6). We deliberately **avoid annualized Calmar** in
the headline because annualizing a 0.41-year window inflates CAGR ~2.4×, which would make the
numbers look far better than they are. Sharpe IS annualized (standard) — note it is therefore
also optimistic on a short window. Trade floor = `MIN_TRADES=30`. Walk-forward: 60/40 split,
tune Ret/DD in-sample with the floor, run the frozen config OOS, and also report the
best-with-hindsight OOS config (if the frozen one is far worse than hindsight, it was fit).

### Cost / leverage note
MES = $5/point. The sweet spot's +10.6% ≈ the equity-curve return of a single-instrument
timing strategy, net of cost — **not** leveraged-futures P&L on margin. On a single contract
the winner made roughly the equivalent point total reported in the HTML render's stat cards.
Return-on-margin would be several times larger (and so would the drawdown).

---

## Test plan

| Test | Scenario / input | Expected |
|---|---|---|
| load clean | csv with zero-range, zero-vol, dup-timestamp rows | junk dropped, unique sorted timestamps, dup keeps last |
| vwma | 4-bar window, known volumes | equals hand-computed volume-weighted mean; NaN before window |
| vwma weighting | one huge-volume bar | VWMA pulled toward the heavy bar |
| td setup | 30 descending closes | completes "9" exactly at the 13th bar (run hits 9) |
| td reset | descending → up-close → descending | no "9" before the reset |
| td context | dip then rally | True near the bottom, False far above it |
| ribbon bullish | rising / falling series | True late in uptrend, never in downtrend; False before sma44 defined |
| ribbon stacked | long rise | True at end, False before sma120 defined |
| above vwma | jump above VWMA | True at the jump, False in the NaN window |
| fractal fires | hand-built V-bottom reversal | signal True at the constructed bar |
| fractal silent | flat tape / pure uptrend | no signal |
| fractal causal | mutate a FUTURE bar | earlier signal unchanged (no lookahead) |
| confluence subset | strict config vs base | strict entries ⊆ base-fractal entries |
| sim entry timing | signal at bar 0 | entry price = open[1], not close[0] |
| sim cost | flat round-trip | net return = −cost (≈ −1pt/entry) |
| sim session flatten | entry near session end, long time-stop | exits at session_end, not held over |
| rth filter | 03:00 UTC vs 15:00 UTC bar | False overnight, True in RTH |
