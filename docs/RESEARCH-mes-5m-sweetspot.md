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
confluence.* The data is split on which confluence: the **TD7 exhaustion-timing** filter
genuinely improves the risk-adjusted result (the "fire right after a selloff exhausts" read
holds up), while the **trend (VWMA/ribbon)** filters did not add value on this single
up-trending window — see below.

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
samples can't win. We pick the **featured "sweet spot" as the best Ret/DD among configs with
≥60 trades** — deliberately *not* the single top row, because the literal #1 by Ret/DD is a
thin ~32-trade config whose ratio is a small-sample fluke. We do not crown a trade-count
outlier.

**The sweet spot:** the **green triangle + a recent TD7 exhaustion, exited on a 2-ATR stop
with a 3:1 reward target** (`fractal+TD7 · ATR2×RR3`) — **+6.1% return, Ret/DD 3.74,
Sharpe 3.54, 43% win rate, 72 trades.** It beats both **buy-and-hold MES** (≈+7% return but
Ret/DD only ≈0.72 — buy-and-hold makes a similar return while eating a far bigger drawdown)
**and the bare green triangle** (`fractal · time48`: +9.8% return but Ret/DD 3.18, 199
trades) on a risk-adjusted basis. (Ret/DD counts *in-trade* drawdown — the worst the equity
dipped mid-trade — so it is a conservative, honest risk figure.)

**The key finding — confluence DOES earn its keep, via the TD exhaustion filter.** The bare
triangle makes the most raw return (+9.8%) simply by buying every sharp dip in an up-drifting
market and holding. But it takes 199 trades and rides bigger swings to get there. Layering the
**DeMark TD7 "downside-exhausted" timing filter** cuts the trade count to 72, *raises* the
risk-adjusted ratio (Ret/DD 3.18 → 3.74, Sharpe 2.17 → 3.54), and pairs naturally with a tight
2-ATR stop + 3:1 target. This matches the screenshots' intuition: the *best* green triangles
are the ones that fire right after a selloff has exhausted itself. (The VWMA/ribbon trend
filters, by contrast, did **not** add risk-adjusted value here — in a 5-month uptrend there
were few sustained downtrends for them to veto. TD timing helped; trend filtering had little
to do.)

**Walk-forward (the honest read — and the cold shower):** we tuned the risk-adjusted-best
config on the first 60% (warming the rolling indicators from pre-split history, as a real
forward test would) and ran that frozen config on the unseen last 40%. **Out of sample it went
flat — roughly 0% return, Ret/DD ≈ 0 over ~32 trades.** The config the tuner picked on 60% of
the data did NOT generalize. So while the *full-window* leaderboard says "TD-filtered triangle
is the sweet spot," the walk-forward says "don't trust any single tuned config on 5 months."
Both are true at once: the TD filter is a real, sensible edge-improver in-sample, but the
specific tuned config is fragile out-of-sample. The exact numbers are in the HTML render's
walk-forward table.

### Bottom line

> **The operator's green-triangle entry is a real dip-buy signal on MES 5-min. The featured
> sweet spot pairs it with a DeMark TD7 exhaustion filter and a 2-ATR / 3:1 stop-target —
> best risk-adjusted on the full window (Ret/DD 3.74, Sharpe 3.5), beating both buy-and-hold
> and the bare triangle. The TD timing filter genuinely helps; the trend (VWMA/ribbon)
> filters didn't, in this one up-regime. BUT the walk-forward is a cold shower: the tuned
> config went flat out-of-sample. 5 months is one regime. Treat the sweet spot as a
> _hypothesis to forward-test_, not a deployable edge — the in-sample edge is real but the
> specific tuning is fragile.**

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

1. **Forward-test on paper** (the only honest validation on a 5-month sample): run the
   featured sweet spot (triangle + TD7 exhaustion, 2-ATR / 3:1 exit) — and the simpler bare
   triangle + 4h-stop as a control — live-but-simulated for a quarter and compare to this
   study's stats. The walk-forward warns the tuned config is fragile, so the paper test is the
   real arbiter.
2. **Get more data** spanning a real downtrend (e.g. a sustained bear stretch). The TD-timing
   filter helped here; the trend (VWMA/ribbon) filters didn't — but a 5-month uptrend can't
   test them. Only a window with sustained downtrends can tell whether the trend filters earn
   their keep, as the screenshots suggest they should.
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
  DeMark buy-setup: count increments while `close < close[4]`, resets otherwise. `td_buy_context`
  rolls the **completion EVENT** (the bar where the count crosses up through `min_count`), NOT
  the raw running count, and is True within `lookback` bars of an event. (Using `count ≥
  min_count` would stay true for every bar of a long selloff as the count keeps climbing,
  admitting entries long after the actual TD7/9 print.) Rolling, not session-reset (mirrors the
  chart).
- `ribbon_bullish` = `close>sma44 AND sma44 rising`. `ribbon_stacked` = `sma22>sma44>sma120`.
- `hvn_proximity` — VRVP approximation: rolling (240-bar) volume-by-price histogram, flag the
  top-30%-volume bins as HVNs, True when the close sits in an HVN bin. Wired as an **optional
  add-on** on the bare-fractal and full-confluence anchor configs (`require_hvn`), so its
  marginal effect IS in the sweep — but it is not crossed with every combo.
- `in_rth` — US cash session (09:30–16:00 ET), **timezone-aware**: timestamps are converted
  UTC→US/Eastern and gated on ET clock time, so it is correct across the sample's EST→EDT
  change (no fixed UTC offset, which would be wrong by an hour for the ~40% EST portion). The
  TD count and VWMA are intentionally NOT session-reset (the Pine versions are rolling) — only
  end-of-session *flattening* and the time-of-day *filter* are session-aware.

### Simulation (`simulate`)
Long-only, one position at a time, full equity per trade. No lookahead: signal at `close[t]`
→ enter `open[t+1]`. Exits: ATR (`stop=entry-k·ATR(14)`, `target=entry+RR·(entry-stop)`,
intrabar fill at the level — **checked before** the flatten so a stop on a session-last bar
fills at the stop), time (holds exactly `bars` bars incl. the entry bar → exit at
`entry_bar+bars-1`), vwmaCross (close < VWMA55). A `flatten` mask force-closes at the bar
close: it is the **CME equity-futures session-end bars** (the trading day rolls at 17:00 ET,
computed in US/Eastern — NOT at 00:00 UTC, which would force flats mid-session), **plus the
RTH-close bars for RTH-gated configs** (so an RTH entry is closed at the cash-session close,
not held into the overnight book). The
equity curve is **marked-to-market through each hold** (held bars priced at close vs entry),
so max-DD and Sharpe capture in-trade adverse excursion, not just the final outcome. Cost =
`ROUND_TRIP_PTS=1.0` index points per round-trip (fractional on entry).

### Ranking & validation
Primary key = **`mar` = window total-return / max-drawdown** (`_mar_key` treats `inf`
zero-DD as a large-but-finite +1e6 so a real-DD winner isn't beaten by a no-DD fluke; a
negative-return zero-DD config maps to −1e6). We deliberately **avoid annualized Calmar** in
the headline because annualizing a 0.41-year window inflates CAGR ~2.4×, which would make the
numbers look far better than they are. Sharpe IS annualized (standard) — note it is therefore
also optimistic on a short window. Trade floor = `MIN_TRADES=30`; the **featured sweet spot**
(`pick_sweet_spot`) is the best Ret/DD among configs with `n ≥ ROBUST_TRADES=60`, so a thin
~30-trade fluke that tops the raw Ret/DD list is never crowned. Walk-forward: 60/40 split,
tune Ret/DD in-sample with the floor (indicators warmed from `WARMUP_BARS=240` pre-split
bars), run the frozen config OOS, and also report the best-with-hindsight OOS config (if the
frozen one is far worse than hindsight, it was fit).

### Cost / leverage note
MES = $5/point. The sweet spot's +6.1% ≈ the equity-curve return of a single-instrument
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
| td context event | long sustained selloff | True near the completion event, False many bars later |
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
| session boundary | bars straddling 17:00 ET | flatten at 17:00 ET, not 00:00 UTC |
| atr precedence | stop touched on session-last bar | fills at the stop, not the bar close |
| time-stop length | `time3` from bar 0 | holds exactly 3 bars, exits at bar 3 |
| mark-to-market | deep mid-trade dip, flat exit | equity dips mid-hold; max-DD reflects it |
| hvn applied | base vs require_hvn | HVN entries ⊆ base entries; HVN appears in the grid |
| hvn breakout | close above prior range | NOT flagged as a high-volume node |
| rth flatten (EDT) | 19:55 / 20:00 / next-day RTH bars | 15:55 ET is the cash-close flatten bar |
| rth tz-aware | EST 09:00 / EST 10:00 / EDT 10:00 / overnight | correct RTH in both EST and EDT |
| rth entry in-session | signal on the last RTH bar | suppressed (entry would fill after close) |
| warm-up exclusion | flat warm-up bars + a scored trade | score_start lifts exposure; return unchanged |
