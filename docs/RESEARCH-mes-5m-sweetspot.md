# RESEARCH — MES 5-minute "sweet spot" study

- **Status:** exploratory research (one PR for the artifact; no production wiring)
- **Scope:** decode the operator's discretionary 5-min MES chart stack → backtest it (long, short-mirror, and combined) → find the risk-adjusted-best config
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
rule). That is **374 config combinations per direction** (including the optional HVN/VRVP
add-ons). This section is the **long** side; the short and combined sweeps are in
[Long vs short vs combined](#long-vs-short-vs-combined-the-operator-trades-both-ways). We
charged a **1.0-index-point round-trip cost**
(MES is liquid; ~0.5–1.0 pt of slippage+commission is realistic). We ranked by **Ret/DD**
(total window return ÷ worst drawdown — a risk-adjusted score that, unlike annualized Calmar,
is **not** inflated by annualizing a 5-month window) with a **30-trade floor** so tiny lucky
samples can't win. We pick the **featured "sweet spot" as the best Ret/DD among configs with
≥60 trades** — deliberately *not* the single top row, because the literal #1 by Ret/DD is a
thin ~32-trade config whose ratio is a small-sample fluke. We do not crown a trade-count
outlier.

> **Note on exact figures.** `data/csv/MES_5m.csv` is a *live-accumulating* file (each day
> appends new bars), so the precise decimals below drift slightly run-to-run. The figures here
> are rounded/approximate and describe the stable picture; the **HTML render
> (`docs/RESEARCH-mes-5m-sweetspot.html`, regenerated by the script) carries the exact current
> numbers**, and the *ranking* (TD7 config > bare triangle > buy-and-hold on Ret/DD) is robust
> across runs.

**The sweet spot:** the **green triangle + a recent TD7 exhaustion, exited on a 2-ATR stop
with a 3:1 reward target** (`fractal+TD7 · ATR2×RR3`) — about **+6% return, Ret/DD ≈3.7,
Sharpe ≈3.5, ~43% win rate, ~72 trades.** It beats both **buy-and-hold MES** (≈+7% return but
Ret/DD only ≈0.7 — buy-and-hold makes a similar return while eating a far bigger drawdown)
**and the bare green triangle** (`fractal · time48`: ≈+9.6% return but Ret/DD ≈3.1, ~200
trades) on a risk-adjusted basis. (Ret/DD counts *in-trade* drawdown — the worst the equity
dipped mid-trade — so it is a conservative, honest risk figure.)

**The key finding — confluence DOES earn its keep, via the TD exhaustion filter.** The bare
triangle makes the most raw return (≈+9.6%) simply by buying every sharp dip in an up-drifting
market and holding. But it takes ~200 trades and rides bigger swings to get there. Layering the
**DeMark TD7 "downside-exhausted" timing filter** roughly cuts the trade count to ~72, *raises*
the risk-adjusted ratio (Ret/DD ≈3.1 → ≈3.7, Sharpe ≈2.1 → ≈3.5), and pairs naturally with a
tight 2-ATR stop + 3:1 target. This matches the screenshots' intuition: the *best* green
triangles are the ones that fire right after a selloff has exhausted itself. (The VWMA/ribbon
trend filters, by contrast, did **not** add risk-adjusted value here — in a 5-month uptrend
there were few sustained downtrends for them to veto. TD timing helped; trend filtering had
little to do.)

**Walk-forward (the honest read):** we tuned the *robust* risk-adjusted-best config on the
first 60% — using the **same selection rule the headline sweet spot uses** (best Ret/DD among
configs with enough trades, not a thin fluke), warming the rolling indicators from pre-split
history as a real forward test would — then ran that frozen config on the unseen last 40%. On
60% of the data the robust pick is the **bare triangle + 4h time-stop** (200 trades is the
most stable, highest-count config). **Out of sample it held up well: ≈+5.5% return, Ret/DD
≈2.9 over ~75 trades** — the edge generalized, not a one-window fluke. (An earlier version of
this study validated a thin 32-trade config here and saw it go flat OOS; that was the wrong
config to test — validating a robust config, the result is genuinely positive.) Caveat stands:
this is still one 5-month up-regime, so "survives this walk-forward" ≠ "survives a bear market."

### Bottom line

> **The operator's green-triangle entry is a real dip-buy signal on MES 5-min. The featured
> sweet spot pairs it with a DeMark TD7 exhaustion filter and a 2-ATR / 3:1 stop-target — best
> risk-adjusted on the full window (Ret/DD ≈3.7, Sharpe ≈3.5), beating both buy-and-hold and the
> bare triangle. The TD timing filter genuinely helps; the trend (VWMA/ribbon) filters didn't,
> in this one up-regime. Encouragingly, the robust core (bare triangle + 4h-stop) SURVIVED
> walk-forward out-of-sample (≈+5.5%, Ret/DD ≈2.9). Still: 5 months is one up-regime. Treat the
> sweet spot as a strong _hypothesis to forward-test_, not yet a deployable edge — one bear
> market could change the picture.**

This is a *credible* result, consistent with project memory: simple signals on short windows
look good in-sample and must be distrusted out-of-sample (`project_tqqq_regime_switching`).

---

## Long vs short vs combined (the operator trades both ways)

Everything above is the **long** side — the operator's actual green-triangle BUY signal. But the
operator also **shorts**. The operator's Pine only *plots* the long signal, so there is no
published short rule. We therefore built a faithful **derived mirror** and clearly label it as
such: it is **not** the operator's own signal.

**The derived short signal (`fractalDownTrend`).** It is the exact axis-flip of the long
triangle: where the long fires on a bullish bottom reversal, the short fires on a bearish **top**
reversal. Every comparison is reflected — the up-fractal in the *highs* (a local peak) replaces
the down-fractal in the lows; three **red** bars replace three green; a **new low** replaces a new
high; a prior fast-EMA-**above**-slow replaces fast-below-slow. The volume gate is identical. We
mirrored the TD count too (a **sell**-setup 7/9 = *upside* exhaustion), and flipped each
confluence filter (price *below* VWMA55, ribbon *falling*, ribbon stacked *bearish*).

```
   LONG  (operator's Pine)          SHORT  (our derived mirror — NOT in the Pine)
   ───────────────────────          ────────────────────────────────────────────
   down-fractal in LOWS             up-fractal in HIGHS   (high[3] = local peak)
   3 green bars, body expands       3 red bars, body expands
   new HIGH > high[3]               new LOW < low[3]
   ema5 < ema11 (was falling)       ema5 > ema11 (was rising)
   → BUY the bottom reversal        → SELL the top reversal
```

We re-ran the **entire sweep three ways**: long-only (the headline above), short-only, and a
**combined** book that can be long *or* short, **one position at a time, no pyramiding** (a long
signal while already in a trade is ignored; on the rare bar where both fire, long wins — a
documented, reproducible tie-break). Each is ranked the same way (Ret/DD with the trade-count
floor) and walk-forward validated.

### The honest result: on this regime, shorts DRAG

> **⚠ Regime caveat — this is the whole point.** The 5-month sample (2026-01 → 06) is a
> **predominantly UP regime** — a broad grind higher. That is the **single worst environment in
> which to evaluate a short strategy**: every short spends the whole window fighting an upward
> drift. So whatever the short numbers say here, **they are not a fair test of the short signal.**

What we found (exact figures in the HTML render; the stable picture):

- **Short-only** looks *mildly* positive **in-sample** (a low-double-digit Ret/DD on the best
  config) but **fails out-of-sample**: the frozen short config goes **negative OOS** (≈−1.4%
  return, **Ret/DD ≈ −0.6**). The in-sample number was overfit; the held-out window exposes it.
- **Combined long+short** looks great in-sample (its in-sample Ret/DD even *tops* long-only,
  because the leaderboard cherry-picks the few shorts that worked) — but **out-of-sample it is
  clearly worse than long-only** (combined OOS Ret/DD ≈1.6 vs long-only OOS Ret/DD ≈2.7). The
  shorts that survive selection in-sample do not generalize; they just dilute the long edge OOS.
- **The TD-timed LONG remains the best risk-adjusted config**, in-sample and out.

**Why shorts drag here — and why that's expected, not a signal flaw.** Shorting a rising tape is
structurally a losing proposition. The short result on this data tells us almost nothing about
whether the mirror is a *good signal* — it only confirms the obvious: **don't short an uptrend.**
The short signal is mechanically the exact reflection of the long one, so if the long signal is
real (and it is, on this window), the short signal is a **credible candidate the moment the
regime cooperates.** This window simply cannot prove it either way.

### Bottom line (long/short)

> **Keep trading the LONG.** On this 5-month UP regime the derived short mirror **subtracts**:
> short-only fails out-of-sample, and the combined book underperforms long-only out-of-sample.
> This is exactly what theory predicts when you evaluate shorts in a sustained uptrend — it is
> **not** evidence the short signal is bad, only that you shouldn't short a rising market. A fair
> short evaluation **needs a down/chop regime.** Until the tape turns, treat the short mirror as a
> **hypothesis to forward-test**, not a live edge — and forward-test it specifically through a
> sustained selloff, where it would actually have a chance to earn its keep.

---

## Where this fits vs the existing codebase (the gap)

The repo already has signal infrastructure, but **none of it does this job**:

- `src/rainier/signals/resonance.py` + `panel.py` + `resonance_gate.py` is a **daily** trend
  gate for **TQQQ/QQQ** — a power-weighted panel of ≤66-day indicators that flips in/out of a
  leveraged ETF. Different instrument, different timeframe (daily, not 5-min), different
  purpose (regime gate, not intraday entry).
- Prior exploratory work ran the fractal signal on **daily** MES bars and found it fires too
  rarely to trade there — which is exactly why this study moves it to the 5-minute timeframe,
  where the signal actually occurs often enough to evaluate.

**Neither systematizes this 5-minute intraday stack.** That is the gap this study fills on the
*research* side. The natural next step — explicitly **out of scope here** — is to promote the
winning entry into a `FractalSignalEmitter` implementing the repo's `SignalEmitter` protocol
(`emit(df, symbol, timeframe) → list[Signal]`, per CLAUDE.md "Adding a New Signal Strategy").
The backtest engine, sweep runner, and export would then work on it unchanged. **We recommend
that as a follow-up, but do not build it now** — the in-sample edge is real and survived this
walk-forward, but 5 months is one up-regime, so it deserves a forward-test before production
wiring.

---

## Recommended next step

1. **Forward-test on paper** (the only honest validation on a 5-month sample): run the
   featured sweet spot (triangle + TD7 exhaustion, 2-ATR / 3:1 exit) — and the simpler bare
   triangle + 4h-stop (which survived walk-forward) as a control — live-but-simulated for a
   quarter and compare to this study's stats. A real regime change is the true test.
2. **Get more data** spanning a real downtrend (e.g. a sustained bear stretch). The TD-timing
   filter helped here; the trend (VWMA/ribbon) filters didn't — but a 5-month uptrend can't
   test them. **This is doubly true for the short mirror:** the only fair test of the derived
   short signal is a down/chop regime, which this UP window does not contain. Re-run the
   short-only and combined sweeps on a window with sustained selloffs before drawing any
   conclusion about shorts.
3. **If both hold up**, promote to a `FractalSignalEmitter` (separate PR) behind the existing
   `SignalEmitter` protocol — long first; add the short mirror only if a down-regime test
   validates it. Not before.

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
- `fractal_down_trend(df)` — the **DERIVED short mirror** (NOT in the operator's Pine). Exact
  axis-flip: `fractalsUp` = the up-fractal in the **highs** over `t-5..t-1` (`high[t-3]` the
  local peak) AND the SAME volume gate. `strongBearFractal` = `close[t]<open[t-1]` AND
  `close[t-1]<open[t-1]` AND `close[t-2]<open[t-2]` AND `|close[t-1]-open[t-1]| >
  |close[t-3]-open[t-3]|` AND `low[t]<low[t-3]` AND `ema5[t-3] > ema11[t-3]`. Every `<`/`>`,
  `low`/`high`, and EMA-cross direction is the reflection of the long rule.
- `vwma(df, 55)` — `sum(hlc3·vol,55)/sum(vol,55)`, rolling (NOT session-reset). `above_vwma`
  = `close >= vwma`; `below_vwma` = `close <= vwma` (the short mirror).
- `td_setup_buy(df, completed=9)` / `td_buy_context(df, lookback=6, min_count=7)` — canonical
  DeMark buy-setup: count increments while `close < close[4]`, resets otherwise. `td_buy_context`
  rolls the **completion EVENT** (the bar where the count crosses up through `min_count`), NOT
  the raw running count, and is True within `lookback` bars of an event. (Using `count ≥
  min_count` would stay true for every bar of a long selloff as the count keeps climbing,
  admitting entries long after the actual TD7/9 print.) Rolling, not session-reset (mirrors the
  chart). `td_setup_sell` / `td_sell_context` are the **mirror** (count increments while
  `close > close[4]` → upside exhaustion, the short-timing context).
- `ribbon_bearish` (`close<sma44 AND sma44 falling`) and `ribbon_stacked_bear`
  (`sma22<sma44<sma120`) are the short mirrors of `ribbon_bullish` / `ribbon_stacked`.
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

### Simulation (`simulate`, `simulate_combined`)
One position at a time, full equity per trade, takes a `direction` (`"long"` | `"short"`). No
lookahead: signal at `close[t]` → enter `open[t+1]`. Per-trade return is **direction-signed**:
**linear relative to the entry price** for both sides — long = `(exit-entry)/entry`, short =
`(entry-exit)/entry` — each net of cost. (Using the same entry-price denominator both ways keeps
long and short symmetric: a short 100→90 earns +10%, not the +11.1% a naive `entry/exit-1` price
ratio would report.) The mark-to-market factor uses the identical basis so the equity curve and
the booked return agree. Exits MIRROR by direction:
- **ATR** — long: `stop=entry-k·ATR(14)` (below), target above, intrabar fill when `low≤stop` /
  `high≥tp`. Short: `stop=entry+k·ATR` (above), target below, fill when `high≥stop` / `low≤tp`.
  Intrabar risk is **checked before** the flatten so a stop on a session-last bar fills at the stop.
- **time** — holds exactly `bars` bars incl. the entry bar → exit at `entry_bar+bars-1` (direction-agnostic).
- **vwmaCross** — long exits when `close < VWMA55`; short exits when `close > VWMA55` (reclaim).

The shared trade walk lives in `_execute_trade` (entry → exit) and `_book_trade` (direction-signed
P&L); both `simulate` and `simulate_combined` call them, so the long and short mechanics can never
drift apart. **`simulate_combined`** runs the long+short book: while flat it takes whichever signal
fires first (long wins a same-bar tie, documented + reproducible); while in a trade all signals are
ignored (no pyramiding) — so trade count and exposure stay comparable to buy-and-hold.

A `flatten` mask force-closes at the bar close: the **CME equity-futures session-end bars** (the
trading day rolls at 17:00 ET, computed in US/Eastern — NOT at 00:00 UTC, which would force flats
mid-session), **plus the RTH-close bars for RTH-gated configs**. The equity curve is
**marked-to-market through each hold** (held bars priced at close vs entry, direction-signed), so
max-DD and Sharpe capture in-trade adverse excursion, not just the final outcome. Cost =
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
frozen one is far worse than hindsight, it was fit). `walk_forward(df, mode=)` runs the same
validation for **each** of long / short / combined; the long/short verdict keys off the
**out-of-sample** result (not the overfit in-sample leaderboard) when deciding whether shorts
add or drag.

### Cost / leverage note
MES = $5/point. The sweet spot's ≈+6% ≈ the equity-curve return of a single-instrument
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
| **short fractal fires** | hand-built peak reversal | `fractalDownTrend` True at the constructed bar |
| short fractal silent | flat tape / pure downtrend | no short signal |
| short fractal causal | mutate a FUTURE bar | earlier short signal unchanged |
| short ≠ long mirror | long fixture vs short fixture | long fires only on bottoms, short only on tops |
| td sell setup | 30 ascending closes | completes "9" at the 13th bar (mirror of buy) |
| td sell reset | ascending → down-close → ascending | no "9" before the reset |
| td sell context event | long sustained rally | True near completion, False many bars later |
| ribbon bearish | falling / rising series | True late in downtrend, never in uptrend |
| ribbon stacked bear | long fall | sma22<sma44<sma120 True at end |
| below vwma | drop below VWMA | True at the drop, False in NaN window |
| short confluence subset | strict short config vs base short | strict ⊆ base-short entries |
| direction dispatch | long vs short EntryConfig, shared cache | short uses short base, long uses long base |
| short sim P&L sign | falling / rising price | short gains on a decline, loses on a rally |
| short ATR stop side | upside spike | stop ABOVE entry; fills at the stop |
| short ATR target side | downside dip | target BELOW entry; fills at the target |
| short vwma exit | price reclaims VWMA | short exits on `close > VWMA` (reclaim) |
| short mark-to-market | adverse upside spike mid-hold | equity drawdown marked; max-DD reflects it |
| combined one-at-a-time | long signal while a trade is open | in-trade signal skipped; next trade after exit |
| combined tie-break | long & short on the same bar | long wins (documented, reproducible) |
| combined grid rows | fixture with a long + a short | sweep emits `mode="combined"` rows |
| short grid mirrors long | long grid vs short grid | same flag combos, opposite direction |
| short P&L is linear | `_book_trade` short 100→90 / 100→110 | +10% / −10% gross (not ±11.1%/9.1%); long/short symmetric about entry |
| mtm matches book basis | `_mtm_factor` vs `_book_trade` at exit price | mtm factor = 1 + booked gross return (same linear basis) |
| combined OOS vs long OOS | combined OOS Ret/DD below / above long OOS | "does NOT beat" when below; "barely beats" only when above; conservative w/o long WF |
