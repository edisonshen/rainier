# Leveraged/inverse ETF appearances in QU100 vs QQQ/SPY tops and bottoms

Question: when SQQQ / TQQQ (and the SPY equivalents SPXS/SPXU/SDS/SH, SPXL/UPRO/SSO) show up
repeatedly in the QU100 list - e.g. "3 times in a week" - is that an early signal of a
bottom (uptrend start) or a top?

Data: `money_flow_snapshots`, 2020-10-01 .. 2026-09-25, 1494 QU100 dates, latest capture per
`(data_date, ranking_type, symbol)`. Benchmarks: QQQ / SPY daily closes (yfinance).
Reproduce with `uv run python scripts/analyze_leveraged_etf_signal.py` (writes `reports/etf_signal/`).

## How often do they appear

| symbol | top100 dates | bottom100 dates |
|---|---|---|
| TQQQ | 1088 | 179 |
| SQQQ | 753 | 89 |
| SOXL | 856 | 129 |
| SPXL | 180 | 129 |
| SPXS | 151 | 49 |
| SPXU | 107 | 46 |
| UPRO | 78 | 129 |
| SOXS | 187 | 54 |

TQQQ is in the top100 on ~73% of all days and SQQQ on ~50%, so a single appearance carries
almost no information. Only the *count in the trailing 5 sessions* and the *combination*
(bear ETF in, bull ETF out) separate regimes.

## Method

* Signal = symbol present in `top100` (or `bottom100`) on >= N of the last 5 sessions.
* Signal days cluster, so stats are computed at the **episode** level: first day of a
  cluster, with a 20-session cooldown before a new episode can start.
* Forward QQQ/SPY return over 5/10/20 sessions vs. unconditional baseline, plus a bootstrap
  p-value (`p_20d` = probability that a random sample of the same number of days has a mean
  20-day return >= the observed one).
* Location features: drawdown from the 60-day high on the signal day, worst close over the
  next 10 sessions (`min_fwd_10d`), and whether the day is within +/-3 sessions of a centered
  20-day swing low / high.

## Results (episode level)

### QQQ (baseline: mean 20d return +1.5%, win rate 64%, near swing low 22%, near swing high 21%)

| signal | episodes | mean 20d | win 20d | p_20d | dd from 60d high | worst next 10d | near swing low | near swing high |
|---|---|---|---|---|---|---|---|---|
| SQQQ top100 >= 3 of 5 | 52 | +2.1% | 67% | 0.23 | -6.3% | -1.8% | 29% | 15% |
| SQQQ top100 all 5 | 27 | +2.2% | 63% | 0.25 | -8.4% | -2.3% | 30% | 7% |
| **bear ETF top100 >= 3 of 5 and TQQQ/QLD not in top100** | 42 | **+2.7%** | 69% | **0.085** | -6.7% | -2.7% | 26% | 7% |
| TQQQ bottom100 >= 3 of 5 | 13 | +0.6% | 62% | 0.70 | -7.3% | -2.4% | 31% | 0% |
| TQQQ top100 >= 3 of 5 | 70 | +1.2% | 64% | 0.68 | -4.3% | -2.1% | 27% | 26% |
| no bear ETF in top100 for 5 sessions and TQQQ in top100 | 29 | +1.1% | 59% | 0.64 | -1.3% | -2.1% | 21% | 28% |

### SPY (baseline: mean 20d return +1.3%, win rate 68%, near swing low 20%, near swing high 19%)

| signal | episodes | mean 20d | win 20d | p_20d | dd from 60d high | worst next 10d | near swing low | near swing high |
|---|---|---|---|---|---|---|---|---|
| SPXS top100 >= 3 of 5 | 14 | +2.5% | 50% | 0.13 | -9.0% | -1.8% | 29% | 7% |
| SPXU top100 >= 3 of 5 | 10 | +3.0% | 80% | 0.08 | -8.0% | -1.1% | 30% | 0% |
| any SPY bear ETF top100 >= 3 of 5 and bull ETF not in top100 | 17 | +2.2% | 65% | 0.16 | -7.7% | -2.0% | 18% | 6% |
| UPRO bottom100 >= 3 of 5 | 9 | +2.0% | 78% | 0.31 | -6.2% | -1.3% | 33% | 0% |
| SPXL top100 >= 3 of 5 | 13 | +1.3% | 62% | 0.52 | -3.5% | -1.8% | 15% | 31% |
| no bear ETF in top100 for 5 sessions and bull ETF in top100 | 42 | +1.2% | 69% | 0.54 | -2.0% | -1.3% | 10% | 26% |

### Timing check: does the burst mark *the* low?

For each "bear ETF >= 3 of 5" episode, where is the lowest close of the next 30 sessions?

| | signal episodes | random days |
|---|---|---|
| median sessions until the 30d low | 8-9 | 8-9 |
| low is on the signal day itself | 15-24% | 14% |
| low within 5 sessions | 40-43% | 38-41% |
| median further drop to that low (QQQ) | -3.1% to -4.7% | -2.7% |

Reverse test - first SQQQ/SPXS/SPXU appearance after >= 10 sessions absent, as a *top*
warning: 14-28 episodes, 20d forward return -0.5% to +0.9% vs baseline +1.3..1.5%, p 0.68-0.90.
Weakly negative, not significant.

## Interpretation

1. **The hypothesis is directionally right but weak.** When SQQQ (or SPXS/SPXU) is in the
   top100 on 3+ of the last 5 sessions, QQQ/SPY are already ~6-9% below their 60-day high,
   and the following 20 sessions return roughly +2.1..3.0% vs. +1.3..1.5% on an average day.
   The best variant (bear ETF crowding in *and* TQQQ dropped out of the top100) reaches
   p ~= 0.085 on 42 episodes - suggestive, not conventionally significant, and it is the best
   of ~10 variants tried, so some of that edge is selection.
2. **It does not time the bottom.** Swing-low proximity (26-30%) is only modestly above the
   22% base rate, the low of the next month is typically still 8-9 sessions away (same as a
   random day), and the index usually falls another 2-5% first. The extra forward return is
   mostly the ordinary mean reversion of being in a pullback, not a precise turn signal.
3. **TQQQ presence says nothing.** TQQQ is in the top100 most of the time; "TQQQ 3 of 5" is
   indistinguishable from baseline. TQQQ in the *bottom100* (outflows) 3 of 5 is rare (13
   episodes) and has *below*-baseline follow-through - the opposite of a buy signal.
4. **Top signals are weaker than bottom signals.** "No bear ETF for a week and bull ETF
   present" (complacency) has slightly lower forward return and higher swing-high proximity
   (26-28% vs 19-21%), but nothing statistically meaningful. First-appearance-of-SQQQ as a
   sell warning is also flat.
5. **SPY family samples are small** (9-17 episodes); directions match QQQ but treat the
   numbers as anecdotal.

Practical use: treat "bear leveraged ETF in QU100 top100 on >= 3 of 5 sessions while the bull
ETF is absent" as a *pullback-in-progress / capitulation-building* context flag that tilts the
20-day outlook positive, and expect further downside of a few percent before the actual low.
Do not use it as a standalone entry trigger or as a top signal.

## Follow-up: does SQQQ mark the *start* of a big downtrend instead?

Reframed test: signal = SQQQ (or SPXS) *first* top100 appearance after 5/10/20 sessions absent,
count rising from 0 to >= 2 in 5 sessions, rank <= 20, TQQQ dropping out, bear-ETF breadth, etc.
Outcomes = P(index falls >= 5% within 20 sessions), P(falls >= 10% within 40), P(new 60-day low
within 20), P(20-day return < 0), vs. unconditional base rates. Episode-level, 20-session cooldown.

| signal | n | P(fall >= 5% / 20d) | P(new 60d low / 20d) | P(fwd20 < 0) |
|---|---|---|---|---|
| **QQQ baseline** | | 28% | 20% | 37% |
| SQQQ first top100 after 5d absence | 43 | 21% | 12% | 42% |
| SQQQ first top100 after 10d absence | 14 | 21% | 7% | 50% |
| SQQQ count5 rises 0 -> >= 2 | 18 | 28% | 11% | 33% |
| SQQQ rank <= 20 | 26 | 27% | 46% | 27% |
| TQQQ first bottom100 (outflow) after 10d | 34 | 18% | 9% | 21% |
| # bear/vol ETFs in top100 rises to >= 4 | 16 | 38% | 44% | 31% |
| **SPY baseline** | | 16% | 17% | 32% |
| SPXS first top100 after 5d absence | 29 | 31% (p=0.04) | 34% | 45% |
| SPXS count5 rises 0 -> >= 2 | 12 | 17% | 42% | 67% (p=0.02) |
| SPXS \| SPXU first top100 after 5d absence | 36 | 22% | 28% | 39% |

Robustness check on the SPXS result (the only one that looked real): split 2020-22 vs 2023+:
fall>=5% 44% vs 25% base in 2020-22, but 15% vs 11% in 2023+; adding SPXU to the definition
removes the edge (21% vs 25% base in 2020-22). 16 of the 29 episodes are in 2021-22. It is
mostly "2022 was a bear market", not a transferable signal.

Recall check: of the 10 QQQ local peaks followed by a >= 10% drop, SQQQ was in the top100
within +/-5 sessions of all 10 - but SQQQ is in the top100 in 94% of *all* 11-session windows,
so that is no information. SPXS: 3 of 5 SPY peaks vs 35% base (n too small).

Broader search for a "trending down" group signal:
* `long_short == "Short in"` share of the top100: no edge at 80/90/95th percentile.
* Every symbol with >= 15 first-appearance episodes (488 symbols): best z-score 2.3
  (APPS, ALB, RIO ...), which is exactly what the maximum of 488 noise draws looks like.
  No individual symbol is a usable downtrend predictor; random-pick groupings will overfit.
* Sector share of top100 at >= 90th percentile: Energy-crowded days -> 42% fall>=5% (n=26,
  concentrated in 2022); Tech-crowded (n=23) / Financials-crowded (n=32) days -> 4% / 9%,
  i.e. when Tech or Financials dominate inflows the index rarely drops 5% in the next month.
  Regime-dependent and small-n; worth watching, not proven.

Conclusion: SQQQ appearing in QU100 does **not** mark the start of a big downtrend for QQQ. It
is in the list roughly half of all days and its first appearance is followed by *fewer*
5%+ drops than average. SQQQ with a strong rank (<= 20) or 4+ bear ETFs together means the
market is already ~9% off its high and volatile (new 60-day lows 44-46% vs 20%), but 20-day
returns are still above average - a late-decline / capitulation marker, not an early one.
