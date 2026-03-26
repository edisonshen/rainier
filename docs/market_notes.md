# Market Notes

## 2026-03-26 — Bear Regime Assessment

### Current Conditions
- Market character: bear market / cyclical correction
- Strong open, fade into close pattern
- Major averages below 200-day moving average
- RSI over 75 — overbought in a bear regime, expect failure/pullback in April 2026
- Sentiment: rising pessimism but NOT full capitulation
- VIX elevated but below true washout extremes (no washout yet)
- No Follow-Through Day (FTD) confirmed

### Bull Scenario
- War ends
- Oil prices recede
- Stagflation concerns ease
- Central banks continue easing trajectory

Expected outcome: broadening advance, new leadership from sound bases, confirmed FTD with no immediate distribution, significant vol drop.

### Bear Scenario
- War persists or escalates
- Strait of Hormuz remains disrupted
- Oil prices make new highs
- Stagflation becomes evident in hard economic data

Expected outcome: breakout stocks fail, breadth deteriorates, limited rally attempts, elevated vol/distribution. Would need higher pessimism for durable bottom.

### Key Insights
- A bottom can form without a VIX washout if macro pressures simply ease
- RSI >75 in a bear regime (below 200 DMA) is a classic bull trap signal
- This is a "sit on hands" market until FTD + successful breakouts + breadth improvement confirm

### Regime Detection Plan
Quantitative regime detection using hard signals for backtest engine (Phase B):

| Signal | Bear | Neutral | Bull |
|---|---|---|---|
| SPY vs 200 DMA | Below | Near | Above |
| VIX level | >25 | 15-25 | <15 |
| QU100 pattern win rate (rolling 30d) | <50% | 50-65% | >65% |
| FTD confirmed (IBD rules) | No | Attempted | Yes + holding |
| Breakout failure rate | >60% fail | Mixed | <30% fail |

Rolling QU100 pattern win rate is the most interesting — it's endogenous to our system. If FBW is historically 84% but drops to 50%, that IS the regime signal. No need to interpret macro news; patterns already encode it.
