# Backtest Results

Tracking all backtesting experiments, parameters, and findings for the Rainier trading system.

---

## QU100 Pattern-Filtered Backtest (2026-03-26)

**Setup:** QU100 universe (1,313 symbols), daily data 2024-12-30 to 2026-03-26 (310 trading days).
Pattern detection via Caisen methodology on daily OHLCV. Buy at next day open, hold for 5 trading days.

### Best Patterns (5-day hold)

Tested all 12 Caisen patterns. Three patterns performed best:

| Pattern | Trades | Win Rate | Avg Return | Notes |
|---|---|---|---|---|
| **False Breakdown W Bottom** | 823 | **84.6%** | **+8.56%** | Dominant pattern, highest conviction |
| False Breakdown | 626 | 52.6% | +1.41% | Solid, consistent |
| Bull Flag | 82 | 39.0% | -0.16% | Weakest of the three, negative avg return |

### Top-N Comparison (3 best patterns, 5-day hold)

How many stocks to trade per day? Ranked by pattern confidence, pick top N.

| Top N | Trades | Win Rate | Avg Return | Sharpe | Max Drawdown |
|---|---|---|---|---|---|
| Top 1 | 308 | 69.5% | +7.32% | 2.16 | 17.4% |
| **Top 2** | **616** | **71.1%** | **+6.88%** | **3.14** | **13.6%** |
| **Top 3** | **923** | **69.4%** | **+6.51%** | **3.85** | **10.5%** |
| Top 5 | 1,531 | 69.0% | +5.17% | 4.44 | 11.2% |
| Top 10 | 3,030 | 67.9% | +4.55% | 5.29 | 11.5% |
| All matches | 13,435 | 67.2% | +3.02% | 5.05 | 13.6% |

### Key Findings

1. **False Breakdown W Bottom** is the dominant signal — 84%+ win rate across all top-N levels
2. **Top 2-3 is the sweet spot**: Top 2 has best win rate (71.1%), Top 3 has lowest drawdown (10.5%)
3. More stocks per day = better Sharpe (diversification) but dilutes per-trade returns
4. Top 1 has highest per-trade return but worst drawdown (concentration risk)
5. **Bull Flag may not be worth including** — consistently negative or near-zero avg return

### Recommended Strategy

- Patterns: False Breakdown W Bottom + False Breakdown (drop Bull Flag)
- Top N: 2-3 stocks per day
- Hold: 5 trading days
- Expected: ~70% win rate, +6.5-7% avg return, ~10-14% max drawdown

### Command

```bash
uv run rainier backtest qu100 --patterns --pattern-top-n 3 --hold 5
uv run rainier backtest qu100 --patterns --pattern-top-n 2 --hold 5
```

---

## QU100 Money Flow Ranking Backtest (prior results)

_Baseline strategy: buy top-ranked QU100 "Long in" stocks by money flow, no pattern filter._

Results from parameter sweep and signal tuning variations are available via:
```bash
uv run rainier backtest qu100 --sweep
uv run rainier backtest qu100 --variations
```

---

## Backtest TODO

Priority order. See `docs/strategy_ideas.md` for full hypothesis and backtest plans.

1. [ ] **SPY monthly RSI regime filter** — split results by RSI > 70 vs < 70
2. [ ] **Drop Bull Flag** — re-run with only FB + FBW
3. [ ] **Multi-day "Long In" streak boost** — streak >= 3 days, streak + pattern combo
4. [ ] **Combine pattern + money flow rank** — rank 1-20 AND pattern match
5. [ ] **Hold period per pattern** — sweep 3/5/7/10/15/20d per pattern type
6. [ ] **Stop-loss / trailing stop** — use pattern SL levels
7. [ ] **Multi-timeframe pattern detection** — weekly/monthly bars for QU100 (DELL case)
8. [ ] **Walk-forward validation** — check for overfitting
9. [ ] **Per-sector breakdown** — pattern performance by sector
