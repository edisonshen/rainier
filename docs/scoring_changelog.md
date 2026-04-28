# QU100 Screener Scoring Changelog

All changes to composite scoring weights, components, and thresholds.

---

## v2 — 2026-04-01: Add rank scoring component

**Trigger**: MAR (Marriott) ranked #1 Long in QU100 was classified as "watch" instead of "buy" because pattern confidence (0.55 for forming w_bottom) dominated the composite at 65% weight.

**Change**: Added rank as a 4th scoring component. Higher QU100 rank = higher score, with a bonus for top-20 stocks.

### Components (4 layers)

| Component | Weight | Description |
|---|---|---|
| Money Flow | 0.20 | Long/short, capital flow direction, rank bucket, rank change, days in top100 |
| Sector | 0.10 | +0.1 boost for stocks in bullish sectors (net_sentiment > 0.30) |
| **Rank** | **0.15** | **NEW — linear scale: rank 1 = 1.0, rank 100 = 0.0. Top-20 bonus: +0.15 (capped at 1.0)** |
| Pattern | 0.55 | Best actionable pattern confidence (Caisen methodology) |

### Thresholds

| Tier | Old | New |
|---|---|---|
| strong_buy | >= 0.80 | >= 0.75 |
| buy | >= 0.65 | >= 0.60 |
| watch | >= 0.50 | >= 0.45 |

### Rank score formula

```
base = (100 - rank) / 99          # linear: rank 1 → 1.0, rank 100 → 0.0
if rank <= 20: base += 0.15       # top-20 bonus
score = min(base, 1.0)            # cap at 1.0
```

### Example: MAR rank 1, pattern conf 0.55

| | v1 | v2 |
|---|---|---|
| money_flow | 0.25 * 0.85 = 0.2125 | 0.20 * 0.85 = 0.1700 |
| sector | 0.10 * 0.10 = 0.0100 | 0.10 * 0.10 = 0.0100 |
| rank | — | 0.15 * 1.00 = 0.1500 |
| pattern | 0.65 * 0.55 = 0.3559 | 0.55 * 0.55 = 0.3011 |
| **composite** | **0.578 (watch)** | **0.631 (buy)** |

### Other changes
- Discord report now shows recommendation tier (S-BUY/BUY/WATCH) and QU100 rank in summary table

---

## v1 — Initial (pre-2026-04-01)

### Components (3 layers)

| Component | Weight | Description |
|---|---|---|
| Money Flow | 0.25 | Long/short, capital flow direction, rank <= 30 bonus, rank change, days in top100 |
| Sector | 0.10 | +0.1 boost for stocks in bullish sectors |
| Pattern | 0.65 | Best actionable pattern confidence |

### Thresholds

| Tier | Threshold |
|---|---|
| strong_buy | >= 0.80 |
| buy | >= 0.65 |
| watch | >= 0.50 |
