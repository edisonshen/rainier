# TODOS

## QU100 Stock Screener (蔡森 多空转折一手抓)

Design doc: `docs/stock_screener_design.md`
Book: 蔡森 "多空转折一手抓" at `/Users/pinkbear/Downloads/多空转折一手抓(高清).pdf`

### Phase 1 — Foundation ✓

- [x] **#1 pattern_primitives.py** — Swing points, necklines, breakout detection, volume-price
- [x] **#2 target_calculator.py** — 涨幅/跌幅满足 price target calculations (7 compute functions)
- [x] **#4 core/types.py** — Added PatternSignal, MoneyFlowSignal, SectorTrend, StockScreenResult
- [x] **#3 stock_patterns.py** — All 12 蔡森 pattern detectors (W底, M头, 破底翻, 假突破, 破底翻W底, 假突破头肩顶, 下飘旗形, 上飘旗形, 头肩底, 头肩顶, 收敛三角形底部, 收敛三角形头部)
- [x] **#5 Tests** — 19 tests in test_pattern_primitives.py + test_stock_patterns.py
- [ ] **#6 Visual validation gate** — Overlay swing points/necklines on Plotly charts, compare to TradingView on 10 stocks

### Phase 2 — Money Flow + Sector ✓

- [x] **#7 stock_screener.py** — 3-layer pipeline with screen_stocks() entry point
- [x] **#8 sector_analyzer.py** — Sector trend analysis + 0.1 bullish sector boost
- [x] **#StockScreenerConfig** — Added to config.py + settings.yaml

### Phase 3 — Remaining Bullish Patterns ✓ (merged into Phase 1)

- [x] **#9 All 6 bullish patterns implemented** in stock_patterns.py

### Phase 4 — Bearish + Scoring ✓ (merged into Phase 1)

- [x] **#10 All 6 bearish mirrors + composite scoring** implemented
- [x] **Scheduler wired up** — screen_stocks() called after every scrape, top 20 sent to Discord
  - Composite: 0.25 money_flow + 0.10 sector + 0.65 pattern

### Phase 5 — LLM Vision Validation

- [ ] **#11 llm_validator.py** — Anthropic API pattern validation [blocked by #3, #6]
  - Render chart PNG → Claude Sonnet → confidence 1-10
  - Combined = 0.6 * rule + 0.4 * LLM, fallback on error
  - Add `anthropic` + `kaleido` to pyproject.toml

### Phase 6 — Persistence + Report

- [ ] **#12 Database tables** — StockPatternSignal, StockScreenRecord [blocked by #4]
- [ ] **#13 Report + CLI + Discord** [blocked by #10, #11, #12]
  - `rainier screen` command, Discord alerts, 16:30 PST daily scheduler

### Phase 7 — Visualization

- [ ] **#14 Interactive pattern overlay charts** [blocked by #6, #10]

### Pattern Weights (Tier 1-4)

| Tier | Patterns | Weight |
|------|----------|--------|
| 1 | 破底翻, 假突破 | 1.0 |
| 2 | W底, M头, 头肩底/顶 | 0.80-0.85 |
| 3 | 旗形, 假突破+头肩 | 0.75 |
| 4 | 三角形 | 0.65 |

---

## AI/ML Pipeline (Futures)

### ScoringStrategy protocol
**What:** Build BookScorer wrapping existing `score_setup`, MLScorer interface. Pipeline-first: validate end-to-end with BookScorer before swapping in ML.
**Priority:** P1
**Depends on:** FeatureExtractor ✓

### Feature Store (Parquet export)
**What:** Export features + labels to Parquet files for ML training/backtesting.
**Priority:** P1

### HMM Regime Detector
**What:** Train on historical NQ/ES/GC data. 4 regimes: trending-up, trending-down, ranging, volatile.
**Priority:** P2

### XGBoost Pattern Scorer
**What:** Single model with regime as feature. Train on labeled features from backtest.
**Priority:** P2
**Depends on:** Feature Store, HMM

## Completed

### Wire up real multi-TF confluence scoring ✓
Replaced placeholder `confluence_score = 0.5` in scorer with `_multi_tf_confluence_score()`. Generator now passes `sr_levels` to `score_setup()`.

### Update design doc to reflect eng review decisions ✓
All 10 eng review decisions incorporated into design doc.

### FeatureExtractor ✓
Built `src/quant/features/extractor.py` — transforms AnalysisResult + OHLCV into ~30 ML-ready features. NaN policy: fill with meaningful defaults + assert clean.

### LabelGenerator ✓
Built `src/quant/features/labels.py` — generates training labels from backtest trades. TP=1, SL=0, end-of-data excluded by default.

### Strict pin bar detection (book rules) ✓
Updated `pinbar.py` to match 小酱 methodology: dominant wick > 2/3, no spinning tops (secondary wick < 1/3 of dominant), wick >= 2x body, visually prominent check. Reduced 5m pin bars from 1,673 to 178.

### Trading TF skips own S/R levels ✓
5m chart only shows S/R from 1H+ (book: higher TF for key levels, trading TF for signals only).

### Multi-TF level merging ✓
Higher TF absorbs nearby lower TF levels, refines price using lower TF precision, boosts strength for confluence.

### Tabbed multi-TF chart ✓
Built `create_tabbed_chart()` with 1D/4H/1H/5m tabs, pin bar + signal toggles, TradingView-style watermark, light/dark theme toggle. All candlestick visibility issues resolved.

### Chart redesign: TradingView-style ✓
- **Root cause fix**: Plotly CDN `plotly-latest.min.js` served v1.58.5, incompatible with plotly.py 6.6.0 (v3.4.0). Pinned to `plotly-3.4.0.min.js`.
- **Candlestick rendering**: Switched from `fig.to_html()` (binary encoding) to `fig.to_json()` + `Plotly.newPlot()` for reliable rendering.
- **Per-TF bar limits**: 1D=120, 4H=120, 1H=168, 5m=500.
- **TradingView watermark**: Large faded "MES, 5m" symbol in chart center.
- **Y-axis right-side**: Clean layout with S/R TF labels inside chart area.
- **Light/dark theme toggle**: Dark (black bg, green/red candles, cyan S/R) and Light (white bg, black/white candles, teal S/R).
- **Signal toggle**: Show/hide entry diamonds + TP/SL boxes.
- **Pin bar toggle**: Works across all TFs (1D=0, 4H=10, 1H=15, 5m=178).
- **X-axis session times**: 5m uses fixed futures session times (4:00, 6:30, 9:30, 12:30, 15:00, 23:00).
- **S/R line styling**: Cyan color, ends before TF label, no overlap with y-axis.
- **Inter font**: Clean, modern typography for title/labels.

### Data: yfinance 5m pull ✓
4 weeks of MES 5m data via yfinance. Timestamps converted to US/Pacific.

### Chart: no-gap x-axis ✓
Sequential integer x-axis eliminates weekend/maintenance gaps. Date labels on ticks.

### Config updates ✓
- `min_confidence`: 0.85 → 0.60
- `min_rr_ratio`: 2.0 → 1.5 (book minimum)
- `max_sr_levels`: 12 → 10
- `min_touches` (MES/ES): 5 → 4
- Pin bar config: strict book rules in settings.yaml
