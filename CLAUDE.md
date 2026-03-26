# CLAUDE.md

## Project Overview

Rainier — trading analysis platform combining futures price action (Xiaojiang pin bar methodology) with stock money flow data (QuantUnicorn QU100). Evolving from rule-based to AI-adaptive with a three-layer hybrid architecture.

## Module Map

```
src/rainier/
├── core/           types.py (shared dataclasses), protocols.py (boundary contracts),
│                   config.py (unified Pydantic), models.py (10 ORM tables), database.py (singleton + TimescaleDB)
├── data/           provider.py (protocol), csv_provider.py, yfinance_provider.py
├── analysis/       analyzer.py (orchestrator), pivots.py, pinbar.py, sr_horizontal.py, sr_diagonal.py, bias.py, inside_bar.py
├── features/       extractor.py (AnalysisResult→ML features), labels.py (TradeRecord→training labels)
├── signals/        generator.py (entry/SL/TP), scorer.py (weighted confidence), emitter.py (SignalEmitter adapters)
├── backtest/       engine.py (event-driven, protocol-based), sweep.py (parameter optimization),
│                   export.py (CSV/Parquet), report.py (text + Plotly)
├── viz/            charts.py (Plotly interactive)
├── scrapers/       base.py (BaseScraper ABC), browser.py (Playwright), qu/ (QuantUnicorn scraper)
├── notifications/  notifier.py (Apprise multi-channel)
├── alerts/         discord.py (webhook notifications)
├── reports/        daily.py (daily review + next-day outlook)
├── scheduler/      jobs.py (cron.yaml → system crontab), service.py (APScheduler for scraping)
├── trader/         (Phase 3 placeholder — IB TWS execution)
└── dashboard/      (placeholder — Streamlit)
```

## Commands

```bash
# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/

# Futures data + analysis
uv run rainier fetch --symbol MES --plot
uv run rainier scan --symbol MES --timeframe 1H --csv data/csv/MES_1H.csv
uv run rainier daytrade --symbol MES --data-dir data/csv
uv run rainier backtest --symbol MES --timeframe 1H --csv data/csv/MES_1H.csv
uv run rainier backtest --symbol MES --timeframe 1H --csv data/csv/MES_1H.csv --trades
uv run rainier backtest --symbol MES --timeframe 1H --csv data/csv/MES_1H.csv --export trades.csv
uv run rainier backtest --symbol MES --timeframe 1H --csv data/csv/MES_1H.csv --export trades.parquet
uv run rainier backtest --symbol MES --timeframe 1H --csv data/csv/MES_1H.csv --sweep
uv run rainier backtest --symbol MES --timeframe 1H --csv data/csv/MES_1H.csv --slippage 0.001 --commission 5.0
uv run rainier chart --symbol MES --timeframe 1H --csv data/csv/MES_1H.csv
uv run rainier report --symbol MES --timeframe 1H --csv data/csv/MES_1H.csv

# QU100 backtesting
uv run rainier backtest-qu100                              # default: rank 1-50, 5d hold
uv run rainier backtest-qu100 --sweep                      # parameter sweep (rank x hold)
uv run rainier backtest-qu100 --variations                 # all signal tuning variants
uv run rainier backtest-qu100 --patterns                   # pattern-filtered (best 3 patterns, top 5)
uv run rainier backtest-qu100 --patterns --pattern-top-n 10  # pattern-filtered, top 10

# QuantUnicorn scraping
uv run rainier scrape qu --session morning --headed
uv run rainier scrape qu --session morning --cdp http://localhost:9222
uv run rainier scrape qu-detail --symbols TSLA,NVDA

# Scheduler + jobs
uv run rainier run                                    # start APScheduler (4x daily Mon-Fri)
uv run rainier run --once morning                     # single immediate scrape
uv run rainier jobs list                              # show cron.yaml jobs
uv run rainier jobs sync                              # sync to system crontab
uv run rainier jobs stop --name fetch-mes

# Recovery (after restart/power outage)
uv run rainier recover                                # check services + re-run missed jobs
uv run rainier recover --dry-run                      # show what would be done

# Database
uv run rainier db init                                # create tables + hypertables
```

## Module Contracts & Dependency Rules

All boundary contracts live in `core/protocols.py`. Modules depend on protocols, not on each other.

### Dependency Graph (arrows = "depends on")

```
                    ┌──────────┐
                    │  core/   │  types.py, protocols.py, config.py, database.py
                    └────┬─────┘
          ┌──────────┬───┴───┬──────────┬──────────┐
          │          │       │          │          │
     analysis/  signals/  backtest/  features/  trader/
     (pure fn)  (pure fn) (offline)  (ML prep)  (live exec)
```

**Hard rules:**
- `backtest/` imports ONLY from `core/` — never from `analysis/` or `signals/`
- `signals/` imports from `core/` and `analysis/` — never from `backtest/` or `trader/`
- `features/` imports ONLY from `core/` — never from `backtest/` directly
- `trader/` (future) imports ONLY from `core/` — receives signals via protocol
- `cli.py` is the **composition root** — the only place that wires modules together

### Protocol Contracts (in `core/protocols.py`)

| Protocol | Input | Output | Implemented by |
|---|---|---|---|
| `SignalEmitter` | `(DataFrame, symbol, timeframe)` | `list[Signal]` | `signals/emitter.py:PinBarSignalEmitter`, future ML emitter |
| `Analyzer` | `(DataFrame, symbol, timeframe)` | `AnalysisResult` | `analysis/analyzer.py:analyze()` |

### Data Contracts

| Type | Location | Used by | Purpose |
|---|---|---|---|
| `Signal` | `core/types.py` | signals → backtest, trader | Trade signal with entry/SL/TP/confidence |
| `AnalysisResult` | `core/types.py` | analysis → signals, features | Pivots, S/R levels, pin bars, bias |
| `TradeRecord` | `core/protocols.py` | backtest → features, export | Flat trade output (CSV/Parquet-friendly) |
| `BacktestMetrics` | `core/protocols.py` | backtest → report, CLI | Aggregate stats + trade log + equity curve |
| `BacktestConfig` | `core/config.py` | backtest engine | Slippage, commission, position limits, sweep ranges |

### Data Flow Patterns

```
BACKTEST (offline, no DB, no side effects):
  CSV/DataFrame → SignalEmitter.emit() → engine → BacktestMetrics → export/report

LIVE SIGNAL (online, with DB + notifications):
  scheduler → scraper → DB → analysis → signals → notifications/discord

LIVE TRADING (future, Phase 3):
  signal stream → TradeExecutor protocol → IB TWS → position management
```

### Adding a New Signal Strategy

1. Create a new class implementing the `SignalEmitter` protocol (just needs `emit()` method)
2. Wire it in `cli.py` or wherever the composition happens
3. The backtest engine, sweep runner, and export all work automatically — zero changes needed

## Key Conventions

- All shared types live in `src/rainier/core/types.py` — do not scatter dataclasses across modules
- All boundary protocols live in `src/rainier/core/protocols.py` — contracts between modules
- Scraper dataclasses (QU100Row, CapitalFlowRow) stay in `scrapers/qu/parsers.py` (domain-specific)
- Config via `config/settings.yaml` + `.env` (secrets), loaded by Pydantic in `core/config.py`
- Per-symbol overrides in `config/watchlists/default.yaml` (tick_size, point_value, min_touches)
- Database: singleton engine via `get_settings()` → `get_engine()` → `get_session()` context manager
- TimescaleDB hypertables for time-series tables (money_flow_snapshots, stock_capital_flow, etc.)
- Tests use synthetic fixtures from `tests/conftest.py`
- Python 3.12+, ruff for linting, line length 100

## Design Decisions (from eng review 2026-03-22)

- Pipeline-first: build full pipeline with book strategy scorer, then swap in ML models
- FeatureExtractor consumes AnalysisResult (not parallel extraction)
- ScoringStrategy protocol: book scorer and ML scorer interchangeable in generator
- Single XGBoost model with regime as feature (split to per-regime at 200+ samples/regime)
- Walk-forward cross-validation mandatory for all ML models (no random splits)
- NaN policy: fill with meaningful defaults + assert no NaN before model input
- LLM output: Pydantic validation + 3x retry
- Feature store: Parquet files for training/backtesting
- Scraper: Playwright with CDP mode for bot-detection bypass
- Raw JSONB: always store raw scraped data as safety net for re-parsing
