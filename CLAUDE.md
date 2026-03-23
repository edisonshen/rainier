# CLAUDE.md

## Project Overview

Rainier — trading analysis platform combining futures price action (小酱 pin bar methodology) with stock money flow data (QuantUnicorn QU100). Evolving from rule-based to AI-adaptive with a three-layer hybrid architecture.

## Module Map

```
src/rainier/
├── core/           types.py (shared dataclasses), config.py (unified Pydantic), models.py (10 ORM tables), database.py (singleton + TimescaleDB)
├── data/           provider.py (protocol), csv_provider.py, yfinance_provider.py
├── analysis/       analyzer.py (orchestrator), pivots.py, pinbar.py, sr_horizontal.py, sr_diagonal.py, bias.py, inside_bar.py
├── features/       extractor.py (AnalysisResult→ML features), labels.py (backtest→training labels)
├── signals/        generator.py (entry/SL/TP), scorer.py (weighted confidence + multi-TF confluence)
├── backtest/       engine.py (event-driven, S/R recompute every 50 bars)
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
uv run rainier chart --symbol MES --timeframe 1H --csv data/csv/MES_1H.csv
uv run rainier report --symbol MES --timeframe 1H --csv data/csv/MES_1H.csv

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

# Database
uv run rainier db init                                # create tables + hypertables
```

## Key Conventions

- All shared types live in `src/rainier/core/types.py` — do not scatter dataclasses across modules
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
