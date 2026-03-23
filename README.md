# rainier

Trading analysis platform — futures price action (小酱 pin bar methodology) + stock money flow (QuantUnicorn QU100).

## What it does

**Futures Price Action:**
1. **S/R Detection** — Horizontal support/resistance via pivot clustering + diagonal trendlines
2. **Pin Bar Scanner** — Identifies pin bars forming near S/R levels
3. **Signal Generation** — Entry, SL, TP, R:R ratio, and confidence score
4. **Multi-TF Analysis** — 1D + 4H + 1H + 5m confluence
5. **Backtesting** — Event-driven backtest engine
6. **Charts** — Interactive Plotly charts with tabbed TF switcher

**Stock Money Flow:**
7. **QuantUnicorn Scraper** — QU100 Top/Bottom 100 rankings (Playwright + CDP mode)
8. **Capital Flow Detail** — Per-ticker daily/weekly capital flow data
9. **Scheduled Scraping** — 4x daily (Mon-Fri) via APScheduler
10. **Notifications** — Apprise (email, Slack, Telegram, Discord, etc.)

## Quickstart

```bash
# requires python 3.12+, uv
uv sync

# fetch latest futures data from yfinance
uv run rainier fetch --symbol MES --plot

# multi-TF day trading analysis
uv run rainier daytrade --symbol MES --data-dir data/csv

# scrape QuantUnicorn QU100 rankings
uv run rainier scrape qu --session morning --headed

# start scraper scheduler (4x daily, Mon-Fri)
uv run rainier run

# initialize database
uv run rainier db init
```

## Scheduled Jobs

Jobs are defined in `config/cron.yaml` and synced to system crontab:

```bash
uv run rainier jobs list
uv run rainier jobs sync
uv run rainier jobs stop --name fetch-mes
```

## Project Structure

```
rainier/
├── config/
│   ├── settings.yaml              # All config (price action + scraping + LLM)
│   ├── cron.yaml                  # Scheduled jobs
│   ├── watchlists/default.yaml    # Futures instruments
│   └── prompts/                   # LLM prompt templates
├── src/rainier/
│   ├── core/                      # Types, config, database, ORM models (10 tables)
│   ├── data/                      # DataProvider protocol, CSV reader, yfinance fetcher
│   ├── analysis/                  # Pivots, S/R, pin bars, inside bars, bias
│   ├── features/                  # ML feature extraction
│   ├── signals/                   # Confidence scorer, signal generator
│   ├── backtest/                  # Event-driven backtest engine
│   ├── viz/                       # Plotly interactive charts
│   ├── scrapers/                  # QuantUnicorn web scraper (Playwright)
│   ├── notifications/             # Apprise multi-channel notifications
│   ├── alerts/                    # Discord webhooks
│   ├── reports/                   # Daily review + next-day outlook
│   ├── scheduler/                 # Crontab manager + APScheduler service
│   └── trader/                    # (Phase 3) IB TWS execution
├── docker-compose.yaml            # TimescaleDB
└── tests/                         # 142 tests
```

## Data Sources

| Source | Use | Status |
|--------|-----|--------|
| yfinance | Automated futures OHLCV (15-min delay) | Active |
| QuantUnicorn | Stock money flow rankings (QU100) | Active |
| Interactive Brokers TWS | Real-time futures + execution | Planned |

## Tech Stack

Python 3.12 · pandas/numpy · SQLAlchemy 2.0 + TimescaleDB · Playwright · plotly · APScheduler · Apprise · pydantic-settings · click · structlog · pytest

## License

Private — not open source.
