"""Market-breadth substrate for the S&P 500 breadth-webpage child series.

Two modules:

  * ``universe_loader`` — parses ``config/sp500_universe.yaml`` (the seed
    artifact owned by ``scripts/seed_sp500_universe.py``) into a flat
    ``list[(symbol, sector)]``. Wikipedia spelling is preserved (``BRK.B``
    stays with the dot); symbol dot→dash translation happens later, at
    yfinance fetch time inside ``ohlcv_backfill``.

  * ``ohlcv_backfill`` — yfinance chunked + retried OHLCV fetch into
    ``data/cache/sp500_universe.parquet``. Two modes: one-shot backfill
    over ``[since, today]`` and ``--incremental`` (last 5 calendar days,
    upsert). Atomic parquet write via ``.tmp`` + ``os.replace``.

Parent design:
    docs/DESIGN-market-breadth-webpage.md §3.2 (APPROVED 2026-05-25)
Task plan:
    docs/TASK-PLAN-sp500-ohlcv-backfill-47b8.md
"""

from rainier.market_breadth import ohlcv_backfill, universe_loader

__all__ = ["ohlcv_backfill", "universe_loader"]
