"""Generate `tests/fixtures/yfinance_aapl_2024.parquet`.

One-shot fixture maker. Run once; the parquet ships in the repo alongside
the tests. NOT executed by pytest. Re-run if the canonical schema grows.

Shape: 50 rows of synthetic OHLCV for symbol=AAPL across the first 50
business days of 2024. Mirrors the long-format schema from
`data/cache/thematic_universe.parquet`:

    symbol            string
    date              date32[day]
    open / high / low / close   double
    volume            int64
    fetched_at        timestamp[ns, tz=UTC]
    yfinance_version  string

Why ship a fixture file at all? Round-trip tests that read a parquet from
disk catch dtype regressions that pure in-memory construction misses —
particularly the date32 vs timestamp choice and tz-awareness of
`fetched_at`. ~6 KB on disk.

Usage:
    python scripts/_make_yfinance_aapl_fixture.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "yfinance_aapl_2024.parquet"


_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("date", pa.date32()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.int64()),
        ("fetched_at", pa.timestamp("ns", tz="UTC")),
        ("yfinance_version", pa.string()),
    ]
)


def main() -> Path:
    dates = pd.bdate_range("2024-01-02", periods=50).date.tolist()
    rows = []
    base = 185.0
    for i, d in enumerate(dates):
        # Monotonic-ish walk — deterministic, no random.
        px = base + i * 0.5
        rows.append(
            {
                "symbol": "AAPL",
                "date": d,
                "open": px,
                "high": px + 1.0,
                "low": px - 1.0,
                "close": px + 0.25,
                "volume": 50_000_000 + i * 100_000,
                "fetched_at": datetime(2024, 3, 15, 12, 0, 0, tzinfo=timezone.utc),
                "yfinance_version": "0.2.40",
            }
        )

    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, OUT, compression="snappy")
    print(f"wrote fixture {len(df)} rows -> {OUT}")
    return OUT


if __name__ == "__main__":
    main()
