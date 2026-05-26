"""Indicator orchestrator — load OHLCV parquet → compute 12 indicators → write long parquet.

Flow:

    +-----------------------+
    | data/cache/sp500_     |    long-format OHLCV
    |   universe.parquet    |    (symbol, date, open, high, low, close, volume, ...)
    +-----------+-----------+
                |
                v
    +-----------+-----------+
    | indicators.* (×12)    |    pct_above_ma_{5,10,20,50,100,200},
    |   pivot once,         |    new_52w_high/low, ad_diff, ad_cumulative,
    |   rolling math        |    mcclellan_oscillator, mcclellan_summation
    +-----------+-----------+
                |
                v
    +-----------+-----------+    long-format
    | data/cache/sp500_     |    (asof_date date32, indicator string,
    |   breadth_daily       |     value float64)
    |    .parquet           |    one row per (date, indicator)
    +-----------------------+

Idempotency: rows are sorted by (asof_date, indicator); pyarrow writes the
columns in the same order with the same compression every call. Same input
→ byte-identical output.

Atomic write: pyarrow writes to ``<out>.tmp`` then ``os.replace`` onto the
destination so a mid-write crash leaves the prior parquet (if any) intact.

Epoch semantics: ``ad_cumulative`` and ``mcclellan_summation`` start
accumulating on the first business day STRICTLY AFTER ``epoch``. Rows
≤ epoch carry NaN for those two indicators and are dropped from the
output frame.

Design refs:
    docs/DESIGN-market-breadth-webpage.md §4.2
    docs/TASK-PLAN-market-breadth-indicator-e7fb.md §Acceptance
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rainier.market_breadth import indicators as ind

# Output schema — pinned for byte-identical reproducibility.
_SCHEMA = pa.schema(
    [
        ("asof_date", pa.date32()),
        ("indicator", pa.string()),
        ("value", pa.float64()),
    ]
)


DEFAULT_EPOCH = date(2020, 1, 1)
DEFAULT_OUT = Path("data/cache/sp500_breadth_daily.parquet")


def compute_indicators(
    input_path: str | Path,
    output_path: str | Path = DEFAULT_OUT,
    *,
    epoch: date = DEFAULT_EPOCH,
) -> Path:
    """Compute all 12 breadth indicators over ``input_path`` and write a
    long-format parquet to ``output_path``.

    Parameters
    ----------
    input_path:
        Path to the OHLCV parquet (long format) produced by
        ``ohlcv_backfill.backfill``.
    output_path:
        Destination parquet. Atomic-written.
    epoch:
        Cumulative-indicator anchor. ``ad_cumulative`` and
        ``mcclellan_summation`` start at zero on the first business day
        strictly AFTER ``epoch``. Default: 2020-01-01.

    Returns
    -------
    Path
        The path actually written (equal to ``output_path``).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_parquet(input_path)
    # Normalize the date column: parquet round-trips as `date` objects
    # already, but tests pass DataFrames written from a pandas Timestamp
    # which round-trips as `Timestamp`. Coerce both to `date` so the
    # downstream index is uniform.
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Compute each indicator into a Series indexed by date.
    series_map: dict[str, pd.Series] = {}
    for w in ind.MA_WINDOWS:
        series_map[f"pct_above_ma_{w}"] = ind.pct_above_ma(df, window=w)
    series_map["new_52w_high"] = ind.new_52w_high(df)
    series_map["new_52w_low"] = ind.new_52w_low(df)
    series_map["ad_diff"] = ind.ad_diff(df)
    series_map["ad_cumulative"] = ind.ad_cumulative(df, epoch=epoch)
    series_map["mcclellan_oscillator"] = ind.mcclellan_oscillator(df)
    series_map["mcclellan_summation"] = ind.mcclellan_summation(df, epoch=epoch)

    # Stack into long format. One row per (date, indicator) regardless of
    # NaN warm-up state — the parquet schema stays uniform so downstream
    # renderers can `groupby('indicator')` without worrying about which
    # series start when. NaN values round-trip cleanly through parquet
    # float64.
    frames: list[pd.DataFrame] = []
    for name in ind.indicator_names():
        s = series_map[name]
        sub = pd.DataFrame(
            {
                "asof_date": s.index,
                "indicator": name,
                "value": s.values,
            }
        )
        frames.append(sub)

    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["asof_date", "indicator", "value"])

    # Sort by (asof_date, indicator) for deterministic row order.
    out = out.sort_values(["asof_date", "indicator"]).reset_index(drop=True)

    # Cast value column to float64 explicitly — pandas can promote to
    # object if any series in the concat is object-typed (e.g., int64
    # ad_diff alongside float64 pct_above_ma).
    out["value"] = out["value"].astype("float64")

    _write_parquet_atomic(out, output_path)
    return output_path


def _write_parquet_atomic(df: pd.DataFrame, out_path: Path) -> None:
    """Write ``df`` to ``<out_path>.tmp`` then ``os.replace`` onto destination.

    Mid-write crash leaves the prior parquet (if any) untouched.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    table = pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False)
    try:
        pq.write_table(table, tmp, compression="snappy")
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
