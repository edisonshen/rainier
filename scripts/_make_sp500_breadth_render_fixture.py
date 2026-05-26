"""Generate the long-format breadth fixture parquet for render.py tests.

12 indicators × ~750 trading days ending 2026-05-25 (the canonical render
asof_date used across the dashboard test suite). The 750-day window is
deliberately ~3 years so the renderer's default 2y (~504 trading days)
window slicing has rows on either side of the cutoff — tests assert the
slice actually drops the older third.

Output schema (matches src/rainier/market_breadth/compute.py):

    asof_date (date32) | indicator (string) | value (float64)

Latest-day numerics are crafted so the header snapshot has stable display
values for `test_today_snapshot_values_match_input`:

    pct_above_ma_5            = 64.0
    pct_above_ma_10           = 60.0
    pct_above_ma_20           = 65.0
    pct_above_ma_50           = 70.0
    pct_above_ma_100          = 68.0
    pct_above_ma_200          = 62.0
    new_52w_high              = 24
    new_52w_low               = 7
    ad_diff                   = +217
    ad_cumulative             = +14893
    mcclellan_oscillator      = +18.4
    mcclellan_summation       = +1841.2

This is a one-shot generator. Re-run when the schema or load-bearing
header asserts change. NOT run by pytest.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd

# Load-bearing tail values — must match test_today_snapshot_values_match_input.
TAIL_VALUES: dict[str, float] = {
    "pct_above_ma_5": 64.0,
    "pct_above_ma_10": 60.0,
    "pct_above_ma_20": 65.0,
    "pct_above_ma_50": 70.0,
    "pct_above_ma_100": 68.0,
    "pct_above_ma_200": 62.0,
    "new_52w_high": 24.0,
    "new_52w_low": 7.0,
    "ad_diff": 217.0,
    "ad_cumulative": 14893.0,
    "mcclellan_oscillator": 18.4,
    "mcclellan_summation": 1841.2,
}


def _walk(name: str, days: int, tail_value: float) -> list[float]:
    """Deterministic per-indicator walk ending at ``tail_value`` on the last day.

    The walk is just a sine-wave around a baseline that linearly drifts toward
    ``tail_value``. Caller-visible: walks are deterministic, byte-stable, and
    distinct per indicator so the rendered SVG paths differ.
    """
    out: list[float] = []
    # Hash the name to a stable phase so different indicators don't trace the
    # same shape.
    phase = (sum(ord(c) for c in name) % 13) * 0.31
    for i in range(days):
        t = i / max(1, days - 1)
        baseline = (1.0 - t) * (tail_value * 0.6) + t * tail_value
        amplitude = max(1.0, abs(tail_value) * 0.05)
        out.append(baseline + amplitude * math.sin(phase + 0.07 * i))
    # Pin the last value exactly to the tail value so header snapshot is stable.
    out[-1] = tail_value
    return out


def build_fixture() -> pd.DataFrame:
    end = date(2026, 5, 25)
    # 750 trading days so render slicing has data on either side of the 504d
    # default window AND ad_cumulative grows to a sensible tail magnitude.
    all_days = pd.bdate_range(end=end, periods=750).date.tolist()

    rows: list[dict] = []
    for indicator, tail_value in TAIL_VALUES.items():
        walk = _walk(indicator, len(all_days), tail_value)
        for d, v in zip(all_days, walk):
            rows.append({"asof_date": d, "indicator": indicator, "value": v})
    df = pd.DataFrame(rows)
    df = df.sort_values(["asof_date", "indicator"]).reset_index(drop=True)
    return df


def main() -> None:
    df = build_fixture()
    out_dir = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "sp500_breadth_small.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows, {df['indicator'].nunique()} indicators)")

    # Sanity print: tail values per indicator.
    end_date = df["asof_date"].max()
    tail = df[df["asof_date"] == end_date].sort_values("indicator")
    print(f"tail asof_date={end_date}:")
    print(tail[["indicator", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()
