"""Generate the deterministic fixture parquet for market-breadth indicator tests.

10 symbols x 300 trading days ending 2024-06-14 (Friday). Latest 5d/10d/20d/
50d/100d/200d MAs are all defined at the tail. The price walks are crafted so:

  * On 2024-06-03 (Monday, day index 295) — EXACTLY 6 of 10 symbols have
    close > 5d SMA. This is the load-bearing assertion in
    `test_pct_above_ma_5_known_fixture`.
  * Symbol BULL hits a new 52w high on 2024-06-03 (and only on that date in
    the tail) — load-bearing for `test_new_52w_high_match`.
  * Symbol BEAR hits a new 52w low on 2024-06-03 (and only on that date) —
    load-bearing for `test_new_52w_low_match`.

The fixture mirrors `data/cache/sp500_universe.parquet` schema:
  symbol, date, open, high, low, close, volume, fetched_at, yfinance_version

This is a one-shot generator. Re-run only when the schema or the
load-bearing assertions in test_indicators.py change. NOT run by pytest.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd


def _trading_days(start: date, end: date) -> list[date]:
    """Return business days in [start, end]. Holidays not stripped — for
    fixture purposes weekday-only is enough; the holiday-handling test
    uses a separate in-memory frame."""
    return [d.date() for d in pd.bdate_range(start, end)]


def build_fixture() -> pd.DataFrame:
    end = date(2024, 6, 14)  # Friday
    # 300 trading days backwards ~ 14 months, plenty for 200d MA + 52w lookback.
    all_days = pd.bdate_range(end=end, periods=300).date.tolist()
    target = date(2024, 6, 3)  # 5th-from-last business day in the window
    target_idx = all_days.index(target)

    symbols = [
        "BULL",  # rising trend; new 52w high on target date (counted "above" → 1)
        "BEAR",  # falling trend; new 52w low on target date (counted "below")
        "UPA",   # synthetic "above 5d MA on target" (→ 2)
        "UPB",   # synthetic "above 5d MA on target" (→ 3)
        "UPC",   # synthetic "above 5d MA on target" (→ 4)
        "UPD",   # synthetic "above 5d MA on target" (→ 5)
        "UPE",   # synthetic "above 5d MA on target" (→ 6) — 6/10 target
        "DNA",   # synthetic "below 5d MA on target"
        "DNB",   # synthetic "below 5d MA on target"
        "FLAT",  # flat line → close == 5d MA → NOT above (strict >)
    ]

    rows: list[dict] = []
    fetched_at = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)

    for sym in symbols:
        # Base price + per-symbol deterministic walk.
        # The walk anchors a known relationship between close_T and SMA_5(T).
        base = 100.0
        prices: list[float] = []
        for i, d in enumerate(all_days):
            if sym == "BULL":
                # Long slow ramp + a small bump on target date that takes it
                # above its prior 251-day rolling max. Make sure the bump is
                # confined to target_idx so other dates don't ALSO mint a new high.
                p = base + 0.1 * i
                if i == target_idx:
                    p += 5.0  # ensures new 52w high
            elif sym == "BEAR":
                p = base + 200.0 - 0.1 * i  # falling
                if i == target_idx:
                    p -= 5.0  # ensures new 52w low
            elif sym in ("UPA", "UPB", "UPC", "UPD", "UPE"):
                # 5 "above 5d MA on target" symbols (+ BULL = 6 total).
                # Make recent 5 days = flat-ish then a bump on the target day.
                p = base + 0.05 * i
                if i >= target_idx - 4 and i < target_idx:
                    # The prior 4 days sit flat below the bump.
                    p = base + 0.05 * (target_idx - 5)
                if i == target_idx:
                    p = base + 0.05 * (target_idx - 5) + 3.0  # pop above 5d MA
                if i > target_idx:
                    p = base + 0.05 * i  # restore normal walk after target
            elif sym in ("DNA", "DNB"):
                # 2 "below 5d MA on target" symbols.
                p = base + 50.0 - 0.05 * i
                if i >= target_idx - 4 and i < target_idx:
                    p = base + 50.0 - 0.05 * (target_idx - 5)
                if i == target_idx:
                    p = base + 50.0 - 0.05 * (target_idx - 5) - 3.0  # drop below 5d MA
                if i > target_idx:
                    p = base + 50.0 - 0.05 * i
            elif sym == "FLAT":
                # FLAT sits at exactly its own 5d MA on target → NOT counted as above.
                p = base + 75.0
            else:
                raise AssertionError(sym)
            prices.append(p)

        for i, d in enumerate(all_days):
            close = prices[i]
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "open": close - 0.25,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1_000_000,
                    "fetched_at": fetched_at,
                    "yfinance_version": "fixture",
                }
            )

    df = pd.DataFrame(rows)
    df["volume"] = df["volume"].astype("int64")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def main() -> None:
    df = build_fixture()
    out_dir = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "sp500_ohlcv_small.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows, {df['symbol'].nunique()} symbols)")

    # Sanity print: row counts per symbol, last 6 closes.
    target = pd.Timestamp("2024-06-03").date()
    last = df[df["date"] == target].sort_values("symbol")
    print("rows on 2024-06-03:")
    print(last[["symbol", "close"]].to_string(index=False))

    # Compute pct_above_ma_5 on target — must equal 60.0.
    wide = df.pivot(index="date", columns="symbol", values="close")
    sma5 = wide.rolling(5).mean()
    above = (wide > sma5).loc[target]
    pct = above.sum() / above.count() * 100
    print(f"pct_above_ma_5 on 2024-06-03 = {pct:.2f}% "
          f"(above: {above[above].index.tolist()}, "
          f"not-above: {above[~above].index.tolist()})")


if __name__ == "__main__":
    main()
