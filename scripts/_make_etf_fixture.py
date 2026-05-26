"""Generate deterministic fixture parquet for etf-dashboard-renderer tests.

3 sectors × 4 tickers × 35 days. Latest asof_date is 2026-05-25.

This is a one-shot generator. Run once; the .parquet ships in the repo
alongside the test that loads it. The script is kept so we can regen
the fixture if the schema grows. NOT run by pytest.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd


def build_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features_df, registry_df)."""
    # 3 sectors × 4 tickers each = 12 tickers.
    sectors = [
        (1, "energy", ["XLE", "XOP", "OIH", "XES"]),
        (2, "finance", ["KRE", "XLF", "KBE", "IAI"]),
        (3, "technology", ["XLK", "VGT", "SOXX", "SMH"]),
    ]
    end = date(2026, 5, 25)
    days = 35
    asof_dates = [end - timedelta(days=days - 1 - i) for i in range(days)]

    # Each ticker gets a stable per-ticker rank baseline that walks up/down
    # day-by-day to give the sparkline real variation. The LAST day's row is
    # what the table renders.
    # Layout:
    #  - One Top-15 (rank ≥ 85) per sector → 3 rows in the Top 15 tab.
    #  - One big mover per sector (|Δ1d| ≥ 10 OR |Δ5d| ≥ 15) → 3 rows in Movers.
    #  - At least one row with NaN sparkline history (truncated to <30d) for
    #    the missing-history test.
    rows = []
    for sec_id, sec_name, tickers in sectors:
        for tk_i, sym in enumerate(tickers):
            base_rank = [92, 70, 55, 30][tk_i]  # one Top-15 (≥85), one mid, etc.
            for day_i, asof in enumerate(asof_dates):
                # Walk rank toward base_rank with small modulation.
                rank = base_rank + ((day_i % 7) - 3)  # ±3 oscillation
                rank = max(0, min(99, rank))
                # Latest-day movers: tk_i==1 in each sector gets a +12 1d delta
                # so |Δ1d| ≥ 10. tk_i==3 gets a -16 5d delta so |Δ5d| ≥ 15.
                rank_delta_1d = 0
                rank_delta_5d = 0
                if day_i == days - 1:
                    if tk_i == 1:
                        rank_delta_1d = 12
                    if tk_i == 3:
                        rank_delta_5d = -16
                # Ticker XES (sector 1, tk_i=3) gets only 20 days of history
                # (forces the "<30 days" sparkline path).
                if sym == "XES" and day_i < days - 20:
                    continue
                rows.append(
                    {
                        "asof_date": asof,
                        "symbol": sym,
                        "sector_id": sec_id,
                        "rank": rank,
                        "rank_delta_1d": rank_delta_1d,
                        "rank_delta_5d": rank_delta_5d,
                        "r_5": (rank + 2) % 100,
                        "r_10": (rank + 5) % 100,
                        "r_20": (rank + 8) % 100,
                        "ret_ytd": 0.20 + (tk_i * 0.04) + (sec_id * 0.01),
                        "top15_streak": max(0, rank - 84),
                    }
                )
    features = pd.DataFrame(rows)
    # Match prod parquet dtypes loosely (ints, asof_date as object/string-day).
    features["asof_date"] = features["asof_date"].astype("string")
    features["rank"] = features["rank"].astype("int8")
    features["rank_delta_1d"] = features["rank_delta_1d"].astype("int8")
    features["rank_delta_5d"] = features["rank_delta_5d"].astype("int8")
    features["r_5"] = features["r_5"].astype("int8")
    features["r_10"] = features["r_10"].astype("int8")
    features["r_20"] = features["r_20"].astype("int8")
    features["sector_id"] = features["sector_id"].astype("int8")
    features["top15_streak"] = features["top15_streak"].astype("int16")
    features["ret_ytd"] = features["ret_ytd"].astype("float32")

    registry = pd.DataFrame(
        [
            {"sector_id": sec_id, "sector_name": sec_name, "first_seen": "2026-05-01"}
            for sec_id, sec_name, _ in sectors
        ]
    )
    registry["sector_id"] = registry["sector_id"].astype("int16")
    return features, registry


def main() -> None:
    features, registry = build_fixture()
    out_dir = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_dir / "etf_features_small.parquet", index=False)
    registry.to_parquet(out_dir / "etf_sector_registry_small.parquet", index=False)
    print(f"wrote {out_dir/'etf_features_small.parquet'} ({len(features)} rows)")
    print(f"wrote {out_dir/'etf_sector_registry_small.parquet'} ({len(registry)} rows)")
    # Quick sanity check.
    asof_max = features["asof_date"].max()
    last = features.loc[features["asof_date"] == asof_max]
    print(f"asof_max={asof_max} latest_rows={len(last)}")
    print("top-15 rows:")
    print(last.loc[last["rank"] >= 85, ["symbol", "sector_id", "rank"]].to_string())
    print("movers:")
    movers = last.loc[
        (last["rank_delta_1d"].abs() >= 10) | (last["rank_delta_5d"].abs() >= 15)
    ]
    print(movers[["symbol", "sector_id", "rank_delta_1d", "rank_delta_5d"]].to_string())


if __name__ == "__main__":
    main()
