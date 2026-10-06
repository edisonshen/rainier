"""
Leveraged-ETF appearance signal study.

Question: when SQQQ / TQQQ (and the SPY-family leveraged ETFs) show up in the
QU100 top100 / bottom100 money-flow lists, does that mark a top or bottom in
the underlying index (QQQ / SPY)?

Reads `money_flow_snapshots` from the local rainier Postgres (LEGACY_DATABASE_URL,
same DSN the scraper writes to), pulls QQQ/SPY closes from yfinance, and reports:

  1. Every appearance (date, list, rank, long_short) as CSV.
  2. Per (symbol, list): mean/median forward return of the benchmark at
     5/10/20 trading days vs. the unconditional baseline, hit-rate (% positive),
     and how often the signal day lands within +/-3 days of a 20-day swing
     low / high of the benchmark.
  3. "Burst" signals: >= N appearances of a symbol in a trailing 5-session
     window (default N=3), same forward-return stats.
  4. Where the benchmark sits on signal days (percentile of close within the
     trailing 60-day range) -- low percentile = near a bottom.

Usage:
  uv run python scripts/analyze_leveraged_etf_signal.py
  uv run python scripts/analyze_leveraged_etf_signal.py --out reports/etf_signal
  uv run python scripts/analyze_leveraged_etf_signal.py --csv etf_appearances.csv   # offline
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text

DEFAULT_DSN = "postgresql://rainier:rainier_dev@localhost:5432/rainier"

# ETF -> (benchmark, direction). direction=+1 bull, -1 bear.
ETF_MAP: dict[str, tuple[str, int]] = {
    "TQQQ": ("QQQ", 1),
    "QLD": ("QQQ", 1),
    "SQQQ": ("QQQ", -1),
    "QID": ("QQQ", -1),
    "PSQ": ("QQQ", -1),
    "SPXL": ("SPY", 1),
    "UPRO": ("SPY", 1),
    "SSO": ("SPY", 1),
    "SPXS": ("SPY", -1),
    "SPXU": ("SPY", -1),
    "SDS": ("SPY", -1),
    "SH": ("SPY", -1),
    "SOXL": ("SOXX", 1),
    "SOXS": ("SOXX", -1),
}
HORIZONS = (5, 10, 20)
SWING_WINDOW = 20
SWING_TOL = 3


def load_appearances_db(dsn: str) -> pd.DataFrame:
    sql = text(
        """
        select distinct on (data_date, ranking_type, symbol)
               data_date, ranking_type, symbol, rank, long_short, capture_session, captured_at
        from money_flow_snapshots
        where symbol = any(:symbols)
        order by data_date, ranking_type, symbol, captured_at desc
        """
    )
    engine = create_engine(dsn)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"symbols": list(ETF_MAP)})
    return df


def load_appearances_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["symbol"].isin(ETF_MAP)]
    if "captured_at" in df.columns:
        df = df.sort_values("captured_at")
    return df.drop_duplicates(["data_date", "ranking_type", "symbol"], keep="last")


def load_prices(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    raw = yf.download(
        symbols,
        start=start - pd.Timedelta(days=120),
        end=end + pd.Timedelta(days=45),
        auto_adjust=True,
        progress=False,
    )
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(raw.columns, pd.MultiIndex):
        close.columns = symbols
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close.dropna(how="all")


def benchmark_features(close: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame({"close": close})
    for h in HORIZONS:
        f[f"fwd_{h}d"] = close.shift(-h) / close - 1
    lo = close.rolling(SWING_WINDOW, center=True).min()
    hi = close.rolling(SWING_WINDOW, center=True).max()
    f["is_swing_low"] = close == lo
    f["is_swing_high"] = close == hi
    f["near_swing_low"] = f["is_swing_low"].rolling(2 * SWING_TOL + 1, center=True).max().astype(bool)
    f["near_swing_high"] = (
        f["is_swing_high"].rolling(2 * SWING_TOL + 1, center=True).max().astype(bool)
    )
    rng_lo = close.rolling(60).min()
    rng_hi = close.rolling(60).max()
    f["pct_of_60d_range"] = (close - rng_lo) / (rng_hi - rng_lo)
    f["ret_20d_back"] = close / close.shift(20) - 1
    f["dd_from_60d_high"] = close / rng_hi - 1
    f["min_fwd_10d"] = close.rolling(10).min().shift(-10) / close - 1
    return f


def episodes(mask: pd.Series, cooldown: int) -> pd.DatetimeIndex:
    """First day of each signal cluster; a new episode needs `cooldown` sessions of gap."""
    out: list[int] = []
    last = -10**9
    for i, v in enumerate(mask.to_numpy()):
        if v and i - last > cooldown:
            out.append(i)
            last = i
    return mask.index[out]


def bootstrap_p(feat: pd.DataFrame, ev: pd.DatetimeIndex, col: str, n_boot: int = 5000) -> float:
    """P(random same-size day sample has mean >= observed)."""
    if len(ev) == 0:
        return np.nan
    rng = np.random.default_rng(0)
    pool = feat[col].dropna().to_numpy()
    obs = feat.loc[ev, col].mean()
    sims = rng.choice(pool, size=(n_boot, len(ev))).mean(axis=1)
    return float((sims >= obs).mean())


def episode_block(feat: pd.DataFrame, mask: pd.Series, label: str, cooldown: int) -> dict:
    ev = episodes(mask, cooldown)
    sub = feat.loc[ev]
    row = {"signal": label, "signal_days": int(mask.sum()), "episodes": len(ev)}
    for h in HORIZONS:
        row[f"mean_{h}d"] = sub[f"fwd_{h}d"].mean()
        row[f"win_{h}d"] = (sub[f"fwd_{h}d"] > 0).mean() if len(sub) else np.nan
    row["p_20d"] = bootstrap_p(feat, ev, "fwd_20d")
    row["dd_from_60d_high"] = sub["dd_from_60d_high"].mean() if len(sub) else np.nan
    row["min_fwd_10d"] = sub["min_fwd_10d"].mean() if len(sub) else np.nan
    row["near_swing_low"] = sub["near_swing_low"].mean() if len(sub) else np.nan
    row["near_swing_high"] = sub["near_swing_high"].mean() if len(sub) else np.nan
    return row


def stats_block(feat: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    sub = feat[mask]
    row = {"signal": label, "n": int(len(sub))}
    for h in HORIZONS:
        col = sub[f"fwd_{h}d"].dropna()
        row[f"mean_{h}d"] = col.mean()
        row[f"median_{h}d"] = col.median()
        row[f"win_{h}d"] = (col > 0).mean() if len(col) else np.nan
    row["near_swing_low"] = sub["near_swing_low"].mean() if len(sub) else np.nan
    row["near_swing_high"] = sub["near_swing_high"].mean() if len(sub) else np.nan
    row["pct_of_60d_range"] = sub["pct_of_60d_range"].mean() if len(sub) else np.nan
    row["ret_20d_back"] = sub["ret_20d_back"].mean() if len(sub) else np.nan
    return row


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols), *body])


def fmt(df: pd.DataFrame) -> str:
    out = df.copy()
    for c in out.columns:
        if c.startswith(("mean_", "median_", "win_", "near_", "pct_", "ret_", "dd_", "min_")):
            out[c] = (out[c] * 100).round(1).astype(str) + "%"
        elif c.startswith("p_"):
            out[c] = out[c].round(3)
    return md_table(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("LEGACY_DATABASE_URL", DEFAULT_DSN))
    ap.add_argument("--csv", type=Path, help="offline: appearances CSV instead of DB")
    ap.add_argument("--out", type=Path, default=Path("reports/etf_signal"))
    ap.add_argument("--burst-n", type=int, default=3)
    ap.add_argument("--burst-window", type=int, default=5)
    ap.add_argument("--cooldown", type=int, default=20,
                    help="sessions of gap before a signal day starts a new episode")
    args = ap.parse_args()

    app = load_appearances_csv(args.csv) if args.csv else load_appearances_db(args.dsn)
    if app.empty:
        raise SystemExit("no leveraged-ETF rows found in money_flow_snapshots")
    app["data_date"] = pd.to_datetime(app["data_date"]).dt.normalize()
    app["benchmark"] = app["symbol"].map(lambda s: ETF_MAP[s][0])
    app["direction"] = app["symbol"].map(lambda s: ETF_MAP[s][1])
    args.out.mkdir(parents=True, exist_ok=True)
    app.sort_values(["data_date", "symbol"]).to_csv(args.out / "appearances.csv", index=False)

    benchmarks = sorted(app["benchmark"].unique())
    close = load_prices(benchmarks, app["data_date"].min(), app["data_date"].max())
    sessions = close.index[(close.index >= app["data_date"].min())
                           & (close.index <= app["data_date"].max())]

    lines = [
        "# Leveraged-ETF appearance signal study",
        "",
        f"Data window: {app['data_date'].min().date()} -> {app['data_date'].max().date()} "
        f"({len(sessions)} sessions, {len(app)} appearance rows)",
        "",
        "## Appearance counts",
        "",
        md_table(app.groupby(["symbol", "ranking_type"]).size().rename("days").reset_index()),
        "",
    ]

    for bm in benchmarks:
        feat = benchmark_features(close[bm]).loc[sessions]
        rows = [stats_block(feat, pd.Series(True, index=feat.index), f"{bm} baseline (all days)")]
        sub_app = app[app["benchmark"] == bm]

        for (sym, rt), g in sub_app.groupby(["symbol", "ranking_type"]):
            days = feat.index.isin(g["data_date"])
            rows.append(stats_block(feat, pd.Series(days, index=feat.index), f"{sym} in {rt}"))

        # appearance count in trailing window, bucketed (0 .. window)
        ep_rows = []
        for (sym, rt), g in sub_app.groupby(["symbol", "ranking_type"]):
            present = pd.Series(feat.index.isin(g["data_date"]), index=feat.index).astype(int)
            cnt = present.rolling(args.burst_window).sum()
            for k in range(args.burst_window + 1):
                rows.append(stats_block(feat, cnt == k,
                                        f"{sym} {rt}: {k} of last {args.burst_window} sessions"))
            burst = cnt >= args.burst_n
            ep_rows.append(episode_block(
                feat, burst,
                f"{sym} {rt} >= {args.burst_n} of {args.burst_window}", args.cooldown))
            ep_rows.append(episode_block(
                feat, cnt == args.burst_window,
                f"{sym} {rt} all {args.burst_window}", args.cooldown))
        # bear-ETF inflow while bull ETF has dropped out of top100
        bear_top = sub_app[(sub_app["direction"] == -1) & (sub_app["ranking_type"] == "top100")]
        bull_top = sub_app[(sub_app["direction"] == 1) & (sub_app["ranking_type"] == "top100")]
        bt = pd.Series(feat.index.isin(bear_top["data_date"]), index=feat.index)
        lt = pd.Series(feat.index.isin(bull_top["data_date"]), index=feat.index)
        bt_cnt = bt.astype(int).rolling(args.burst_window).sum()
        ep_rows.append(episode_block(
            feat, (bt_cnt >= args.burst_n) & ~lt,
            f"bear ETF top100 >= {args.burst_n} of {args.burst_window} & bull ETF not in top100",
            args.cooldown))
        ep_rows.append(episode_block(
            feat, (bt_cnt == 0) & lt,
            f"no bear ETF in top100 for {args.burst_window} sessions & bull ETF in top100",
            args.cooldown))
        # combined bull-vs-bear presence on same day
        for direction, lab in ((1, "bull ETF"), (-1, "bear ETF")):
            d = sub_app[sub_app["direction"] == direction]["data_date"]
            rows.append(stats_block(feat, pd.Series(feat.index.isin(d), index=feat.index),
                                    f"any {lab} on {bm} (any list)"))

        table = pd.DataFrame(rows)
        table.to_csv(args.out / f"stats_{bm}.csv", index=False)
        ep_table = pd.DataFrame(ep_rows)
        ep_table.to_csv(args.out / f"episodes_{bm}.csv", index=False)
        lines += [f"## {bm}", "", "### Daily conditional stats", "", fmt(table), "",
                  f"### Episode-level (first day of each cluster, {args.cooldown}-session cooldown)",
                  "", fmt(ep_table), ""]

        # event timeline
        feat_rows = feat.reset_index()
        feat_rows = feat_rows.rename(columns={feat_rows.columns[0]: "data_date"})
        ev = sub_app.merge(feat_rows, on="data_date", how="left")
        ev_cols = ["data_date", "symbol", "ranking_type", "rank", "long_short", "close",
                   "pct_of_60d_range", "ret_20d_back", "fwd_5d", "fwd_10d", "fwd_20d",
                   "near_swing_low", "near_swing_high"]
        ev = ev[[c for c in ev_cols if c in ev.columns]].sort_values("data_date")
        ev.to_csv(args.out / f"events_{bm}.csv", index=False)
        ev["data_date"] = ev["data_date"].dt.date
        lines += [f"### {bm} event timeline", "", md_table(ev.round(4)), ""]

    lines += [
        "## How to read",
        "",
        "- `win_Nd` above baseline for a *bear* ETF appearance (SQQQ/SPXS...) => capitulation / "
        "bottom signal. Below baseline for a *bull* ETF => euphoria / top signal.",
        "- `near_swing_low` = share of signal days within +/-3 sessions of a 20-day swing low; "
        "compare against the baseline row, not against 0.",
        "- `pct_of_60d_range` near 0% = index at bottom of its 60-day range on signal day.",
        "- Daily rows over-count because signal days cluster; the episode table de-clusters and "
        "`p_20d` is a bootstrap P(random sample of the same size has mean fwd_20d >= observed).",
        "- `min_fwd_10d` = worst close over the next 10 sessions (how much further it falls).",
        "- With episodes < ~15 treat any edge as anecdotal.",
    ]
    (args.out / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {args.out}/REPORT.md, appearances.csv, stats_*.csv, events_*.csv")


if __name__ == "__main__":
    main()
