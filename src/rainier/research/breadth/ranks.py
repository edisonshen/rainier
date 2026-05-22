"""Compute Layer A point-in-time features + Layer B forward-return labels.

Layer A: provably look-ahead-clean per-(asof_date, symbol) row of stationary
features. Layer B: per-(asof_date, symbol) row of T+N forward returns. Both
deterministic on byte-identical input.

Design ref:
    docs/DESIGN-thematic-ranks-dashboard.md §5.1 (Layer A schema),
                                            §5.2 (Layer B schema),
                                            §3   (REL/R/Rank arithmetic),
                                            [D-002] windows {5,10,20},
                                            [D-003] rank = round(mean(r_5,r_10,r_20)),
                                            [D-008] centile scaling 0-99,
                                            [D-013] vol-norm fixed N=20,
                                            [D-014] top15_streak threshold rank>=85.

ASCII flow (Layer A, single asof_date T):

    panel(symbol, date, close, ...)
            |
            v
    for each window N in {5,10,20}:
        REL_N(s, T) = close(s, T) / close(s, T - N_trading_days) - 1
            |
            v
        rank within universe (scipy rankdata method='average')
            |
            v
        R_N = round((rank - 1) / (n - 1) * 99)  in [0, 99]
            |
            v
    Rank = round(mean(R_5, R_10, R_20))
            |
            v
    rank_delta_1d, rank_delta_5d, top15_streak  (from prev_features)
            |
            v
    DataFrame[(symbol, asof_date), schema-stable columns]

ASCII flow (Layer B):

    panel(symbol, date, close, ...)
            |
            v
    for each asof_date T:
        fwd_Nd_ret(s, T) = close(s, T+N) / close(s, T) - 1
            |
            v
        fwd_5d_excess_ret(s, T) = fwd_5d_ret(s, T) - mean(fwd_5d_ret(*, T))
            |
            v
        rolling max(close[T:T+10]) -> runup
        rolling min(close[T:T+10]) -> drawdown
            |
            v
    label_complete_through = max asof_date with all fwd_30d defined
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats

# Windows per [D-002]
_REL_WINDOWS: tuple[int, ...] = (5, 10, 20)
_VOL_WINDOW: int = 20  # [D-013] fixed N=20 vol denominator
_PERSISTENCE_THRESHOLD: int = 85  # [D-014] top 15%
_RANK_DELTA_5D_LOOKBACK: int = 5

_FORWARD_RETURN_HORIZONS: tuple[int, ...] = (3, 5, 10, 20, 30)  # [D-012]
_FORWARD_EXCESS_HORIZONS: tuple[int, ...] = (5, 10, 20)
_DRAWDOWN_RUNUP_WINDOW: int = 10

# Sentinel for "no rank computable" (e.g. ticker with <20d history). Stored as
# int8 because the Rank composite column is int8; a NaN would force float64.
_RANK_SENTINEL_MISSING: int = -1

# ---------------------------------------------------------------------------
# Layer A — point-in-time features
# ---------------------------------------------------------------------------


def _centile_rank(values: pd.Series) -> pd.Series:
    """Map values to centiles 0..99 via scipy.stats.rankdata(method='average').

    NaNs propagate (return NaN). Per [D-008]:
        centile = round((rank - 1) / (n - 1) * 99)
    where n = number of NON-NaN values in the series.
    """
    out = pd.Series(index=values.index, dtype="float64")
    notna = values.notna()
    sub = values[notna]
    n = len(sub)
    if n == 0:
        out[:] = np.nan
        return out
    if n == 1:
        # Single observation gets the top centile by convention (mid-range
        # would also be defensible but produces a single-value series of
        # NaN under the (n-1) denominator — pick 99 to avoid div-by-zero).
        out.loc[notna] = 99.0
        out.loc[~notna] = np.nan
        return out
    ranks = stats.rankdata(sub.values, method="average")
    centiles = np.round((ranks - 1.0) / (n - 1.0) * 99.0)
    out.loc[notna] = centiles
    out.loc[~notna] = np.nan
    return out


def _trading_day_ordinal(d: date) -> int:
    """Epoch-anchored integer ordinal (counts every calendar day, not just
    trading days). We use calendar days to keep the function pure — no
    NYSE-calendar dep needed — and the transformer only requires monotonicity.
    """
    epoch = date(1970, 1, 1)
    return (d - epoch).days


def compute_thematic_features(
    panel: pd.DataFrame,
    asof: date,
    sector_map: Mapping[str, str],
    ticker_registry: Mapping[str, int],
    sector_registry: Mapping[str, int],
    universe_yaml_sha: str,
    prev_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute Layer A features for one ``asof_date``.

    Parameters
    ----------
    panel:
        Long-format OHLCV (columns ``symbol``, ``date``, ``open``, ``high``,
        ``low``, ``close``, ``volume``). At minimum must contain rows for
        ``asof`` and (asof - 20 trading days) per symbol.
    asof:
        The T date being computed.
    sector_map:
        ``symbol -> sector_name``. Used for ``rank_minus_sector_mean``.
    ticker_registry:
        ``symbol -> ticker_id``. Stable per [D-015].
    sector_registry:
        ``sector_name -> sector_id``. Stable per [D-015].
    universe_yaml_sha:
        SHA-1 of the YAML universe in effect at ``asof``. Propagated to the
        ``universe_yaml_sha`` column (audit + cache-key).
    prev_features:
        Optional output of a previous ``compute_thematic_features`` call (for
        the most recent prior trading day). Required only for ``rank_delta_*``
        and ``top15_streak`` — first-call use is permitted; deltas + streak
        default to 0 in that case.

    Returns
    -------
    pd.DataFrame with §5.1 schema, one row per symbol in the universe at
    ``asof``.
    """
    if panel.empty:
        return _empty_features_frame()

    # Slice to data <= asof (defensive — never look forward).
    cols_needed = {"symbol", "date", "close"}
    missing = cols_needed - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    panel = panel.loc[panel["date"] <= asof].copy()
    if panel.empty:
        return _empty_features_frame()

    # All trading days seen across the input (per-symbol may differ but the
    # universe trading calendar is the union).
    all_days = sorted(panel["date"].unique())
    if asof not in panel["date"].values:
        # asof is not a trading day per the panel; return empty.
        return _empty_features_frame()
    asof_idx = all_days.index(asof)

    # Pivot to wide close: index=date, columns=symbol.
    wide_close = (
        panel.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
        .sort_index()
    )

    # Symbols present in the asof row (drop symbols without a close on asof).
    asof_close = wide_close.loc[asof].dropna()
    symbols_asof = asof_close.index.tolist()
    # Restrict to the active YAML universe so a cached panel that retains
    # removed tickers does NOT pollute cross-sectional ranks or universe_size
    # (codex iter-7 [P2]). When sector_map is empty (defensive), keep all
    # symbols — the explicit YAML cohort is the source of truth.
    if sector_map:
        active = set(sector_map.keys())
        symbols_asof = [s for s in symbols_asof if s in active]
        asof_close = asof_close.loc[symbols_asof]

    # ---- Raw returns and REL_N ----
    rows: list[dict] = []
    rel_values: dict[int, pd.Series] = {}  # window -> series indexed by symbol
    for window in _REL_WINDOWS:
        prior_idx = asof_idx - window
        if prior_idx < 0:
            # Not enough history to compute this window — all-NaN series.
            rel_values[window] = pd.Series(np.nan, index=symbols_asof)
            continue
        prior_day = all_days[prior_idx]
        prior_row = wide_close.loc[prior_day]
        # Per-symbol REL_N: NaN if prior is NaN (insufficient history).
        rel = asof_close / prior_row - 1.0
        rel_values[window] = rel.reindex(symbols_asof)

    # ---- ret_1d ----
    if asof_idx >= 1:
        ret_1d_row = asof_close / wide_close.loc[all_days[asof_idx - 1]] - 1.0
    else:
        ret_1d_row = pd.Series(np.nan, index=symbols_asof)
    ret_1d_row = ret_1d_row.reindex(symbols_asof)

    # ---- Realized vol over fixed 20-day window (std of daily log returns) ----
    # Per [D-013] — fixed N=20 vol denominator for all relvol_N.
    if asof_idx >= _VOL_WINDOW:
        # Use simple returns (close/close.shift - 1) over the window.
        # fill_method=None: do NOT forward-fill internal NaNs before pct_change.
        # The default 'pad' behaviour silently turns gap days into 0% returns,
        # under-estimating vol_20 for sparse thematic ETFs (codex iter-6 [P2]).
        # NaN returns propagate to std() which is what we want.
        window_close = wide_close.iloc[asof_idx - _VOL_WINDOW : asof_idx + 1]
        rets = window_close.pct_change(fill_method=None).iloc[1:]
        vol_20 = rets.std(ddof=0)  # population std, deterministic
    else:
        vol_20 = pd.Series(np.nan, index=symbols_asof)
    vol_20 = vol_20.reindex(symbols_asof)

    # ---- YTD return ----
    asof_year = asof.year
    # Find last trading day of prior year present in the panel.
    prior_year_days = [d for d in all_days if d.year < asof_year]
    if prior_year_days:
        last_prior_year = prior_year_days[-1]
        ytd_base = wide_close.loc[last_prior_year]
    else:
        # If no prior-year data in panel, use the first available day for each
        # symbol (best-effort; dashboard context only).
        ytd_base = wide_close.iloc[0]
    ret_ytd_row = (asof_close / ytd_base - 1.0).reindex(symbols_asof)

    # ---- Centile ranks R_N within universe ----
    r_values: dict[int, pd.Series] = {}
    for window in _REL_WINDOWS:
        r_values[window] = _centile_rank(rel_values[window])

    # ---- Composite Rank = round(mean(R_5, R_10, R_20)) per [D-003] ----
    # NaN propagates: if any R_N is NaN, Rank is NaN.
    rank_df = pd.DataFrame(
        {f"r_{w}": r_values[w] for w in _REL_WINDOWS},
        index=symbols_asof,
    )
    rank_mean = rank_df.mean(axis=1, skipna=False)
    rank_composite = rank_mean.round()

    # ---- Sector mean and rank_minus_sector_mean ----
    sector_means = {}
    for sec_name in sorted({sector_map.get(s, "_unknown") for s in symbols_asof}):
        sec_syms = [s for s in symbols_asof if sector_map.get(s, "_unknown") == sec_name]
        sec_ranks = rank_composite.loc[sec_syms]
        # NaN-skip mean for the sector reference.
        sector_means[sec_name] = sec_ranks.mean(skipna=True)

    # ---- prev_features lookup for rank_delta_1d / top15_streak ----
    # rank_delta_1d uses previous asof_date's rank; if prev_features is the
    # output of the immediately-prior asof, we use that.
    prev_rank_by_sym: dict[str, float] = {}
    prev_streak_by_sym: dict[str, int] = {}
    if prev_features is not None and not prev_features.empty:
        for row in prev_features.itertuples(index=False):
            prev_rank_by_sym[row.symbol] = float(row.rank)
            prev_streak_by_sym[row.symbol] = int(row.top15_streak)

    # ---- rank_delta_5d: compute the composite rank 5 trading days ago and diff ----
    # This is a single-shot recomputation against the same panel (no prev_features
    # chain required). Cheap (one extra cross-sectional rankdata pass) and stays
    # deterministic on the same input.
    rank_5d_ago: pd.Series
    prior_5_idx = asof_idx - _RANK_DELTA_5D_LOOKBACK
    if prior_5_idx >= 0:
        prior_5_day = all_days[prior_5_idx]
        prior_5_close = wide_close.loc[prior_5_day]
        # Recompute REL_N at prior_5 using the same windows; this gives us the
        # historical rank at that day under the *current* universe membership
        # (correct for rank_delta_5d under the [D-015] stable-id discipline).
        prior_r: dict[int, pd.Series] = {}
        for window in _REL_WINDOWS:
            pp_idx = prior_5_idx - window
            if pp_idx < 0:
                prior_r[window] = pd.Series(np.nan, index=symbols_asof)
                continue
            pp_day = all_days[pp_idx]
            rel_pp = prior_5_close / wide_close.loc[pp_day] - 1.0
            prior_r[window] = _centile_rank(rel_pp.reindex(symbols_asof))
        prior_rank_df = pd.DataFrame(prior_r, index=symbols_asof)
        rank_5d_ago = prior_rank_df.mean(axis=1, skipna=False).round()
    else:
        rank_5d_ago = pd.Series(np.nan, index=symbols_asof)

    universe_size = len(symbols_asof)
    computed_at = pd.Timestamp(datetime.now(timezone.utc))

    for sym in symbols_asof:
        sec = sector_map.get(sym, "_unknown")
        ticker_id = ticker_registry.get(sym, 0)
        sector_id = sector_registry.get(sec, 0)
        r5 = r_values[5].get(sym, np.nan)
        r10 = r_values[10].get(sym, np.nan)
        r20 = r_values[20].get(sym, np.nan)
        rank_v = rank_composite.get(sym, np.nan)
        prev_rank = prev_rank_by_sym.get(sym)
        prev_streak = prev_streak_by_sym.get(sym, 0)

        # rank_delta_1d: treat the int8 -1 sentinel from prior_features as
        # "missing rank, neutral delta" — otherwise the first day a ticker
        # becomes rankable would record a spurious `rank_v + 1` jump (codex
        # iter-3 [P2]).
        if (
            pd.notna(rank_v)
            and prev_rank is not None
            and int(prev_rank) >= 0
        ):
            rd1 = int(rank_v) - int(prev_rank)
        else:
            rd1 = 0

        rank_5d = rank_5d_ago.get(sym, np.nan)
        if pd.notna(rank_v) and pd.notna(rank_5d):
            rd5 = int(rank_v) - int(rank_5d)
        else:
            rd5 = 0

        # Persistence streak: increments on rank>=85, resets to 0 otherwise.
        if pd.notna(rank_v) and rank_v >= _PERSISTENCE_THRESHOLD:
            streak = prev_streak + 1
        else:
            streak = 0

        sec_mean = sector_means.get(sec, np.nan)
        if pd.notna(rank_v) and pd.notna(sec_mean):
            rms = float(rank_v) - float(sec_mean)
        else:
            rms = 0.0

        # Relvol N: rel_N / vol_20 (vol_20 in raw returns space). Defensive
        # zero-vol guard: if vol_20 == 0 we get inf; clamp to NaN.
        def _relvol(rel: float, v: float) -> float:
            if pd.isna(rel) or pd.isna(v) or v == 0:
                return float("nan")
            return rel / v

        rel5 = rel_values[5].get(sym, np.nan)
        rel10 = rel_values[10].get(sym, np.nan)
        rel20 = rel_values[20].get(sym, np.nan)
        v20 = vol_20.get(sym, np.nan)

        # Convert NaN ranks to sentinel for int8 storage.
        def _r_int8(x: float) -> int:
            if pd.isna(x):
                return _RANK_SENTINEL_MISSING
            return int(x)

        rows.append(
            {
                "asof_date": asof,
                "trading_day_ordinal": np.int32(_trading_day_ordinal(asof)),
                "symbol": sym,
                "ticker_id": np.int16(ticker_id),
                "sector_id": np.int8(sector_id),
                "close": float(asof_close[sym]),
                "ret_1d": np.float32(ret_1d_row.get(sym, np.nan)),
                "rel_5": np.float32(rel5 if pd.notna(rel5) else np.nan),
                "rel_10": np.float32(rel10 if pd.notna(rel10) else np.nan),
                "rel_20": np.float32(rel20 if pd.notna(rel20) else np.nan),
                "ret_ytd": np.float32(ret_ytd_row.get(sym, np.nan)),
                "relvol_5": np.float32(_relvol(rel5, v20)),
                "relvol_10": np.float32(_relvol(rel10, v20)),
                "relvol_20": np.float32(_relvol(rel20, v20)),
                "r_5": np.int8(_r_int8(r5)),
                "r_10": np.int8(_r_int8(r10)),
                "r_20": np.int8(_r_int8(r20)),
                "rank": np.int8(_r_int8(rank_v)),
                "rank_delta_1d": np.int8(rd1),
                "rank_delta_5d": np.int8(rd5),
                "top15_streak": np.int16(streak),
                "rank_minus_sector_mean": np.float32(rms),
                "universe_size": np.int16(universe_size),
                "universe_yaml_sha": universe_yaml_sha,
                "computed_at": computed_at,
            }
        )

    df = pd.DataFrame(rows)
    # Stable sort key: (symbol, asof_date)
    df = df.sort_values(["symbol", "asof_date"]).reset_index(drop=True)
    # Enforce dtypes (DataFrame ctor may upcast on mixed-NaN handling).
    df = _enforce_features_dtypes(df)
    return df


def _empty_features_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the Layer A schema."""
    cols = {
        "asof_date": pd.Series(dtype="object"),
        "trading_day_ordinal": pd.Series(dtype="int32"),
        "symbol": pd.Series(dtype="object"),
        "ticker_id": pd.Series(dtype="int16"),
        "sector_id": pd.Series(dtype="int8"),
        "close": pd.Series(dtype="float64"),
        "ret_1d": pd.Series(dtype="float32"),
        "rel_5": pd.Series(dtype="float32"),
        "rel_10": pd.Series(dtype="float32"),
        "rel_20": pd.Series(dtype="float32"),
        "ret_ytd": pd.Series(dtype="float32"),
        "relvol_5": pd.Series(dtype="float32"),
        "relvol_10": pd.Series(dtype="float32"),
        "relvol_20": pd.Series(dtype="float32"),
        "r_5": pd.Series(dtype="int8"),
        "r_10": pd.Series(dtype="int8"),
        "r_20": pd.Series(dtype="int8"),
        "rank": pd.Series(dtype="int8"),
        "rank_delta_1d": pd.Series(dtype="int8"),
        "rank_delta_5d": pd.Series(dtype="int8"),
        "top15_streak": pd.Series(dtype="int16"),
        "rank_minus_sector_mean": pd.Series(dtype="float32"),
        "universe_size": pd.Series(dtype="int16"),
        "universe_yaml_sha": pd.Series(dtype="object"),
        "computed_at": pd.Series(dtype="datetime64[ns, UTC]"),
    }
    return pd.DataFrame(cols)


def _enforce_features_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    casts = {
        "trading_day_ordinal": "int32",
        "ticker_id": "int16",
        "sector_id": "int8",
        "close": "float64",
        "ret_1d": "float32",
        "rel_5": "float32",
        "rel_10": "float32",
        "rel_20": "float32",
        "ret_ytd": "float32",
        "relvol_5": "float32",
        "relvol_10": "float32",
        "relvol_20": "float32",
        "r_5": "int8",
        "r_10": "int8",
        "r_20": "int8",
        "rank": "int8",
        "rank_delta_1d": "int8",
        "rank_delta_5d": "int8",
        "top15_streak": "int16",
        "rank_minus_sector_mean": "float32",
        "universe_size": "int16",
    }
    for col, dt in casts.items():
        if col in df.columns:
            df[col] = df[col].astype(dt)
    return df


# ---------------------------------------------------------------------------
# Layer B — forward-return labels
# ---------------------------------------------------------------------------


def compute_forward_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute Layer B forward-return labels for every (asof_date, symbol)
    pair in ``panel``.

    The freshest computable row is `panel_end - max_horizon` (where
    max_horizon = 30 trading days). Rows beyond that have NaN forward returns.

    Returns one DataFrame with the §5.2 schema.
    """
    if panel.empty:
        return _empty_labels_frame()

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.date

    # Wide close pivot
    wide = (
        panel.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
        .sort_index()
    )
    all_days = sorted(wide.index.tolist())
    n_days = len(all_days)

    # Determine label_complete_through: last asof_idx such that
    # asof_idx + max_horizon < n_days. When the panel is shorter than the
    # longest forward horizon, NO row has a complete fwd_30d_ret — represent
    # that as None rather than `all_days[0]`, otherwise downstream consumers
    # that filter on this metadata would treat incomplete rows as trainable
    # (codex iter-6 [P2]).
    max_h = max(_FORWARD_RETURN_HORIZONS)
    if n_days <= max_h:
        label_complete_through = None
    else:
        label_complete_through = all_days[n_days - max_h - 1]

    rows: list[dict] = []
    symbols = list(wide.columns)

    # Pre-compute forward returns matrices
    # fwd_ret[N][asof_idx, sym] = close(T+N) / close(T) - 1, NaN if out-of-range
    fwd_ret: dict[int, pd.DataFrame] = {}
    for N in _FORWARD_RETURN_HORIZONS:
        shifted = wide.shift(-N)
        fwd_ret[N] = shifted / wide - 1.0

    # Excess returns vs universe mean per horizon
    fwd_excess: dict[int, pd.DataFrame] = {}
    for N in _FORWARD_EXCESS_HORIZONS:
        # axis=1 mean across symbols at each date.
        universe_mean = fwd_ret[N].mean(axis=1, skipna=True)
        fwd_excess[N] = fwd_ret[N].sub(universe_mean, axis=0)

    # Drawdown + runup within the 10-day window:
    #   fwd_10d_max_drawdown(T) = min over k in [1..10] of close(T+k)/close(T) - 1
    #                              but only if negative — represented as positive
    #                              magnitude (0 if no drawdown).
    #   fwd_10d_max_runup(T)    = max over k in [1..10] of close(T+k)/close(T) - 1
    #                              (0 if no runup).
    drawdown = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns)
    runup = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns)
    W = _DRAWDOWN_RUNUP_WINDOW
    for i in range(n_days - W):
        slice_ = wide.iloc[i : i + W + 1]
        base = slice_.iloc[0]
        # Pct vs base for each day in [1..W]
        rel = slice_.iloc[1:].div(base, axis=1) - 1.0
        # max drawdown = absolute value of most-negative; clipped to >=0
        dd = (-rel.min(axis=0)).clip(lower=0)
        ru = rel.max(axis=0).clip(lower=0)
        drawdown.iloc[i] = dd
        runup.iloc[i] = ru

    for asof_dt in all_days:
        for sym in symbols:
            close_t = wide.at[asof_dt, sym]
            if pd.isna(close_t):
                # Symbol not present on this asof — skip.
                continue
            row = {
                "asof_date": asof_dt,
                "symbol": sym,
                "label_complete_through": label_complete_through,
            }
            for N in _FORWARD_RETURN_HORIZONS:
                row[f"fwd_{N}d_ret"] = np.float32(fwd_ret[N].at[asof_dt, sym])
            for N in _FORWARD_EXCESS_HORIZONS:
                row[f"fwd_{N}d_excess_ret"] = np.float32(fwd_excess[N].at[asof_dt, sym])
            row["fwd_10d_max_drawdown"] = np.float32(drawdown.at[asof_dt, sym])
            row["fwd_10d_max_runup"] = np.float32(runup.at[asof_dt, sym])
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["symbol", "asof_date"]).reset_index(drop=True)
    return _enforce_labels_dtypes(df)


def _empty_labels_frame() -> pd.DataFrame:
    cols = {
        "asof_date": pd.Series(dtype="object"),
        "symbol": pd.Series(dtype="object"),
        "fwd_3d_ret": pd.Series(dtype="float32"),
        "fwd_5d_ret": pd.Series(dtype="float32"),
        "fwd_10d_ret": pd.Series(dtype="float32"),
        "fwd_20d_ret": pd.Series(dtype="float32"),
        "fwd_30d_ret": pd.Series(dtype="float32"),
        "fwd_5d_excess_ret": pd.Series(dtype="float32"),
        "fwd_10d_excess_ret": pd.Series(dtype="float32"),
        "fwd_20d_excess_ret": pd.Series(dtype="float32"),
        "fwd_10d_max_drawdown": pd.Series(dtype="float32"),
        "fwd_10d_max_runup": pd.Series(dtype="float32"),
        "label_complete_through": pd.Series(dtype="object"),
    }
    return pd.DataFrame(cols)


def _enforce_labels_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    float32_cols = (
        "fwd_3d_ret",
        "fwd_5d_ret",
        "fwd_10d_ret",
        "fwd_20d_ret",
        "fwd_30d_ret",
        "fwd_5d_excess_ret",
        "fwd_10d_excess_ret",
        "fwd_20d_excess_ret",
        "fwd_10d_max_drawdown",
        "fwd_10d_max_runup",
    )
    for c in float32_cols:
        if c in df.columns:
            df[c] = df[c].astype("float32")
    return df
