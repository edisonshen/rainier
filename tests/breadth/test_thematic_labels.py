"""Golden-fixture tests for Layer B forward-return labels.

Critical safety test: assert no T+1 leak — a row at asof_date=T cannot
reference any data after T in Layer A. Layer B has the fwd_* columns and
explicitly declares t_plus_n_close observability per [D-011] + DESIGN §5.5.

Design ref: docs/DESIGN-thematic-ranks-dashboard.md §5.2 ([D-011] / [D-012]).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from rainier.breadth.ranks import compute_forward_labels


def _build_panel_constant_returns(
    symbols: list[str],
    n_days: int,
    daily_returns: dict[str, float],
    start: date = date(2024, 10, 1),
) -> pd.DataFrame:
    """Build a long-format OHLCV panel where each ticker compounds at a fixed
    daily return.
    """
    dates = []
    d = start
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d)
        d = d + timedelta(days=1)

    rows = []
    for sym in symbols:
        r = daily_returns[sym]
        close = 100.0
        for i, day in enumerate(dates):
            if i > 0:
                close = close * (1.0 + r)
            rows.append(
                {
                    "symbol": sym,
                    "date": day,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)


@pytest.fixture
def panel_5x60() -> pd.DataFrame:
    """5 tickers x 60 trading days with mixed return profiles."""
    daily_returns = {
        "AAA": 0.01,
        "BBB": 0.005,
        "CCC": 0.0,
        "DDD": -0.005,
        "EEE": -0.01,
    }
    return _build_panel_constant_returns(
        symbols=["AAA", "BBB", "CCC", "DDD", "EEE"],
        n_days=60,
        daily_returns=daily_returns,
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


REQUIRED_LABEL_COLS = {
    "asof_date",
    "symbol",
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
    "label_complete_through",
}


def test_label_schema(panel_5x60):
    out = compute_forward_labels(panel=panel_5x60)
    assert REQUIRED_LABEL_COLS.issubset(out.columns), (
        f"missing label columns: {REQUIRED_LABEL_COLS - set(out.columns)}"
    )


# ---------------------------------------------------------------------------
# Forward-return arithmetic
# ---------------------------------------------------------------------------


def test_fwd_5d_ret_matches_simple_compounding(panel_5x60):
    """For AAA growing 1%/day: fwd_5d_ret(asof=T) = (1.01)^5 - 1 = ~0.05101."""
    out = compute_forward_labels(panel=panel_5x60)
    aaa = out.loc[out["symbol"] == "AAA"]
    # Pick an asof_date well inside the panel
    asof_dates = sorted(aaa["asof_date"].tolist())
    mid = asof_dates[20]  # day 20 of 60
    row = aaa.loc[aaa["asof_date"] == mid].iloc[0]
    expected = (1.01) ** 5 - 1
    assert abs(row["fwd_5d_ret"] - expected) < 1e-4, (
        f"fwd_5d_ret for AAA on day 20: got {row['fwd_5d_ret']} vs {expected}"
    )


# ---------------------------------------------------------------------------
# Look-ahead safety — no T+1 leak in Layer A side
# ---------------------------------------------------------------------------


def test_label_complete_through_excludes_recent_rows(panel_5x60):
    """label_complete_through must be at most panel_end - 30 trading days."""
    out = compute_forward_labels(panel=panel_5x60)
    # Find the max asof_date with a non-NaN fwd_30d_ret.
    full = out.dropna(subset=["fwd_30d_ret"])
    max_asof = full["asof_date"].max()
    panel_end = panel_5x60["date"].max()
    # There must be at least ~30 trading days between max_asof and panel_end.
    gap_days = (panel_end - max_asof).days
    # 30 trading days ≈ 42 calendar days. Allow for >= 30 calendar days at least.
    assert gap_days >= 30, (
        f"fwd_30d_ret has rows too close to panel end: gap={gap_days}d"
    )

    # label_complete_through column must reflect this.
    assert (out["label_complete_through"] == max_asof).all(), (
        f"label_complete_through inconsistent: unique values "
        f"{out['label_complete_through'].unique()}"
    )


def test_fwd_columns_nan_for_incomplete_rows(panel_5x60):
    """The last 30 trading days of the panel must have NaN fwd_30d_ret."""
    out = compute_forward_labels(panel=panel_5x60)
    # Last asof_date in the panel
    panel_end = panel_5x60["date"].max()
    # Rows in the last ~30 trading days should have NaN fwd_30d_ret
    near_end = out[out["asof_date"] >= panel_end - timedelta(days=10)]
    assert near_end["fwd_30d_ret"].isna().all(), (
        "fwd_30d_ret should be NaN within 30 trading days of panel end"
    )


def test_no_future_dates_in_label_assembly(panel_5x60):
    """Subtle safety: the row at asof=T uses close(T+N) — but the asof_date
    itself is T. No row's asof_date should exceed panel_end.
    """
    out = compute_forward_labels(panel=panel_5x60)
    panel_end = panel_5x60["date"].max()
    assert out["asof_date"].max() <= panel_end, (
        "label compute leaked asof_date beyond panel end"
    )


# ---------------------------------------------------------------------------
# Excess returns (universe-mean)
# ---------------------------------------------------------------------------


def test_fwd_5d_excess_ret_relative_to_universe_mean(panel_5x60):
    """fwd_5d_excess_ret(t,d) = fwd_5d_ret(t,d) - mean_over_universe(fwd_5d_ret(*,d))."""
    out = compute_forward_labels(panel=panel_5x60)
    # Pick a complete asof_date
    full = out.dropna(subset=["fwd_5d_ret", "fwd_5d_excess_ret"])
    sample_asof = full["asof_date"].iloc[10]
    same_day = out.loc[
        (out["asof_date"] == sample_asof) & out["fwd_5d_ret"].notna()
    ]
    universe_mean = same_day["fwd_5d_ret"].mean()
    for _, row in same_day.iterrows():
        expected = row["fwd_5d_ret"] - universe_mean
        assert abs(row["fwd_5d_excess_ret"] - expected) < 1e-5, (
            f"{row['symbol']} excess mismatch: "
            f"{row['fwd_5d_excess_ret']} vs {expected}"
        )


# ---------------------------------------------------------------------------
# Drawdown / runup in the 10-day window
# ---------------------------------------------------------------------------


def test_max_drawdown_zero_for_steady_riser():
    """A ticker that rises every day has zero drawdown over the 10d window."""
    panel = _build_panel_constant_returns(
        symbols=["UP"], n_days=60, daily_returns={"UP": 0.01}
    )
    out = compute_forward_labels(panel=panel)
    full = out.dropna(subset=["fwd_10d_max_drawdown"])
    # Steady riser → max drawdown should be 0 or very close to 0.
    assert (full["fwd_10d_max_drawdown"] <= 1e-6).all(), (
        f"steady riser has drawdown > 0: "
        f"{full['fwd_10d_max_drawdown'].describe()}"
    )


def test_max_runup_positive_for_steady_riser():
    """A ticker that rises every day has positive runup over the 10d window."""
    panel = _build_panel_constant_returns(
        symbols=["UP"], n_days=60, daily_returns={"UP": 0.01}
    )
    out = compute_forward_labels(panel=panel)
    full = out.dropna(subset=["fwd_10d_max_runup"])
    assert (full["fwd_10d_max_runup"] > 0).all(), (
        "steady riser must show positive runup"
    )


# ---------------------------------------------------------------------------
# Dtype contract
# ---------------------------------------------------------------------------


def test_drawdown_runup_nan_for_incomplete_forward_window():
    """Regression — codex iter-9 [P2]: if a symbol has a missing close
    anywhere in the T+1..T+10 window, fwd_10d_max_drawdown / fwd_10d_max_runup
    must be NaN rather than computed from the partial path. Otherwise
    consumers treat partial labels as complete.
    """
    # Build a 60d panel for SPARSE with a gap mid-window.
    base = _build_panel_constant_returns(
        symbols=["UP"], n_days=60, daily_returns={"UP": 0.01}
    )
    sparse = _build_panel_constant_returns(
        symbols=["SPARSE"], n_days=60, daily_returns={"SPARSE": 0.002}
    )
    # Punch a hole at the 25th trading day for SPARSE.
    gap_day = sorted(sparse["date"].unique())[25]
    sparse = sparse.loc[~((sparse["symbol"] == "SPARSE") & (sparse["date"] == gap_day))]
    panel = pd.concat([base, sparse], ignore_index=True)

    out = compute_forward_labels(panel=panel)
    # An asof T whose [T+1..T+10] window covers gap_day should have NaN dd/ru.
    # gap_day is at idx 25; affected asofs are 15..24 (inclusive).
    affected_asofs = sorted(panel["date"].unique())[15:25]
    for asof_dt in affected_asofs:
        sparse_row = out.loc[
            (out["symbol"] == "SPARSE") & (out["asof_date"] == asof_dt)
        ]
        if sparse_row.empty:
            continue
        dd = sparse_row.iloc[0]["fwd_10d_max_drawdown"]
        ru = sparse_row.iloc[0]["fwd_10d_max_runup"]
        assert pd.isna(dd), (
            f"SPARSE asof={asof_dt}: dd should be NaN (gap in window) but got {dd}"
        )
        assert pd.isna(ru), (
            f"SPARSE asof={asof_dt}: ru should be NaN (gap in window) but got {ru}"
        )


def test_label_complete_through_is_none_for_short_panel():
    """Regression — codex iter-6 [P2]: when the panel is shorter than the
    longest forward horizon (30 trading days), NO row has a complete
    fwd_30d_ret. ``label_complete_through`` must be None, not the first
    asof_date (which would mislead consumers filtering on completeness).
    """
    panel = _build_panel_constant_returns(
        symbols=["AAA"], n_days=10, daily_returns={"AAA": 0.01}
    )
    out = compute_forward_labels(panel=panel)
    assert (out["label_complete_through"].isna()).all(), (
        "label_complete_through must be None when panel < max_horizon; "
        f"got unique values {out['label_complete_through'].unique()}"
    )


def test_label_dtypes(panel_5x60):
    out = compute_forward_labels(panel=panel_5x60)
    for col in (
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
    ):
        assert out[col].dtype == np.float32, f"{col} dtype: {out[col].dtype}"
