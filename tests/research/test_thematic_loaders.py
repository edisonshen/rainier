"""Loader tests — long-format read, wide-panel pivot, supervised join.

Design ref: docs/DESIGN-thematic-ranks-dashboard.md §5.6 ([D-016]).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rainier.research.breadth.loaders import (
    load_supervised,
    load_thematic_features,
    load_thematic_panel,
)


# ---------------------------------------------------------------------------
# Fixture builders — synthesize Layer A and Layer B parquets
# ---------------------------------------------------------------------------


def _make_features_df(symbols: list[str], dates: list[date]) -> pd.DataFrame:
    rows = []
    for s_idx, sym in enumerate(symbols):
        for d_idx, dt in enumerate(dates):
            rows.append(
                {
                    "asof_date": dt,
                    "trading_day_ordinal": np.int32(d_idx),
                    "symbol": sym,
                    "ticker_id": np.int16(s_idx + 1),
                    "sector_id": np.int8(1),
                    "close": np.float64(100.0 + s_idx + d_idx),
                    "ret_1d": np.float32(0.01),
                    "rel_5": np.float32(0.02 + s_idx * 0.01),
                    "rel_10": np.float32(0.03 + s_idx * 0.01),
                    "rel_20": np.float32(0.04 + s_idx * 0.01),
                    "ret_ytd": np.float32(0.1),
                    "relvol_5": np.float32(0.5),
                    "relvol_10": np.float32(0.5),
                    "relvol_20": np.float32(0.5),
                    "r_5": np.int8(20 + s_idx * 15),
                    "r_10": np.int8(20 + s_idx * 15),
                    "r_20": np.int8(20 + s_idx * 15),
                    "rank": np.int8(20 + s_idx * 15),
                    "rank_delta_1d": np.int8(0),
                    "rank_delta_5d": np.int8(0),
                    "top15_streak": np.int16(0),
                    "rank_minus_sector_mean": np.float32(0.0),
                    "universe_size": np.int16(len(symbols)),
                    "universe_yaml_sha": "abc123",
                    "computed_at": pd.Timestamp("2024-11-08T16:30:00Z"),
                }
            )
    return pd.DataFrame(rows)


def _make_labels_df(symbols: list[str], dates: list[date]) -> pd.DataFrame:
    rows = []
    # Last 30 trading days of dates are incomplete (NaN fwd_30d_ret).
    cutoff_idx = max(0, len(dates) - 30)
    label_complete_through = dates[cutoff_idx - 1] if cutoff_idx > 0 else dates[0]
    for s_idx, sym in enumerate(symbols):
        for d_idx, dt in enumerate(dates):
            incomplete = d_idx >= cutoff_idx
            rows.append(
                {
                    "asof_date": dt,
                    "symbol": sym,
                    "fwd_3d_ret": np.float32(
                        np.nan if d_idx >= len(dates) - 3 else 0.02
                    ),
                    "fwd_5d_ret": np.float32(
                        np.nan if d_idx >= len(dates) - 5 else 0.03
                    ),
                    "fwd_10d_ret": np.float32(
                        np.nan if d_idx >= len(dates) - 10 else 0.04
                    ),
                    "fwd_20d_ret": np.float32(
                        np.nan if d_idx >= len(dates) - 20 else 0.05
                    ),
                    "fwd_30d_ret": np.float32(np.nan if incomplete else 0.06),
                    "fwd_5d_excess_ret": np.float32(
                        np.nan if d_idx >= len(dates) - 5 else 0.01
                    ),
                    "fwd_10d_excess_ret": np.float32(
                        np.nan if d_idx >= len(dates) - 10 else 0.01
                    ),
                    "fwd_20d_excess_ret": np.float32(
                        np.nan if d_idx >= len(dates) - 20 else 0.01
                    ),
                    "fwd_10d_max_drawdown": np.float32(
                        np.nan if d_idx >= len(dates) - 10 else 0.0
                    ),
                    "fwd_10d_max_runup": np.float32(
                        np.nan if d_idx >= len(dates) - 10 else 0.05
                    ),
                    "label_complete_through": label_complete_through,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_cache(tmp_path: Path) -> dict[str, Path]:
    """Write Layer A and Layer B parquets to a temp dir."""
    symbols = ["AAA", "BBB", "CCC"]
    dates = []
    d = date(2024, 10, 1)
    while len(dates) < 60:
        if d.weekday() < 5:
            dates.append(d)
        d = d + timedelta(days=1)

    feats = _make_features_df(symbols, dates)
    labels = _make_labels_df(symbols, dates)

    feats_path = tmp_path / "thematic_features_daily.parquet"
    labels_path = tmp_path / "thematic_labels_daily.parquet"

    feats.to_parquet(feats_path)
    labels.to_parquet(labels_path)
    return {"features": feats_path, "labels": labels_path}


# ---------------------------------------------------------------------------
# load_thematic_features
# ---------------------------------------------------------------------------


def test_load_features_returns_dataframe(tmp_cache):
    df = load_thematic_features(
        start=date(2024, 10, 1),
        end=date(2024, 12, 31),
        path=tmp_cache["features"],
    )
    assert isinstance(df, pd.DataFrame)
    assert "asof_date" in df.columns
    assert "symbol" in df.columns
    assert len(df) > 0


def test_load_features_date_filter(tmp_cache):
    df = load_thematic_features(
        start=date(2024, 10, 1),
        end=date(2024, 10, 10),
        path=tmp_cache["features"],
    )
    assert df["asof_date"].max() <= date(2024, 10, 10)
    assert df["asof_date"].min() >= date(2024, 10, 1)


def test_load_features_deterministic_sort(tmp_cache):
    df = load_thematic_features(
        start=date(2024, 10, 1),
        end=date(2024, 12, 31),
        path=tmp_cache["features"],
    )
    # Stable sort key: (symbol, asof_date) ascending
    for sym in df["symbol"].unique():
        sub = df.loc[df["symbol"] == sym, "asof_date"].tolist()
        assert sub == sorted(sub)


# ---------------------------------------------------------------------------
# load_thematic_panel — wide pivot
# ---------------------------------------------------------------------------


def test_load_panel_returns_wide_shape(tmp_cache):
    """Panel is a wide DataFrame indexed by (asof_date, symbol)."""
    panel = load_thematic_panel(
        features=["r_5", "r_10", "rank"],
        start=date(2024, 10, 1),
        end=date(2024, 12, 31),
        path=tmp_cache["features"],
    )
    # Could be MultiIndex columns or DataFrame with feature columns.
    # Spec: shape (n_dates, n_tickers, n_features). For a flat DataFrame,
    # we expect MultiIndex (asof_date, symbol) as rows and feature cols.
    assert "r_5" in panel.columns or any(
        "r_5" in str(c) for c in panel.columns
    )
    # n_unique asof_dates = 60; n_unique symbols = 3
    n_dates = panel.index.get_level_values(0).nunique()
    n_syms = panel.index.get_level_values(1).nunique()
    assert n_dates > 0 and n_syms == 3


def test_load_panel_preserves_feature_ordering(tmp_cache):
    """The features list determines column order in the panel output."""
    panel = load_thematic_panel(
        features=["r_5", "rank", "r_10"],
        start=date(2024, 10, 1),
        end=date(2024, 12, 31),
        path=tmp_cache["features"],
    )
    assert list(panel.columns) == ["r_5", "rank", "r_10"]


# ---------------------------------------------------------------------------
# load_supervised — A + B join
# ---------------------------------------------------------------------------


def test_load_supervised_joins_features_and_labels(tmp_cache):
    """load_supervised returns rows present in BOTH features and labels."""
    df = load_supervised(
        feature_cols=["r_5", "rank"],
        label_cols=["fwd_5d_ret", "fwd_30d_ret"],
        start=date(2024, 10, 1),
        end=date(2024, 12, 31),
        features_path=tmp_cache["features"],
        labels_path=tmp_cache["labels"],
    )
    for col in ("asof_date", "symbol", "r_5", "rank", "fwd_5d_ret", "fwd_30d_ret"):
        assert col in df.columns, f"missing column: {col}"


def test_load_supervised_drop_incomplete_labels(tmp_cache):
    """drop_incomplete_labels=True excludes rows where fwd_30d_ret is NaN."""
    df = load_supervised(
        feature_cols=["r_5"],
        label_cols=["fwd_5d_ret", "fwd_30d_ret"],
        start=date(2024, 10, 1),
        end=date(2024, 12, 31),
        features_path=tmp_cache["features"],
        labels_path=tmp_cache["labels"],
        drop_incomplete_labels=True,
    )
    # No NaN in fwd_30d_ret after filtering.
    assert not df["fwd_30d_ret"].isna().any(), (
        "drop_incomplete_labels=True should remove NaN fwd_30d_ret rows"
    )


def test_load_supervised_keeps_incomplete_when_flag_false(tmp_cache):
    """drop_incomplete_labels=False keeps NaN rows for inspection."""
    df = load_supervised(
        feature_cols=["r_5"],
        label_cols=["fwd_30d_ret"],
        start=date(2024, 10, 1),
        end=date(2024, 12, 31),
        features_path=tmp_cache["features"],
        labels_path=tmp_cache["labels"],
        drop_incomplete_labels=False,
    )
    # Should have some NaN fwd_30d_ret rows (the last ~30 trading days).
    assert df["fwd_30d_ret"].isna().any(), (
        "drop_incomplete_labels=False should keep NaN rows"
    )


def test_load_supervised_deterministic(tmp_cache):
    """Identical calls yield identical DataFrames."""
    df_a = load_supervised(
        feature_cols=["r_5", "rank"],
        label_cols=["fwd_5d_ret"],
        start=date(2024, 10, 1),
        end=date(2024, 12, 31),
        features_path=tmp_cache["features"],
        labels_path=tmp_cache["labels"],
        drop_incomplete_labels=True,
    )
    df_b = load_supervised(
        feature_cols=["r_5", "rank"],
        label_cols=["fwd_5d_ret"],
        start=date(2024, 10, 1),
        end=date(2024, 12, 31),
        features_path=tmp_cache["features"],
        labels_path=tmp_cache["labels"],
        drop_incomplete_labels=True,
    )
    pd.testing.assert_frame_equal(df_a, df_b)
