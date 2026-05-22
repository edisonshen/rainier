"""Read-only loaders for Layer A features + Layer A/B supervised joins.

Pure functions of `(date_range, paths)` -> `pd.DataFrame`. Identical inputs
return identical outputs (deterministic sort + filter).

Design ref:
    docs/DESIGN-thematic-ranks-dashboard.md §5.6 ([D-016] long-format base +
    builder-function API).
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

DEFAULT_FEATURES_PATH = Path("data/cache/thematic_features_daily.parquet")
DEFAULT_LABELS_PATH = Path("data/cache/thematic_labels_daily.parquet")


# ---------------------------------------------------------------------------
# Coercion helper
# ---------------------------------------------------------------------------


def _coerce_date(value: str | date | datetime | pd.Timestamp) -> date:
    if isinstance(value, datetime) or isinstance(value, pd.Timestamp):
        return pd.to_datetime(value).date()
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


# ---------------------------------------------------------------------------
# Layer A reader — long format
# ---------------------------------------------------------------------------


def load_thematic_features(
    start: str | date,
    end: str | date,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load Layer A features filtered to ``start <= asof_date <= end``.

    Returns a DataFrame sorted by ``(symbol, asof_date)`` ascending so that
    byte-identical inputs return byte-identical outputs.
    """
    p = Path(path) if path is not None else DEFAULT_FEATURES_PATH
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_parquet(p)
    start_d = _coerce_date(start)
    end_d = _coerce_date(end)
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.date
    df = df.loc[(df["asof_date"] >= start_d) & (df["asof_date"] <= end_d)]
    df = df.sort_values(["symbol", "asof_date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Wide panel — MultiIndex (asof_date, symbol) x features
# ---------------------------------------------------------------------------


def load_thematic_panel(
    features: list[str],
    start: str | date,
    end: str | date,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Return a wide panel indexed by ``(asof_date, symbol)`` with the
    requested feature columns in the order supplied.

    Shape: ``(n_dates * n_tickers, n_features)`` rows; downstream code can
    convert to torch.Tensor via ``df.values.reshape(n_dates, n_tickers, n_feat)``.

    Parameters
    ----------
    features:
        List of feature column names. Order is preserved in the output columns.
    start, end:
        Inclusive date filter.
    path:
        Override for the Layer A parquet location.
    """
    long = load_thematic_features(start=start, end=end, path=path)
    # Validate requested columns exist.
    missing = [c for c in features if c not in long.columns]
    if missing:
        raise ValueError(f"requested features not in Layer A: {missing}")

    panel = long.set_index(["asof_date", "symbol"])[features]
    panel = panel.sort_index()
    return panel


# ---------------------------------------------------------------------------
# Supervised — Layer A + B join
# ---------------------------------------------------------------------------


def load_supervised(
    feature_cols: list[str],
    label_cols: list[str],
    start: str | date,
    end: str | date,
    features_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    drop_incomplete_labels: bool = True,
) -> pd.DataFrame:
    """Join Layer A features with Layer B labels for supervised ML.

    Parameters
    ----------
    feature_cols, label_cols:
        Column lists from Layer A / Layer B respectively. Order is preserved
        in the output: ``[asof_date, symbol, *feature_cols, *label_cols]``.
    start, end:
        Inclusive ``asof_date`` filter.
    features_path, labels_path:
        Cache-path overrides.
    drop_incomplete_labels:
        When True (default), rows lacking a complete set of labels (any NaN
        in the most-restrictive label, ``fwd_30d_ret``) are filtered out.
        When False, the join keeps incomplete rows so callers can inspect.
    """
    f = Path(features_path) if features_path is not None else DEFAULT_FEATURES_PATH
    l = Path(labels_path) if labels_path is not None else DEFAULT_LABELS_PATH
    if not f.exists():
        raise FileNotFoundError(f)
    if not l.exists():
        raise FileNotFoundError(l)

    start_d = _coerce_date(start)
    end_d = _coerce_date(end)

    features = pd.read_parquet(f)
    labels = pd.read_parquet(l)
    features["asof_date"] = pd.to_datetime(features["asof_date"]).dt.date
    labels["asof_date"] = pd.to_datetime(labels["asof_date"]).dt.date

    # Project to requested cols + identity
    feat_proj = features.loc[
        (features["asof_date"] >= start_d) & (features["asof_date"] <= end_d),
        ["asof_date", "symbol", *feature_cols],
    ]
    label_proj = labels.loc[
        (labels["asof_date"] >= start_d) & (labels["asof_date"] <= end_d),
        ["asof_date", "symbol", *label_cols],
    ]

    joined = feat_proj.merge(label_proj, on=["asof_date", "symbol"], how="inner")
    if drop_incomplete_labels:
        # Most-restrictive label: prefer fwd_30d_ret if requested, else the
        # longest-horizon column among label_cols (largest N in fwd_Nd_ret).
        gate_col = _most_restrictive_label(label_cols)
        if gate_col and gate_col in joined.columns:
            joined = joined.loc[joined[gate_col].notna()].reset_index(drop=True)

    joined = joined.sort_values(["symbol", "asof_date"]).reset_index(drop=True)
    return joined


def _most_restrictive_label(label_cols: list[str]) -> str | None:
    """Pick the label column with the longest forward horizon.

    Heuristic: among requested labels, prefer ``fwd_30d_ret`` (the most-
    restrictive in the §5.2 schema). If not requested, scan for any
    ``fwd_<N>d_ret`` and return the largest N.
    """
    if "fwd_30d_ret" in label_cols:
        return "fwd_30d_ret"
    candidates: list[tuple[int, str]] = []
    for c in label_cols:
        if c.startswith("fwd_") and c.endswith("d_ret"):
            try:
                n = int(c.removeprefix("fwd_").removesuffix("d_ret"))
                candidates.append((n, c))
            except ValueError:
                continue
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]
