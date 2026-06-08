"""Synthetic OHLCV fixtures for resonance-gate tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rainier.signals.panel import PanelInputs


def _ohlc_from_close(close: np.ndarray) -> pd.DataFrame:
    """Build a clean OHLC frame from a close path (small synthetic ranges)."""
    close = np.asarray(close, dtype=float)
    high = close * 1.005
    low = close * 0.995
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


@pytest.fixture
def n_bars() -> int:
    return 400


@pytest.fixture
def trending_inputs(n_bars) -> PanelInputs:
    """Steady uptrend → every trend member should fire risk-on once warmed up."""
    t = np.arange(n_bars)
    close = 100.0 * (1.0 + 0.001) ** t  # smooth ~0.1%/day uptrend
    qqq = _ohlc_from_close(close)
    vix = pd.Series(np.full(n_bars, 15.0))  # calm
    spy = pd.Series(100.0 * (1.0 + 0.0008) ** t)
    breadth = pd.Series(np.full(n_bars, 0.8))
    return PanelInputs(qqq=qqq, vix=vix, spy=spy, breadth=breadth)


@pytest.fixture
def choppy_inputs(n_bars) -> PanelInputs:
    """Flat/choppy path → trend members mostly risk-off."""
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0, 0.2, n_bars)) * 0.0
    close = 100.0 + rng.normal(0, 1.0, n_bars)  # mean-reverting noise around 100
    qqq = _ohlc_from_close(close)
    vix = pd.Series(np.full(n_bars, 28.0))
    spy = pd.Series(100.0 + rng.normal(0, 1.0, n_bars))
    breadth = pd.Series(np.full(n_bars, 0.3))
    return PanelInputs(qqq=qqq, vix=vix, spy=spy, breadth=breadth)
