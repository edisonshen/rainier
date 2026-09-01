"""Characterization + unit tests for the QU100 portfolio backtest.

The golden test locks the FULL output of ``run_qu100_portfolio_backtest`` on a
deterministic synthetic dataset across parameter modes, so refactors are
provably behavior-preserving. Regenerate the golden (only when a behavior
change is intended) with:

    RAINIER_REGEN_GOLDEN=1 uv run pytest tests/test_qu100_portfolio_backtest.py
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rainier.backtest import qu100_portfolio as qp

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden" / "qu100_portfolio_backtest.json"

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


@dataclass
class _FakePattern:
    pattern_type: str
    confidence: float
    stop_loss: float
    target_wave1: float


def _price_index() -> pd.DatetimeIndex:
    return pd.bdate_range("2025-01-06", periods=90)


def _synthetic_prices() -> pd.DataFrame:
    """Deterministic multi-symbol OHLCV frame in yfinance MultiIndex layout."""
    rng = np.random.default_rng(42)
    idx = _price_index()
    frames = {}
    for si, sym in enumerate(SYMBOLS + ["SPY"]):
        base = 50.0 + si * 20.0
        rets = rng.normal(0.0005, 0.03, len(idx))
        close = base * np.exp(np.cumsum(rets))
        open_ = np.concatenate([[base], close[:-1]]) * (
            1 + rng.normal(0, 0.005, len(idx))
        )
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.02, len(idx))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.02, len(idx))))
        vol = rng.integers(1_000, 100_000, len(idx)).astype(float)
        for field, vals in (
            ("Open", open_), ("High", high), ("Low", low),
            ("Close", close), ("Volume", vol),
        ):
            frames[(field, sym)] = pd.Series(vals, index=idx)
    df = pd.DataFrame(frames)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _synthetic_rankings() -> pd.DataFrame:
    """QU100 rankings covering the back half of the price window."""
    idx = _price_index()
    rows = []
    for di, ts in enumerate(idx[40:]):
        d = ts.date()
        for ri, sym in enumerate(SYMBOLS):
            rows.append({
                "data_date": d,
                "symbol": sym,
                "rank": ((ri + di) % 25) + 1,
                "ranking_type": "top100",
                "long_short": "Long in",
                "sector": "Tech",
                "industry": "Software",
            })
        # noise rows the backtest must filter out
        rows.append({
            "data_date": d, "symbol": "ZZZ", "rank": 1,
            "ranking_type": "bottom100", "long_short": "Long in",
            "sector": "X", "industry": "Y",
        })
        rows.append({
            "data_date": d, "symbol": SYMBOLS[0], "rank": 2,
            "ranking_type": "top100", "long_short": "Short in",
            "sector": "X", "industry": "Y",
        })
    return pd.DataFrame(rows)


def _fake_detect(symbol, df, config, pattern_filter=None):
    """Deterministic pattern detector keyed off the last close.

    The modulo condition flips as prices move, which naturally exercises the
    pattern_invalidated exit path; SL/TP around the last close exercise the
    stop_loss / target_hit paths given the ±3% daily synthetic moves.
    """
    last_close = float(df["close"].iloc[-1])
    if int(last_close * 100) % 3 != 0:
        return []
    ptype = "false_breakdown" if int(last_close) % 2 == 0 else "false_breakdown_w_bottom"
    return [_FakePattern(
        pattern_type=ptype,
        confidence=round(last_close % 1.0, 4),
        stop_loss=last_close * 0.96,
        target_wave1=last_close * 1.05,
    )]


@pytest.fixture
def synthetic_env(monkeypatch):
    monkeypatch.setattr(qp, "load_rankings_from_db", _synthetic_rankings)
    prices = _synthetic_prices()
    monkeypatch.setattr(qp, "fetch_all_prices", lambda symbols, start, end: prices)


# ---------------------------------------------------------------------------
# Golden characterization test
# ---------------------------------------------------------------------------

SCENARIOS = {
    "default": {},
    "top_n_3_max_pos_2": {"max_positions": 2, "top_n": 3},
    "max_hold_1": {"max_hold_days": 1},
    "hard_stop_close": {"hard_stop_pct": 0.05, "use_close_price": True},
    "hard_stop_stop_limit": {
        "hard_stop_pct": 0.05, "use_close_price": True, "use_stop_limit": True,
    },
    "start_date_clamped": {"start_date_str": "2025-04-01"},
}


def _result_to_jsonable(result) -> dict:
    d = asdict(result)
    d["trades"] = [
        {k: (v.isoformat() if isinstance(v, date) else _round(v))
         for k, v in asdict(t).items()}
        for t in result.trades
    ]
    d["equity_curve"] = [_round(v) for v in result.equity_curve]
    d["equity_dates"] = [x.isoformat() for x in result.equity_dates]
    d["start_date"] = result.start_date.isoformat()
    d["end_date"] = result.end_date.isoformat()
    for k, v in d.items():
        if isinstance(v, float):
            d[k] = _round(v)
    return d


def _round(v):
    return round(v, 10) if isinstance(v, float) else v


def _run_all_scenarios() -> dict:
    out = {}
    for name, kwargs in SCENARIOS.items():
        result = qp.run_qu100_portfolio_backtest(_fake_detect, **kwargs)
        out[name] = _result_to_jsonable(result)
    return out


def test_characterization_golden(synthetic_env):
    actual = _run_all_scenarios()

    if os.environ.get("RAINIER_REGEN_GOLDEN"):
        GOLDEN_PATH.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n")
        pytest.skip("golden regenerated")

    expected = json.loads(GOLDEN_PATH.read_text())
    assert actual == expected


def test_synthetic_dataset_exercises_all_exit_paths(synthetic_env):
    """Guard: the golden covers every exit reason, so it can catch regressions
    in each exit branch."""
    reasons = set()
    for name, kwargs in SCENARIOS.items():
        result = qp.run_qu100_portfolio_backtest(_fake_detect, **kwargs)
        reasons |= {t.exit_reason for t in result.trades}
        assert result.total_trades == len(result.trades)
    assert {
        "stop_loss", "target_hit", "pattern_invalidated", "max_hold",
        "end_of_backtest",
    } <= reasons
