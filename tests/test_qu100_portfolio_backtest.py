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


# ---------------------------------------------------------------------------
# Unit tests for the extracted steps (hand-computed expectations)
# ---------------------------------------------------------------------------


def _params(**kw) -> qp.StrategyParams:
    return qp.StrategyParams(**kw)


def _pos(**kw) -> qp.Position:
    defaults = dict(
        symbol="AAA", pattern_type="false_breakdown",
        entry_date=date(2025, 3, 3), entry_price=100.0,
        shares=1.0, allocated_amount=100.0,
        stop_loss=95.0, target_price=110.0,
        confidence=0.9, qu100_rank=1,
    )
    defaults.update(kw)
    return qp.Position(**defaults)


def _mini_prices(symbols=("AAA", "SPY")) -> pd.DataFrame:
    """3-day frame with hand-picked OHLC per symbol."""
    idx = pd.DatetimeIndex(["2025-03-03", "2025-03-04", "2025-03-05"])
    data = {
        "AAA": {
            "Open": [100.0, 102.0, 101.0],
            "High": [105.0, 106.0, 104.0],
            "Low": [98.0, 99.0, 97.0],
            "Close": [102.0, 104.0, 100.0],
        },
        "SPY": {
            "Open": [400.0, 402.0, 404.0],
            "High": [401.0, 403.0, 405.0],
            "Low": [399.0, 401.0, 403.0],
            "Close": [400.0, 402.0, 410.0],
        },
    }
    frames = {}
    for sym in symbols:
        for fld, vals in data[sym].items():
            frames[(fld, sym)] = pd.Series(vals, index=idx)
        frames[("Volume", sym)] = pd.Series([1000.0] * 3, index=idx)
    df = pd.DataFrame(frames)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_build_cohorts_filters_and_dates():
    rankings = _synthetic_rankings()
    universe = qp.build_cohorts(rankings)
    assert set(universe.top100["ranking_type"]) == {"top100"}
    assert set(universe.top100["long_short"]) == {"Long in"}
    assert universe.top100["rank"].max() <= 20
    assert "ZZZ" not in universe.symbols
    assert universe.start_date == universe.dates[0]
    assert universe.end_date == universe.dates[-1]

    clamped = qp.build_cohorts(rankings, start_date_str="2025-04-01")
    assert clamped.start_date >= date(2025, 4, 1)


def test_build_cohorts_needs_two_dates():
    rankings = _synthetic_rankings()
    only_last = rankings["data_date"].max().isoformat()
    with pytest.raises(ValueError, match="at least 2 dates"):
        qp.build_cohorts(rankings, start_date_str=only_last)


def test_build_price_lookups_multiindex():
    lookups = qp.build_price_lookups(_mini_prices(), ["AAA"])
    assert lookups.open.loc["2025-03-04", "AAA"] == 102.0
    assert lookups.close.loc["2025-03-05", "SPY"] == 410.0
    assert lookups.date_to_idx[date(2025, 3, 4)] == 1
    assert lookups.price_dates == [
        date(2025, 3, 3), date(2025, 3, 4), date(2025, 3, 5),
    ]


def test_compute_allocation():
    # 20% of portfolio value (cash 60 + positions 40 = 100) → 20
    assert qp.compute_allocation(60.0, 40.0, 5) == 20.0
    # capped at available cash
    assert qp.compute_allocation(10.0, 90.0, 5) == 10.0


def test_make_position_pattern_stop_vs_hard_stop():
    entry = {
        "symbol": "AAA", "pattern_type": "false_breakdown",
        "confidence": 0.8, "stop_loss": 95.0, "target_price": 110.0,
        "qu100_rank": 3,
    }
    pos = qp.make_position(entry, 100.0, date(2025, 3, 3), 20.0, 0.0)
    assert pos.stop_loss == 95.0
    assert pos.shares == 0.2  # 20 / 100

    pos_hard = qp.make_position(entry, 100.0, date(2025, 3, 3), 20.0, 0.05)
    assert pos_hard.stop_loss == pytest.approx(95.0)  # 100 * (1 - 0.05)


def test_close_position_pnl():
    trade = qp.close_position(
        _pos(shares=2.0), 110.0, "target_hit", date(2025, 3, 5),
    )
    assert trade.exit_price == 110.0
    assert trade.return_pct == pytest.approx(0.10)
    assert trade.pnl == pytest.approx(20.0)
    assert trade.exit_reason == "target_hit"
    assert trade.exit_date == date(2025, 3, 5)


def test_evaluate_price_exit_intraday():
    pos = _pos()  # entry 100, SL 95, TP 110
    p = _params()
    # stop loss on low
    assert qp.evaluate_price_exit(pos, 94.0, 105.0, 96.0, 1, p) == (95.0, "stop_loss")
    # SL takes precedence over TP
    assert qp.evaluate_price_exit(pos, 94.0, 111.0, 96.0, 1, p) == (95.0, "stop_loss")
    # target on high
    assert qp.evaluate_price_exit(pos, 99.0, 111.0, 108.0, 1, p) == (110.0, "target_hit")
    # no exit
    assert qp.evaluate_price_exit(pos, 99.0, 105.0, 102.0, 1, p) is None


def test_evaluate_price_exit_max_hold():
    pos = _pos()
    p = _params(max_hold_days=3)
    assert qp.evaluate_price_exit(pos, 99.0, 105.0, 102.0, 2, p) is None
    assert qp.evaluate_price_exit(pos, 99.0, 105.0, 102.0, 3, p) == (102.0, "max_hold")
    # price exits win over max hold
    assert qp.evaluate_price_exit(pos, 94.0, 105.0, 102.0, 3, p) == (95.0, "stop_loss")


def test_evaluate_price_exit_close_mode():
    pos = _pos()
    # close-only hard stop: -5% at close triggers, intraday low does not
    p = _params(use_close_price=True, hard_stop_pct=0.05)
    assert qp.evaluate_price_exit(pos, 80.0, 105.0, 96.0, 1, p) is None
    assert qp.evaluate_price_exit(pos, 80.0, 105.0, 95.0, 1, p) == (95.0, "stop_loss")
    # target checked at close, not high
    assert qp.evaluate_price_exit(pos, 99.0, 120.0, 108.0, 1, p) is None
    assert qp.evaluate_price_exit(pos, 99.0, 120.0, 110.0, 1, p) == (110.0, "target_hit")


def test_evaluate_price_exit_stop_limit_mode():
    pos = _pos()  # stop_loss 95
    p = _params(use_close_price=True, use_stop_limit=True, hard_stop_pct=0.05)
    # intraday low through the stop fills at the exact stop price
    assert qp.evaluate_price_exit(pos, 94.0, 105.0, 102.0, 1, p) == (95.0, "stop_loss")
    assert qp.evaluate_price_exit(pos, 96.0, 105.0, 102.0, 1, p) is None
    # stop-limit requires hard_stop_pct > 0
    p_no_hs = _params(use_close_price=True, use_stop_limit=True)
    assert qp.evaluate_price_exit(pos, 94.0, 105.0, 102.0, 1, p_no_hs) is None


def test_is_pattern_invalidated():
    from rainier.core.config import StockScreenerConfig
    config = StockScreenerConfig()
    prices = _synthetic_prices()
    end_ts = prices.index[-1]
    pos = _pos(symbol="AAA", pattern_type="w_bottom")

    def detect_none(sym, df, cfg, pattern_filter=None):
        return []

    def detect_match(sym, df, cfg, pattern_filter=None):
        return [_FakePattern("w_bottom", 0.9, 1.0, 2.0)]

    assert qp.is_pattern_invalidated(pos, prices, end_ts, detect_none, config)
    assert not qp.is_pattern_invalidated(pos, prices, end_ts, detect_match, config)
    # too little history → not invalidated (detector never runs)
    short_ts = prices.index[config.min_pattern_bars - 2]
    assert not qp.is_pattern_invalidated(pos, prices, short_ts, detect_none, config)


def test_find_entry_candidates_sorted_by_confidence():
    from rainier.core.config import StockScreenerConfig
    config = StockScreenerConfig()
    prices = _synthetic_prices()
    end_ts = prices.index[-1]
    day_stocks = pd.DataFrame([
        {"symbol": "AAA", "rank": 5},
        {"symbol": "BBB", "rank": 1},
        {"symbol": "CCC", "rank": 2},
    ])
    conf = {"AAA": 0.5, "BBB": 0.7, "CCC": 0.9}

    def detect(sym, df, cfg, pattern_filter=None):
        if sym == "BBB" and len(df) == 0:
            return []
        return [_FakePattern("false_breakdown", conf[sym], 10.0, 20.0)]

    candidates = qp.find_entry_candidates(day_stocks, prices, end_ts, detect, config)
    assert [c["symbol"] for c in candidates] == ["CCC", "BBB", "AAA"]
    assert candidates[0] == {
        "symbol": "CCC", "pattern_type": "false_breakdown",
        "confidence": 0.9, "stop_loss": 10.0, "target_price": 20.0,
        "qu100_rank": 2,
    }


def test_execute_pending_entries_fill_and_skips():
    lookups = qp.build_price_lookups(_mini_prices(), ["AAA"])
    state = qp.PortfolioState(cash=100.0, pending_entries=[
        # invalid: target below tomorrow's open (102)
        {"symbol": "AAA", "pattern_type": "false_breakdown", "confidence": 0.9,
         "stop_loss": 95.0, "target_price": 101.0, "qu100_rank": 1},
        # valid fill at day-1 open 102
        {"symbol": "AAA", "pattern_type": "false_breakdown", "confidence": 0.8,
         "stop_loss": 95.0, "target_price": 120.0, "qu100_rank": 2},
        # invalid: stop above entry price
        {"symbol": "AAA", "pattern_type": "false_breakdown", "confidence": 0.7,
         "stop_loss": 103.0, "target_price": 120.0, "qu100_rank": 3},
        # unknown symbol skipped
        {"symbol": "NOPE", "pattern_type": "false_breakdown", "confidence": 0.6,
         "stop_loss": 1.0, "target_price": 9.0, "qu100_rank": 4},
    ])
    qp.execute_pending_entries(state, lookups, 1, date(2025, 3, 4), _params())

    assert len(state.positions) == 1
    pos = state.positions[0]
    assert pos.entry_price == 102.0
    assert pos.allocated_amount == 20.0  # 100 / 5 slots
    assert pos.shares == pytest.approx(20.0 / 102.0)
    assert state.cash == 80.0
    assert state.pending_entries == []


def test_execute_pending_entries_respects_max_positions():
    lookups = qp.build_price_lookups(_mini_prices(), ["AAA"])
    entry = {"symbol": "AAA", "pattern_type": "false_breakdown", "confidence": 0.9,
             "stop_loss": 95.0, "target_price": 120.0, "qu100_rank": 1}
    state = qp.PortfolioState(
        cash=100.0,
        positions=[_pos(symbol="BBB")],
        pending_entries=[entry],
    )
    qp.execute_pending_entries(state, lookups, 1, date(2025, 3, 4), _params(max_positions=1))
    assert len(state.positions) == 1  # full book → entry dropped
    assert state.cash == 100.0


def test_apply_exits_skips_entry_day_and_closes_stop():
    from rainier.core.config import StockScreenerConfig
    config = StockScreenerConfig()
    lookups = qp.build_price_lookups(_mini_prices(), ["AAA"])
    same_day = _pos(symbol="AAA", entry_date=date(2025, 3, 5))
    # day-2 low is 97 → stop at 98 triggers
    stopped = _pos(symbol="AAA", entry_date=date(2025, 3, 3), stop_loss=98.0, shares=2.0)
    state = qp.PortfolioState(cash=0.0, positions=[same_day, stopped])

    qp.apply_exits(
        state, lookups, 2, date(2025, 3, 5), _params(),
        lambda *a, **k: [], config,
    )

    assert state.positions == [same_day]
    assert len(state.closed_trades) == 1
    trade = state.closed_trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == 98.0
    assert state.cash == pytest.approx(2.0 * 98.0)


def test_close_remaining_positions():
    lookups = qp.build_price_lookups(_mini_prices(), ["AAA"])
    universe = qp.BacktestUniverse(
        top100=pd.DataFrame(),
        dates=[date(2025, 3, 3), date(2025, 3, 5)],
        symbols=["AAA"],
    )
    state = qp.PortfolioState(cash=10.0, positions=[_pos(shares=2.0)])
    qp.close_remaining_positions(state, universe, lookups)

    assert state.positions == []
    assert len(state.closed_trades) == 1
    trade = state.closed_trades[0]
    assert trade.exit_reason == "end_of_backtest"
    assert trade.exit_price == 100.0  # day-2 close
    assert state.cash == pytest.approx(10.0 + 200.0)


def test_compute_benchmark_return_pct():
    lookups = qp.build_price_lookups(_mini_prices(), ["AAA"])
    # SPY open day0 = 400, close day2 = 410 → +2.5%
    bench = qp.compute_benchmark_return_pct(
        lookups, date(2025, 3, 3), date(2025, 3, 5),
    )
    assert bench == pytest.approx(2.5)
    # start date not in price index → None
    assert qp.compute_benchmark_return_pct(
        lookups, date(2025, 3, 2), date(2025, 3, 5),
    ) is None


def test_simulate_day_non_trading_day_carries_equity():
    from rainier.core.config import StockScreenerConfig
    config = StockScreenerConfig()
    lookups = qp.build_price_lookups(_mini_prices(), ["AAA"])
    universe = qp.BacktestUniverse(
        top100=pd.DataFrame(columns=["data_date", "symbol", "rank"]),
        dates=[date(2025, 3, 3), date(2025, 3, 8)],
        symbols=["AAA"],
    )
    state = qp.PortfolioState(cash=100.0, equity_curve=[100.0])
    qp.simulate_day(
        state, date(2025, 3, 8), universe, lookups, _params(),
        lambda *a, **k: [], config,
    )
    assert state.equity_curve == [100.0, 100.0]
    assert state.equity_dates == [date(2025, 3, 8)]


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
