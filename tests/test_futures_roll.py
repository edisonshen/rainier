"""Spec: front-month futures bars are back-adjusted at every quarterly roll.

Synthetic series: price drifts 1 point a bar; at each roll the front month jumps up by that
roll's gap (the next contract's carry premium). Back-adjusting must remove every jump and
leave the newest bars untouched, wherever inside the roll window the source switched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rainier.core.types import Timeframe
from rainier.data.futures_roll import (
    CARRY,
    back_adjust,
    carry_gap,
    contract_gaps,
    fill_roll_gaps,
    quarterly_expiries,
)
from rainier.data.persistence import merge_with_existing

GAPS = pd.DataFrame({"expiry": ["20250919", "20251219"], "gap": [55.0, 58.5],
                     "source": ["ibkr", "ibkr"], "price": [None] * 2, "rate": [None] * 2,
                     "asof": [None] * 2})


def _front_month(freq: str, switches: dict[str, list[str]]) -> pd.DataFrame:
    """Bars Aug 2025 - Feb 2026 jumping by each gap at the given switch time(s) per expiry.

    A list with several times flips old/new/old... like a source that wavers near a roll.
    """
    ts = pd.date_range("2025-08-01", "2026-02-01", freq=freq, tz="UTC")
    true = 6000.0 + np.arange(len(ts))
    premium = np.zeros(len(ts))
    for (exp, gap), times in zip(zip(GAPS.expiry, GAPS.gap, strict=True), switches.values(),
                                 strict=True):
        on = np.zeros(len(ts), dtype=bool)
        for k, t in enumerate(times):
            flip = (ts >= pd.Timestamp(t, tz="UTC"))
            on = on ^ flip if k else flip
        premium -= gap * ~on  # bars before the switch sit `gap` below today's contract
    close = true + premium
    return pd.DataFrame({"timestamp": ts, "open": close, "high": close + 0.5,
                         "low": close - 0.5, "close": close, "volume": 1.0}), true


def _contracts(bars: pd.DataFrame, true: np.ndarray) -> dict[str, pd.Series]:
    """Each contract's closes at every bar, `true` less the premium of the rolls after it."""
    idx = pd.DatetimeIndex(bars["timestamp"])
    return {"20250919": pd.Series(true - 55.0 - 58.5, index=idx),
            "20251219": pd.Series(true - 58.5, index=idx),
            "20260320": pd.Series(true, index=idx)}


@pytest.mark.parametrize("freq, switches, cached", [
    ("1D", {"sep": ["2025-09-22"], "dec": ["2025-12-22"]}, False),    # yfinance: after expiry
    ("1h", {"sep": ["2025-09-14 22:00"], "dec": ["2025-12-14 23:00"]}, False),  # volume roll
    ("1h", {"sep": ["2025-09-14 22:00", "2025-09-15 03:00", "2025-09-15 07:00"],
            "dec": ["2025-12-15 01:00"]}, True),         # wavers: needs per-contract closes
])
def test_back_adjust_removes_each_roll_jump_and_keeps_latest_bars(freq, switches, cached):
    bars, true = _front_month(freq, switches)
    adj = back_adjust(bars, GAPS, _contracts(bars, true) if cached else None)
    offset = adj["close"].to_numpy() - true
    assert np.allclose(offset, offset[-1]) and offset[-1] == 0
    assert np.allclose(adj["high"] - adj["close"], 0.5), "OHLC shift together"


def test_contract_closes_pin_the_roll_when_a_real_move_is_as_large_as_the_gap():
    bars, true = _front_month("1D", {"sep": ["2025-09-22"], "dec": ["2025-12-22"]})
    rally = (bars["timestamp"] >= pd.Timestamp("2025-12-12", tz="UTC")).to_numpy()
    for col in ("open", "high", "low", "close"):
        bars[col] += np.where(rally, 50.0, 0.0)  # a 50-pt up day a week before the roll
    true = true + np.where(rally, 50.0, 0.0)
    adj = back_adjust(bars, GAPS, _contracts(bars, true))
    assert np.allclose(adj["close"], true)


def test_roll_still_in_progress_is_not_applied_yet():
    bars, _ = _front_month("1h", {"sep": ["2025-09-14 22:00"], "dec": ["2025-12-14 23:00"]})
    mid_dec = bars[bars["timestamp"] < pd.Timestamp("2025-12-16", tz="UTC")]
    adj = back_adjust(mid_dec, GAPS)
    assert adj["close"].iloc[-1] == mid_dec["close"].iloc[-1], "latest price never moves"
    sep_jump = adj["close"].diff()[adj["timestamp"] == pd.Timestamp("2025-09-14 22:00", tz="UTC")]
    assert sep_jump.abs().max() < 5, "the finished Sep roll is adjusted"


def test_quarterly_expiries_are_third_fridays():
    got = quarterly_expiries(pd.Timestamp("2025-01-01", tz="UTC"),
                             pd.Timestamp("2026-12-31", tz="UTC"))
    assert [e.strftime("%Y%m%d") for e in got] == [
        "20250321", "20250620", "20250919", "20251219",
        "20260320", "20260619", "20260918", "20261218"]


def test_contract_gaps_measure_next_minus_expiring_at_the_volume_roll():
    ts = pd.date_range("2025-12-01", "2025-12-19", freq="1h", tz="UTC")
    roll = ts >= pd.Timestamp("2025-12-14 23:00", tz="UTC")
    z5 = pd.DataFrame({"timestamp": ts, "close": 6800.0, "volume": np.where(roll, 10, 1000)})
    h6 = pd.DataFrame({"timestamp": ts, "close": 6858.5, "volume": np.where(roll, 1000, 10)})
    assert contract_gaps({"20251219": z5, "20260320": h6}) == {"20251219": 58.5}


def test_unobserved_rolls_get_a_same_quarter_carry_estimate():
    observed = GAPS.iloc[[1]].assign(price=6900.0, rate=0.04)
    expiries = quarterly_expiries(pd.Timestamp("2024-12-01", tz="UTC"),
                                  pd.Timestamp("2025-12-31", tz="UTC"))
    out = fill_roll_gaps(observed, expiries, price_at=lambda t: 6000.0, rate_at=lambda t: 0.04)
    assert out["expiry"].tolist() == ["20241220", "20250321", "20250620", "20250919", "20251219"]
    assert out.set_index("expiry").loc["20251219", "source"] == "ibkr"
    dec24 = out.set_index("expiry").loc["20241220"]
    assert dec24["source"] == CARRY
    # Same dividend yield as Dec 2025, scaled to Dec 2024's price.
    assert dec24["gap"] == pytest.approx(58.5 * 6000 / 6900, abs=0.25)
    assert carry_gap(6000, 0.04, 0.04, 91) == 0


def test_daily_bars_of_one_session_from_two_sources_merge_into_one(tmp_path):
    csv = tmp_path / "MES_1D.csv"
    pd.DataFrame({"timestamp": pd.to_datetime(["2025-12-04 05:00", "2025-12-05 00:00"],
                                              utc=True),
                  "close": [6850.0, 6990.5]}).to_csv(csv, index=False)
    new = pd.DataFrame({"timestamp": pd.to_datetime(["2025-12-05 05:00"], utc=True),
                        "close": [6878.25]})
    out = merge_with_existing(new, csv, daily=True)
    assert out["close"].tolist() == [6850.0, 6878.25]
    assert len(merge_with_existing(new, csv)) == 3, "intraday bars dedupe on exact timestamp"


def test_roll_gaps_cli_builds_the_table_that_back_adjusts_loaded_csvs(tmp_path, monkeypatch):
    """`rainier roll-gaps`: IBKR-cached contracts give observed gaps, Yahoo the live spread,
    a carry estimate the rest; the chart loader then reads every CSV in today's contract."""
    from click.testing import CliRunner

    from rainier.cli import cli
    from rainier.cli.futures import _load_mtf_csvs
    from rainier.data import yfinance_provider

    data, raw = tmp_path / "csv", tmp_path / "raw"
    data.mkdir(), raw.mkdir()
    bars, true = _front_month("1D", {"sep": ["2025-09-22"], "dec": ["2025-12-22"]})
    bars.to_csv(data / "MES_1D.csv", index=False)
    ts = pd.date_range("2025-12-01", "2025-12-19", freq="1h", tz="UTC")
    roll = ts >= pd.Timestamp("2025-12-14 23:00", tz="UTC")
    for exp, px, vol in (("20251219", 6800.0, np.where(roll, 10, 1000)),
                         ("20260320", 6858.5, np.where(roll, 1000, 10))):
        pd.DataFrame({"timestamp": ts, "open": px, "high": px, "low": px, "close": px,
                      "volume": vol}).to_csv(raw / f"MES_{exp}_5m.csv", index=False)
    monkeypatch.setattr(yfinance_provider, "front_roll_spread", lambda sym, now: None)
    monkeypatch.setattr(yfinance_provider, "rate_history", lambda start: pd.Series(
        0.04, index=pd.date_range("2025-01-01", periods=1, tz="UTC")))
    r = CliRunner().invoke(cli, ["roll-gaps", "--data-dir", str(data), "--raw-dir", str(raw)])
    assert r.exit_code == 0, r.output
    table = pd.read_csv(data / "MES_rolls.csv", dtype={"expiry": str}).set_index("expiry")
    assert table.loc["20251219", ["gap", "source"]].tolist() == [58.5, "ibkr"]
    assert table.loc["20250919", "source"] == CARRY

    loaded = _load_mtf_csvs(data, "MES")[Timeframe.D1]
    jump = loaded["close"].diff().abs()
    assert jump.max() < 5, "no roll steps left"
    assert loaded["close"].iloc[-1] == bars["close"].iloc[-1]
