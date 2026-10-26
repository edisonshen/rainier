"""Spec tests: Xiaojiang transaction rules trading the chart's own S/R zones.

Hand-drawn zone scenarios pin each rule (fill, hard stop, target, touch count, one position at
a time, open position at the data end); the real-MES ``sr-levels`` run checks that the chart's
A/B trades sit on point-in-time ``sr_zones`` and never use bars after the decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from rainier.core.config import ChartPublishConfig, SRZoneConfig, XJSetupConfig
from rainier.core.types import Timeframe
from rainier.research.xiaojiang.chart import chart_setups
from rainier.research.xiaojiang.sim import (
    COMMISSION_USD,
    POINT_USD,
    Market,
    next_orders,
    simulate,
)
from rainier.research.xiaojiang.zones import Lines

FIXTURE = Path(__file__).parent / "fixtures" / "mes_sr_small.parquet"
SUPPORT = Lines(price=[100.0], lo=[99.0], hi=[101.0], rank=[3], nconf=[2], touches=[2])


class _OneZoneBook:
    """The same support zone 99-101 (center 100) for every bar."""

    def __init__(self, n: int):
        self.hour_of_bar = np.zeros(n, dtype=int)
        self.snapshots = [SUPPORT]


def _market(bars: list[tuple[str, float, float, float, float]]) -> Market:
    m5 = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close"])
    m5["timestamp"] = pd.to_datetime(m5.timestamp)  # PT wall clock, like CSVProvider
    m5["volume"] = 100.0
    return Market({Timeframe.M5: m5}, _OneZoneBook(len(m5)))


# Tue Oct 13 2026 (PT). Session 1 ends at the 14:00 halt; session 2 opens 15:00.
TWO_TOUCHES = [
    ("2026-10-13 13:00", 110, 111, 109, 110),
    ("2026-10-13 13:05", 110, 110.5, 106, 107),
    ("2026-10-13 13:10", 107, 107.5, 103, 104),
    ("2026-10-13 13:15", 104, 104.5, 99.5, 100.5),  # 1st touch: buy limit 100 fills
    ("2026-10-13 13:20", 100.5, 105, 100, 104),
    ("2026-10-13 13:25", 104, 110.5, 103.5, 110.2),  # 2R target 110 traded through
    ("2026-10-13 15:00", 111.5, 113, 111.2, 112),  # left the zone by 10+ pts
    ("2026-10-13 15:05", 112, 112, 105, 106),
    ("2026-10-13 15:10", 106, 106, 99.5, 100.5),  # 2nd touch
    ("2026-10-13 15:15", 100.5, 100.6, 94.5, 95),  # hard stop 95
    ("2026-10-13 15:20", 95, 96, 94, 95.5),
]
LEFT = XJSetupConfig(entry="left", level="center", stop_pts=5, exit="rr2", min_rank=3,
                     min_conf=2, max_tests=3)


def test_left_limit_fills_at_the_zone_then_target_then_hard_stop():
    tr = simulate(_market(TWO_TOUCHES), LEFT).trades
    assert tr[["dir", "line", "entry", "stop", "tp", "exit", "px_out", "touch"]].values.tolist() == [
        [1, 100.0, 100.0, 95.0, 110.0, "target", 110.0, 1],
        [1, 100.0, 100.0, 95.0, 110.0, "stop", 94.75, 2],  # stop 95 less 0.25 slippage
    ]
    assert tr.t_in.dt.strftime("%H:%M").tolist() == ["13:15", "15:10"]
    assert tr.usd.round(2).tolist() == [round(10 * POINT_USD - COMMISSION_USD, 2),
                                        round(-5.25 * POINT_USD - COMMISSION_USD, 2)]
    assert ((tr.zone_lo <= tr.line) & (tr.line <= tr.zone_hi)).all()


def test_first_touch_only_skips_the_second_touch():
    tr = simulate(_market(TWO_TOUCHES), LEFT.model_copy(update={"max_tests": 1})).trades
    assert tr.exit.tolist() == ["target"]


def test_resting_order_then_open_position_blocks_new_orders():
    flat = _market(TWO_TOUCHES[:3])
    res = simulate(flat, LEFT)
    assert res.trades.empty and res.open is None
    assert next_orders(flat, res, LEFT, SUPPORT) == [
        {"dir": 1, "price": 100.0, "stop": 95.0, "tp": 110.0, "kind": "limit",
         "zone_lo": 99.0, "zone_hi": 101.0}]
    held = _market(TWO_TOUCHES[:5])  # filled at 13:15, data ends mid-week
    res = simulate(held, LEFT)
    assert res.trades.empty
    assert {k: res.open[k] for k in ("dir", "entry", "stop", "tp")} == {
        "dir": 1, "entry": 100.0, "stop": 95.0, "tp": 110.0}
    assert next_orders(held, res, LEFT, SUPPORT) == []


def test_right_side_enters_on_the_rejection_close_with_the_stop_beyond_the_zone():
    bars = [
        ("2026-10-13 06:30", 110, 111, 109, 110),
        ("2026-10-13 06:35", 110, 110.5, 104, 105),
        ("2026-10-13 06:40", 105, 105, 100.5, 101.5),  # trades through the 101 edge
        ("2026-10-13 06:45", 101.5, 104, 101, 103.5),  # closes back above it, up candle
        ("2026-10-13 06:50", 103.5, 105, 103, 104),
    ]
    p = XJSetupConfig(entry="right", level="edge", stop_pts=12, exit="rr2", min_rank=2,
                      max_tests=1)
    mkt = _market(bars)
    res = simulate(mkt, p)
    assert res.trades.empty and res.open is not None
    assert res.open["t_in"] == pd.Timestamp("2026-10-13 06:45")
    assert (res.open["line"], res.open["entry"], res.open["stop"]) == (101.0, 103.75, 89.0)
    assert res.open["tp"] == pytest.approx(103.75 + 2 * (103.75 - 89.0))
    assert next_orders(mkt, res, p, SUPPORT) == []


def test_friday_close_flattens_the_position():
    bars = [(t.replace("2026-10-13 13", "2026-10-16 13"), *px) for t, *px in TWO_TOUCHES[:5]]
    bars.append(("2026-10-16 13:55", 104, 105, 103, 104.5))
    tr = simulate(_market(bars), LEFT).trades
    assert tr.exit.tolist() == ["session_end"] and tr.px_out.iloc[0] == 104.25


@pytest.fixture(scope="module")
def mes() -> dict[Timeframe, pd.DataFrame]:
    df = pd.read_parquet(FIXTURE)
    return {
        Timeframe(tf): g.drop(columns="timeframe").sort_values("timestamp").reset_index(drop=True)
        for tf, g in df.groupby("timeframe")
    }


def test_chart_ab_trades_sit_on_point_in_time_chart_zones(mes, tmp_path):
    """`sr-levels` (what the cron publishes) carries A and B for the page's toggle."""
    from rainier.cli import cli
    from rainier.cli.futures import _load_mtf_csvs

    for tf, df in mes.items():
        df.to_csv(tmp_path / f"MES_{tf.value}.csv", index=False)
    out = tmp_path / "payload.json"
    r = CliRunner().invoke(cli, ["sr-levels", "--data-dir", str(tmp_path), "--format", "json",
                                 "--output", str(out)])
    assert r.exit_code == 0, r.output
    payload = json.loads(out.read_text())
    setups = {s["key"]: s for s in payload["xj"]["setups"]}
    cfg = ChartPublishConfig().xj_setups
    assert list(setups) == ["A", "B"]
    candle_t = {c[0] for c in payload["candles"]["5m"]}
    for key, s in setups.items():
        p = cfg[key]
        assert s["label"] == p.label and s["trades"], key
        prev_out = 0
        for t_in, t_out, d, level, entry, px, stop, tp, why, usd, zlo, zhi in s["trades"]:
            assert t_in in candle_t and t_out in candle_t and t_in > prev_out  # one at a time
            prev_out = t_out
            assert zlo <= level <= zhi  # every order sits on a chart zone (edge or center)
            assert level - d * p.stop_pts == stop  # fixed-point hard stop, never moved
            assert why in {"stop", "target", "session_end"}
            assert usd == pytest.approx(d * (px - entry) * POINT_USD - COMMISSION_USD, abs=0.01)
            if p.exit.startswith("rr"):
                assert tp == pytest.approx(entry + d * float(p.exit[2:]) * abs(entry - stop),
                                           abs=0.01)
        bands = [(z["low"], z["high"]) for z in payload["zones"]]
        for d, price, stop, tp, kind, zlo, zhi in s["orders"]:
            assert s["open"] is None
            assert any(abs(zlo - lo) <= 0.125 and abs(zhi - hi) <= 0.125 for lo, hi in bands)
            assert kind == ("limit" if p.entry == "left" else "rejection")
            assert price - d * p.stop_pts == stop and d * (payload["price"] - price) > 0

    # No look-ahead: cut the data at T and every trade decided and closed before T is unchanged.
    data = _load_mtf_csvs(tmp_path, "MES")
    since = data[Timeframe.M5].timestamp.tail(2880).iloc[0]
    full = chart_setups(data, cfg, [], since, SRZoneConfig())
    cut = pd.Timestamp("2026-10-14 06:00")
    past = chart_setups({tf: df[df.timestamp < cut] for tf, df in data.items()}, cfg, [], since,
                        SRZoneConfig())
    for a, b in zip(full, past, strict=True):
        before = a["trades"][a["trades"].t_out < cut].reset_index(drop=True)
        assert len(before) > 0
        pd.testing.assert_frame_equal(before, b["trades"][b["trades"].t_out < cut]
                                      .reset_index(drop=True))


def test_sr_levels_without_daily_bars_still_publishes_zones_and_setups(mes, tmp_path):
    """Setup B's daily trend filter turns off instead of blocking the chart."""
    from rainier.cli import cli

    for tf, df in mes.items():
        if tf != Timeframe.D1:
            df.to_csv(tmp_path / f"MES_{tf.value}.csv", index=False)
    out = tmp_path / "payload.json"
    r = CliRunner().invoke(cli, ["sr-levels", "--data-dir", str(tmp_path), "--format", "json",
                                 "--output", str(out)])
    assert r.exit_code == 0, r.output
    payload = json.loads(out.read_text())
    assert payload["zones"] and [s["key"] for s in payload["xj"]["setups"]] == ["A", "B"]
