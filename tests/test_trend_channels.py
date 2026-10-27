"""Spec tests for Xiaojiang-style trend channels + trend lines (chart overlay, 4H/1D).

``trend/tsla_1d_2021_2024-02-15.csv``: TSLA daily bars up to Feb 15 2024, the date of a
Xiaojiang TSLA chart with a down channel from the Apr 2022 high and an up channel from the
Jan 2023 low that breaks in Jan 2024. ``mes_sr_small.parquet``: real MES bars (see
test_sr_zones.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from rainier.analysis.trend_channels import find_channels, trend_overlays
from rainier.core.config import ChartPublishConfig, TrendTimeframeConfig
from rainier.core.types import Timeframe
from rainier.viz.chart_payload import sr_chart_payload

FIXTURES = Path(__file__).parent / "fixtures"
TSLA = FIXTURES / "trend" / "tsla_1d_2021_2024-02-15.csv"


@pytest.fixture(scope="module")
def mes() -> dict[Timeframe, pd.DataFrame]:
    df = pd.read_parquet(FIXTURES / "mes_sr_small.parquet")
    return {
        Timeframe(tf): g.drop(columns="timeframe").sort_values("timestamp").reset_index(drop=True)
        for tf, g in df.groupby("timeframe")
    }


def test_reproduces_xiaojiang_tsla_channels():
    df = pd.read_csv(TSLA, parse_dates=["timestamp"])
    since_2022 = int((df.timestamp >= "2022-01-01").sum())
    cfg = TrendTimeframeConfig(leg_atr=15, swing_bars=10, lookback=since_2022, legs=3)
    day = lambda j: str(df.timestamp[j].date())  # noqa: E731

    down, up = find_channels(df, cfg)

    # Down leg: from the Apr 2022 high through the Sep 2022 high, ~145 wide, broken upward.
    assert (down.direction, day(down.anchor), round(down.anchor_price, 1)) == (-1, "2022-04-05",
                                                                               384.3)
    assert [day(j) for j in down.base_touches] == ["2022-04-05", "2022-09-21"]
    assert -150 < down.width < -140 and day(down.broken) == "2023-06-05"
    # Up leg: from the Jan 2023 low (101.8) through the Oct 31 2023 low, broken on Jan 22 2024.
    assert (up.direction, day(up.anchor), round(up.anchor_price, 1)) == (1, "2023-01-06", 101.8)
    assert "2023-10-31" in [day(j) for j in up.base_touches]
    assert len(up.parallel_touches) >= 2 and 100 < up.width < 106
    assert day(up.broken) == "2024-01-22"


def _mes_overlays(data):
    asof = data[Timeframe.M5].timestamp.iloc[-1]
    return trend_overlays(data, ChartPublishConfig().trend, asof), asof


def test_mes_overlays_use_only_closed_bars(mes):
    found, asof = _mes_overlays(mes)
    assert set(found) == {Timeframe.H4, Timeframe.D1}
    assert all(chs or lines for chs, lines in found.values())

    # The still-forming 4H/1D bar (closes after asof) must not move anything: spike it.
    spiked = dict(mes)
    for tf in found:
        df = mes[tf].copy()
        assert df.timestamp.iloc[-1] + pd.Timedelta(hours=4) > asof  # forming bar present
        df.loc[df.index[-1], ["high", "low", "close"]] = [1e5, 1.0, 1e5]
        spiked[tf] = df
    assert _mes_overlays(spiked)[0] == found


def test_chart_payload_draws_channels_and_lines_in_the_window(mes):
    found, asof = _mes_overlays(mes)
    bars = ChartPublishConfig().bars
    payload = sr_chart_payload("MES", asof.to_pydatetime(), 7800.0, [], mes, bars,
                               trend=found, trend_extend_bars=10)
    trend = json.loads(json.dumps(payload))["trend"]
    assert set(trend) == {"4H", "1D"}
    for tf, overlay in trend.items():
        candle_t = [r[0] for r in payload["candles"][tf]]
        step = candle_t[-1] - candle_t[-2]
        for item in overlay["channels"] + overlay["lines"]:
            t = [r[0] for r in item["rows"]]
            assert t == sorted(set(t)) and t[0] >= candle_t[0] and t[0] in candle_t
            if item.get("broken"):  # stops at the break bar, no projection
                assert len(t) == 2 and t[1] in candle_t
            else:  # last bar, then one row per projected bar
                assert t[1] == candle_t[-1] and t[2:] == [candle_t[-1] + step * k
                                                          for k in range(1, 11)]
        for ch in overlay["channels"]:
            widths = [par - base for _, base, par in ch["rows"]]  # parallel: one width
            assert np.ptp(widths) <= 0.02 and np.sign(widths[0]) == ch["dir"]
    # As of Oct 20: the 1D up channel from the Jun 11 low is intact, both 4H down channels broke.
    assert [(c["dir"], c["broken"]) for c in trend["1D"]["channels"]][-1] == (1, False)
    assert all(c["broken"] for c in trend["4H"]["channels"])


def test_cli_sr_levels_json_carries_trend(mes, tmp_path):
    from rainier.cli import cli

    for tf, df in mes.items():
        out = df.copy()
        out["timestamp"] = out.timestamp.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
        out.to_csv(tmp_path / f"MES_{tf.value}.csv", index=False)
    js = tmp_path / "out.json"
    res = CliRunner().invoke(cli, ["sr-levels", "--symbol", "MES", "--data-dir", str(tmp_path),
                                   "--format", "json", "--output", str(js)])
    assert res.exit_code == 0, res.output
    trend = json.loads(js.read_text())["trend"]
    assert set(trend) == {"4H", "1D"} and trend["1D"]["channels"]
