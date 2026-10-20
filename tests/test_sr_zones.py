"""Spec tests for multi-TF S/R zones (1D -> 4H -> 1H, consolidated) on real MES bars.

Fixture ``mes_sr_small.parquet``: MES 1D (Jul 2025+), 4H (Feb 2026+), 1H (Jul 2026+) and
5m (Sep 8 - Oct 20 2026) from the canonical candles table + yfinance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from rainier.analysis.analyzer import analyze_multi_tf
from rainier.analysis.sr_eval import closed_bars, scorecard, walk_forward
from rainier.analysis.sr_zones import Zone, consolidate_zones, detect_mtf_zones, zones_to_levels
from rainier.core.config import AnalysisConfig, SRZoneConfig
from rainier.core.types import SRRole, Timeframe

FIXTURE = Path(__file__).parent / "fixtures" / "mes_sr_small.parquet"
HTF = (Timeframe.D1, Timeframe.H4, Timeframe.H1)


@pytest.fixture(scope="module")
def mes() -> dict[Timeframe, pd.DataFrame]:
    df = pd.read_parquet(FIXTURE)
    return {
        Timeframe(tf): g.drop(columns="timeframe").sort_values("timestamp").reset_index(drop=True)
        for tf, g in df.groupby("timeframe")
    }


def _htf(data):
    return {tf: data[tf] for tf in HTF}


def test_zones_are_point_in_time(mes):
    """Zones drawn at T are identical whether or not bars after T exist."""
    cfg = SRZoneConfig()
    now = pd.Timestamp("2026-10-01 14:00", tz="UTC")
    truncated = {tf: df[df.timestamp < now] for tf, df in _htf(mes).items()}
    a = detect_mtf_zones(closed_bars(_htf(mes), now), cfg, 6700.0)
    b = detect_mtf_zones(closed_bars(truncated, now), cfg, 6700.0)
    assert a and [(z.price, z.low, z.high) for z in a] == [(z.price, z.low, z.high) for z in b]
    last_closed = max(df.timestamp.max() for df in closed_bars(_htf(mes), now).values())
    assert last_closed < now


def test_consolidation_tunes_close_lines_and_bands_nearby_ones():
    cfg = SRZoneConfig()  # ref ATR 10 -> tight 2.5, band 7.5
    zones = consolidate_zones({
        Timeframe.D1: [Zone(100.0, 99.0, 101.0, 1.0, 3, 1)],
        Timeframe.H4: [Zone(101.0, 100.5, 101.5, 1.0, 2, 0)],   # within tight: same line
        Timeframe.H1: [Zone(105.0, 104.5, 105.5, 1.0, 2, 0),    # within band: widen
                       Zone(130.0, 129.0, 131.0, 1.0, 2, 0)],   # far: its own zone
    }, ref_atr=10.0, cfg=cfg)
    assert len(zones) == 2
    top = next(z for z in zones if z.price == 100.0)  # highest TF keeps the price
    assert top.timeframes == [Timeframe.D1, Timeframe.H4, Timeframe.H1]
    assert top.low == 99.0 and top.high == 105.5
    assert next(z for z in zones if z.price == 130.0).timeframes == [Timeframe.H1]


def test_levels_have_role_bounds_and_timeframes(mes):
    price = float(mes[Timeframe.M5].close.iloc[-1])
    levels = zones_to_levels(detect_mtf_zones(_htf(mes), SRZoneConfig(), price), price)
    assert levels
    for lv in levels:
        assert lv.zone_low <= lv.price <= lv.zone_high
        assert lv.role == (SRRole.SUPPORT if lv.price < price else SRRole.RESISTANCE)
        assert lv.timeframes and lv.source_tf == lv.timeframes[0]
    assert any(len(set(lv.timeframes)) >= 2 for lv in levels)


def test_analyze_multi_tf_uses_zones_by_default_and_legacy_on_request(mes):
    zones = analyze_multi_tf(mes, "MES", Timeframe.M5, AnalysisConfig())
    assert zones.sr_levels and all(lv.zone_low is not None for lv in zones.sr_levels)
    legacy = analyze_multi_tf(mes, "MES", Timeframe.M5, AnalysisConfig(mtf_sr_method="legacy"))
    assert all(lv.zone_low is None for lv in legacy.sr_levels)


def test_walk_forward_scorecard_on_5m(mes):
    """The user's acceptance test: higher-TF zones touched by a 5m rejection candle, scored
    after they were drawn, vs the same zones randomly shifted. Earlier = tune, later = holdout."""
    cfg = SRZoneConfig()
    outcomes = walk_forward(
        _htf(mes), mes[Timeframe.M5], lambda past, px: detect_mtf_zones(past, cfg, px),
        pd.Timestamp("2026-10-01", tz="UTC"), step=96,
    )
    card = scorecard(outcomes)
    assert set(card.index.get_level_values("period")) == {"tune", "holdout"}
    assert (card.touches > 10).all()
    assert card.bounce_rate.between(0, 1).all() and card.random_rate.between(0, 1).all()
    # Random control is seeded per step, so the scorecard is reproducible.
    again = scorecard(walk_forward(
        _htf(mes), mes[Timeframe.M5], lambda past, px: detect_mtf_zones(past, cfg, px),
        pd.Timestamp("2026-10-01", tz="UTC"), step=96,
    ))
    pd.testing.assert_frame_equal(card, again)


def test_cli_sr_levels_json_and_pine(mes, tmp_path):
    from rainier.cli import cli

    for tf, df in mes.items():
        df.to_csv(tmp_path / f"MES_{tf.value}.csv", index=False)
    runner = CliRunner()
    js = tmp_path / "zones.json"
    r = runner.invoke(cli, ["sr-levels", "--data-dir", str(tmp_path), "--format", "json",
                            "--output", str(js)])
    assert r.exit_code == 0, r.output
    payload = json.loads(js.read_text())
    assert payload["symbol"] == "MES" and payload["zones"]
    assert {"price", "low", "high", "role", "timeframes"} <= payload["zones"][0].keys()

    pine = tmp_path / "zones.pine"
    r = runner.invoke(cli, ["sr-levels", "--data-dir", str(tmp_path), "--format", "pine",
                            "--output", str(pine)])
    assert r.exit_code == 0, r.output
    src = pine.read_text()
    assert src.startswith("//@version=5") and src.count("line.new(") == len(payload["zones"])
