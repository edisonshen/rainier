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
from rainier.analysis.pivots import compute_atr
from rainier.analysis.sr_eval import closed_bars, scorecard, walk_forward
from rainier.analysis.sr_zones import Zone, consolidate_zones, detect_mtf_zones, zones_to_levels
from rainier.core.config import AnalysisConfig, ChartPublishConfig, SRZoneConfig
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


def test_chart_keeps_nearest_level_on_each_side_beyond_distance_window(mes):
    """Chart config always shows a level above and below price; signal config does not."""
    price = float(mes[Timeframe.M5].close.iloc[-1])
    all_zones = detect_mtf_zones(_htf(mes), SRZoneConfig())
    above = min((z.price for z in all_zones if z.price > price), key=lambda p: p - price)
    below = max(z.price for z in all_zones if z.price <= price)

    narrow = SRZoneConfig(max_distance_atr_mult=0.01)
    assert not detect_mtf_zones(_htf(mes), narrow, price)
    chart = narrow.model_copy(update={"keep_nearest_each_side": True})
    assert sorted(z.price for z in detect_mtf_zones(_htf(mes), chart, price)) == [below, above]

    # Price above every zone: only a support can be shown, and it is the closest one.
    top = max(z.price for z in all_zones) + 500
    (z,) = detect_mtf_zones(_htf(mes), chart, top)
    assert z.price == max(z.price for z in all_zones)


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


def test_cli_sr_levels_publishes_chart_payload_to_kv(mes, tmp_path, monkeypatch):
    """fengshen.dev /trading/mes reads this KV value: candles per TF + zones."""
    from rainier.cli import cli
    from rainier.publish import cloudflare_kv

    for tf, df in mes.items():
        df.to_csv(tmp_path / f"MES_{tf.value}.csv", index=False)
    calls, store = [], {}

    class _Resp:
        def __init__(self, status_code=200, text=""):
            self.status_code, self.text = status_code, text

        def raise_for_status(self):
            pass

    def _put(url, **kw):
        calls.append((url, kw))
        store[url] = kw["content"].decode()
        return _Resp()

    monkeypatch.setattr(cloudflare_kv.httpx, "put", _put)
    monkeypatch.setattr(cloudflare_kv.httpx, "get",
                        lambda url, **kw: _Resp(200, store[url]) if url in store else _Resp(404))
    runner = CliRunner()
    r = runner.invoke(cli, ["sr-levels", "--data-dir", str(tmp_path), "--publish-kv"])
    assert r.exit_code != 0 and "CLOUDFLARE_API_TOKEN" in r.output and not calls

    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    r = runner.invoke(cli, ["sr-levels", "--data-dir", str(tmp_path), "--publish-kv"])
    assert r.exit_code == 0, r.output
    (url, kw), = calls
    assert url.endswith("/accounts/acct/storage/kv/namespaces/"
                        "0d0281e0d0ff49b4868a42aa2a39f4ec/values/mes%3Asr")
    assert kw["headers"]["Authorization"] == "Bearer tok"
    payload = json.loads(kw["content"])
    assert set(payload["candles"]) == {"5m", "1H", "4H", "1D"}
    assert len(payload["candles"]["5m"]) == 2880  # ~2 weeks of 23h sessions
    t, o, h, lo, c = payload["candles"]["5m"][-1]
    assert lo <= min(o, c) <= max(o, c) <= h
    # The page formats times as PT itself, so the payload must carry real UTC, not the
    # loader's naive PT wall clock (that showed "as of 04:35 PT" for an 11:35 PT bar).
    last_bar = pd.Timestamp(mes[Timeframe.M5].timestamp.iloc[-1])
    assert t == int(last_bar.timestamp())
    assert pd.Timestamp(payload["asof"]) == last_bar and payload["asof"].endswith("+00:00")
    assert payload["zones"] and payload["price"] == mes[Timeframe.M5].close.iloc[-1]
    # The chart keeps levels the 4-ATR signal filter drops (e.g. a 4H+1H support 8 ATR below).
    atr = float(compute_atr(mes[Timeframe.H1].iloc[-500:]).iloc[-1])
    dist = [abs(z["price"] - payload["price"]) / atr for z in payload["zones"]]
    assert max(dist) > SRZoneConfig().max_distance_atr_mult
    assert max(dist) <= ChartPublishConfig().max_distance_atr_mult
    assert {z["role"] for z in payload["zones"]} == {"support", "resistance"}
    # Next 5-min run with no new bar (daily halt, holiday): identical payload, no KV write.
    r = runner.invoke(cli, ["sr-levels", "--data-dir", str(tmp_path), "--publish-kv"])
    assert r.exit_code == 0 and "unchanged" in r.output and len(calls) == 1
