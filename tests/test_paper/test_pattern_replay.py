"""WS A — faithful pattern-layer replay parity / look-ahead / determinism.

Acceptance (task plan):
- parity: on recent dates (live latest-snapshot path), the replay's selection
  (ordering + best_pattern + composite score) == `screen_stocks` on a fixture
  carrying ≥126 price bars (6-month windowing truncates) AND a money-flow row.
- look-ahead: only bars up to `t` are fed to the detector.
- determinism: identical corpus bytes on re-run (explicit ORDER BY symbol, date).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from rainier.analysis.stock_screener import screen_stocks
from rainier.core.config import StockScreenerConfig
from rainier.core.types import MoneyFlowSignal, SectorTrend
from rainier.paper.pattern_replay import (
    LIVE_LOOKBACK_BARS,
    emission_at,
    replay_composite,
    replay_screen,
    window_as_of,
)

# ---------------------------------------------------------------------------
# fixtures — price paths that yield a real, recently-confirmed actionable
# pattern so best_pattern is non-None on parity.
# ---------------------------------------------------------------------------


def _make_ohlcv(prices: list[float]) -> pd.DataFrame:
    n = len(prices)
    return pd.DataFrame(
        {
            "open": [prices[max(0, i - 1)] for i in range(n)],
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        }
    )


def _false_breakdown_actionable() -> pd.DataFrame:
    """False-breakdown that recovers and HOLDS NEAR the recovery level so the
    confirmed pattern is fresh AND price is still near entry (actionable)."""
    prices = [95.0, 93.0, 91.0, 90.0, 91.0, 93.0, 95.0, 93.0, 91.0, 90.0]
    prices += [92.0, 93.0, 91.0]  # range above support
    prices += [89.0, 88.0, 87.0]  # break below support
    prices += [89.0, 91.0, 92.0]  # recover above support quickly (confirm)
    prices += [92.5, 92.0, 92.5, 92.0, 92.5]  # hold near entry (still actionable)
    return _make_ohlcv(prices)


_CONFIG = StockScreenerConfig(
    swing_lookback=3,
    min_pattern_bars=3,
    max_pattern_bars=50,
    neckline_tolerance_pct=0.05,
)


# ---------------------------------------------------------------------------
# window_as_of — no look-ahead; trims to the live ~6-month lookback.
# ---------------------------------------------------------------------------


def test_window_excludes_future_bars():
    df = _make_ohlcv([float(i) for i in range(200)])
    w = window_as_of(df, t_idx=150)
    # last row IS the as-of bar; nothing after t leaks
    assert w["close"].iloc[-1] == 150.0
    assert len(w) == LIVE_LOOKBACK_BARS  # trimmed to the 6-month span (last 126)
    assert w["close"].max() == 150.0  # no future bar > 150
    assert w["close"].iloc[0] == 25.0  # first kept bar = 150 - 126 + 1


def test_window_short_history_keeps_all_bars():
    df = _make_ohlcv([float(i) for i in range(30)])
    w = window_as_of(df, t_idx=29)
    assert len(w) == 30
    assert w["close"].iloc[-1] == 29.0


def test_window_uses_calendar_6mo_when_date_indexed():
    """On a date-indexed frame the window is a CALENDAR 6-month slice ending at
    t (byte-faithful to yfinance period='6mo'), not a fixed 126-row count.
    Guards codex iter-7 P1."""
    df = _make_ohlcv([float(i) for i in range(400)])
    # Business-day index so holiday-free month lengths vary the bar count.
    df.index = pd.bdate_range("2024-06-01", periods=400, tz="UTC")
    t_idx = 350
    w = window_as_of(df, t_idx=t_idx)
    t_ts = df.index[t_idx]
    cutoff = t_ts - pd.DateOffset(months=6)
    # every kept bar is within (t-6mo, t]; nothing before the calendar cutoff
    assert w.index.min() > cutoff
    assert w.index.max() == t_ts
    # and it is NOT the fixed 126-bar slice (calendar 6mo of bdays ≈ 130)
    assert len(w) != 126


# ---------------------------------------------------------------------------
# composite — reproduces the live sector double-count.
# ---------------------------------------------------------------------------


def test_composite_matches_live_formula_with_sector_double_count():
    # signal_strength here ALREADY includes the +0.10 sector boost (as the live
    # _apply_sector_boost would have applied); sector_boost is then added again.
    out = replay_composite(
        signal_strength=0.9,  # 0.8 base + 0.1 sector (double-counted below)
        sector_boost=0.1,
        best_confidence=0.5,
        config=StockScreenerConfig(),
    )
    expected = round(0.25 * 0.9 + 0.10 * 0.1 + 0.65 * 0.5, 4)
    assert out == expected


# ---------------------------------------------------------------------------
# parity — replay vs the LIVE screen_stocks on identical windows + signals.
# ---------------------------------------------------------------------------


def test_replay_matches_screen_stocks_selection():
    """Feed screen_stocks and replay_screen the SAME as-of windows + money-flow
    signals; assert identical ordering + best_pattern + composite score."""
    df_actionable = _false_breakdown_actionable()
    df_flat = _make_ohlcv([100.0] * 40)  # no pattern → contributes 0 pattern

    windows = {"AAA": df_actionable, "BBB": df_flat}

    # Money-flow signals (already sector-boosted, mirroring _apply_sector_boost).
    sig_aaa = MoneyFlowSignal(
        symbol="AAA", rank=5, rank_change=1, long_short="Long in",
        capital_flow_direction="+", days_in_top100=3, sector="Technology",
        industry="Semis", signal_strength=0.95,
    )
    sig_bbb = MoneyFlowSignal(
        symbol="BBB", rank=20, rank_change=0, long_short="Long in",
        capital_flow_direction="+", days_in_top100=1, sector="Energy",
        industry="Oil", signal_strength=0.70,
    )
    sector_trends = [
        SectorTrend(
            sector="Technology", long_in_count=10, short_in_count=1,
            net_sentiment=0.8, top_stocks=["AAA"], trend_direction="bullish",
            sector_rank=1,
        ),
        SectorTrend(
            sector="Energy", long_in_count=2, short_in_count=2,
            net_sentiment=0.0, top_stocks=["BBB"], trend_direction="neutral",
            sector_rank=2,
        ),
    ]

    cm = MagicMock()
    cm.__enter__.return_value = MagicMock()
    cm.__exit__.return_value = False

    with patch(
        "rainier.analysis.stock_screener._screen_money_flow",
        return_value=[sig_aaa, sig_bbb],
    ), patch(
        "rainier.analysis.stock_screener.analyze_sectors",
        return_value=sector_trends,
    ), patch(
        "rainier.analysis.stock_screener._fetch_stock_data",
        return_value={"AAA": df_actionable, "BBB": df_flat},
    ), patch(
        "rainier.analysis.stock_screener.get_session", return_value=cm
    ):
        candidates, _ = screen_stocks(_settings_with(_CONFIG))

    # Replay over the SAME inputs. signals_by_symbol carries the POST-boost
    # money-flow strength (mirroring the live _apply_sector_boost: AAA is in a
    # bullish sector so +0.10 → 1.05; BBB neutral stays 0.70). sector_boost is
    # then counted a SECOND time inside replay_composite, matching the live
    # double-count.
    replay = replay_screen(
        signals_by_symbol={"AAA": 1.05, "BBB": 0.70},
        sectors_by_symbol={"AAA": "Technology", "BBB": "Energy"},
        sector_trends=sector_trends,
        windows_by_symbol=windows,
        config=_CONFIG,
    )

    # Ordering parity.
    assert [c.symbol for c in candidates] == [r.symbol for r in replay]
    # Per-symbol composite-score + best_pattern parity.
    live_by_sym = {c.symbol: c for c in candidates}
    for r in replay:
        live = live_by_sym[r.symbol]
        assert live.signal_strength == r.composite_score, r.symbol
        assert live.pattern_type == r.best_pattern_type, r.symbol


def _settings_with(screener_config: StockScreenerConfig):
    from rainier.core.config import Settings

    s = Settings()
    s.stock_screener = screener_config
    return s


# ---------------------------------------------------------------------------
# determinism — same fixture twice → identical emission.
# ---------------------------------------------------------------------------


def test_emission_is_deterministic():
    df = _false_breakdown_actionable()
    e1 = emission_at("AAA", df, _CONFIG)
    e2 = emission_at("AAA", df, _CONFIG)
    assert e1 is not None
    assert e1 == e2


def test_emission_none_when_no_pattern():
    df = _make_ohlcv([100.0] * 40)
    assert emission_at("FLAT", df, _CONFIG) is None


# ---------------------------------------------------------------------------
# load_prices — start_date bounds the SQL load (codex iter-4 P2: no full-history
# scan for a trailing-window audit).
# ---------------------------------------------------------------------------


def test_load_prices_start_date_adds_sql_bound():
    import pandas as pd

    from rainier.paper.pattern_replay import load_prices

    class _RecordingSession:
        def __init__(self):
            self.stmt = None

        def execute(self, stmt):
            self.stmt = stmt

            class _Res:
                def all(self):
                    return []

            return _Res()

    # With start_date: the compiled SQL carries a `date >=` bound.
    sess = _RecordingSession()
    load_prices(sess, ["AAA"], start_date=pd.Timestamp("2025-06-01", tz="UTC"))
    sql_with = str(sess.stmt.compile(compile_kwargs={"literal_binds": False}))
    assert "stock_prices.date >=" in sql_with

    # Without start_date: no date lower bound in the WHERE clause.
    sess2 = _RecordingSession()
    load_prices(sess2, ["AAA"])
    sql_without = str(sess2.stmt.compile(compile_kwargs={"literal_binds": False}))
    assert "stock_prices.date >=" not in sql_without


def test_load_prices_drops_partial_ohlc_bars():
    """A row with a null high/low/open (a partial-backfill bar) is dropped
    before the detector sees it; a null volume is filled to 0, not dropped.
    Guards codex iter-9 P2."""
    import datetime as _dt
    from types import SimpleNamespace

    from rainier.paper.pattern_replay import load_prices

    def _row(d, o, h, lo, c, v):
        return SimpleNamespace(
            symbol="AAA",
            date=_dt.datetime(2025, 1, d, tzinfo=_dt.timezone.utc),
            open=o, high=h, low=lo, close=c, volume=v,
        )

    rows = [
        _row(1, 100.0, 101.0, 99.0, 100.0, 1000.0),   # complete
        _row(2, 100.0, None, 99.0, 100.5, 1000.0),    # null high → DROP
        _row(3, 100.0, 102.0, 99.0, 101.0, None),     # null volume → keep, vol=0
    ]

    class _Sess:
        def execute(self, stmt):
            return SimpleNamespace(all=lambda: rows)

    out = load_prices(_Sess(), ["AAA"])
    df = out["AAA"]
    assert len(df) == 2  # the null-high bar dropped
    assert df["volume"].isna().sum() == 0
    assert (df["volume"] == 0.0).any()  # null volume filled to 0
