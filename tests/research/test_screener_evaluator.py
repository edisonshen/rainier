"""Replay + score evaluator — corpus driver, BasketOutcomes, base scorecard.

Task ab-replay-evaluator-edb5 (design §4.4/§4.5/§11.4). Composes the shipped
pattern replay + as-of selectors into per-day BasketOutcomes → rewards → the
candidate-agnostic base scorecard, driven by an experiment spec.

Each test names the contract/bug it guards:
- parity: driver selection == live screen_stocks on one pinned fixture.
- champion baseline: challenger differs from the live champion only in the
  overridden key, NOT from pydantic code-defaults (the §4.5 silent no-op).
- basket coverage: money-flow-only selected symbol → fwd_return[H] key present
  (None), never omitted; dodged_loss sees the wider screened pool.
- co-evaluation: one run emits champion + all challengers with equal corpus_hash.
- no look-ahead: bars after t never move the as-of ranking.
- regime slices: per-regime scores match a hand-sliced fixture; DSR is null.
- windows: train/holdout disjoint; embargo purge gap respected.
- unknown reward name → loud error.
- scorecard validates against base_scorecard (no LLM fields).
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from rainier.core.config import Settings, StockScreenerConfig
from rainier.core.types import MoneyFlowSignal, SectorTrend
from rainier.research import output_schema
from rainier.research.evaluator import screener
from rainier.research.evaluator.screener import (
    build_basket_outcomes,
    corpus_hash,
    evaluator_sha,
    load_base_overrides,
    run_experiment,
    split_windows,
)
from rainier.research.experiment import ExperimentWindow, materialize_challengers, parse_spec
from rainier.research.rewards.basket import dodged_loss

# ---------------------------------------------------------------------------
# fixtures — date-indexed OHLCV; money-flow + sector inputs pinned so the
# ranking is identical between the live path and the driver.
# ---------------------------------------------------------------------------

# min_daily_bars=10: the synthetic fixtures below are 24-40 bars by design, so
# pin the live Layer-3 gate below them to keep them pattern-eligible (the driver
# mirrors _fetch_stock_data's min_daily_bars cut — see the short-history test).
_CONFIG = StockScreenerConfig(
    swing_lookback=3, min_pattern_bars=3, max_pattern_bars=50,
    neckline_tolerance_pct=0.05, min_daily_bars=10,
)


def _ohlcv(prices: list[float], *, end: str = "2026-03-31") -> pd.DataFrame:
    n = len(prices)
    df = pd.DataFrame(
        {
            "open": [prices[max(0, i - 1)] for i in range(n)],
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        }
    )
    df.index = pd.date_range(end=end, periods=n, freq="D", tz="UTC")
    return df


def _false_breakdown() -> list[float]:
    p = [95.0, 93.0, 91.0, 90.0, 91.0, 93.0, 95.0, 93.0, 91.0, 90.0]
    p += [92.0, 93.0, 91.0, 89.0, 88.0, 87.0, 89.0, 91.0, 92.0]
    p += [92.5, 92.0, 92.5, 92.0, 92.5]
    return p


def _signal(symbol, strength, *, rank=5, sector="Technology") -> MoneyFlowSignal:
    return MoneyFlowSignal(
        symbol=symbol, rank=rank, rank_change=1, long_short="Long in",
        capital_flow_direction="+", days_in_top100=3, sector=sector,
        industry="X", signal_strength=strength,
    )


def _sector_trends() -> list[SectorTrend]:
    return [
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


def _settings_with(config: StockScreenerConfig) -> Settings:
    s = Settings()
    s.stock_screener = config
    return s


def _spec(challengers, *, primary="sharpe", guardrails=("max_drawdown",),
          train="2026-01-01..2026-03-31", holdout="2026-05-01..2026-06-25",
          embargo_days=20):
    raw = {
        "id": "exp-test",
        "status": "active",
        "champion": "champion.yaml",
        "layer": "layer_weights",
        "challengers": challengers,
        "primary": primary,
        "guardrails": list(guardrails),
        "window": {"train": train, "holdout": holdout, "embargo_days": embargo_days},
    }
    return parse_spec(raw, source="<test>")


# ---------------------------------------------------------------------------
# parity — driver's per-day selection == live screen_stocks (pinned fixture).
# ---------------------------------------------------------------------------


def test_parity_driver_selection_matches_live_screen_stocks():
    df_a = _ohlcv(_false_breakdown())              # actionable pattern
    df_b = _ohlcv([100.0] * 40)                    # flat → no pattern
    frames = {"AAA": df_a, "BBB": df_b}
    as_of = df_a.index[-1].date()

    raw = [_signal("AAA", 0.95, sector="Technology"),
           _signal("BBB", 0.70, rank=20, sector="Energy")]
    trends = _sector_trends()

    cm = MagicMock()
    cm.__enter__.return_value = MagicMock()
    cm.__exit__.return_value = False
    with patch("rainier.analysis.stock_screener._screen_money_flow", return_value=raw), \
         patch("rainier.analysis.stock_screener.analyze_sectors", return_value=trends), \
         patch("rainier.analysis.stock_screener._fetch_stock_data", return_value=frames), \
         patch("rainier.analysis.stock_screener.get_session", return_value=cm):
        from rainier.analysis.stock_screener import screen_stocks
        candidates, _ = screen_stocks(_settings_with(_CONFIG))

    with patch.object(screener, "_screen_money_flow_as_of", return_value=raw), \
         patch.object(screener, "analyze_sectors_at", return_value=trends):
        outcomes = build_basket_outcomes(
            session=object(), config=_CONFIG, days=[as_of],
            prices_by_symbol=frames, regime_fn=lambda d: "bull",
            basket_size=10,
        )

    day = outcomes.days[0]
    # Identical ranking (ordering) between the driver basket and the live screen.
    assert day.symbols == [c.symbol for c in candidates]


# ---------------------------------------------------------------------------
# champion baseline — the §4.5 silent no-op guard: a challenger differs from the
# LIVE champion only in the overridden key, never from pydantic code-defaults.
# ---------------------------------------------------------------------------


def test_champion_baseline_from_live_settings_not_code_defaults(tmp_path):
    # settings.yaml carries a non-default layer weight; champion.yaml a
    # non-default threshold. The live champion is their merge — NOT the defaults.
    (tmp_path / "settings.yaml").write_text(
        "stock_screener:\n  layer_weight_pattern: 0.70\n"
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "champion.yaml").write_text(
        "version: 1\nstrong_buy_threshold: 0.85\n"
    )

    base = load_base_overrides(tmp_path / "settings.yaml")
    assert base["layer_weight_pattern"] == 0.70   # from settings.yaml
    assert base["strong_buy_threshold"] == 0.85   # from champion.yaml

    spec = _spec([{"id": "bt60", "override": {"buy_threshold": 0.60}}])
    champ, challengers = materialize_challengers(spec, base)
    chall = challengers["bt60"]

    # champion inherits the LIVE layer, not code-defaults (default lw_pattern=0.65).
    assert champ.layer_weight_pattern == 0.70
    assert champ.strong_buy_threshold == 0.85
    # challenger differs from champion ONLY in the overridden key.
    assert chall.buy_threshold == 0.60
    assert chall.layer_weight_pattern == champ.layer_weight_pattern
    assert chall.strong_buy_threshold == champ.strong_buy_threshold


# ---------------------------------------------------------------------------
# basket coverage — money-flow-only selected symbol; wider screened pool.
# ---------------------------------------------------------------------------


def test_money_flow_only_symbol_and_dodged_loss_pool():
    df_win = _ohlcv([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])   # SEL: +5d
    df_loser = _ohlcv([100.0, 99.0, 98.0, 97.0, 90.0, 80.0])      # LOSER: -20%
    frames = {"SEL": df_win, "LOSER": df_loser}    # MFONLY has NO price frame
    as_of = df_win.index[0].date()   # t=0 so 5 bars ahead exist

    # MFONLY outranks LOSER on money-flow so it lands in the top-2 basket
    # despite having no price (best_confidence=0), exactly like the live screen.
    raw = [_signal("SEL", 0.95), _signal("MFONLY", 0.90), _signal("LOSER", 0.30)]

    with patch.object(screener, "_screen_money_flow_as_of", return_value=raw), \
         patch.object(screener, "analyze_sectors_at", return_value=[]):
        outcomes = build_basket_outcomes(
            session=object(), config=_CONFIG, days=[as_of],
            prices_by_symbol=frames, regime_fn=lambda d: "bull",
            basket_size=2, horizons=(5,),
        )

    day = outcomes.days[0]
    assert "MFONLY" in day.symbols              # selected despite no price
    # every selected symbol is a fwd_return key (None for the price-less one)
    assert "MFONLY" in day.fwd_return[5]
    assert day.fwd_return[5]["MFONLY"] is None
    # the wider screened pool carries the non-selected LOSER, so dodged_loss > 0
    assert "LOSER" in day.fwd_return[5]
    assert "LOSER" not in day.symbols
    assert dodged_loss(outcomes, horizon=5) > 0.0


# ---------------------------------------------------------------------------
# co-evaluation — one run scores champion + all challengers, equal corpus_hash.
# ---------------------------------------------------------------------------


def _patch_selectors(raw, trends=()):
    return (
        patch.object(screener, "_screen_money_flow_as_of", return_value=raw),
        patch.object(screener, "analyze_sectors_at", return_value=list(trends)),
    )


def test_co_evaluation_equal_corpus_hash_all_candidates():
    frames = {"AAA": _ohlcv([100.0] * 40)}
    days = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    raw = [_signal("AAA", 0.9)]
    spec = _spec(
        [
            {"id": "mf35", "override": {"layer_weight_money_flow": 0.35}},
            {"id": "mf40", "override": {"layer_weight_money_flow": 0.40}},
        ],
        primary="total_return", guardrails=(),
    )
    p1, p2 = _patch_selectors(raw)
    with p1, p2:
        cards = run_experiment(
            spec, base_overrides=None, session=object(),
            prices_by_symbol=frames, trading_days=days,
            regime_fn=lambda d: "bull", segment="train",
        )
    ids = {c["candidate_id"] for c in cards}
    assert ids == {"champion", "mf35", "mf40"}
    hashes = {c["corpus_hash"] for c in cards}
    assert len(hashes) == 1                    # one shared corpus snapshot
    # corpus_hash is the shared (symbol,date,close) content hash, not per-basket.
    assert next(iter(hashes)) == corpus_hash(frames)


# ---------------------------------------------------------------------------
# no look-ahead — bars strictly after t never move the as-of ranking.
# ---------------------------------------------------------------------------


def test_no_look_ahead_future_bars_do_not_change_selection():
    base_prices = _false_breakdown()
    df = _ohlcv(base_prices)
    as_of = df.index[-1].date()
    raw = [_signal("AAA", 0.95), _signal("BBB", 0.70, rank=20)]
    frames_now = {"AAA": df, "BBB": _ohlcv([100.0] * 40)}

    with (p := _patch_selectors(raw))[0], p[1]:
        before = build_basket_outcomes(
            session=object(), config=_CONFIG, days=[as_of],
            prices_by_symbol=frames_now, regime_fn=lambda d: "bull", basket_size=10,
        ).days[0]

    # append wildly different FUTURE bars (dates strictly after as_of)
    future = _ohlcv(base_prices + [200.0, 5.0, 250.0, 1.0, 300.0], end="2026-04-07")
    frames_future = {"AAA": future, "BBB": _ohlcv([100.0] * 45, end="2026-04-07")}
    with (p := _patch_selectors(raw))[0], p[1]:
        after = build_basket_outcomes(
            session=object(), config=_CONFIG, days=[as_of],
            prices_by_symbol=frames_future, regime_fn=lambda d: "bull", basket_size=10,
        ).days[0]

    # the as-of ranking is identical — only bars <= t fed the detector.
    assert after.symbols == before.symbols


# ---------------------------------------------------------------------------
# regime slices — per-regime scores match a hand-sliced fixture; DSR null.
# ---------------------------------------------------------------------------


def test_regime_scores_match_hand_sliced_fixture_and_dsr_null():
    frames = {"AAA": _ohlcv([100.0 + i for i in range(60)])}
    days = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]
    raw = [_signal("AAA", 0.9)]
    # two bull days, two bear days
    regime_map = {days[0]: "bull", days[1]: "bull", days[2]: "bear", days[3]: "bear"}
    spec = _spec(
        [{"id": "c1", "override": {"layer_weight_money_flow": 0.35}}],
        primary="total_return", guardrails=(),
    )
    p1, p2 = _patch_selectors(raw)
    with p1, p2:
        cards = run_experiment(
            spec, base_overrides=None, session=object(),
            prices_by_symbol=frames, trading_days=days,
            regime_fn=lambda d: regime_map[d], segment="train", primary_horizon=5,
        )
    champ = next(c for c in cards if c["candidate_id"] == "champion")
    assert set(champ["regime_scores"]) == {"bull", "bear"}
    # each regime slice carries the primary reward, hand-verifiable (non-null).
    assert "total_return" in champ["regime_scores"]["bull"]
    assert champ["deflated_sharpe"] is None       # null until the DSR spec lands


# ---------------------------------------------------------------------------
# windows — train/holdout disjoint; embargo purge gap respected; holdout intact.
# ---------------------------------------------------------------------------


def test_split_windows_disjoint_and_embargo_purge():
    # 30 consecutive business days spanning the train and holdout ranges.
    days = [d.date() for d in pd.bdate_range("2026-01-01", periods=40, tz="UTC")]
    window = ExperimentWindow(
        train=f"{days[0]}..{days[19]}",
        holdout=f"{days[25]}..{days[39]}",
        embargo_days=3,
    )
    split = split_windows(days, window)

    train_set, hold_set = set(split.train_days), set(split.holdout_days)
    assert train_set.isdisjoint(hold_set)          # no leakage across the split
    assert hold_set == set(days[25:40])            # holdout matches its range
    # embargo=3 purges the 3 train days immediately before the holdout start.
    # holdout starts at days[25]; the 3 trading days before it are days[22..24],
    # but those already fall outside the train range (train ends days[19]).
    # Purge bites when it overlaps train: shift holdout adjacent to train.
    window2 = ExperimentWindow(
        train=f"{days[0]}..{days[19]}",
        holdout=f"{days[20]}..{days[39]}",
        embargo_days=3,
    )
    split2 = split_windows(days, window2)
    # the last 3 train days (days[17,18,19]) are purged — within 3 trading days
    # of the holdout start days[20].
    assert days[19] not in split2.train_days
    assert days[17] not in split2.train_days
    assert days[16] in split2.train_days           # just outside the embargo gap
    assert set(split2.holdout_days) == set(days[20:40])   # holdout untouched


# ---------------------------------------------------------------------------
# unknown reward name → loud error (symmetric to unknown-override-key rejection).
# ---------------------------------------------------------------------------


def test_unknown_reward_name_raises():
    frames = {"AAA": _ohlcv([100.0] * 30)}
    days = [date(2026, 1, 5), date(2026, 1, 6)]
    raw = [_signal("AAA", 0.9)]
    spec = _spec(
        [{"id": "c1", "override": {"layer_weight_money_flow": 0.35}}],
        primary="no_such_reward", guardrails=(),
    )
    p1, p2 = _patch_selectors(raw)
    with p1, p2, pytest.raises(ValueError, match="no_such_reward"):
        run_experiment(
            spec, base_overrides=None, session=object(),
            prices_by_symbol=frames, trading_days=days,
            regime_fn=lambda d: "bull", segment="train",
        )


# ---------------------------------------------------------------------------
# scorecard contract — validates against base_scorecard with no LLM fields.
# ---------------------------------------------------------------------------


def test_scorecard_validates_against_base_schema():
    frames = {"AAA": _ohlcv([100.0 + i for i in range(30)])}
    days = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    raw = [_signal("AAA", 0.9)]
    spec = _spec(
        [{"id": "c1", "override": {"layer_weight_money_flow": 0.35}}],
        primary="sharpe", guardrails=("max_drawdown", "turnover"),
    )
    p1, p2 = _patch_selectors(raw)
    with p1, p2:
        cards = run_experiment(
            spec, base_overrides=None, session=object(),
            prices_by_symbol=frames, trading_days=days,
            regime_fn=lambda d: "bull", segment="train", primary_horizon=5,
        )
    for card in cards:
        # strict validation: every base_scorecard field present, no extras,
        # and crucially NO llm_extension fields on the screener evaluator.
        output_schema.validate("base_scorecard", card, strict=True)
        assert card["candidate_type"] == "screener"
        assert card["evaluator_sha"] == evaluator_sha()
        # H recorded inside a permissive field, not a new top-level key.
        assert card["rewards"]["horizon"] == 5
        assert "sharpe" in card["rewards"]["primary"]
        assert set(card["rewards"]["guardrails"]) == {"max_drawdown", "turnover"}


# ---------------------------------------------------------------------------
# CLI wiring (composition root) — spec load + LIVE base_overrides + output write.
# ---------------------------------------------------------------------------


def test_cli_experiment_run_resolves_champion_base_and_writes(tmp_path):
    from click.testing import CliRunner

    from rainier.cli import cli

    # Hermetic champion layer: settings.yaml (non-default weight) + champion.yaml.
    (tmp_path / "settings.yaml").write_text(
        "stock_screener:\n  layer_weight_pattern: 0.70\n"
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "champion.yaml").write_text("version: 1\n")
    spec_file = tmp_path / "exp.yaml"
    spec_file.write_text(
        "id: cli-test\n"
        "status: active\n"
        "champion: champion.yaml\n"
        "layer: layer_weights\n"
        "challengers:\n"
        "  - id: c1\n"
        "    override: {layer_weight_money_flow: 0.35}\n"
        "primary: sharpe\n"
        "guardrails: [max_drawdown]\n"
        "window: {train: 2026-01-01..2026-03-31, holdout: 2026-05-01..2026-06-25, "
        "embargo_days: 20}\n"
    )
    out = tmp_path / "cards.json"
    fixture_cards = [{"candidate_id": "champion", "n_selection_days": 3}]

    cm = MagicMock()
    cm.__enter__.return_value = MagicMock()
    cm.__exit__.return_value = False
    with patch("rainier.paper.pattern_audit.universe_symbols", return_value=["AAA"]), \
         patch("rainier.paper.pattern_replay.load_prices",
               return_value={"AAA": _ohlcv([100.0] * 30)}), \
         patch("rainier.core.database.get_session", return_value=cm), \
         patch.object(screener, "run_experiment", return_value=fixture_cards) as mock_run:
        result = CliRunner().invoke(
            cli,
            ["experiment", "run", str(spec_file),
             "--config", str(tmp_path / "settings.yaml"),
             "--output", str(out)],
        )

    assert result.exit_code == 0, result.output
    assert out.exists()
    # The driver received the LIVE champion base layer (settings lw_pattern=0.70),
    # NOT None/code-defaults — the §4.5 silent-no-op guard at the composition root.
    base_arg = mock_run.call_args.args[1]
    assert base_arg["layer_weight_pattern"] == 0.70


# ---------------------------------------------------------------------------
# short-history parity — a symbol whose as-of window has fewer than
# config.min_daily_bars bars is money-flow-only in the driver, exactly as the
# live _fetch_stock_data(symbols, config.min_daily_bars) gate drops it. Without
# the cut it would earn pattern credit the live ranking never grants.
# ---------------------------------------------------------------------------


def test_short_history_symbol_is_money_flow_only_like_live_gate():
    # STRONG carries a strong actionable pattern but only 24 bars; PLAIN is flat
    # (no pattern) with 80 bars and a slightly higher money-flow strength.
    strong = _ohlcv(_false_breakdown())            # 24 bars, w_bottom pattern
    plain = _ohlcv([100.0] * 80)                   # 80 flat bars, no pattern
    frames = {"STRONG": strong, "PLAIN": plain}
    as_of = strong.index[-1].date()
    raw = [_signal("STRONG", 0.50), _signal("PLAIN", 0.55)]

    eligible = StockScreenerConfig(
        swing_lookback=3, min_pattern_bars=3, max_pattern_bars=50,
        neckline_tolerance_pct=0.05, min_daily_bars=10,  # 24-bar STRONG passes
    )
    gated = eligible.model_copy(update={"min_daily_bars": 60})  # 24-bar STRONG fails

    p1, p2 = _patch_selectors(raw)
    with p1, p2:
        pattern_credited = build_basket_outcomes(
            session=object(), config=eligible, days=[as_of],
            prices_by_symbol=frames, regime_fn=lambda d: "bull",
            basket_size=2, horizons=(5,),
        ).days[0]
        money_flow_only = build_basket_outcomes(
            session=object(), config=gated, days=[as_of],
            prices_by_symbol=frames, regime_fn=lambda d: "bull",
            basket_size=2, horizons=(5,),
        ).days[0]

    # min_daily_bars=10: STRONG's pattern credit (w_pattern=0.65) lifts it first.
    assert pattern_credited.symbols == ["STRONG", "PLAIN"]
    # min_daily_bars=60: STRONG has too little history → money-flow-only, so its
    # lower money-flow strength ranks it BELOW PLAIN — the live-gate parity.
    assert money_flow_only.symbols == ["PLAIN", "STRONG"]


# ---------------------------------------------------------------------------
# horizon guard — a non-positive primary horizon is a look-ahead footgun
# (forward_return would read the as-of bar or wrap to a future bar); reject it.
# ---------------------------------------------------------------------------


def test_run_experiment_rejects_nonpositive_primary_horizon():
    frames = {"AAA": _ohlcv([100.0] * 30)}
    days = [date(2026, 1, 5), date(2026, 1, 6)]
    raw = [_signal("AAA", 0.9)]
    spec = _spec(
        [{"id": "c1", "override": {"layer_weight_money_flow": 0.35}}],
        primary="total_return", guardrails=(),
    )
    for bad_horizon in (0, -5):
        p1, p2 = _patch_selectors(raw)
        with p1, p2, pytest.raises(ValueError, match="positive"):
            run_experiment(
                spec, base_overrides=None, session=object(),
                prices_by_symbol=frames, trading_days=days,
                regime_fn=lambda d: "bull", segment="train",
                primary_horizon=bad_horizon,
            )


# ---------------------------------------------------------------------------
# baseline integrity — a missing --config must NOT silently baseline on
# code-defaults (§4.5); the champion id is reserved from challenger reuse.
# ---------------------------------------------------------------------------


def test_load_base_overrides_missing_config_raises(tmp_path):
    # A typo'd/absent settings file would return {} → challengers baseline on
    # code-defaults with no champion — the exact silent no-op §4.5 prevents.
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_base_overrides(tmp_path / "nope.yaml")


def test_run_experiment_rejects_challenger_named_champion():
    frames = {"AAA": _ohlcv([100.0] * 30)}
    days = [date(2026, 1, 5), date(2026, 1, 6)]
    raw = [_signal("AAA", 0.9)]
    # parse_spec allows id: champion; the driver must reject it so the two arms
    # do not collide on candidate_id in the co-evaluation output/registry.
    spec = _spec(
        [{"id": "champion", "override": {"layer_weight_money_flow": 0.35}}],
        primary="total_return", guardrails=(),
    )
    p1, p2 = _patch_selectors(raw)
    with p1, p2, pytest.raises(ValueError, match="reserved"):
        run_experiment(
            spec, base_overrides=None, session=object(),
            prices_by_symbol=frames, trading_days=days,
            regime_fn=lambda d: "bull", segment="train",
        )
