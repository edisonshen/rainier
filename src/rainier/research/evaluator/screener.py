"""Screener replay + score evaluator — the A/B substrate's corpus driver.

Design: docs/DESIGN-qu100-ab-testing.md §4.4, §4.5, §11.4.

Nothing else *runs* an experiment. This module walks each trading day of the
corpus, reproduces the full live 3-layer ranking as-of that day for each
candidate (champion + challengers), collects the selected basket + forward
returns, and reduces them to comparable base scorecards — driven by the
experiment spec. It COMPOSES the shipped pieces, it does not rebuild them:

    for t in scored days:                (per candidate: champion + challengers)
      _screen_money_flow_as_of(t)   ─┐   Layer 1 (as-of, casing-normalized)
      analyze_sectors_at(t)          ├─► _apply_sector_boost ─► signals_by_symbol
      window_as_of(prices, t)       ─┘   Layer 3 windows (bars <= t only)
              │
              ▼  replay_screen (live composite, sector double-count) ─► ranking
      BasketDay(t, top-N basket, fwd_return[H]=screened-pool returns, regime)
              ▼
      BasketOutcomes ─► rewards (task b3b8) ─► base_scorecard (task 5a0f schema)

Key invariants (see the task plan):
- ``base_overrides`` is REQUIRED by :func:`materialize_challengers`; the driver
  resolves the LIVE champion layer via :func:`load_base_overrides` so every
  challenger baselines on the live champion, not pydantic code-defaults (§4.5).
- forward returns are computed PER SELECTED SYMBOL (incl. money-flow-only names
  with no price frame → ``None``, never omitted) plus the WIDER screened pool so
  ``dodged_loss`` isn't 0-by-construction.
- ``corpus_hash`` is over the SHARED ``(symbol, date, close)`` snapshot, so every
  candidate in one run carries the same hash (co-evaluation, §4.5).
- reward horizon is pinned to H=5 and recorded INSIDE ``rewards`` (a permissive
  dict) — not a new top-level scorecard key (strict ``base_scorecard`` rejects
  extras).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from rainier.analysis.sector_analyzer import analyze_sectors_at
from rainier.analysis.stock_screener import (
    _apply_sector_boost,
    _screen_money_flow_as_of,
)
from rainier.core.config import StockScreenerConfig
from rainier.paper.pattern_audit import (
    HORIZONS,
    as_of_index,
    forward_returns_by_symbol,
)
from rainier.paper.pattern_replay import replay_screen, window_as_of
from rainier.research.experiment import ExperimentSpec, ExperimentWindow, materialize_challengers
from rainier.research.rewards import REGISTRY, score
from rainier.research.rewards.basket import BasketDay, BasketOutcomes

# The scorecard's candidate KIND (screener-config vs LLM-skill). Distinct from
# the reward registry's input_type ("basket") — the screener candidate emits a
# BasketOutcomes payload, so its rewards are scored under the "basket" shape.
SCREENER_CANDIDATE_TYPE = "screener"
BASKET_INPUT_TYPE = "basket"

# Champion arm's candidate_id. Its config identity (version) is provenance
# recorded by the registry (task 5), never a comparison baseline (§4.5).
CHAMPION_CANDIDATE_ID = "champion"

# Coordinator default: the primary reward horizon (shipped basket-reward
# default). 10/20d returns are retained as provenance; per-horizon scoring is a
# future spec extension. Recorded inside the scorecard's `window`/`rewards`.
DEFAULT_PRIMARY_HORIZON = 5

# The live LLM-fed actionable set is the top-5 (pipeline/post_scrape.py); the
# basket mirrors that. Parameterized — NOT pinned by the spec schema.
DEFAULT_BASKET_SIZE = 5


RegimeFn = Callable[[date], str]


def _default_regime_fn(as_of: date) -> str:
    from rainier.llm_thesis.research import compute_market_regime

    return compute_market_regime(as_of=as_of)


# ---------------------------------------------------------------------------
# Provenance hashes.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def evaluator_sha() -> str:
    """Content SHA of THIS evaluator module.

    Stamped on every scorecard so a result traces to the exact scoring code
    (registry §10.5); it moves whenever the evaluator's logic changes. Cached —
    the module bytes are fixed for the process lifetime.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def corpus_hash(prices_by_symbol: dict[str, pd.DataFrame]) -> str:
    """SHA of the SHARED ``(symbol, date, close)`` corpus snapshot.

    The common input universe scored across ALL candidates in a run — NOT any
    per-candidate basket. Co-evaluation (§4.5) requires every candidate in one
    run to carry the SAME hash; the registry refuses champion-vs-challenger
    comparisons across mismatched hashes. Deterministic for a fixed snapshot
    (symbols sorted, rows date-ascending).
    """
    h = hashlib.sha256()
    for symbol in sorted(prices_by_symbol):
        df = prices_by_symbol[symbol]
        for ts, close in zip(df.index, df["close"]):
            iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            h.update(f"{symbol}|{iso}|{float(close)!r}".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Walk-forward windows — train/holdout split with a purged embargo gap.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowSplit:
    """Trading days partitioned into train + holdout with the embargo applied."""

    train_days: list[date]
    holdout_days: list[date]
    embargo_days: int


def _parse_range(range_str: str) -> tuple[date, date]:
    # The spec interpreter already validated the "start..end" shape at parse
    # time (experiment.parse_spec); here it is safe to split.
    start_s, end_s = range_str.split("..")
    return date.fromisoformat(start_s), date.fromisoformat(end_s)


def split_windows(trading_days, window: ExperimentWindow) -> WindowSplit:
    """Partition ``trading_days`` into train + holdout with a purged embargo.

    The holdout is the trailing out-of-sample window; it is NEVER read during
    challenger selection (the driver scores the ``train`` segment by default).
    The embargo PURGES the last ``embargo_days`` TRADING days of the train set
    that fall within ``embargo_days`` trading days before the first holdout day
    — so a train day's forward-return window (overlapping horizons up to H=20)
    cannot reach into the holdout and leak it into selection.
    """
    days = sorted(trading_days)
    train_start, train_end = _parse_range(window.train)
    hold_start, hold_end = _parse_range(window.holdout)
    train = [d for d in days if train_start <= d <= train_end]
    holdout = [d for d in days if hold_start <= d <= hold_end]
    if holdout and window.embargo_days > 0:
        first_hold_idx = days.index(holdout[0])
        embargo_zone = set(days[max(0, first_hold_idx - window.embargo_days):first_hold_idx])
        train = [d for d in train if d not in embargo_zone]
    return WindowSplit(
        train_days=train, holdout_days=holdout, embargo_days=window.embargo_days
    )


def _window_str(window: ExperimentWindow, segment: str) -> str:
    """Human/registry-readable window label for the scored segment."""
    rng = window.train if segment == "train" else window.holdout
    return f"{rng} [{segment}]"


# ---------------------------------------------------------------------------
# Per-day replay — the live 3-layer ranking as-of t, then basket + fwd returns.
# ---------------------------------------------------------------------------


def build_basket_outcomes(
    session: Any,
    config: StockScreenerConfig,
    *,
    days,
    prices_by_symbol: dict[str, pd.DataFrame],
    regime_fn: RegimeFn = _default_regime_fn,
    basket_size: int = DEFAULT_BASKET_SIZE,
    horizons: tuple[int, ...] = HORIZONS,
    normalize_casing: bool = True,
) -> BasketOutcomes:
    """Replay the live 3-layer ranking as-of each day → per-day ``BasketDay``.

    Layer 1 (money flow) and Layer 2 (sector) are config-independent — they read
    the DB via the as-of selectors (``normalize_casing=True`` on the replay path,
    §10.1). Layer 3 (patterns) + the composite are config-dependent, so the whole
    ranking is re-run per candidate config (``replay_screen``). The selected
    basket is the top-``basket_size`` of the composite ranking; ``fwd_return[H]``
    carries the ENTIRE screened pool's returns so ``dodged_loss`` sees the
    non-selected losers, and every selected symbol is guaranteed a key.
    """
    horizons = tuple(horizons)
    basket_days: list[BasketDay] = []
    for as_of in days:
        signals = _screen_money_flow_as_of(
            session, as_of, normalize_casing=normalize_casing
        )
        sector_trends = analyze_sectors_at(
            as_of, session, normalize_casing=normalize_casing
        )
        boosted = _apply_sector_boost(signals, sector_trends)

        signals_by_symbol: dict[str, float] = {}
        sectors_by_symbol: dict[str, str] = {}
        windows_by_symbol: dict[str, pd.DataFrame] = {}
        for sig in boosted:
            signals_by_symbol[sig.symbol] = sig.signal_strength
            sectors_by_symbol[sig.symbol] = sig.sector
            df = prices_by_symbol.get(sig.symbol)
            if df is None:
                continue  # money-flow-only symbol — ranks with best_confidence=0
            pos = as_of_index(df, as_of)
            if pos is None:
                continue  # no bar on/before t — money-flow-only, no pattern credit
            windowed = window_as_of(df, pos)
            # Mirror the live Layer-3 gate: screen_stocks feeds patterns via
            # _fetch_stock_data(symbols, config.min_daily_bars), which DROPS any
            # symbol whose as-of window carries < min_daily_bars bars and treats
            # it as money-flow-only. Without this same cut a short-history symbol
            # would earn pattern credit here that the live ranking never grants —
            # a parity break in the A/B measuring instrument (best_confidence=0).
            if len(windowed) >= config.min_daily_bars:
                windows_by_symbol[sig.symbol] = windowed

        selections = replay_screen(
            signals_by_symbol=signals_by_symbol,
            sectors_by_symbol=sectors_by_symbol,
            sector_trends=sector_trends,
            windows_by_symbol=windows_by_symbol,
            config=config,
        )
        screened_pool = [s.symbol for s in selections]
        basket = screened_pool[:basket_size]
        fwd = forward_returns_by_symbol(
            prices_by_symbol, as_of, screened_pool, horizons=horizons
        )
        basket_days.append(
            BasketDay(
                date=as_of,
                symbols=basket,
                fwd_return=fwd,
                regime=regime_fn(as_of),
            )
        )
    return BasketOutcomes(days=basket_days)


# ---------------------------------------------------------------------------
# Scorecard reduction — reward values by role + per-regime slices.
# ---------------------------------------------------------------------------


def _secondary_basket_rewards() -> list[str]:
    """Registered ``basket`` rewards whose role is ``secondary`` (reported)."""
    return sorted(
        name
        for name, spec in REGISTRY.items()
        if spec.input_type == BASKET_INPUT_TYPE and spec.role == "secondary"
    )


def _build_scorecard(
    *,
    candidate_id: str,
    outcomes: BasketOutcomes,
    spec: ExperimentSpec,
    window_str: str,
    corpus_hash_value: str,
    horizon: int,
    evaluator_sha_value: str,
) -> dict[str, Any]:
    """Reduce one candidate's ``BasketOutcomes`` to a base scorecard payload.

    Reward keys resolve against the registry at scoring time — an unknown name
    (or an input-type mismatch, e.g. a ``trades`` reward on a screener basket)
    raises a loud ``ValueError`` here, symmetric to the interpreter's
    unknown-override-key rejection.
    """

    def _score(name: str, payload: BasketOutcomes) -> float:
        return score(BASKET_INPUT_TYPE, name, payload, horizon=horizon)

    primary = {spec.primary: _score(spec.primary, outcomes)}
    guardrails = {g: _score(g, outcomes) for g in spec.guardrails}
    secondary_names = _secondary_basket_rewards()
    secondary = {name: _score(name, outcomes) for name in secondary_names}
    rewards = {
        # H recorded in a permissive field, NOT a new top-level scorecard key.
        "horizon": horizon,
        "primary": primary,
        "guardrails": guardrails,
        "secondary": secondary,
    }

    # Per-regime slices: the same reward set over each regime's day subset.
    scored_names: list[str] = []
    for name in (spec.primary, *spec.guardrails, *secondary_names):
        if name not in scored_names:
            scored_names.append(name)
    regime_scores: dict[str, dict[str, float]] = {}
    for regime in sorted({d.regime for d in outcomes.days}):
        sub = BasketOutcomes(days=[d for d in outcomes.days if d.regime == regime])
        regime_scores[regime] = {name: _score(name, sub) for name in scored_names}

    return {
        "candidate_id": candidate_id,
        "candidate_type": SCREENER_CANDIDATE_TYPE,
        "window": window_str,
        "n_selection_days": sum(1 for d in outcomes.days if d.symbols),
        "corpus_hash": corpus_hash_value,
        "rewards": rewards,
        "regime_scores": regime_scores,
        # ALWAYS null until the §11-task-6 DSR spec lands (design §4.4).
        "deflated_sharpe": None,
        "evaluator_sha": evaluator_sha_value,
    }


# ---------------------------------------------------------------------------
# The driver — champion + challengers co-evaluated over one corpus snapshot.
# ---------------------------------------------------------------------------


def run_experiment(
    spec: ExperimentSpec,
    base_overrides: dict[str, Any] | None,
    *,
    session: Any,
    prices_by_symbol: dict[str, pd.DataFrame],
    trading_days,
    regime_fn: RegimeFn = _default_regime_fn,
    basket_size: int = DEFAULT_BASKET_SIZE,
    primary_horizon: int = DEFAULT_PRIMARY_HORIZON,
    segment: str = "train",
    horizons: tuple[int, ...] = HORIZONS,
) -> list[dict[str, Any]]:
    """Co-evaluate champion + all challengers over ONE corpus snapshot (§4.5).

    ``base_overrides`` is the REQUIRED live champion layer (``None`` is test-only
    — it baselines on code-defaults). Every candidate is materialized from ONE
    :func:`materialize_challengers` call, scored over the SAME segment days, and
    stamped with the SAME ``corpus_hash`` — stored scores are provenance, never a
    comparison baseline. Returns one base_scorecard dict per candidate (champion
    first, then challengers in spec order).
    """
    champion_cfg, challenger_cfgs = materialize_challengers(spec, base_overrides)

    split = split_windows(trading_days, spec.window)
    if segment == "train":
        scored_days = split.train_days
    elif segment == "holdout":
        scored_days = split.holdout_days
    else:
        raise ValueError(
            f"unknown segment {segment!r} (expected 'train' or 'holdout')"
        )

    corpus_hash_value = corpus_hash(prices_by_symbol)
    evaluator_sha_value = evaluator_sha()
    window_str = _window_str(spec.window, segment)
    # Score at primary_horizon; ensure the driver populates it even if the
    # caller's horizons omit it (else _require_horizon would raise).
    horizons_used = tuple(sorted(set(horizons) | {primary_horizon}))

    candidates: list[tuple[str, StockScreenerConfig]] = [
        (CHAMPION_CANDIDATE_ID, champion_cfg),
        *challenger_cfgs.items(),
    ]
    cards: list[dict[str, Any]] = []
    for candidate_id, config in candidates:
        outcomes = build_basket_outcomes(
            session,
            config,
            days=scored_days,
            prices_by_symbol=prices_by_symbol,
            regime_fn=regime_fn,
            basket_size=basket_size,
            horizons=horizons_used,
        )
        cards.append(
            _build_scorecard(
                candidate_id=candidate_id,
                outcomes=outcomes,
                spec=spec,
                window_str=window_str,
                corpus_hash_value=corpus_hash_value,
                horizon=primary_horizon,
                evaluator_sha_value=evaluator_sha_value,
            )
        )
    return cards


# ---------------------------------------------------------------------------
# Champion baseline resolution — the live settings + champion layer.
# ---------------------------------------------------------------------------


def load_base_overrides(config_path: str | Path = "config/settings.yaml") -> dict[str, Any]:
    """Resolve the champion-effective override layer for the driver.

    Mirrors ``core.config.load_settings`` EXACTLY: ``settings.yaml:stock_screener``
    deep-merged UNDER ``champion.yaml`` (champion wins per key), resolving the
    champion model dir as ``<settings_dir>/model``. The returned dict is the SAME
    layer the live ``StockScreenerConfig`` is built from — passing it as the
    REQUIRED ``base_overrides`` is what keeps challengers baselined on the LIVE
    champion, not code-defaults (§4.5).
    """
    import yaml

    from rainier.core.champion import (
        load_champion_overrides,
        merge_stock_screener_config,
    )

    path = Path(config_path)
    yaml_config: dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            yaml_config = yaml.safe_load(f) or {}
    yaml_screener = yaml_config.get("stock_screener")
    champion_overrides = load_champion_overrides(path.parent / "model")
    return merge_stock_screener_config(yaml_screener, champion_overrides)
