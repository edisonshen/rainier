"""BasketOutcomes (design §10.3, pinned) + the screener basket reward set.

``BasketOutcomes`` is what the replay evaluator (ab-replay-evaluator-edb5)
emits per candidate: one ``BasketDay`` per selection day, carrying the
selected symbols in rank order and forward returns per horizon. Forward
returns are ``None`` near the window end — never 0 — and every aggregation
here EXCLUDES ``None`` entries rather than zeroing them.

All rewards are pure and deterministic. Per-day basket return = equal-weight
mean of the selected symbols' non-None forward returns at the chosen
horizon; days with no available return are skipped entirely (they do not
enter any mean, stdev, or hit-rate denominator).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date as date_type

from rainier.research.rewards import register

#: Default forward-return horizon (trading days) used by the basket rewards.
DEFAULT_HORIZON = 5

#: Guardrail bounds consumed by the promote-time gate (ab-registry-promote-84a7).
#: max_drawdown: peak-to-trough fraction of basket equity; breach above 20%.
MAX_DRAWDOWN_THRESHOLD = 0.20
#: turnover: mean day-over-day fraction of the basket replaced; breach above 50%.
TURNOVER_THRESHOLD = 0.50


@dataclass(frozen=True)
class BasketDay:
    """One selection day t (design §10.3, pinned)."""

    date: date_type
    symbols: list[str]  # selected basket, rank order
    fwd_return: dict[int, dict[str, float | None]]  # H → symbol → return; None near window end
    regime: str  # from compute_market_regime


@dataclass(frozen=True)
class BasketOutcomes:
    """Per-day basket + forward returns for one candidate over one window."""

    days: list[BasketDay]


def _day_returns(outcomes: BasketOutcomes, horizon: int) -> list[float]:
    """Equal-weight basket return per day; days with no data are skipped."""
    out: list[float] = []
    for day in outcomes.days:
        returns = day.fwd_return.get(horizon, {})
        available = [r for s in day.symbols if (r := returns.get(s)) is not None]
        if available:
            out.append(sum(available) / len(available))
    return out


@register("total_return", input_type="basket", role="secondary", direction="increase")
def total_return(outcomes: BasketOutcomes, horizon: int = DEFAULT_HORIZON) -> float:
    """Arithmetic sum of per-day equal-weight basket returns."""
    return sum(_day_returns(outcomes, horizon))


@register("sharpe", input_type="basket", role="primary", direction="increase")
def sharpe(outcomes: BasketOutcomes, horizon: int = DEFAULT_HORIZON) -> float:
    """Mean over sample stdev (ddof=1) of daily basket returns; 0.0 if undefined."""
    day_returns = _day_returns(outcomes, horizon)
    if len(day_returns) < 2:
        return 0.0
    stdev = statistics.stdev(day_returns)
    if stdev == 0.0:
        return 0.0
    return statistics.mean(day_returns) / stdev


@register("hit_rate", input_type="basket", role="secondary", direction="increase")
def hit_rate(outcomes: BasketOutcomes, horizon: int = DEFAULT_HORIZON) -> float:
    """Fraction of aggregated days with a positive basket return."""
    day_returns = _day_returns(outcomes, horizon)
    if not day_returns:
        return 0.0
    return sum(1 for r in day_returns if r > 0) / len(day_returns)


@register(
    "max_drawdown",
    input_type="basket",
    role="guardrail",
    direction="decrease",
    threshold=MAX_DRAWDOWN_THRESHOLD,
)
def max_drawdown(outcomes: BasketOutcomes, horizon: int = DEFAULT_HORIZON) -> float:
    """Max peak-to-trough decline of compounded basket equity, as a positive fraction."""
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for r in _day_returns(outcomes, horizon):
        equity *= 1.0 + r
        peak = max(peak, equity)
        worst = max(worst, (peak - equity) / peak)
    return worst


@register(
    "turnover",
    input_type="basket",
    role="guardrail",
    direction="decrease",
    threshold=TURNOVER_THRESHOLD,
)
def turnover(outcomes: BasketOutcomes, horizon: int = DEFAULT_HORIZON) -> float:
    """Mean day-over-day fraction of the basket replaced.

    For consecutive days (t-1, t): |symbols_t \\ symbols_{t-1}| / |symbols_t|.
    Horizon-independent (basket composition only); the parameter exists so
    every basket reward shares one call shape.
    """
    days = [d for d in outcomes.days if d.symbols]
    if len(days) < 2:
        return 0.0
    fractions = []
    for prev, curr in zip(days, days[1:]):
        entered = set(curr.symbols) - set(prev.symbols)
        fractions.append(len(entered) / len(curr.symbols))
    return sum(fractions) / len(fractions)


@register("dodged_loss", input_type="basket", role="secondary", direction="increase")
def dodged_loss(outcomes: BasketOutcomes, horizon: int = DEFAULT_HORIZON) -> float:
    """Sum of losses avoided by NOT selecting a symbol.

    For each day, sums |negative forward return| over symbols present in
    ``fwd_return[H]`` but absent from the selected basket. Requires the
    evaluator to populate ``fwd_return`` with the wider screened pool; if it
    only carries the selected symbols, dodged_loss is 0.0.
    """
    total = 0.0
    for day in outcomes.days:
        selected = set(day.symbols)
        for symbol, ret in day.fwd_return.get(horizon, {}).items():
            if symbol not in selected and ret is not None and ret < 0:
                total += -ret
    return total
