"""Selection-loop reward bodies (R1) — pure functions, no DB, no side effects.

Design: docs/DESIGN-qu100-selection-reward-loop.md §3.1.

Per-decision reward is the **R-multiple against the committed plan**::

    risk_per_share = entry - stop            (must be > 0, else no reward)
    R              = (exit - entry) / risk_per_share

Cohort objectives are the operator's three axes, reported SEPARATELY so the
trade-offs stay visible, plus one composite for ranking:

    total return   total_R = sum(R), total_pnl_usd
    risk           max_drawdown_R (peak-to-trough of the cumulative-R curve),
                   std_R
    win rate       wins / matured decisions          (win = R > 0)

    composite_score = total_R − λ · max_drawdown_R    (λ = 1.0 default)
    win-rate floor  challenger.win_rate ≥ champion.win_rate − 5pp

Supporting stats: n, expectancy_R = mean(R), t_stat_R = mean / (std / √n).

``REWARD_VERSION`` is pinned; any formula change here bumps it so ledger rows
computed under the old formula are recomputed (same convention as
``feature_version`` in paper/features.py).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

REWARD_VERSION = 1

#: Drawdown penalty weight in the composite score (★ operator default).
DEFAULT_DRAWDOWN_LAMBDA = 1.0
#: A challenger may not lose more than this much win rate vs the champion.
DEFAULT_WIN_RATE_FLOOR_PP = 0.05
#: Counterfactual horizon cap, in trading sessions (entry day = session 1).
DEFAULT_COUNTERFACTUAL_HORIZON_DAYS = 20


# ---------------------------------------------------------------------------
# Per-decision
# ---------------------------------------------------------------------------


def r_multiple(entry: float, stop: float, exit_price: float) -> float | None:
    """R-multiple of an exit against the plan's risk. None when the plan has
    no positive risk (stop at/above entry) or any input is non-finite."""
    if not all(math.isfinite(x) for x in (entry, stop, exit_price)):
        return None
    risk = entry - stop
    if risk <= 0:
        return None
    return (exit_price - entry) / risk


def sleeve_pnl(entry: float, exit_price: float, shares: int) -> float:
    """Share-based sleeve P&L — same convention as paper/exit.py (F14)."""
    return shares * (exit_price - entry)


# ---------------------------------------------------------------------------
# Cohort aggregates — each takes the matured R series (chronological order)
# ---------------------------------------------------------------------------


def total_r(rs: Sequence[float]) -> float:
    return float(sum(rs))


def total_pnl(pnls: Sequence[float]) -> float:
    return float(sum(pnls))


def win_rate(rs: Sequence[float]) -> float | None:
    if not rs:
        return None
    return sum(1 for r in rs if r > 0) / len(rs)


def expectancy_r(rs: Sequence[float]) -> float | None:
    if not rs:
        return None
    return float(sum(rs)) / len(rs)


def std_r(rs: Sequence[float]) -> float | None:
    """Sample standard deviation (n-1); None below two observations."""
    n = len(rs)
    if n < 2:
        return None
    mean = sum(rs) / n
    return math.sqrt(sum((r - mean) ** 2 for r in rs) / (n - 1))


def t_stat_r(rs: Sequence[float]) -> float | None:
    """mean / (std / √n); None when undefined (n<2 or zero dispersion)."""
    sd = std_r(rs)
    if sd is None or sd == 0:
        return None
    return (sum(rs) / len(rs)) / (sd / math.sqrt(len(rs)))


def max_drawdown_r(rs: Sequence[float]) -> float:
    """Worst peak-to-trough decline of the cumulative-R curve, in R (>= 0).

    The curve starts at 0 before the first decision, so a losing opening run
    counts as drawdown from the flat start. ``rs`` must be in outcome order.
    """
    peak = 0.0
    cum = 0.0
    worst = 0.0
    for r in rs:
        cum += r
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > worst:
            worst = dd
    return worst


def composite_score(
    total_r: float, max_dd_r: float, *, lam: float = DEFAULT_DRAWDOWN_LAMBDA
) -> float:
    """total return penalized by risk: ``total_R − λ · max_drawdown_R``."""
    return total_r - lam * max_dd_r


def win_rate_floor_ok(
    challenger_win_rate: float | None,
    champion_win_rate: float | None,
    *,
    floor_pp: float = DEFAULT_WIN_RATE_FLOOR_PP,
) -> bool:
    """Promotion constraint: the challenger may not buy total return with
    materially more losers. Vacuously true when either side has no sample."""
    if challenger_win_rate is None or champion_win_rate is None:
        return True
    return challenger_win_rate >= champion_win_rate - floor_pp


def summarize(
    rs: Sequence[float],
    pnls: Sequence[float] | None = None,
    *,
    lam: float = DEFAULT_DRAWDOWN_LAMBDA,
) -> dict[str, float | int | None]:
    """All three axes + supporting stats for one matured cohort."""
    tr = total_r(rs)
    mdd = max_drawdown_r(rs)
    return {
        "n": len(rs),
        "total_R": tr,
        "total_pnl_usd": total_pnl(pnls) if pnls is not None else None,
        "win_rate": win_rate(rs),
        "expectancy_R": expectancy_r(rs),
        "std_R": std_r(rs),
        "t_stat_R": t_stat_r(rs),
        "max_drawdown_R": mdd,
        "composite_score": composite_score(tr, mdd, lam=lam),
        "reward_version": REWARD_VERSION,
    }


#: Registry key → aggregate body over a matured, chronologically ordered R
#: series. Kept module-local (not pushed into ``research.rewards.REGISTRY``)
#: so this module stays decoupled from the A/B registry's typed-spec surface.
SELECTION_REWARDS: dict[str, Callable[[Sequence[float]], float | None]] = {
    "total_R": total_r,
    "win_rate": win_rate,
    "expectancy_R": expectancy_r,
    "std_R": std_r,
    "t_stat_R": t_stat_r,
    "max_drawdown_R": max_drawdown_r,
}
