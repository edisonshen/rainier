"""Daily mark-to-market portfolio sim for ``WeightStrategy`` gates.

Productionized from ``scripts/regime_switch_study.run_portfolio`` (per-asset
daily MTM, no lookahead, turnover + T-bill cost). The sim owns the **+1 shift**:
a decision weight at ``close[t]`` applies to the t→t+1 return. The first bar is
flat (no prior decision to act on).

```
   decision weights w[t] (from a WeightStrategy, un-shifted)
            │  shift +1  (w[t] decided on close[t] applies to t→t+1 return)
            ▼
   for each day: port_ret = Σ wᵢ·retᵢ + (1-Σwᵢ)·cash_rate  −  turnover·cost
            ▼
   equity curve  ──►  metrics (return, CAGR, maxDD, Calmar, Sharpe, switches)
```

**Costs — no double-counting (design §5.5).** On *real adjusted TQQQ* the price
already embeds the 3× daily-reset financing and decay, so we charge ONLY
turnover/slippage on each buy/sell plus the T-bill rate the cash sleeve earns.
The synthetic ``3·r − 2·rate − fee`` formula is reserved for the §6.2 pre-2010
OOS test and lives in the study layer, NOT here.

Cash rate can be a scalar (constant T-bill) or a per-day series (historical
3-month T-bill), matching the spec's pre-registered pre-2010 cost params.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ANNUAL = 252
DEFAULT_CASH_APY = 0.04
DEFAULT_ONE_WAY_COST = 0.0003  # 3 bps per unit of turnover (one-way)


def max_drawdown(curve: np.ndarray) -> float:
    """Largest peak-to-trough fractional drop over an equity curve."""
    peak = -np.inf
    mdd = 0.0
    for v in curve:
        if v > peak:
            peak = v
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd


def shift_decision(weights: np.ndarray) -> np.ndarray:
    """Apply the +1 no-lookahead shift: decision at close[t] acts on t→t+1.

    The first bar becomes flat (0) — there is no prior-day decision to act on.
    """
    out = np.zeros_like(weights, dtype=float)
    out[1:] = weights[:-1]
    return out


@dataclass
class PortfolioResult:
    name: str
    equity: np.ndarray
    total_return: float = 0.0
    cagr: float = 0.0
    max_dd: float = 0.0
    calmar: float = 0.0
    sharpe: float = 0.0
    exposure: float = 0.0
    switches: int = 0


def _cash_daily(cash_apy) -> np.ndarray | float:
    """Convert an annual cash yield (scalar or per-day series) to a daily rate."""
    arr = np.asarray(cash_apy, dtype=float)
    return (1.0 + arr) ** (1.0 / ANNUAL) - 1.0


def run_portfolio(
    name: str,
    weights: dict[str, np.ndarray],
    rets: dict[str, np.ndarray],
    n_years: float,
    *,
    cash_apy=DEFAULT_CASH_APY,
    one_way_cost: float = DEFAULT_ONE_WAY_COST,
    pre_shifted: bool = False,
) -> PortfolioResult:
    """Run the daily-MTM sim over per-asset decision weights.

    Parameters
    ----------
    weights : dict[asset -> ndarray]
        Daily **decision** weights (un-shifted) per asset; remainder is cash.
        Set ``pre_shifted=True`` only if the caller already applied the +1 shift.
    rets : dict[asset -> ndarray]
        Daily simple returns per asset, aligned to ``weights``.
    n_years : float
        Span in years (for CAGR). Calmar = CAGR / maxDD.
    cash_apy : float | ndarray
        Annual cash yield — scalar (constant T-bill) or per-day series
        (historical 3-month T-bill). Earned on the un-invested fraction.
    one_way_cost : float
        Cost per unit of turnover (one-way), charged on each rebalance.
    """
    assets = list(weights.keys())
    if not assets:
        raise ValueError("weights is empty")
    n = len(next(iter(weights.values())))
    for a in assets:
        if len(weights[a]) != n:
            raise ValueError(f"weight series length mismatch for {a}")
        if len(rets[a]) != n:
            raise ValueError(f"return series length mismatch for {a}")

    shifted = {a: (weights[a] if pre_shifted else shift_decision(weights[a])) for a in assets}
    cash_daily = _cash_daily(cash_apy)
    cash_arr = np.full(n, cash_daily) if np.isscalar(cash_daily) else np.asarray(cash_daily)

    eq = np.ones(n)
    cur = 1.0
    prev = {a: 0.0 for a in assets}
    exposure_days = 0.0
    switches = 0
    for t in range(n):
        w = {a: float(shifted[a][t]) for a in assets}
        invested = sum(w.values())
        port_ret = sum(w[a] * rets[a][t] for a in assets) + (1.0 - invested) * cash_arr[t]
        turnover = sum(abs(w[a] - prev[a]) for a in assets)
        if turnover > 1e-9:
            switches += 1
        cur *= 1.0 + port_ret - turnover * one_way_cost
        eq[t] = cur
        prev = w
        exposure_days += invested

    res = PortfolioResult(name, eq)
    res.total_return = eq[-1] - 1.0
    res.cagr = eq[-1] ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
    res.max_dd = max_drawdown(eq)
    res.calmar = res.cagr / res.max_dd if res.max_dd > 0 else float("inf")
    dr = np.diff(eq) / eq[:-1]
    if dr.std() > 0:
        res.sharpe = float(dr.mean() / dr.std() * np.sqrt(ANNUAL))
    res.exposure = exposure_days / n
    res.switches = switches
    return res
