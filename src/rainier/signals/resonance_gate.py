"""ResonanceGate — dual-threshold state machine over the resonance score.

```
score r[t]  ──►  if state==CASH  and r[t] ≥ BUY  → state := TQQQ
            ──►  if state==TQQQ  and r[t] ≤ SELL → state := CASH
            ──►  else                            → hold state
       BUY > SELL ; the gap (BUY-SELL) is the hysteresis band.
```

The gate is deterministic: a fixed score sequence yields a fixed state
sequence (unit-tested). It decides on ``close[t]`` and returns the **decision**
weight series — the +1 shift that makes the close[t] decision apply to the
t→t+1 return is owned by the sim (``backtest/daily_mtm.py``), NOT here. This is
the no-lookahead boundary: nothing in this module reads ``r[t+1]``.

Boot (§5.4): before ``WARMUP_BARS`` the gate is forced CASH (EMA-family
members haven't converged — entering on noise there would make ``t0``
data-dependent). At the first valid bar ``t0 = WARMUP_BARS`` the state boots to
TQQQ iff ``r[t0] ≥ BUY`` else CASH, then the hysteresis machine runs.

Implements the ``WeightStrategy`` protocol: ``weights(df, symbol, timeframe)``
returns ``{symbol: ndarray}`` of {0.0, 1.0} (binary in/out, remainder cash).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rainier.core.types import Timeframe
from rainier.signals.panel import WARMUP_BARS, PanelInputs, SignalPanel
from rainier.signals.resonance import ResonanceScorer

CASH = 0.0
TQQQ = 1.0


def run_state_machine(
    score: np.ndarray, buy: float, sell: float, warmup: int = WARMUP_BARS
) -> np.ndarray:
    """Deterministic dual-threshold state machine → {0,1} decision weights.

    Forced CASH for ``t < warmup``; boots at ``t0 = warmup`` (CASH→TQQQ iff
    ``score[t0] ≥ buy``). ``buy`` must be > ``sell`` (hysteresis).
    """
    if not buy > sell:
        raise ValueError(f"BUY ({buy}) must be strictly greater than SELL ({sell})")
    n = len(score)
    w = np.zeros(n, dtype=float)
    state = CASH
    t0 = min(warmup, n)
    for t in range(t0, n):
        r = score[t]
        if state == CASH and r >= buy:
            state = TQQQ
        elif state == TQQQ and r <= sell:
            state = CASH
        # else: hold
        w[t] = state
    return w


class ResonanceGate:
    """Power-weighted dual-threshold in/out gate. Implements ``WeightStrategy``.

    Parameters
    ----------
    scorer : ResonanceScorer
        Produces the daily resonance score from a panel.
    buy, sell : float
        Dual thresholds in [0,1]; ``buy`` > ``sell``.
    asset : str
        Target asset name for the returned weight dict (default ``"TQQQ"``).
    warmup : int
        Frozen warmup before the gate may enter (default ``WARMUP_BARS``).
    """

    def __init__(
        self,
        scorer: ResonanceScorer,
        buy: float = 0.6,
        sell: float = 0.4,
        asset: str = "TQQQ",
        warmup: int = WARMUP_BARS,
    ) -> None:
        if not buy > sell:
            raise ValueError(f"BUY ({buy}) must be strictly greater than SELL ({sell})")
        self.scorer = scorer
        self.buy = buy
        self.sell = sell
        self.asset = asset
        self.warmup = warmup

    @property
    def panel(self) -> SignalPanel:
        return self.scorer.panel

    def score_series(self, inputs: PanelInputs) -> np.ndarray:
        return self.scorer.score(self.panel.compute(inputs))

    def decide(self, inputs: PanelInputs) -> np.ndarray:
        """Decision weight series (un-shifted) from panel inputs."""
        score = self.score_series(inputs)
        return run_state_machine(score, self.buy, self.sell, self.warmup)

    # -- WeightStrategy protocol --------------------------------------------

    def weights(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
    ) -> dict[str, np.ndarray]:
        """Per-asset daily target weights from an OHLCV+context DataFrame.

        ``df`` must carry QQQ OHLC (columns ``open/high/low/close``) plus a
        ``vix`` column; optional ``spy`` and ``breadth`` columns enable the
        cross-asset and breadth panel members. ``symbol`` and ``timeframe`` are
        part of the ``WeightStrategy`` contract but the gate keys its output on
        ``self.asset`` (the leveraged instrument it trades).
        """
        inputs = PanelInputs(
            qqq=df[["open", "high", "low", "close"]],
            vix=df["vix"],
            spy=df["spy"] if "spy" in df.columns else None,
            breadth=df["breadth"] if "breadth" in df.columns else None,
        )
        return {self.asset: self.decide(inputs)}
