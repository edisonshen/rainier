"""HMM-based market regime detection.

Uses a Gaussian Hidden Markov Model to identify market regimes
from returns and volatility features. Maps HMM states to
MarketRegime enum values based on state characteristics.

Design decisions (eng review 2026-03-22):
- 3 states by default (trending up, trending down, range/high-vol)
- BIC/AIC comparison for 2-4 states shows 3 is optimal
- Fit on 5 years of daily data, predict on current window
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from rainier.analysis.pivots import compute_atr
from rainier.core.types import MarketRegime

logger = logging.getLogger(__name__)

# Default number of HMM states
N_STATES = 3


class HMMRegimeDetector:
    """Gaussian HMM regime detector.

    Features used for HMM observations:
    1. Log returns (directional signal)
    2. Realized volatility (ATR ratio — current ATR / rolling mean ATR)
    3. ADX (trend strength)

    All features are z-score normalized before fitting to prevent
    scale-dominant features (ADX ~0-100) from drowning out small-scale
    features (log returns ~0.001).
    """

    def __init__(self, n_states: int = N_STATES, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.model: GaussianHMM | None = None
        self._state_map: dict[int, MarketRegime] = {}
        self._scaler: StandardScaler | None = None
        self._raw_means: np.ndarray | None = None  # unscaled means for state mapping

    def fit(self, df: pd.DataFrame) -> HMMRegimeDetector:
        """Fit the HMM on historical OHLCV data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume.
                Should be daily bars, ideally 2+ years.

        Returns:
            self (for chaining).
        """
        raw_obs = self._extract_observations(df)

        # Z-score normalize to prevent scale dominance
        self._scaler = StandardScaler()
        obs = self._scaler.fit_transform(raw_obs)

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=200,
            random_state=self.random_state,
            tol=1e-4,
        )
        self.model.fit(obs)
        logger.info(
            "HMM fitted: %d states, %d observations, converged=%s",
            self.n_states, len(obs), self.model.monitor_.converged,
        )

        # Map HMM states to MarketRegime using raw (unscaled) means
        self._raw_means = self._compute_raw_state_means(raw_obs, obs)
        self._state_map = self._map_states_to_regimes(raw_obs)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict regime for each bar.

        Returns a Series of MarketRegime values with the same index as df.
        """
        if self.model is None or self._scaler is None:
            raise RuntimeError("Model not fitted — call fit() first")

        raw_obs = self._extract_observations(df)
        obs = self._scaler.transform(raw_obs)
        hidden_states = self.model.predict(obs)

        regimes = pd.Series(
            [self._state_map.get(s, MarketRegime.RANGE_BOUND) for s in hidden_states],
            index=df.index[len(df) - len(obs):],
        )
        # Pad early bars (lost to feature computation) with first regime
        if len(regimes) < len(df):
            pad = pd.Series(
                [regimes.iloc[0]] * (len(df) - len(regimes)),
                index=df.index[:len(df) - len(regimes)],
            )
            regimes = pd.concat([pad, regimes])

        return regimes

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """Fit and predict in one call."""
        self.fit(df)
        return self.predict(df)

    def _extract_observations(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix for HMM from OHLCV data.

        Features: [log_return, atr_ratio, adx]
        """
        from rainier.analysis.regime import compute_adx

        close = df["close"].values.astype(np.float64)

        # Log returns
        log_returns = np.diff(np.log(close))

        # ATR ratio
        atr = compute_atr(df, period=14).values
        atr_mean = pd.Series(atr).rolling(window=20, min_periods=1).mean().values
        atr_ratio = np.where(atr_mean > 0, atr / atr_mean, 1.0)

        # ADX
        adx = compute_adx(df, period=14).values

        # Align lengths (log_returns is 1 shorter)
        n = len(log_returns)
        obs = np.column_stack([
            log_returns,
            atr_ratio[1:n + 1],
            adx[1:n + 1],
        ])

        # Replace any NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _compute_raw_state_means(
        self, raw_obs: np.ndarray, scaled_obs: np.ndarray,
    ) -> np.ndarray:
        """Compute mean of raw (unscaled) observations per HMM state."""
        states = self.model.predict(scaled_obs)
        means = np.zeros((self.n_states, raw_obs.shape[1]))
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() > 0:
                means[s] = raw_obs[mask].mean(axis=0)
        return means

    def _map_states_to_regimes(self, raw_obs: np.ndarray) -> dict[int, MarketRegime]:
        """Map HMM states to MarketRegime based on raw observation means per state.

        Logic:
        - State with highest mean return → TRENDING_UP
        - State with lowest mean return → TRENDING_DOWN
        - Remaining state → RANGE_BOUND (or HIGH_VOLATILITY if ATR ratio is extreme)
        """
        if self._raw_means is None:
            raise RuntimeError("Raw means not computed")

        means = self._raw_means
        # Feature order: [log_return, atr_ratio, adx]
        mean_returns = means[:, 0]
        mean_atr_ratio = means[:, 1]

        state_map: dict[int, MarketRegime] = {}

        up_state = int(np.argmax(mean_returns))
        down_state = int(np.argmin(mean_returns))

        state_map[up_state] = MarketRegime.TRENDING_UP
        state_map[down_state] = MarketRegime.TRENDING_DOWN

        # Remaining state(s)
        for s in range(self.n_states):
            if s not in state_map:
                if mean_atr_ratio[s] > 1.3:
                    state_map[s] = MarketRegime.HIGH_VOLATILITY
                else:
                    state_map[s] = MarketRegime.RANGE_BOUND

        logger.info(
            "HMM state mapping: %s (raw means: returns=%s, atr_ratio=%s)",
            {s: r.value for s, r in state_map.items()},
            [f"{m:.4f}" for m in mean_returns],
            [f"{m:.2f}" for m in mean_atr_ratio],
        )
        return state_map

    def save(self, path: Path) -> None:
        """Save fitted model to disk."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "state_map": self._state_map,
                "n_states": self.n_states,
                "scaler": self._scaler,
                "raw_means": self._raw_means,
            }, f)
        logger.info("HMM model saved to %s", path)

    def load(self, path: Path) -> None:
        """Load fitted model from disk."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self._state_map = data["state_map"]
        self.n_states = data["n_states"]
        self._scaler = data["scaler"]
        self._raw_means = data.get("raw_means")
        logger.info("HMM model loaded from %s", path)

    def regime_summary(self, regimes: pd.Series) -> dict:
        """Return regime distribution and average duration statistics."""
        counts = regimes.value_counts()
        total = len(regimes)

        # Compute average regime duration (consecutive bars in same regime)
        durations: dict[MarketRegime, list[int]] = {}
        current = regimes.iloc[0]
        run_length = 1
        for i in range(1, len(regimes)):
            if regimes.iloc[i] == current:
                run_length += 1
            else:
                durations.setdefault(current, []).append(run_length)
                current = regimes.iloc[i]
                run_length = 1
        durations.setdefault(current, []).append(run_length)

        return {
            "distribution": {r.value: int(counts.get(r, 0)) for r in MarketRegime},
            "pct": {r.value: f"{counts.get(r, 0) / total:.1%}" for r in MarketRegime},
            "avg_duration": {
                r.value: f"{np.mean(runs):.1f} bars"
                for r, runs in durations.items()
            },
            "total_bars": total,
        }
