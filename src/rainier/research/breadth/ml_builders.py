"""ML/RL panel builders + gym-style trading environment.

Lazy gymnasium import — the gymnasium dep is loaded ONLY inside this module
when ``ThematicTradingEnv`` is constructed. Importing this module itself does
NOT require gymnasium; only the env class does.

Design ref:
    docs/DESIGN-thematic-ranks-dashboard.md §5.6 ([D-016]).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Wide panel builder (3D tensor-friendly)
# ---------------------------------------------------------------------------


def build_panel_tensor(
    panel: pd.DataFrame,
) -> tuple[np.ndarray, list, list, list]:
    """Convert a wide-format panel (from ``load_thematic_panel``) into a
    numpy 3D tensor of shape ``(n_dates, n_tickers, n_features)`` plus the
    axis labels.

    Returns
    -------
    arr:
        ``np.ndarray`` of shape ``(n_dates, n_tickers, n_features)``.
    dates, tickers, features:
        Axis labels in the same order as the tensor dimensions.
    """
    if panel.empty:
        return np.empty((0, 0, 0), dtype=np.float32), [], [], []

    dates = sorted({d for d, _ in panel.index})
    tickers = sorted({t for _, t in panel.index})
    features = list(panel.columns)

    arr = np.full(
        (len(dates), len(tickers), len(features)), np.nan, dtype=np.float32
    )
    for i, d in enumerate(dates):
        if d not in panel.index.get_level_values(0):
            continue
        sub = panel.loc[d]
        for j, t in enumerate(tickers):
            if t in sub.index:
                arr[i, j, :] = sub.loc[t].values
    return arr, dates, tickers, features


# ---------------------------------------------------------------------------
# Gym-style env (lazy gymnasium import)
# ---------------------------------------------------------------------------


class ThematicTradingEnv:
    """A minimal gym-style env over the thematic panel.

    Out of scope for this PR: actual training loop or RL agent (the env is the
    deliverable; training is consumer responsibility).

    Action space: ``long_short_topk`` — at each step the agent receives a
    feature vector and produces a length-``n_tickers`` vector of {-1, 0, +1}
    indicating short / hold / long. Reward = portfolio's realized
    ``fwd_<reward_horizon>_excess_ret`` over the next ``reward_horizon`` days.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        reward: str = "fwd_5d_excess_ret",
        action_space: str = "long_short_topk",
        k: int = 10,
    ):
        # Lazy gymnasium import — only enforced if the operator opts in.
        # The env still works without gymnasium (the API is gym-shaped);
        # consumers that want a strict gym.Env should subclass and inject.
        try:
            import gymnasium  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ThematicTradingEnv requires the `gymnasium` package. "
                "Install via: uv pip install gymnasium  (or "
                "pip install '.[rl]')"
            ) from exc

        if action_space != "long_short_topk":
            raise ValueError(f"unknown action_space: {action_space}")
        # `_after_cost` reward variants are reserved for a future cost-model
        # injection. Accepting them today and silently stripping the suffix
        # would alias them to the pre-cost label and hide that no cost is
        # applied (codex iter-3 / iter-5 [P2]). Reject explicitly until the
        # cost-adjusted labels land.
        if reward not in (
            "fwd_5d_excess_ret",
            "fwd_10d_excess_ret",
            "fwd_20d_excess_ret",
        ):
            if reward.endswith("_after_cost"):
                raise NotImplementedError(
                    f"reward={reward!r} is reserved for cost-adjusted "
                    f"labels which are not yet implemented. Use the "
                    f"pre-cost variant (e.g. 'fwd_5d_excess_ret') or "
                    f"inject a cost model in a follow-up PR."
                )
            raise ValueError(f"unknown reward: {reward}")

        self.features = features.copy()
        self.labels = labels.copy()
        self.reward = reward
        self.action_space_kind = action_space
        self.k = int(k)
        self._step_idx = 0
        self._dates = sorted(features.index.get_level_values(0).unique())

    def reset(self) -> tuple[np.ndarray, dict]:
        self._step_idx = 0
        return self._observation(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step.

        Parameters
        ----------
        action:
            Length-n_tickers vector of {-1, 0, +1}.

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        # Reward = sum over tickers of action_i * fwd_excess_ret_i
        if self._step_idx >= len(self._dates):
            return self._observation(), 0.0, True, False, {}
        cur_date = self._dates[self._step_idx]
        # Lookup excess returns on the labels frame for cur_date. The
        # constructor rejects `_after_cost` reward variants until cost-
        # adjusted labels exist, so `self.reward` is always a real column.
        reward_col = self.reward
        sub = self.labels.loc[self.labels["asof_date"] == cur_date]
        if not sub.empty and reward_col in sub.columns:
            # Align action to symbol order.
            sub_sorted = sub.sort_values("symbol").reset_index(drop=True)
            rets = sub_sorted[reward_col].fillna(0.0).to_numpy(dtype=np.float32)
            if action.shape[0] != rets.shape[0]:
                # Pad / truncate to match.
                if action.shape[0] < rets.shape[0]:
                    pad = np.zeros(rets.shape[0] - action.shape[0], dtype=action.dtype)
                    action = np.concatenate([action, pad])
                else:
                    action = action[: rets.shape[0]]
            r = float(np.dot(action, rets))
        else:
            r = 0.0
        self._step_idx += 1
        terminated = self._step_idx >= len(self._dates)
        return self._observation(), r, terminated, False, {"date": str(cur_date)}

    def _observation(self) -> np.ndarray:
        if self._step_idx >= len(self._dates):
            return np.zeros(0, dtype=np.float32)
        cur_date = self._dates[self._step_idx]
        sub = self.features.loc[cur_date]
        return sub.values.astype(np.float32).flatten()
