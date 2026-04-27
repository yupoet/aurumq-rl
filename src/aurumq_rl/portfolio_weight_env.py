"""Continuous-weight portfolio optimization RL environment.

Difference vs StockPickingEnv
-----------------------------
* StockPickingEnv:    action = priority scores → top-k discrete selection
* PortfolioWeightEnv: action = continuous weights, sum(w) = 1.0,
                      per-stock weight in [0, max_position_pct]

Reward types
------------
* return        — simple forward return
* sharpe        — rolling Sharpe ratio
* sortino       — rolling Sortino ratio
* mean_variance — Markowitz mean - λ * variance

State
-----
Observation = concat([factor_panel[t].flatten(), current_weights])
"""

from __future__ import annotations

import datetime
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import gymnasium as gym  # noqa: F811
    from gymnasium import spaces  # noqa: F811

from aurumq_rl.env import (
    LIMIT_PCT_THRESHOLD,
    NEW_STOCK_PROTECT_DAYS,
    _apply_trading_mask,
)
from aurumq_rl.reward_functions import (
    mean_variance_reward,
    sharpe_reward,
    sortino_reward,
)


# Constants
DEFAULT_MAX_POSITION_PCT: float = 0.05
DEFAULT_MAX_INDUSTRY_PCT: float = 0.30
_SOFTMAX_TEMP: float = 1.0
_MIN_WEIGHT_SUM: float = 1e-8


@dataclass(frozen=True)
class PortfolioWeightConfig:
    """Configuration for the continuous-weight portfolio env."""

    start_date: datetime.date = field(default_factory=lambda: datetime.date(2020, 1, 1))
    end_date: datetime.date = field(default_factory=lambda: datetime.date(2024, 12, 31))
    n_factors: int = 64
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT
    max_industry_pct: float = DEFAULT_MAX_INDUSTRY_PCT
    rolling_window: int = 20
    forward_period: int = 10
    reward_type: Literal["return", "sharpe", "sortino", "mean_variance"] = "sharpe"
    risk_aversion: float = 1.0
    cost_bps: float = 30.0
    turnover_penalty: float = 0.001

    def __post_init__(self) -> None:
        if self.start_date >= self.end_date:
            raise ValueError(
                f"start_date={self.start_date} must be before end_date={self.end_date}"
            )
        if self.n_factors < 1:
            raise ValueError(f"n_factors={self.n_factors} must be >= 1")
        if self.forward_period < 1:
            raise ValueError(f"forward_period={self.forward_period} must be >= 1")
        if not 0.0 < self.max_position_pct <= 1.0:
            raise ValueError(
                f"max_position_pct={self.max_position_pct} must be in (0, 1]"
            )
        if not 0.0 < self.max_industry_pct <= 1.0:
            raise ValueError(
                f"max_industry_pct={self.max_industry_pct} must be in (0, 1]"
            )
        if self.reward_type not in {"return", "sharpe", "sortino", "mean_variance"}:
            raise ValueError(f"reward_type={self.reward_type!r} not supported")
        if self.risk_aversion < 0.0:
            raise ValueError(f"risk_aversion={self.risk_aversion} must be >= 0")
        if self.rolling_window < 1:
            raise ValueError(f"rolling_window={self.rolling_window} must be >= 1")


def _project_weights(
    raw_action: np.ndarray,
    max_position_pct: float,
    max_industry_pct: float,
    industry_codes: np.ndarray | None = None,
    trading_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Project raw action into a constrained weight vector.

    Steps:
        1. Zero out untradeable stocks (if trading_mask given)
        2. Softmax over tradeable stocks
        3. Project to simplex with upper bounds (per-stock cap)
        4. Iteratively tighten upper bounds for industry cap
        5. Renormalize to sum=1.0
    """
    n = len(raw_action)
    raw = raw_action.astype(np.float64).copy()

    if trading_mask is not None:
        tradeable = trading_mask.astype(bool)
    else:
        tradeable = np.ones(n, dtype=bool)

    n_tradeable = int(np.sum(tradeable))
    if n_tradeable == 0:
        return np.zeros(n, dtype=np.float64)

    upper = np.where(tradeable, max_position_pct, 0.0)

    v = np.full(n, -1e30)
    v[tradeable] = raw[tradeable]
    v_max = np.max(v[tradeable])
    p_raw = np.where(tradeable, np.exp(v - v_max), 0.0)
    total = float(np.sum(p_raw))
    if total < _MIN_WEIGHT_SUM:
        result = np.zeros(n, dtype=np.float64)
        result[tradeable] = 1.0 / n_tradeable
        return result
    target = p_raw / total

    weights = _project_simplex_upper(target, upper)

    # Industry cap (iterative tightening, max 10 rounds)
    if industry_codes is not None:
        unique_industries = np.unique(industry_codes)
        for _outer in range(10):
            industry_violated = False
            for ind in unique_industries:
                mask_ind = industry_codes == ind
                ind_weight = float(np.sum(weights[mask_ind]))
                if ind_weight > max_industry_pct + 1e-9:
                    scale = max_industry_pct / ind_weight
                    upper[mask_ind] = np.minimum(upper[mask_ind], weights[mask_ind] * scale)
                    industry_violated = True
            if not industry_violated:
                break
            weights = _project_simplex_upper(target, upper)

    return weights


def _project_simplex_upper(v: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Project onto simplex with upper bounds: sum(w)=1, 0 <= w_i <= u_i."""
    n = len(v)
    feasible_mask = upper > 0
    n_feasible = int(np.sum(feasible_mask))
    if n_feasible == 0:
        return np.zeros(n, dtype=np.float64)

    total_cap = float(np.sum(upper))
    if total_cap < 1.0 - 1e-9:
        # Infeasible: fill caps and normalize
        w = upper.copy()
        s = float(np.sum(w))
        if s < _MIN_WEIGHT_SUM:
            return np.zeros(n, dtype=np.float64)
        return w / s

    # Bisection to find lambda
    lo = float(np.min(v[feasible_mask])) - float(np.max(upper[feasible_mask]))
    hi = float(np.max(v[feasible_mask]))

    def _f(lam: float) -> float:
        return float(np.sum(np.clip(v - lam, 0.0, upper))) - 1.0

    for _ in range(50):
        mid = (lo + hi) / 2.0
        val = _f(mid)
        if abs(val) < 1e-12:
            break
        if val > 0:
            lo = mid
        else:
            hi = mid

    lam = (lo + hi) / 2.0
    w = np.clip(v - lam, 0.0, upper)
    s = float(np.sum(w))
    if s < _MIN_WEIGHT_SUM:
        return np.zeros(n, dtype=np.float64)
    return w / s


if GYM_AVAILABLE:

    class PortfolioWeightEnv(gym.Env):  # type: ignore[misc]
        """Continuous-weight portfolio optimization environment.

        Observation:
            Box(-inf, inf, shape=(n_stocks * (n_factors + 1),))
            = concat([factor_panel[t].flatten(), current_weights])

        Action:
            Box(0.0, 1.0, shape=(n_stocks,)) — projected via _project_weights.

        Reward:
            Per ``config.reward_type``: return / sharpe / sortino / mean_variance
        """

        metadata: dict[str, Any] = {"render_modes": []}

        def __init__(
            self,
            config: PortfolioWeightConfig,
            factor_panel: np.ndarray,
            return_panel: np.ndarray,
            pct_change_panel: np.ndarray | None = None,
            is_st_panel: np.ndarray | None = None,
            is_suspended_panel: np.ndarray | None = None,
            days_since_ipo_panel: np.ndarray | None = None,
            industry_panel: np.ndarray | None = None,
        ) -> None:
            super().__init__()

            self.config = config
            self.factor_panel = factor_panel.astype(np.float32)
            self.return_panel = return_panel.astype(np.float32)

            n_dates, n_stocks, n_factors = factor_panel.shape
            self.n_dates = n_dates
            self.n_stocks = n_stocks
            self.n_factors = n_factors

            self._pct_change_panel = (
                pct_change_panel.astype(np.float32)
                if pct_change_panel is not None
                else np.zeros((n_dates, n_stocks), dtype=np.float32)
            )
            self._is_st_panel = (
                is_st_panel.astype(np.bool_)
                if is_st_panel is not None
                else np.zeros((n_dates, n_stocks), dtype=np.bool_)
            )
            self._is_suspended_panel = (
                is_suspended_panel.astype(np.bool_)
                if is_suspended_panel is not None
                else np.zeros((n_dates, n_stocks), dtype=np.bool_)
            )
            self._days_since_ipo_panel = (
                days_since_ipo_panel.astype(np.float32)
                if days_since_ipo_panel is not None
                else np.full(
                    (n_dates, n_stocks), NEW_STOCK_PROTECT_DAYS * 2, dtype=np.float32
                )
            )
            self._industry_codes = (
                industry_panel.astype(np.int32) if industry_panel is not None else None
            )

            obs_dim = n_stocks * (n_factors + 1)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(n_stocks,), dtype=np.float32
            )

            self._current_step: int = 0
            self._current_weights: np.ndarray = np.zeros(n_stocks, dtype=np.float64)
            self._weights_history: list[np.ndarray] = []
            self._returns_history: list[np.ndarray] = []
            self._cumulative_reward: float = 0.0
            self._episode_rewards: list[float] = []

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            super().reset(seed=seed)
            self._current_step = 0
            self._current_weights = np.zeros(self.n_stocks, dtype=np.float64)
            self._weights_history = []
            self._returns_history = []
            self._cumulative_reward = 0.0
            self._episode_rewards = []

            obs = self._get_obs(0)
            info = {"step": 0, "n_stocks": self.n_stocks, "n_factors": self.n_factors}
            return obs, info

        def step(
            self, action: np.ndarray
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            t = self._current_step

            trading_mask = self._compute_trading_mask(t)

            weights = _project_weights(
                raw_action=action,
                max_position_pct=self.config.max_position_pct,
                max_industry_pct=self.config.max_industry_pct,
                industry_codes=self._industry_codes,
                trading_mask=trading_mask,
            )

            raw_returns = self.return_panel[t].astype(np.float64)
            masked_returns = _apply_trading_mask(
                returns=raw_returns,
                pct_changes=self._pct_change_panel[t].astype(np.float64),
                is_st=self._is_st_panel[t],
                is_suspended=self._is_suspended_panel[t],
                days_since_ipo=self._days_since_ipo_panel[t].astype(np.float64),
            )

            self._weights_history.append(weights.copy())
            self._returns_history.append(masked_returns.copy())

            reward = self._compute_reward(weights, masked_returns)

            prev_w = self._current_weights
            turnover = float(np.sum(np.abs(weights - prev_w)))
            turnover_cost = self.config.turnover_penalty * turnover
            reward -= turnover_cost

            trade_cost = self.config.cost_bps / 10_000.0 if turnover > 1e-6 else 0.0
            reward -= trade_cost

            self._current_weights = weights.copy()
            self._cumulative_reward += reward
            self._episode_rewards.append(reward)

            self._current_step += 1
            terminated = self._current_step >= self.n_dates - self.config.forward_period
            truncated = False

            if not terminated:
                obs = self._get_obs(self._current_step)
            else:
                obs = self._get_obs(max(0, self._current_step - 1))

            portfolio_return = float(np.dot(weights, masked_returns))
            info = {
                "step": t,
                "portfolio_return": portfolio_return,
                "trade_cost": trade_cost,
                "turnover_cost": turnover_cost,
                "turnover": turnover,
                "n_nonzero": int(np.sum(weights > 1e-6)),
                "max_weight": float(np.max(weights)),
                "cumulative_reward": self._cumulative_reward,
            }
            return obs, reward, terminated, truncated, info

        def render(self) -> None:
            warnings.warn(
                "PortfolioWeightEnv.render() prints text only; no GUI.",
                stacklevel=2,
            )
            print(
                f"Step={self._current_step}, "
                f"CumReward={self._cumulative_reward:.4f}, "
                f"MaxWeight={float(np.max(self._current_weights)):.4f}"
            )

        def _get_obs(self, t: int) -> np.ndarray:
            factors = self.factor_panel[t]
            factors_flat = factors.reshape(-1)
            weights_flat = self._current_weights.astype(np.float32)
            return np.concatenate([factors_flat, weights_flat], axis=0).astype(np.float32)

        def _compute_trading_mask(self, t: int) -> np.ndarray:
            pct = self._pct_change_panel[t].astype(np.float64)
            is_st = self._is_st_panel[t].astype(bool)
            is_susp = self._is_suspended_panel[t].astype(bool)
            days_ipo = self._days_since_ipo_panel[t].astype(np.float64)

            mask = np.ones(self.n_stocks, dtype=bool)
            mask &= np.abs(pct) < LIMIT_PCT_THRESHOLD
            mask &= ~is_st
            mask &= ~is_susp
            mask &= days_ipo >= NEW_STOCK_PROTECT_DAYS
            return mask

        def _compute_reward(
            self,
            weights: np.ndarray,
            masked_returns: np.ndarray,
        ) -> float:
            rtype = self.config.reward_type

            if rtype == "return":
                return float(np.dot(weights, masked_returns))

            wh = np.array(self._weights_history, dtype=np.float64)
            rh = np.array(self._returns_history, dtype=np.float64)

            if rtype == "sharpe":
                return sharpe_reward(
                    weights_history=wh,
                    return_panel=rh,
                    rolling_window=self.config.rolling_window,
                )
            if rtype == "sortino":
                return sortino_reward(
                    weights_history=wh,
                    return_panel=rh,
                    rolling_window=self.config.rolling_window,
                )
            if rtype == "mean_variance":
                return mean_variance_reward(
                    weights=weights,
                    return_panel=rh,
                    risk_aversion=self.config.risk_aversion,
                )

            raise ValueError(f"Unknown reward_type: {rtype!r}")

else:
    class PortfolioWeightEnv:  # type: ignore[no-redef]
        """Placeholder when gymnasium is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "gymnasium is required for PortfolioWeightEnv. "
                "Install with: pip install aurumq-rl[train]"
            )


__all__ = [
    "PortfolioWeightConfig",
    "PortfolioWeightEnv",
    "GYM_AVAILABLE",
    "_project_weights",
    "_project_simplex_upper",
]
