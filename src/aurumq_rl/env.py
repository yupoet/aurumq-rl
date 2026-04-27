"""Daily-frequency A-share stock-picking RL environment.

Design
------
* **State**: shape (n_stocks, n_factors), cross-sectionally z-scored, flattened.
* **Action**: Box(0.0, 1.0, shape=(n_stocks,)) — per-stock priority scores.
* **Reward**: r_t = mean_return(top_k) - cost_bps/10_000 - turnover_penalty * |Δw|
  where mean_return is the mean log-return over forward_period days
  for the top-k highest-scored stocks.

A-share constraints
-------------------
* T+1 (day-N buy → day-N+1 earliest sell) — env-level passthrough; enforced by
  backtest/live execution layer.
* Dynamic price limits (board-aware): main ±10%, ChiNext/STAR ±20%, BSE ±30%,
  ST ±5%. Untradeable stocks contribute 0 to reward.
  Set ``respect_dynamic_price_limits=False`` for legacy ±9.5% behavior.
* ST/*ST exclusion (configurable).
* Suspension (volume == 0) exclusion.
* New-stock protection (60 trading days post-IPO).
* Industry exposure cap: single SW-1 industry ≤ 30% of top-k.

Gymnasium is an optional dependency; if not installed, importing this module
still succeeds but instantiating ``StockPickingEnv`` raises ImportError.
"""

from __future__ import annotations

import datetime
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

# Optional gymnasium import
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


# Constants
LIMIT_PCT_THRESHOLD: float = 0.095  # legacy ±9.5% threshold (decimal form)
NEW_STOCK_PROTECT_DAYS: int = 60
MAX_INDUSTRY_WEIGHT: float = 0.30
LOG_RETURN_CLIP: float = -0.5


@dataclass(frozen=True)
class StockPickingConfig:
    """Static configuration for StockPickingEnv.

    Attributes
    ----------
    start_date / end_date:
        Training/eval date range (informational; data already filtered upstream).
    n_factors:
        Number of factor dimensions in the observation.
    top_k:
        Number of stocks to pick each step.
    forward_period:
        Forward-return window in trading days.
    cost_bps:
        One-side trading cost in basis points (commission + stamp + transfer fee).
    turnover_penalty:
        Coefficient on turnover (||w_t - w_{t-1}||_1 / top_k).
    respect_dynamic_price_limits:
        True (default) → use board-aware limits (requires stock_codes).
        False → use legacy ±9.5% threshold.
    """

    start_date: datetime.date = field(default_factory=lambda: datetime.date(2020, 1, 1))
    end_date: datetime.date = field(default_factory=lambda: datetime.date(2024, 12, 31))
    n_factors: int = 64
    top_k: int = 30
    forward_period: int = 10
    cost_bps: float = 30.0
    turnover_penalty: float = 0.001
    respect_dynamic_price_limits: bool = True

    def __post_init__(self) -> None:
        if self.start_date >= self.end_date:
            raise ValueError(
                f"start_date={self.start_date} must be before end_date={self.end_date}"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k={self.top_k} must be >= 1")
        if self.n_factors < 1:
            raise ValueError(f"n_factors={self.n_factors} must be >= 1")
        if self.forward_period < 1:
            raise ValueError(f"forward_period={self.forward_period} must be >= 1")


# ---------------------------------------------------------------------------
# Helper functions (no gymnasium dependency)
# ---------------------------------------------------------------------------


def _apply_trading_mask(
    returns: np.ndarray,
    pct_changes: np.ndarray,
    is_st: np.ndarray,
    is_suspended: np.ndarray,
    days_since_ipo: np.ndarray,
    stock_codes: list[str] | None = None,
    respect_dynamic_price_limits: bool = True,
) -> np.ndarray:
    """Zero out returns for untradeable stocks.

    Parameters
    ----------
    returns          : (n_stocks,) forward log-returns.
    pct_changes      : (n_stocks,) daily pct change as **decimals** (+10% = 0.10).
    is_st            : (n_stocks,) bool.
    is_suspended     : (n_stocks,) bool.
    days_since_ipo   : (n_stocks,) trading days since IPO.
    stock_codes      : list of Tushare-format codes, used for board detection.
    respect_dynamic_price_limits:
        True  → board-aware threshold via ``aurumq_rl.price_limits``.
        False → legacy ±9.5% threshold.

    Returns
    -------
    masked_returns : (n_stocks,) with untradeable stocks set to 0.
    """
    from aurumq_rl.price_limits import is_at_limit_down, is_at_limit_up

    mask = np.ones(len(returns), dtype=bool)

    if respect_dynamic_price_limits and stock_codes is not None:
        is_st_bool = is_st.astype(bool)
        for i, code in enumerate(stock_codes):
            st_flag = bool(is_st_bool[i])
            pct = float(pct_changes[i])
            try:
                if is_at_limit_up(code, pct, is_st=st_flag) or is_at_limit_down(
                    code, pct, is_st=st_flag
                ):
                    mask[i] = False
            except ValueError:
                # Unknown board → fall back to legacy threshold
                if abs(pct) >= LIMIT_PCT_THRESHOLD:
                    mask[i] = False
    else:
        # Legacy ±9.5% threshold
        mask &= np.abs(pct_changes) < LIMIT_PCT_THRESHOLD

    mask &= ~is_st.astype(bool)
    mask &= ~is_suspended.astype(bool)
    mask &= days_since_ipo >= NEW_STOCK_PROTECT_DAYS

    masked = returns.copy()
    masked[~mask] = 0.0
    return masked


def _apply_industry_constraint(
    scores: np.ndarray,
    top_k: int,
    industry_codes: np.ndarray,
) -> np.ndarray:
    """Greedy top-k selection with single-industry cap.

    Parameters
    ----------
    scores         : (n_stocks,) higher = better.
    top_k          : target portfolio size.
    industry_codes : (n_stocks,) SW-1 industry codes (int).

    Returns
    -------
    selected_indices : (top_k,) indices satisfying industry constraint.
    """
    max_per_industry = max(1, int(top_k * MAX_INDUSTRY_WEIGHT))
    sorted_idx = np.argsort(scores)[::-1]

    selected: list[int] = []
    industry_count: dict[int, int] = {}

    for idx in sorted_idx:
        if len(selected) >= top_k:
            break
        ind = int(industry_codes[idx])
        count = industry_count.get(ind, 0)
        if count < max_per_industry:
            selected.append(idx)
            industry_count[ind] = count + 1

    # Pad with leftover top-scoring if all industries exhausted
    if len(selected) < top_k:
        for idx in sorted_idx:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= top_k:
                break

    return np.array(selected, dtype=np.int64)


# ---------------------------------------------------------------------------
# Gym environment (defined only if gymnasium is available)
# ---------------------------------------------------------------------------

if GYM_AVAILABLE:

    class StockPickingEnv(gym.Env):  # type: ignore[misc]
        """Daily A-share stock-picking RL environment.

        Observation:
            Box(-inf, inf, shape=(n_stocks * n_factors,)) — flattened cross-section.

        Action:
            Box(0.0, 1.0, shape=(n_stocks,)) — per-stock priority scores.

        Reward:
            r_t = mean_log_return(top_k) - cost - turnover_penalty * |turnover|
            where untradeable stocks contribute 0.
        """

        metadata: dict[str, Any] = {"render_modes": []}

        def __init__(
            self,
            config: StockPickingConfig,
            factor_panel: np.ndarray,
            return_panel: np.ndarray,
            pct_change_panel: np.ndarray | None = None,
            is_st_panel: np.ndarray | None = None,
            is_suspended_panel: np.ndarray | None = None,
            days_since_ipo_panel: np.ndarray | None = None,
            industry_codes: np.ndarray | None = None,
            stock_codes: list[str] | None = None,
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
                else np.full((n_dates, n_stocks), NEW_STOCK_PROTECT_DAYS * 2, dtype=np.float32)
            )
            self._industry_codes = (
                industry_codes.astype(np.int32)
                if industry_codes is not None
                else np.zeros(n_stocks, dtype=np.int32)
            )
            self._stock_codes: list[str] | None = stock_codes

            # Gym spaces
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_stocks * n_factors,),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(n_stocks,),
                dtype=np.float32,
            )

            # Internal state
            self._current_step: int = 0
            self._prev_holdings: np.ndarray = np.zeros(n_stocks, dtype=np.float32)
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
            self._prev_holdings = np.zeros(self.n_stocks, dtype=np.float32)
            self._cumulative_reward = 0.0
            self._episode_rewards = []

            obs = self._get_obs(0)
            info = {"step": 0, "n_stocks": self.n_stocks, "n_factors": self.n_factors}
            return obs, info

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            t = self._current_step

            selected_idx = _apply_industry_constraint(
                action, self.config.top_k, self._industry_codes
            )

            # Build equal-weight holdings
            holdings = np.zeros(self.n_stocks, dtype=np.float32)
            if len(selected_idx) > 0:
                holdings[selected_idx] = 1.0 / len(selected_idx)

            raw_returns = self.return_panel[t]
            masked_returns = _apply_trading_mask(
                returns=raw_returns.astype(np.float64),
                pct_changes=self._pct_change_panel[t].astype(np.float64),
                is_st=self._is_st_panel[t],
                is_suspended=self._is_suspended_panel[t],
                days_since_ipo=self._days_since_ipo_panel[t].astype(np.float64),
                stock_codes=self._stock_codes,
                respect_dynamic_price_limits=self.config.respect_dynamic_price_limits,
            )

            portfolio_return = float(np.dot(holdings, masked_returns))

            turnover = float(np.sum(np.abs(holdings - self._prev_holdings)))
            turnover_cost = self.config.turnover_penalty * turnover
            trade_cost = self.config.cost_bps / 10_000.0 if turnover > 1e-6 else 0.0

            reward = portfolio_return - trade_cost - turnover_cost

            self._cumulative_reward += reward
            self._episode_rewards.append(reward)
            self._prev_holdings = holdings.copy()

            self._current_step += 1
            terminated = self._current_step >= self.n_dates - self.config.forward_period
            truncated = False

            if not terminated:
                obs = self._get_obs(self._current_step)
            else:
                obs = self._get_obs(max(0, self._current_step - 1))

            info = {
                "step": t,
                "portfolio_return": portfolio_return,
                "trade_cost": trade_cost,
                "turnover_cost": turnover_cost,
                "turnover": turnover,
                "n_selected": int(np.sum(holdings > 0)),
                "cumulative_reward": self._cumulative_reward,
                "selected_indices": selected_idx.tolist(),
            }

            return obs, reward, terminated, truncated, info

        def render(self) -> None:
            """Print a textual summary; no GUI."""
            warnings.warn(
                "StockPickingEnv.render() prints text only; no GUI.",
                stacklevel=2,
            )
            print(f"Step={self._current_step}, CumReward={self._cumulative_reward:.4f}")

        def _get_obs(self, t: int) -> np.ndarray:
            """Flatten factor cross-section at time t."""
            obs = self.factor_panel[t]
            return obs.reshape(-1).astype(np.float32)

else:
    # Placeholder when gymnasium is missing (import succeeds, instantiation raises)
    class StockPickingEnv:  # type: ignore[no-redef]
        """Placeholder when gymnasium is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "gymnasium is required for StockPickingEnv. "
                "Install with: pip install aurumq-rl[train]"
            )


__all__ = [
    "StockPickingConfig",
    "StockPickingEnv",
    "GYM_AVAILABLE",
    "LIMIT_PCT_THRESHOLD",
    "NEW_STOCK_PROTECT_DAYS",
    "MAX_INDUSTRY_WEIGHT",
]
