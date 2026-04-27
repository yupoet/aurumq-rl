"""RL reward function library.

Four reward types:
  return        — simple forward return
  sharpe        — rolling Sharpe ratio
  sortino       — rolling Sortino ratio
  mean_variance — Markowitz mean - λ * variance
"""

from __future__ import annotations

import numpy as np

# Numerical guards
_MIN_STD: float = 1e-8

# Trading-day annualization factor
ANNUALIZATION_FACTOR: float = 252.0


def _portfolio_return_series(
    weights_history: np.ndarray,
    return_panel: np.ndarray,
) -> np.ndarray:
    """Compute historical portfolio return time series.

    Parameters
    ----------
    weights_history : (T, n_stocks)
    return_panel    : (T, n_stocks)

    Returns
    -------
    port_returns : (T,)
    """
    return np.einsum("ij,ij->i", weights_history, return_panel).astype(np.float64)


def simple_return_reward(
    weights: np.ndarray,
    forward_returns: np.ndarray,
    cost_bps: float,
    turnover_penalty: float,
    prev_weights: np.ndarray,
) -> float:
    """Simple per-step return with cost + turnover penalty."""
    port_return = float(np.dot(weights, forward_returns))
    turnover = float(np.sum(np.abs(weights - prev_weights)))
    trade_cost = cost_bps / 10_000.0 if turnover > 1e-6 else 0.0
    turnover_cost = turnover_penalty * turnover
    return port_return - trade_cost - turnover_cost


def sharpe_reward(
    weights_history: np.ndarray,
    return_panel: np.ndarray,
    rolling_window: int = 20,
    annualization_factor: float = ANNUALIZATION_FACTOR,
) -> float:
    """Rolling Sharpe ratio reward.

    Formula:
        port_returns[t] = weights_history[t] · return_panel[t]
        window = min(rolling_window, T)
        mu     = mean(port_returns[-window:])
        sigma  = std(port_returns[-window:], ddof=1)
        Sharpe = mu / max(sigma, _MIN_STD) * sqrt(annualization_factor)
    """
    if weights_history.ndim != 2 or return_panel.ndim != 2:
        raise ValueError("weights_history and return_panel must be 2D (T, n_stocks)")
    if weights_history.shape != return_panel.shape:
        raise ValueError(
            f"shape mismatch: weights_history={weights_history.shape}, "
            f"return_panel={return_panel.shape}"
        )

    port_returns = _portfolio_return_series(weights_history, return_panel)
    w = min(rolling_window, len(port_returns))
    recent = port_returns[-w:]

    mu = float(np.mean(recent))
    sigma = float(np.std(recent, ddof=min(1, len(recent) - 1)))
    return mu / max(sigma, _MIN_STD) * float(np.sqrt(annualization_factor))


def sortino_reward(
    weights_history: np.ndarray,
    return_panel: np.ndarray,
    rolling_window: int = 20,
    annualization_factor: float = ANNUALIZATION_FACTOR,
    target_return: float = 0.0,
) -> float:
    """Rolling Sortino ratio reward (downside-deviation based).

    Formula:
        downside_std = sqrt(mean(min(port_returns - target, 0)^2))
        Sortino = (mu - target) / max(downside_std, _MIN_STD) * sqrt(annualization)
    """
    if weights_history.ndim != 2 or return_panel.ndim != 2:
        raise ValueError("weights_history and return_panel must be 2D (T, n_stocks)")
    if weights_history.shape != return_panel.shape:
        raise ValueError(
            f"shape mismatch: weights_history={weights_history.shape}, "
            f"return_panel={return_panel.shape}"
        )

    port_returns = _portfolio_return_series(weights_history, return_panel)
    w = min(rolling_window, len(port_returns))
    recent = port_returns[-w:]

    mu = float(np.mean(recent))
    excess = recent - target_return
    downside = np.minimum(excess, 0.0)
    downside_std = float(np.sqrt(np.mean(downside**2)))

    return (mu - target_return) / max(downside_std, _MIN_STD) * float(np.sqrt(annualization_factor))


def mean_variance_reward(
    weights: np.ndarray,
    return_panel: np.ndarray,
    risk_aversion: float = 1.0,
) -> float:
    """Markowitz mean-variance reward.

    Formula:
        mu_vec  = mean(return_panel, axis=0)
        Sigma   = cov(return_panel.T)
        reward  = w · mu_vec - risk_aversion * (w' Σ w)
    """
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1D, got ndim={weights.ndim}")
    if return_panel.ndim != 2:
        raise ValueError(f"return_panel must be 2D (T, n_stocks), got ndim={return_panel.ndim}")
    if len(weights) != return_panel.shape[1]:
        raise ValueError(
            f"weights length {len(weights)} != return_panel columns {return_panel.shape[1]}"
        )

    mu_vec = np.mean(return_panel, axis=0).astype(np.float64)
    w = weights.astype(np.float64)

    expected_return = float(np.dot(w, mu_vec))

    if return_panel.shape[0] < 2:
        return expected_return

    sigma = np.cov(return_panel.T)
    portfolio_variance = float(w @ sigma @ w)

    return expected_return - risk_aversion * portfolio_variance


__all__ = [
    "ANNUALIZATION_FACTOR",
    "simple_return_reward",
    "sharpe_reward",
    "sortino_reward",
    "mean_variance_reward",
]
