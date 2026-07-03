"""RL reward function library.

Five reward types:
  return              — simple forward return
  sharpe              — rolling Sharpe ratio
  sortino             — rolling Sortino ratio
  mean_variance       — Markowitz mean - λ * variance
  differential_sharpe — recurrent per-step Sharpe increment (Moody & Saffell 1998)
"""

from __future__ import annotations

import numpy as np

# Numerical guards (C5)
# Warm-up: below this many samples a rolling std/downside-std is degenerate
# (1 sample → std=0), so risk-adjusted rewards return neutral 0.0 instead.
MIN_RISK_WINDOW: int = 5
# Realistic daily-return volatility floor for Sharpe/Sortino denominators;
# keeps rewards bounded (~1e2) when a window has zero (downside) variance,
# instead of mu/1e-8 spikes (~1e7) that wreck the value-target/GAE scale.
VOLATILITY_FLOOR: float = 1e-4

# Trading-day annualization factor
ANNUALIZATION_FACTOR: float = 252.0

# Default DSR decay rate eta = 1 / rolling_window (default rolling_window=20
# in PortfolioWeightConfig), so the exponential moving average's effective
# memory span matches the rolling Sharpe/Sortino window — keeps the modes
# comparable when swapping reward_type without retuning anything else.
DEFAULT_DSR_ETA: float = 0.05


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
        Sharpe = mu / max(sigma, VOLATILITY_FLOOR) * sqrt(annualization_factor)

    Returns neutral 0.0 while the window holds fewer than MIN_RISK_WINDOW
    samples (warm-up), since the risk estimate is degenerate there.
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

    if len(recent) < MIN_RISK_WINDOW:  # warm-up: risk estimate degenerate (C5)
        return 0.0

    mu = float(np.mean(recent))
    sigma = float(np.std(recent, ddof=1))
    return mu / max(sigma, VOLATILITY_FLOOR) * float(np.sqrt(annualization_factor))


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
        Sortino = (mu - target) / max(downside_std, VOLATILITY_FLOOR) * sqrt(annualization)

    Returns neutral 0.0 while the window holds fewer than MIN_RISK_WINDOW
    samples (warm-up). The VOLATILITY_FLOOR also bounds the reward when the
    window has no downside returns (downside_std == 0, common in rallies).
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

    if len(recent) < MIN_RISK_WINDOW:  # warm-up: risk estimate degenerate (C5)
        return 0.0

    mu = float(np.mean(recent))
    excess = recent - target_return
    downside = np.minimum(excess, 0.0)
    downside_std = float(np.sqrt(np.mean(downside**2)))

    return (
        (mu - target_return)
        / max(downside_std, VOLATILITY_FLOOR)
        * float(np.sqrt(annualization_factor))
    )


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


class DifferentialSharpe:
    """Stateful Differential Sharpe Ratio (Moody & Saffell 1998).

    A recurrent, O(1)-per-step alternative to windowed ``sharpe_reward`` /
    ``sortino_reward``. Instead of a rolling-window Sharpe snapshot (which,
    summed over steps, optimizes neither Sharpe nor return — the C5 issue),
    DSR gives the per-step *marginal contribution* to the Sharpe ratio,
    tracked via exponentially-decayed moving averages of the return (``A``)
    and squared return (``B``):

        dA = R_t - A_{t-1}
        dB = R_t^2 - B_{t-1}
        DSR_t = (B_{t-1} * dA - 0.5 * A_{t-1} * dB) / (B_{t-1} - A_{t-1}^2)^1.5
        A_t = A_{t-1} + eta * dA
        B_t = B_{t-1} + eta * dB

    Summing DSR_t over an episode approximates the change in the batch
    Sharpe ratio (Moody & Saffell, "Reinforcement Learning for Trading",
    NeurIPS 1998), so maximizing cumulative DSR reward directly targets
    end-of-episode Sharpe.

    Numerical safety (the whole point vs. the C5 spike):

    * **Warm-starting A, B from the first observation** (``A_1 = R_1``,
      ``B_1 = R_1^2``; the seeding step itself returns 0.0). This is the
      standard EMA cold-start-bias fix, and it is what keeps DSR bounded on
      a constant stream. Zero-initializing A, B instead (the naive choice)
      makes the EMA-estimated variance ``B - A^2`` decay from its cold-start
      value; on a constant-return stream it crosses VOLATILITY_FLOOR^2 mid-
      trajectory (≈ step 179 at eta=0.05), and right there |DSR| peaks at
      ≈ 0.5 * |R| / VOLATILITY_FLOOR — a *transient* spike that scales
      LINEARLY with |R| (~49 at R=1%, but ~990 at R=20%, exceeding 1e3 at
      A-share limit-up magnitudes). Flooring the denominator does NOT remove
      this peak (it occurs while the apparent variance is above the floor).
      Warm-starting does: a constant stream then has ``A = R``, ``B = R^2``
      exactly from step 1, so every ``dA = dB = 0`` → numerator is exactly
      0 → DSR is exactly 0.0 forever, at ANY magnitude, with no transient.
    * **MIN_RISK_WINDOW warm-up gate**: the first MIN_RISK_WINDOW updates
      return 0.0 (A/B have not accumulated enough history), mirroring
      sharpe_reward/sortino_reward.
    * **VOLATILITY_FLOOR on the denominator**: floors the variance^1.5 term
      so an exactly-constant stream yields 0 / floor = 0.0 instead of a
      0 / 0 NaN, and a genuinely near-degenerate window stays finite —
      the same max(..., VOLATILITY_FLOOR) guard sharpe_reward/sortino_reward
      use. With warm-starting this floor is a belt-and-braces guard, not the
      primary defence (which is the warm-start itself).
    """

    def __init__(self, eta: float = DEFAULT_DSR_ETA) -> None:
        """Create a DSR tracker.

        Parameters
        ----------
        eta : float
            Decay / adaptation rate for the moving averages of R and R^2,
            in (0, 1]. Larger eta adapts faster but is noisier; smaller eta
            is smoother but slower to reflect regime changes. Default
            ``DEFAULT_DSR_ETA`` (1/20, matching the rolling-mode default
            window).
        """
        if not 0.0 < eta <= 1.0:
            raise ValueError(f"eta={eta} must be in (0, 1]")
        self.eta = eta
        self.a: float = 0.0
        self.b: float = 0.0
        self._n: int = 0

    def reset(self) -> None:
        """Reset A, B, and the warm-up counter to their initial state."""
        self.a = 0.0
        self.b = 0.0
        self._n = 0

    def update(self, r: float) -> float:
        """Consume one step's realized return and return this step's DSR.

        Parameters
        ----------
        r : float
            Realized portfolio return for this step (same quantity fed to
            sharpe_reward/sortino_reward's per-step return series).
        """
        self._n += 1

        # Warm-start: seed A, B from the first observation instead of from
        # zero. Avoids the cold-start-bias transient that would otherwise
        # spike |DSR| ≈ 0.5*|R|/VOLATILITY_FLOOR mid-trajectory on a
        # constant stream (see class docstring). The seeding step emits 0.0.
        if self._n == 1:
            self.a = r
            self.b = r * r
            return 0.0

        a_prev, b_prev = self.a, self.b
        d_a = r - a_prev
        d_b = r * r - b_prev

        if self._n <= MIN_RISK_WINDOW:
            dsr = 0.0  # warm-up: A/B have not accumulated enough history
        else:
            variance = b_prev - a_prev * a_prev
            # Floor the denominator so an exactly-constant stream gives
            # 0 / floor = 0.0 rather than a 0 / 0 NaN. With warm-starting
            # this is a belt-and-braces guard, not the primary defence.
            denom = max(variance, VOLATILITY_FLOOR**2) ** 1.5
            dsr = (b_prev * d_a - 0.5 * a_prev * d_b) / denom

        self.a = a_prev + self.eta * d_a
        self.b = b_prev + self.eta * d_b
        return float(dsr)


__all__ = [
    "ANNUALIZATION_FACTOR",
    "DEFAULT_DSR_ETA",
    "MIN_RISK_WINDOW",
    "VOLATILITY_FLOOR",
    "DifferentialSharpe",
    "simple_return_reward",
    "sharpe_reward",
    "sortino_reward",
    "mean_variance_reward",
]
