"""Tests for the reward function library with hand-checked numerical inputs."""

from __future__ import annotations

import numpy as np
import pytest

from aurumq_rl.reward_functions import (
    ANNUALIZATION_FACTOR,
    mean_variance_reward,
    sharpe_reward,
    simple_return_reward,
    sortino_reward,
)


# ---------------------------------------------------------------------------
# simple_return_reward
# ---------------------------------------------------------------------------


def test_simple_return_no_turnover_no_cost() -> None:
    weights = np.array([0.5, 0.5])
    forward = np.array([0.02, 0.04])
    prev = np.array([0.5, 0.5])
    r = simple_return_reward(
        weights=weights,
        forward_returns=forward,
        cost_bps=0.0,
        turnover_penalty=0.0,
        prev_weights=prev,
    )
    assert r == pytest.approx(0.03)  # mean(0.02, 0.04)


def test_simple_return_turnover_penalty_applied() -> None:
    weights = np.array([1.0, 0.0])
    prev = np.array([0.0, 1.0])
    forward = np.array([0.05, 0.05])
    r_no_pen = simple_return_reward(weights, forward, 0.0, 0.0, prev)
    r_with_pen = simple_return_reward(weights, forward, 0.0, 0.1, prev)
    # Turnover = 2.0, penalty 0.1 → expected reduction = 0.2
    assert r_with_pen < r_no_pen
    assert r_with_pen == pytest.approx(0.05 - 0.2)


def test_simple_return_cost_bps_applied_only_when_trading() -> None:
    same = np.array([0.5, 0.5])
    forward = np.array([0.02, 0.02])
    r = simple_return_reward(same, forward, cost_bps=30.0, turnover_penalty=0.0, prev_weights=same)
    # Zero turnover → no cost
    assert r == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# sharpe_reward
# ---------------------------------------------------------------------------


def test_sharpe_constant_returns_uses_min_std() -> None:
    # Constant portfolio returns → std == 0 → guarded by _MIN_STD
    weights_history = np.tile(np.array([1.0, 0.0]), (10, 1))
    return_panel = np.full((10, 2), [0.01, 0.0])
    s = sharpe_reward(weights_history, return_panel, rolling_window=10)
    # mean = 0.01, std ~ 0 → very large positive Sharpe
    assert s > 0
    assert np.isfinite(s)


def test_sharpe_zero_mean_zero_score() -> None:
    rng = np.random.default_rng(42)
    weights_history = np.tile(np.array([0.5, 0.5]), (60, 1))
    # Symmetric returns around 0
    panel = rng.standard_normal((60, 2)) * 0.01
    panel -= panel.mean(axis=0, keepdims=True)
    s = sharpe_reward(weights_history, panel, rolling_window=60)
    assert abs(s) < 1.0  # near zero but with some sqrt(252) scaling possible


def test_sharpe_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        sharpe_reward(
            weights_history=np.zeros((10, 2)),
            return_panel=np.zeros((10, 3)),
        )


def test_sharpe_uses_annualization_factor() -> None:
    weights_history = np.tile(np.array([1.0, 0.0]), (5, 1))
    panel = np.tile(np.array([0.01, 0.0]), (5, 1))
    s = sharpe_reward(weights_history, panel, rolling_window=5, annualization_factor=ANNUALIZATION_FACTOR)
    assert s > 0


# ---------------------------------------------------------------------------
# sortino_reward
# ---------------------------------------------------------------------------


def test_sortino_all_positive_returns_huge() -> None:
    """All returns above target → near-zero downside std → very large Sortino."""
    weights_history = np.tile(np.array([1.0, 0.0]), (5, 1))
    panel = np.tile(np.array([0.02, 0.0]), (5, 1))
    s = sortino_reward(weights_history, panel, rolling_window=5, target_return=0.0)
    assert s > 0


def test_sortino_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        sortino_reward(
            weights_history=np.zeros((5, 2)),
            return_panel=np.zeros((4, 2)),  # mismatched T
        )


def test_sortino_target_return_shifts_score() -> None:
    weights_history = np.tile(np.array([1.0, 0.0]), (10, 1))
    panel = np.tile(np.array([0.01, 0.0]), (10, 1))
    s_low = sortino_reward(weights_history, panel, target_return=0.0)
    s_high = sortino_reward(weights_history, panel, target_return=0.02)
    # raising target_return above the actual return reduces the numerator
    assert s_low > s_high


# ---------------------------------------------------------------------------
# mean_variance_reward
# ---------------------------------------------------------------------------


def test_mean_variance_balances_return_and_risk() -> None:
    rng = np.random.default_rng(0)
    weights = np.array([0.5, 0.5])
    panel = rng.standard_normal((30, 2)) * 0.01
    # Higher risk_aversion → lower (more penalized) score
    r_low = mean_variance_reward(weights, panel, risk_aversion=0.0)
    r_high = mean_variance_reward(weights, panel, risk_aversion=10.0)
    assert r_low >= r_high


def test_mean_variance_single_row_returns_expected() -> None:
    """With only one observation, no covariance → reward == mean dot weights."""
    weights = np.array([1.0, 0.0])
    panel = np.array([[0.05, -0.02]])
    r = mean_variance_reward(weights, panel, risk_aversion=1.0)
    assert r == pytest.approx(0.05)


def test_mean_variance_weights_must_match_panel_cols() -> None:
    with pytest.raises(ValueError):
        mean_variance_reward(
            weights=np.array([1.0, 0.0, 0.0]),
            return_panel=np.zeros((10, 2)),
        )


def test_mean_variance_invalid_dims() -> None:
    with pytest.raises(ValueError):
        mean_variance_reward(
            weights=np.zeros((2, 2)),  # 2D not allowed
            return_panel=np.zeros((10, 2)),
        )
    with pytest.raises(ValueError):
        mean_variance_reward(
            weights=np.zeros(2),
            return_panel=np.zeros(10),  # 1D not allowed
        )
