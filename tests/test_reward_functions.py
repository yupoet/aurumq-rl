"""Tests for the reward function library with hand-checked numerical inputs."""

from __future__ import annotations

import numpy as np
import pytest

from aurumq_rl.reward_functions import (
    ANNUALIZATION_FACTOR,
    MIN_RISK_WINDOW,
    VOLATILITY_FLOOR,
    DifferentialSharpe,
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


def test_sharpe_constant_returns_uses_volatility_floor() -> None:
    # Constant portfolio returns → std == 0 → guarded by VOLATILITY_FLOOR
    weights_history = np.tile(np.array([1.0, 0.0]), (10, 1))
    return_panel = np.full((10, 2), [0.01, 0.0])
    s = sharpe_reward(weights_history, return_panel, rolling_window=10)
    # mean = 0.01, std == 0 → floored denominator, large but bounded Sharpe
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
    s = sharpe_reward(
        weights_history, panel, rolling_window=5, annualization_factor=ANNUALIZATION_FACTOR
    )
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
# C5: div-zero guards (warm-up window + volatility floor)
# ---------------------------------------------------------------------------


def test_sharpe_single_sample_history_returns_zero() -> None:
    """Episode step 0: one-sample history → neutral 0.0, not mu/1e-8 spike."""
    weights_history = np.array([[1.0, 0.0]])
    panel = np.array([[0.001, 0.0]])
    assert sharpe_reward(weights_history, panel) == 0.0


def test_sortino_single_sample_history_returns_zero() -> None:
    """Episode step 0: one-sample history → neutral 0.0, not mu/1e-8 spike."""
    weights_history = np.array([[1.0, 0.0]])
    panel = np.array([[0.001, 0.0]])
    assert sortino_reward(weights_history, panel) == 0.0


def test_warmup_below_min_risk_window_returns_zero() -> None:
    """Every history length below MIN_RISK_WINDOW yields neutral 0.0 reward."""
    for t in range(1, MIN_RISK_WINDOW):
        weights_history = np.tile(np.array([1.0, 0.0]), (t, 1))
        panel = np.tile(np.array([0.01, 0.0]), (t, 1))
        assert sharpe_reward(weights_history, panel) == 0.0
        assert sortino_reward(weights_history, panel) == 0.0


def test_sortino_all_nonnegative_window_bounded_by_floor() -> None:
    """Window >= MIN_RISK_WINDOW with no downside → floored denominator.

    downside_std == 0 whenever all window returns >= target; the reward must
    equal the closed form mu / VOLATILITY_FLOOR * sqrt(annualization), not
    a mu/1e-8 explosion.
    """
    returns = np.array([0.01, 0.0, 0.02, 0.005, 0.0, 0.015])
    weights_history = np.tile(np.array([1.0, 0.0]), (6, 1))
    panel = np.column_stack([returns, np.zeros(6)])
    s = sortino_reward(weights_history, panel, rolling_window=6, target_return=0.0)
    expected = returns.mean() / VOLATILITY_FLOOR * np.sqrt(ANNUALIZATION_FACTOR)
    assert np.isfinite(s)
    assert s == pytest.approx(expected)
    assert abs(s) <= 0.1 / VOLATILITY_FLOOR * np.sqrt(ANNUALIZATION_FACTOR)


def test_sharpe_zero_variance_window_bounded_by_floor() -> None:
    """Constant returns (zero variance) → finite reward equal to floored form."""
    weights_history = np.tile(np.array([1.0, 0.0]), (10, 1))
    panel = np.tile(np.array([0.01, 0.0]), (10, 1))
    s = sharpe_reward(weights_history, panel, rolling_window=10)
    expected = 0.01 / VOLATILITY_FLOOR * np.sqrt(ANNUALIZATION_FACTOR)
    assert np.isfinite(s)
    assert s == pytest.approx(expected)


def test_sharpe_normal_case_unchanged_by_floor() -> None:
    """Genuine variance (std >> floor): value identical to the pre-fix formula."""
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    weights_history = np.tile(np.array([1.0, 0.0]), (5, 1))
    panel = np.column_stack([returns, np.zeros(5)])
    s = sharpe_reward(weights_history, panel, rolling_window=5)
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(ANNUALIZATION_FACTOR)
    assert s == pytest.approx(expected)
    assert s == pytest.approx(3.8824, abs=1e-3)  # pinned pre-fix value


def test_sortino_normal_case_unchanged_by_floor() -> None:
    """Genuine downside variance: value identical to the pre-fix formula."""
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    weights_history = np.tile(np.array([1.0, 0.0]), (5, 1))
    panel = np.column_stack([returns, np.zeros(5)])
    s = sortino_reward(weights_history, panel, rolling_window=5, target_return=0.0)
    downside_std = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
    expected = returns.mean() / downside_std * np.sqrt(ANNUALIZATION_FACTOR)
    assert s == pytest.approx(expected)
    assert s == pytest.approx(6.8873, abs=1e-3)  # pinned pre-fix value


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


# ---------------------------------------------------------------------------
# DifferentialSharpe (Moody & Saffell 1998) — issue #2
# ---------------------------------------------------------------------------

_DSR_RETURNS = [0.01, -0.02, 0.015, 0.03, -0.01, 0.02, 0.005, -0.015]


def test_differential_sharpe_matches_hand_computed_recurrence() -> None:
    """DSR.update reproduces the Moody & Saffell recurrence for post-warm-up steps.

    The first update warm-starts A, B from the seeding observation
    (``A_1 = R_1``, ``B_1 = R_1^2``) and returns 0.0; thereafter A_t, B_t
    evolve via the EMA recurrence every step, but only the reward past the
    MIN_RISK_WINDOW gate is compared here (that is where the raw formula
    applies). We recompute the recurrence — with the same warm-start
    seeding — independently in the test.
    """
    eta = 0.05
    dsr = DifferentialSharpe(eta=eta)
    actual = [dsr.update(r) for r in _DSR_RETURNS]

    a, b = 0.0, 0.0
    expected: list[float] = []
    for i, r in enumerate(_DSR_RETURNS):
        n = i + 1
        if n == 1:  # warm-start seeding step: emit 0.0, seed A/B, no EMA step
            expected.append(0.0)
            a, b = r, r * r
            continue
        d_a = r - a
        d_b = r * r - b
        if n <= MIN_RISK_WINDOW:
            expected.append(0.0)
        else:
            variance = b - a * a
            denom = max(variance, VOLATILITY_FLOOR**2) ** 1.5
            expected.append((b * d_a - 0.5 * a * d_b) / denom)
        a = a + eta * d_a
        b = b + eta * d_b

    for i in range(MIN_RISK_WINDOW, len(_DSR_RETURNS)):
        assert actual[i] == pytest.approx(expected[i])
    # Pinned reference values (guards against silent formula drift).
    assert actual[5] == pytest.approx(0.9626651786208611, rel=1e-6)
    assert actual[6] == pytest.approx(-0.10059759195011124, rel=1e-6)
    assert actual[7] == pytest.approx(-6.292455580242512, rel=1e-6)


def test_differential_sharpe_single_sample_returns_zero() -> None:
    """The seeding update returns 0.0 (no DSR on the warm-start sample)."""
    dsr = DifferentialSharpe()
    assert dsr.update(0.05) == 0.0


def test_differential_sharpe_warmup_below_min_risk_window_returns_zero() -> None:
    dsr = DifferentialSharpe()
    for _ in range(MIN_RISK_WINDOW):
        assert dsr.update(0.01) == 0.0


@pytest.mark.parametrize("r", [0.01, 0.20, -0.20, 0.44])
def test_differential_sharpe_constant_stream_stays_zero_at_any_magnitude(r: float) -> None:
    """Pathological all-equal return stream: true variance is 0 throughout.

    This is the C5 failure class (mu / near-zero-sigma spike) applied to the
    recurrent DSR form. The warm-start (A=R, B=R^2 from the seeding step)
    makes every subsequent dA = dB = 0, so DSR is *exactly* 0.0 forever — at
    ANY magnitude, including A-share limit-up levels (±20%, and the ±44%
    first-day-after-IPO / ST-band extremes).

    Regression guard: the earlier zero-initialized-EMA implementation left a
    phantom-variance transient whose peak |DSR| ≈ 0.5*|R|/VOLATILITY_FLOOR
    scaled LINEARLY with |R| — ≈49 at R=1% but ≈990 at R=20% and >1e3 above,
    which is exactly the C5-class spike this reward mode exists to avoid. The
    r=0.20 case below is the one that would have caught it; it is well under
    1.0 (in fact exactly 0.0) now.
    """
    dsr = DifferentialSharpe()
    values = [dsr.update(r) for _ in range(400)]
    assert all(np.isfinite(v) for v in values)
    assert all(v == 0.0 for v in values)  # exact 0.0, no transient
    assert max(abs(v) for v in values) < 1.0


def test_differential_sharpe_varying_stream_is_finite_and_bounded() -> None:
    """A genuinely varying stream produces finite, O(1)-magnitude DSR values."""
    rng = np.random.default_rng(3)
    returns = rng.standard_normal(300) * 0.01
    dsr = DifferentialSharpe()
    values = [dsr.update(float(r)) for r in returns]
    assert all(np.isfinite(v) for v in values)
    assert max(abs(v) for v in values) < 1e3


def test_differential_sharpe_cumulative_sum_tracks_sharpe_direction() -> None:
    """sum(DSR_t) tracks the *change* in the batch Sharpe ratio — the
    algorithmic justification for DSR as a per-step reward instead of a
    rolling-Sharpe snapshot.

    The correct property is about Sharpe *change*, not level: a stream whose
    risk-adjusted profile IMPROVES over the episode (a poor first half, a
    strong second half) has a positive cumulative DSR; one that DEGRADES
    (strong then poor) has a negative one. (A stationary stream, by contrast,
    has no Sharpe change and a cumulative DSR of ~0 ± noise — which is why an
    earlier version of this test, asserting the sign of a *stationary* good vs
    bad stream, was actually measuring a zero-init cold-start artifact rather
    than the real property; warm-starting removes that artifact.)

    Aggregated over a seed ensemble to stay robust, not brittle.
    """
    improving_sums = []
    degrading_sums = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        improving = np.concatenate([rng.normal(-0.01, 0.01, 100), rng.normal(0.01, 0.01, 100)])
        rng2 = np.random.default_rng(1000 + seed)
        degrading = np.concatenate([rng2.normal(0.01, 0.01, 100), rng2.normal(-0.01, 0.01, 100)])
        d_imp = DifferentialSharpe(eta=0.05)
        d_deg = DifferentialSharpe(eta=0.05)
        improving_sums.append(sum(d_imp.update(float(r)) for r in improving))
        degrading_sums.append(sum(d_deg.update(float(r)) for r in degrading))

    # Aggregate sign is unambiguous (improving positive, degrading negative).
    assert float(np.sum(improving_sums)) > 0
    assert float(np.sum(degrading_sums)) < 0
    assert float(np.mean(improving_sums)) > float(np.mean(degrading_sums))


def test_differential_sharpe_reset_clears_state() -> None:
    dsr = DifferentialSharpe(eta=0.1)
    for r in _DSR_RETURNS:
        dsr.update(r)
    assert dsr.a != 0.0 or dsr.b != 0.0

    dsr.reset()
    assert dsr.a == 0.0
    assert dsr.b == 0.0

    # A fresh instance and a reset instance must behave identically going
    # forward — no leakage of the prior episode's A/B into the next one.
    fresh = DifferentialSharpe(eta=0.1)
    for r in _DSR_RETURNS:
        assert dsr.update(r) == pytest.approx(fresh.update(r))


def test_differential_sharpe_invalid_eta_raises() -> None:
    with pytest.raises(ValueError):
        DifferentialSharpe(eta=0.0)
    with pytest.raises(ValueError):
        DifferentialSharpe(eta=1.5)


# ---------------------------------------------------------------------------
# PortfolioWeightEnv wiring — "differential_sharpe" reward mode (issue #2)
# ---------------------------------------------------------------------------


def test_portfolio_weight_env_differential_sharpe_mode() -> None:
    pytest.importorskip("gymnasium")
    import datetime

    from aurumq_rl.portfolio_weight_env import PortfolioWeightConfig, PortfolioWeightEnv

    n_dates, n_stocks, n_factors = 30, 8, 4
    rng = np.random.default_rng(0)
    factor = rng.standard_normal((n_dates, n_stocks, n_factors)).astype(np.float32)
    ret = (rng.standard_normal((n_dates, n_stocks)) * 0.01).astype(np.float32)

    config = PortfolioWeightConfig(
        start_date=datetime.date(2022, 1, 1),
        end_date=datetime.date(2022, 12, 31),
        n_factors=n_factors,
        forward_period=2,
        reward_type="differential_sharpe",
        cost_bps=0.0,
        turnover_penalty=0.0,
    )
    env = PortfolioWeightEnv(config=config, factor_panel=factor, return_panel=ret)
    env.reset(seed=0)

    action = np.full(n_stocks, 0.5, dtype=np.float32)
    _, reward, *_ = env.step(action)
    assert reward == 0.0  # warm-up: step 0 has degenerate DSR history

    rewards = [reward]
    for _ in range(n_dates - 3):
        _, reward, terminated, _, _ = env.step(action)
        rewards.append(reward)
        if terminated:
            break

    assert all(np.isfinite(r) for r in rewards)
    assert max(abs(r) for r in rewards) < 1e3

    # A fresh episode must not leak the previous episode's DSR state.
    env.reset(seed=0)
    _, reward0, *_ = env.step(action)
    assert reward0 == 0.0
