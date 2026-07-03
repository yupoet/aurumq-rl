"""Tests for src/aurumq_rl/eval_metrics.py (issue #6).

TDD RED-then-GREEN: these tests were written before the implementation and
pin the DIRECTION each statistic must move under the scenarios described in
the issue #6 brief, plus at least one value pinned against a hand/reference
computation for the Deflated Sharpe Ratio.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aurumq_rl.eval_metrics import (
    deflated_sharpe_ratio,
    hac_mean_ci,
    hac_standard_error,
    probability_of_backtest_overfitting,
    spearman_ic,
    spearman_ic_per_date,
    split_selection_confirmation,
)

# ---------------------------------------------------------------------------
# 1. Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


def test_dsr_strong_single_trial_is_high():
    dsr = deflated_sharpe_ratio(
        observed_sharpe=0.15, n_trials=1, returns_skew=0.0, returns_kurtosis=3.0, n_obs=252
    )
    assert dsr > 0.95


def test_dsr_deflation_bites_with_many_trials():
    """Same observed Sharpe, same return distribution, same sample size —
    but selected as the best of 1000 trials instead of 1. The multiple-
    testing correction must materially lower the DSR."""
    kwargs = dict(observed_sharpe=0.15, returns_skew=0.0, returns_kurtosis=3.0, n_obs=252)
    dsr_1 = deflated_sharpe_ratio(n_trials=1, **kwargs)
    dsr_1000 = deflated_sharpe_ratio(n_trials=1000, **kwargs)
    assert dsr_1000 < dsr_1
    assert dsr_1 - dsr_1000 > 0.3  # "materially lower" per the issue #6 brief


def test_dsr_monotonic_in_n_trials():
    kwargs = dict(observed_sharpe=0.15, returns_skew=0.0, returns_kurtosis=3.0, n_obs=252)
    dsrs = [deflated_sharpe_ratio(n_trials=n, **kwargs) for n in (1, 10, 100, 1000, 10000)]
    assert all(a >= b for a, b in zip(dsrs, dsrs[1:], strict=False))


def test_dsr_zero_edge_strategy_near_half():
    dsr = deflated_sharpe_ratio(
        observed_sharpe=0.0, n_trials=1, returns_skew=0.0, returns_kurtosis=3.0, n_obs=252
    )
    assert dsr == pytest.approx(0.5, abs=1e-9)


def test_dsr_zero_edge_with_many_trials_is_at_or_below_half():
    dsr = deflated_sharpe_ratio(
        observed_sharpe=0.0, n_trials=200, returns_skew=0.0, returns_kurtosis=3.0, n_obs=252
    )
    assert dsr <= 0.5


def test_dsr_hand_reference_value():
    """Hand/reference computation pinned against the closed-form Bailey &
    Lopez de Prado (2014) formula (n_trials=1000, trial_sharpe_variance
    defaulted to 1/(n_obs-1), skew=0, kurtosis=3 i.e. Gaussian returns)."""
    n_obs = 252
    observed_sharpe = 0.15
    n_trials = 1000
    trial_sr_std = math.sqrt(1.0 / (n_obs - 1))
    gamma = 0.5772156649015329
    from scipy.stats import norm as _ref_norm  # reference oracle only, not used by the impl

    sr_benchmark = trial_sr_std * (
        (1 - gamma) * _ref_norm.ppf(1 - 1 / n_trials)
        + gamma * _ref_norm.ppf(1 - 1 / (n_trials * math.e))
    )
    se = math.sqrt((1 - 0 * observed_sharpe + (3 - 1) / 4 * observed_sharpe**2) / (n_obs - 1))
    expected = float(_ref_norm.cdf((observed_sharpe - sr_benchmark) / se))

    got = deflated_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        n_trials=n_trials,
        returns_skew=0.0,
        returns_kurtosis=3.0,
        n_obs=n_obs,
    )
    assert got == pytest.approx(expected, abs=1e-6)


def test_dsr_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(0.1, n_trials=0, returns_skew=0, returns_kurtosis=3, n_obs=100)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(0.1, n_trials=1, returns_skew=0, returns_kurtosis=3, n_obs=1)


# ---------------------------------------------------------------------------
# 2. Probability of Backtest Overfitting (CSCV)
# ---------------------------------------------------------------------------


def test_pbo_pure_noise_near_half():
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(128, 20))
    pbo = probability_of_backtest_overfitting(noise, n_splits=8)
    assert 0.3 < pbo < 0.7


def test_pbo_genuine_edge_is_low():
    rng = np.random.default_rng(0)
    perf = rng.normal(size=(128, 20))
    perf[:, 0] = rng.normal(loc=0.5, scale=1.0, size=128)  # persistent OOS edge
    pbo = probability_of_backtest_overfitting(perf, n_splits=8)
    assert pbo < 0.2


def test_pbo_edge_lower_than_noise_same_seed_shape():
    rng = np.random.default_rng(3)
    noise = rng.normal(size=(160, 12))
    edge = noise.copy()
    edge[:, 5] += np.linspace(0, 1.0, 160)  # inject a persistent edge into strategy 5
    pbo_noise = probability_of_backtest_overfitting(noise, n_splits=8)
    pbo_edge = probability_of_backtest_overfitting(edge, n_splits=8)
    assert pbo_edge < pbo_noise


def test_pbo_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((10, 1)), n_splits=4)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((10, 3)), n_splits=3)  # odd
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((4, 3)), n_splits=8)  # T < n_splits


# ---------------------------------------------------------------------------
# 3. Spearman rank IC
# ---------------------------------------------------------------------------


def test_spearman_beats_pearson_on_monotone_nonlinear():
    scores = np.linspace(-3, 3, 200)
    returns = scores**3  # monotone, strongly nonlinear
    sp = spearman_ic(scores, returns)
    pe = float(np.corrcoef(scores, returns)[0, 1])
    assert sp == pytest.approx(1.0, abs=1e-9)
    assert pe < sp
    assert pe < 0.95  # Pearson materially penalised by the nonlinearity


def test_spearman_ic_nan_aware():
    scores = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    returns = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    ic = spearman_ic(scores, returns)
    # valid pairs: (1,1) (2,2) (5,5) -> perfectly monotone
    assert ic == pytest.approx(1.0)


def test_spearman_ic_degenerate_returns_zero():
    scores = np.array([1.0, 1.0, 1.0])
    returns = np.array([1.0, 2.0, 3.0])
    assert spearman_ic(scores, returns) == 0.0
    assert spearman_ic(np.array([1.0]), np.array([1.0])) == 0.0


def test_spearman_ic_shape_mismatch_raises():
    with pytest.raises(ValueError):
        spearman_ic(np.ones(3), np.ones(4))


def test_spearman_ic_per_date_skips_degenerate_days():
    predictions = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [3.0, 1.0, 2.0]])
    returns = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    out = spearman_ic_per_date(predictions, returns)
    assert len(out) == 2  # the constant-prediction day (row 1) is skipped
    assert all(v == pytest.approx(1.0) for v in out)


def test_spearman_ic_per_date_shape_mismatch_raises():
    with pytest.raises(ValueError):
        spearman_ic_per_date(np.ones((2, 3)), np.ones((2, 4)))


# ---------------------------------------------------------------------------
# 4. HAC / Newey-West standard error
# ---------------------------------------------------------------------------


def _ar1_series(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def test_hac_se_exceeds_naive_on_autocorrelated_series():
    x = _ar1_series(500, phi=0.7, seed=1)
    naive_se = x.std(ddof=1) / math.sqrt(len(x))
    hac_se = hac_standard_error(x, lag=9)
    assert hac_se > naive_se


def test_hac_se_lag_zero_equals_naive_se():
    x = _ar1_series(300, phi=0.5, seed=2)
    naive_se = x.std(ddof=1) / math.sqrt(len(x))
    hac_se_0 = hac_standard_error(x, lag=0)
    assert hac_se_0 == pytest.approx(naive_se, rel=1e-9)


def test_hac_se_iid_series_close_to_naive():
    """On an i.i.d. (no autocorrelation) series, HAC SE at a nontrivial lag
    should stay close to the naive SE (no autocorrelation to correct for)."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=500)
    naive_se = x.std(ddof=1) / math.sqrt(len(x))
    hac_se = hac_standard_error(x, lag=5)
    assert hac_se == pytest.approx(naive_se, rel=0.25)


def test_hac_se_drops_non_finite_and_validates_lag():
    x = np.array([1.0, np.nan, 2.0, 3.0, np.inf, 4.0])
    se = hac_standard_error(x, lag=0)
    assert np.isfinite(se) and se > 0
    with pytest.raises(ValueError):
        hac_standard_error(np.arange(5.0), lag=-1)
    with pytest.raises(ValueError):
        hac_standard_error(np.arange(5.0), lag=5)


def test_hac_mean_ci_contains_mean_and_widens_with_autocorrelation():
    x = _ar1_series(500, phi=0.7, seed=1)
    ci_naive = hac_mean_ci(x, lag=0)
    ci_hac = hac_mean_ci(x, lag=9)
    assert ci_naive["ci_low"] <= ci_naive["mean"] <= ci_naive["ci_high"]
    width_naive = ci_naive["ci_high"] - ci_naive["ci_low"]
    width_hac = ci_hac["ci_high"] - ci_hac["ci_low"]
    assert width_hac > width_naive


# ---------------------------------------------------------------------------
# 6. Selection / confirmation split
# ---------------------------------------------------------------------------


def test_split_selection_confirmation_no_overlap_and_tail():
    dates = list(range(100))
    sel, confirm = split_selection_confirmation(dates, confirm_frac=0.3)
    assert set(sel).isdisjoint(confirm)
    assert sel + confirm == dates
    assert confirm == dates[-30:]


def test_split_selection_confirmation_deterministic():
    dates = [f"2024-01-{d:02d}" for d in range(1, 29)]
    a = split_selection_confirmation(dates, confirm_frac=0.25)
    b = split_selection_confirmation(dates, confirm_frac=0.25)
    assert a == b


def test_split_selection_confirmation_leaves_at_least_one_each_side():
    dates = list(range(3))
    sel, confirm = split_selection_confirmation(dates, confirm_frac=0.9)
    assert len(sel) >= 1
    assert len(confirm) >= 1
    assert sel + confirm == dates


def test_split_selection_confirmation_rejects_invalid_args():
    with pytest.raises(ValueError):
        split_selection_confirmation([1], confirm_frac=0.3)
    with pytest.raises(ValueError):
        split_selection_confirmation([1, 2, 3], confirm_frac=0.0)
    with pytest.raises(ValueError):
        split_selection_confirmation([1, 2, 3], confirm_frac=1.0)
