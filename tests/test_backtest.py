"""Tests for src/aurumq_rl/backtest.py."""

from __future__ import annotations

import numpy as np
import pytest

from aurumq_rl.backtest import (
    BacktestResult,
    _top_k_returns_series,
    compute_ic,
    compute_ic_ir,
    compute_ic_spearman,
    compute_top_k_sharpe,
    random_baseline,
    run_backtest,
    run_backtest_with_series,
    top_k_returns_series_cost_adjusted,
)


@pytest.fixture
def perfect_predictions():
    """Predictions that match future returns exactly."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.02, size=(10, 100))
    return returns.copy(), returns


@pytest.fixture
def anti_predictions():
    """Predictions that are negated future returns (worst case)."""
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.02, size=(10, 100))
    return -returns, returns


def test_compute_ic_perfect_predictions(perfect_predictions):
    preds, rets = perfect_predictions
    ic = compute_ic(preds, rets)
    assert ic > 0.99


def test_compute_ic_anti_predictions(anti_predictions):
    preds, rets = anti_predictions
    ic = compute_ic(preds, rets)
    assert ic < -0.99


def test_compute_ic_random_predictions():
    rng = np.random.default_rng(42)
    preds = rng.normal(size=(50, 200))
    rets = rng.normal(size=(50, 200))
    ic = compute_ic(preds, rets)
    assert -0.1 < ic < 0.1


def test_compute_ic_ir_constant_returns_zero():
    preds = np.ones((10, 50))
    rets = np.ones((10, 50)) * 0.01
    ir = compute_ic_ir(preds, rets)
    assert ir == 0.0 or np.isnan(ir)


def test_top_k_sharpe_perfect():
    rng = np.random.default_rng(7)
    rets = rng.normal(0.01, 0.02, size=(60, 100))
    preds = rets.copy()
    sharpe = compute_top_k_sharpe(preds, rets, top_k=10)
    assert sharpe > 2.0


def test_random_baseline_consistent_seed():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.02, size=(60, 100))
    a = random_baseline(rets, top_k=10, n_simulations=50, seed=123)
    b = random_baseline(rets, top_k=10, n_simulations=50, seed=123)
    assert a["mean_sharpe"] == b["mean_sharpe"]


# ---------------------------------------------------------------------------
# M5 — tradeable_mask
# ---------------------------------------------------------------------------


def _best_stock_setup():
    """Stock 0 has a huge return every day and the highest prediction;
    everyone else earns 0.001. Deterministic top-k composition."""
    n_dates, n_stocks = 30, 10
    returns = np.full((n_dates, n_stocks), 0.001, dtype=np.float64)
    returns[:, 0] = 0.10
    preds = np.tile(np.arange(n_stocks, 0, -1, dtype=np.float64), (n_dates, 1))
    # preds: stock 0 highest, then 1, 2, ...
    return preds, returns


def test_tradeable_mask_excludes_best_stock_from_top_k():
    preds, returns = _best_stock_setup()
    mask = np.ones_like(returns, dtype=bool)
    mask[:, 0] = False  # e.g. limit-up at t → not buyable

    _, series_unmasked = run_backtest_with_series(
        preds,
        returns,
        dates=list(range(returns.shape[0])),
        top_k=2,
        n_random_simulations=3,
    )
    _, series_masked = run_backtest_with_series(
        preds,
        returns,
        dates=list(range(returns.shape[0])),
        top_k=2,
        n_random_simulations=3,
        tradeable_mask=mask,
    )
    # Without mask: (0.10 + 0.001) / 2. With mask: (0.001 + 0.001) / 2.
    assert all(r == pytest.approx((0.10 + 0.001) / 2) for r in series_unmasked.top_k_returns)
    assert all(r == pytest.approx(0.001) for r in series_masked.top_k_returns)


def test_tradeable_mask_applies_to_scalar_result_and_ic():
    preds, returns = _best_stock_setup()
    mask = np.ones_like(returns, dtype=bool)
    mask[:, 0] = False
    res_unmasked = run_backtest(preds, returns, top_k=2, n_random_simulations=3)
    res_masked = run_backtest(preds, returns, top_k=2, n_random_simulations=3, tradeable_mask=mask)
    assert res_masked.top_k_cumret < res_unmasked.top_k_cumret
    assert np.isfinite(res_masked.ic)


def test_tradeable_mask_shape_mismatch_raises():
    preds, returns = _best_stock_setup()
    bad_mask = np.ones((5, 5), dtype=bool)
    with pytest.raises(ValueError):
        run_backtest(preds, returns, top_k=2, n_random_simulations=3, tradeable_mask=bad_mask)


def test_random_baseline_respects_tradeable_mask():
    n_dates, n_stocks = 40, 8
    returns = np.zeros((n_dates, n_stocks), dtype=np.float64)
    returns[:, 0] = 0.5  # only the untradeable stock has any return
    mask = np.ones_like(returns, dtype=bool)
    mask[:, 0] = False
    rb_unmasked = random_baseline(returns, top_k=3, n_simulations=20, seed=1)
    rb_masked = random_baseline(returns, top_k=3, n_simulations=20, seed=1, tradeable_mask=mask)
    # Unmasked simulations sometimes pick stock 0 → non-degenerate Sharpe.
    assert rb_unmasked["mean_sharpe"] != 0.0
    # Masked simulations can never pick stock 0 → all returns 0 → Sharpe 0.
    assert rb_masked["mean_sharpe"] == 0.0


def test_backtest_result_to_json_roundtrip(tmp_path):
    result = BacktestResult(
        ic=0.05,
        ic_ir=0.4,
        top_k_sharpe=1.2,
        top_k_cumret=0.18,
        random_baseline={"mean_sharpe": 0.1, "p95_sharpe": 0.5},
        n_dates=60,
        n_stocks=100,
        top_k=10,
    )
    out = tmp_path / "bt.json"
    result.to_json(out)
    loaded = BacktestResult.from_json(out)
    assert loaded.ic == 0.05
    assert loaded.top_k_sharpe == 1.2
    assert loaded.random_baseline["mean_sharpe"] == 0.1


# ---------------------------------------------------------------------------
# Issue #6 — additive: ic_spearman, cost-adjusted top-k series
# ---------------------------------------------------------------------------


def test_compute_ic_spearman_present_and_does_not_change_pearson():
    rng = np.random.default_rng(11)
    scores = rng.normal(size=(30, 60))
    returns = scores**3 + rng.normal(scale=0.001, size=(30, 60))  # monotone-nonlinear + noise
    pearson = compute_ic(scores, returns)
    spearman = compute_ic_spearman(scores, returns)
    assert spearman > pearson  # Spearman survives the nonlinearity better


def test_run_backtest_ic_spearman_field_is_additive():
    rng = np.random.default_rng(12)
    rets = rng.normal(0.001, 0.02, size=(30, 60))
    preds = rets + rng.normal(0, 0.01, size=rets.shape)
    result = run_backtest(preds, rets, top_k=10, n_random_simulations=10)
    assert isinstance(result.ic_spearman, float)
    assert -1.0 <= result.ic_spearman <= 1.0
    # existing fields untouched by the new one being present
    assert result.ic == compute_ic(preds, rets)


def test_top_k_returns_cost_adjusted_default_off_matches_gross_byte_for_byte():
    rng = np.random.default_rng(13)
    rets = rng.normal(0.001, 0.02, size=(40, 50))
    preds = rets + rng.normal(0, 0.01, size=rets.shape)
    gross = _top_k_returns_series(preds, rets, top_k=8)
    cost_off = top_k_returns_series_cost_adjusted(preds, rets, top_k=8, cost_bps=0.0)
    assert cost_off == gross  # byte-for-byte (list of python floats)


def test_top_k_returns_cost_adjusted_zero_turnover_equals_gross():
    """Predictions are identical every day -> the top-K set never changes
    -> zero turnover -> cost-adjusted return equals the gross return even
    with cost_bps > 0."""
    n_dates, n_stocks = 20, 30
    preds = np.tile(np.arange(n_stocks, dtype=np.float64), (n_dates, 1))
    rng = np.random.default_rng(14)
    rets = rng.normal(0.001, 0.02, size=(n_dates, n_stocks))
    gross = _top_k_returns_series(preds, rets, top_k=5)
    cost_adj = top_k_returns_series_cost_adjusted(preds, rets, top_k=5, cost_bps=50.0)
    assert cost_adj == pytest.approx(gross, abs=1e-12)


def test_top_k_returns_cost_adjusted_high_turnover_reduces_return():
    """Predictions completely reshuffle every day -> maximal (or near-
    maximal) turnover every day after the first -> cost-adjusted series
    strictly below gross on those days."""
    n_dates, n_stocks, top_k = 15, 40, 5
    rng = np.random.default_rng(15)
    rets = rng.normal(0.001, 0.02, size=(n_dates, n_stocks))
    # Disjoint top-5 block each day: day t picks stocks [5t : 5t+5).
    preds = np.zeros((n_dates, n_stocks))
    for t in range(n_dates):
        start = (t * top_k) % (n_stocks - top_k)
        preds[t, start : start + top_k] = 1.0
    gross = _top_k_returns_series(preds, rets, top_k=top_k)
    cost_adj = top_k_returns_series_cost_adjusted(preds, rets, top_k=top_k, cost_bps=100.0)
    assert len(cost_adj) == len(gross)
    # First day: no prior portfolio -> no cost -> equal.
    assert cost_adj[0] == pytest.approx(gross[0])
    # Every subsequent day has full turnover -> strictly reduced by the fixed cost.
    for g, c in zip(gross[1:], cost_adj[1:], strict=True):
        assert c == pytest.approx(g - 100.0 / 1e4)


def test_run_backtest_cost_bps_default_zero_leaves_existing_fields_unchanged():
    rng = np.random.default_rng(16)
    rets = rng.normal(0.001, 0.02, size=(30, 60))
    preds = rets + rng.normal(0, 0.01, size=rets.shape)
    baseline = run_backtest(preds, rets, top_k=10, n_random_simulations=10, random_seed=5)
    with_cost_default = run_backtest(
        preds, rets, top_k=10, n_random_simulations=10, random_seed=5, cost_bps=0.0
    )
    assert with_cost_default.top_k_sharpe_cost_adjusted == 0.0
    assert with_cost_default.top_k_cumret_cost_adjusted == 0.0
    assert with_cost_default.cost_bps == 0.0
    assert with_cost_default.ic == baseline.ic
    assert with_cost_default.top_k_sharpe == baseline.top_k_sharpe
    assert with_cost_default.top_k_cumret == baseline.top_k_cumret


def test_run_backtest_with_series_cost_bps_populates_series_field_only_when_positive():
    rng = np.random.default_rng(17)
    rets = rng.normal(0.001, 0.02, size=(20, 40))
    preds = rets + rng.normal(0, 0.01, size=rets.shape)
    dates = list(range(20))
    _, series_off = run_backtest_with_series(preds, rets, dates=dates, top_k=8)
    assert series_off.top_k_returns_cost_adjusted == []
    _, series_on = run_backtest_with_series(preds, rets, dates=dates, top_k=8, cost_bps=30.0)
    assert len(series_on.top_k_returns_cost_adjusted) > 0
    # unaffected fields identical between the two calls
    assert series_on.top_k_returns == series_off.top_k_returns
    assert series_on.ic == series_off.ic


def test_series_skip_degenerate_truncates_trailing_loader_zero_rows():
    """Review fix: FactorPanelLoader leaves the trailing `forward_period`
    rows of return_array as literal 0.0 (finite, so NOT caught by the
    degenerate-day guard). The skip-degenerate / cost-adjusted series must
    truncate them exactly like run_backtest()'s scalar path, otherwise they
    leak in as spurious all-zero observations that understate the HAC SE
    and bias the mean toward zero."""
    from aurumq_rl.backtest import (
        _top_k_returns_series,
        _truncate_trailing_forward_rows,
    )
    from aurumq_rl.eval_metrics import hac_mean_ci, hac_standard_error

    rng = np.random.default_rng(23)
    n_dates, n_stocks, fp, top_k = 60, 40, 10, 8
    rets = rng.normal(0.003, 0.02, size=(n_dates, n_stocks))
    # Fabricate the loader's trailing all-zero forward-return rows.
    rets[n_dates - fp :] = 0.0
    preds = rets + rng.normal(0, 0.01, size=rets.shape)
    dates = list(range(n_dates))

    _, series = run_backtest_with_series(
        preds, rets, dates=dates, top_k=top_k, forward_period=fp, n_random_simulations=5
    )

    # Reference: the correctly-truncated series computed directly.
    tp, tr, _ = _truncate_trailing_forward_rows(preds, rets, fp)
    expected = _top_k_returns_series(tp, tr, top_k)

    # The series must equal the truncated computation, not the untruncated one.
    assert series.top_k_returns_skip_degenerate == expected
    # It is strictly shorter than the untruncated series (the fp zero rows are gone).
    untruncated = _top_k_returns_series(preds, rets, top_k)
    assert len(series.top_k_returns_skip_degenerate) == len(untruncated) - fp
    # None of the retained observations is one of the fabricated all-zero rows.
    assert all(v != 0.0 for v in series.top_k_returns_skip_degenerate[-fp:])

    # HAC on the shipped series == HAC on the correctly-truncated series,
    # and both differ from the (biased) untruncated HAC.
    lag = fp - 1
    assert hac_standard_error(series.top_k_returns_skip_degenerate, lag) == pytest.approx(
        hac_standard_error(expected, lag)
    )
    assert hac_mean_ci(series.top_k_returns_skip_degenerate, lag)["mean"] == pytest.approx(
        hac_mean_ci(expected, lag)["mean"]
    )
    # The untruncated series' HAC SE is understated (extra zero rows shrink it).
    assert hac_standard_error(untruncated, lag) < hac_standard_error(expected, lag)


def test_series_cost_adjusted_also_truncates_trailing_zero_rows():
    """The cost-adjusted series shares the same truncation as skip-degenerate."""
    from aurumq_rl.backtest import (
        _truncate_trailing_forward_rows,
        top_k_returns_series_cost_adjusted,
    )

    rng = np.random.default_rng(24)
    n_dates, n_stocks, fp, top_k = 50, 30, 10, 6
    rets = rng.normal(0.002, 0.02, size=(n_dates, n_stocks))
    rets[n_dates - fp :] = 0.0
    preds = rets + rng.normal(0, 0.01, size=rets.shape)
    dates = list(range(n_dates))

    _, series = run_backtest_with_series(
        preds,
        rets,
        dates=dates,
        top_k=top_k,
        forward_period=fp,
        n_random_simulations=5,
        cost_bps=30.0,
    )
    tp, tr, _ = _truncate_trailing_forward_rows(preds, rets, fp)
    expected = top_k_returns_series_cost_adjusted(tp, tr, top_k=top_k, cost_bps=30.0)
    assert series.top_k_returns_cost_adjusted == expected
