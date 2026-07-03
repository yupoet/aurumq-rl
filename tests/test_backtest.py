"""Tests for src/aurumq_rl/backtest.py."""

from __future__ import annotations

import numpy as np
import pytest

from aurumq_rl.backtest import (
    BacktestResult,
    compute_ic,
    compute_ic_ir,
    compute_top_k_sharpe,
    random_baseline,
    run_backtest,
    run_backtest_with_series,
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
