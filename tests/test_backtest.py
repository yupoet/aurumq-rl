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
