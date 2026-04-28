"""Tests for BacktestSeries dataclass + run_backtest_with_series."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from aurumq_rl.backtest import (
    BacktestResult,
    BacktestSeries,
    run_backtest_with_series,
)


def test_backtest_series_shape_matches_panel():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.02, size=(40, 80))
    preds = rets + rng.normal(0, 0.01, size=rets.shape)
    dates = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(40)]
    result, series = run_backtest_with_series(
        predictions=preds,
        returns=rets,
        dates=dates,
        top_k=10,
        n_random_simulations=20,
    )
    assert isinstance(result, BacktestResult)
    assert isinstance(series, BacktestSeries)
    assert len(series.dates) == 40
    assert len(series.ic) == 40
    assert len(series.top_k_returns) == 40
    assert len(series.equity_curve) == 40
    assert len(series.random_baseline_sharpes) == 20


def test_backtest_series_to_json_roundtrip(tmp_path):
    series = BacktestSeries(
        dates=["2025-01-02", "2025-01-03"],
        ic=[0.01, -0.02],
        top_k_returns=[0.001, 0.002],
        equity_curve=[1.001, 1.003],
        random_baseline_sharpes=[0.1, -0.3, 0.5],
    )
    out = tmp_path / "bs.json"
    series.to_json(out)
    loaded = BacktestSeries.from_json(out)
    assert loaded.dates == ["2025-01-02", "2025-01-03"]
    assert loaded.ic == [0.01, -0.02]
    assert loaded.equity_curve == [1.001, 1.003]


def test_run_backtest_with_series_equity_curve_starts_at_one_or_close():
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0, 0.02, size=(10, 50))
    preds = rng.normal(0.0, 0.02, size=(10, 50))
    dates = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(10)]
    _, series = run_backtest_with_series(
        predictions=preds, returns=rets, dates=dates, top_k=10
    )
    # First entry is 1 + first day's top-k return
    assert abs(series.equity_curve[0] - (1.0 + series.top_k_returns[0])) < 1e-9
    # Equity series is monotonic with cumulative product semantics
    for i in range(1, len(series.equity_curve)):
        expected = series.equity_curve[i - 1] * (1 + series.top_k_returns[i])
        assert abs(series.equity_curve[i] - expected) < 1e-9


def test_run_backtest_with_series_matches_run_backtest_scalars_on_degenerate():
    """Regression: run_backtest_with_series must produce the same scalar
    BacktestResult as run_backtest() (skip-degenerate semantics)."""
    rng = np.random.default_rng(42)
    rets = rng.normal(0.001, 0.02, size=(20, 60))
    preds = rets + rng.normal(0, 0.01, size=rets.shape)
    # Make day 0 degenerate (constant predictions -> std == 0)
    preds[0, :] = 0.0
    # Make day 5 degenerate (NaN-heavy)
    preds[5, :] = np.nan

    from aurumq_rl.backtest import run_backtest, run_backtest_with_series
    canonical = run_backtest(preds, rets, top_k=10, n_random_simulations=20, random_seed=7)
    result, _ = run_backtest_with_series(
        preds, rets,
        dates=[dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(20)],
        top_k=10, n_random_simulations=20, random_seed=7,
    )
    assert result.ic == pytest.approx(canonical.ic)
    assert result.ic_ir == pytest.approx(canonical.ic_ir)
    assert result.top_k_sharpe == pytest.approx(canonical.top_k_sharpe)
    assert result.top_k_cumret == pytest.approx(canonical.top_k_cumret)
