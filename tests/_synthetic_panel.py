"""Synthetic FactorPanel fixtures shared across GPU-framework tests.

Why a separate module (not just conftest fixtures): the same
construction is needed by gpu_env, policy, and factor_importance
tests, plus by some smoke scripts. Keeping it as a plain module
makes it importable from anywhere.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from aurumq_rl.data_loader import FactorPanel


def make_synthetic_panel(
    n_dates: int = 60,
    n_stocks: int = 50,
    n_factors: int = 20,
    seed: int = 0,
    plant_true_factor: bool = False,
    true_factor_index: int = 0,
    true_factor_strength: float = 0.5,
) -> FactorPanel:
    """Build a small in-memory FactorPanel for tests.

    All factor values ~ N(0, 1). Returns are pure noise unless
    ``plant_true_factor=True`` -- in that case
    ``returns[t, s] = strength * factors[t, s, true_factor_index] + noise``,
    so a working factor-importance module must rank that factor first.
    """
    rng = np.random.default_rng(seed)
    factor_array = rng.standard_normal((n_dates, n_stocks, n_factors)).astype(np.float32)
    base_returns = rng.standard_normal((n_dates, n_stocks)).astype(np.float32) * 0.02
    if plant_true_factor:
        signal = true_factor_strength * factor_array[..., true_factor_index]
        return_array = (base_returns + signal).astype(np.float32)
    else:
        return_array = base_returns
    pct_change_array = return_array.copy()
    is_st_array = np.zeros((n_dates, n_stocks), dtype=np.bool_)
    is_suspended_array = np.zeros((n_dates, n_stocks), dtype=np.bool_)
    days_since_ipo_array = np.full((n_dates, n_stocks), 1000, dtype=np.float32)
    base_date = _dt.date(2024, 1, 2)
    dates = [base_date + _dt.timedelta(days=i) for i in range(n_dates)]
    stock_codes = [f"SYN{i:04d}.SH" for i in range(n_stocks)]
    factor_names = [f"alpha_{i:03d}" if i < n_factors // 2 else f"gtja_{i:03d}"
                    for i in range(n_factors)]
    return FactorPanel(
        factor_array=factor_array,
        return_array=return_array,
        pct_change_array=pct_change_array,
        is_st_array=is_st_array,
        is_suspended_array=is_suspended_array,
        days_since_ipo_array=days_since_ipo_array,
        dates=dates,
        stock_codes=stock_codes,
        factor_names=factor_names,
    )
