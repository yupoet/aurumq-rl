"""Tests for env helpers (no gymnasium required) and StockPickingEnv if installed."""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from aurumq_rl.env import (
    LIMIT_PCT_THRESHOLD,
    NEW_STOCK_PROTECT_DAYS,
    StockPickingConfig,
    _apply_industry_constraint,
    _apply_trading_mask,
)


# ---------------------------------------------------------------------------
# Pure helper functions (no gymnasium needed)
# ---------------------------------------------------------------------------


def test_apply_trading_mask_zeroes_st_and_suspended() -> None:
    n = 5
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    pct_changes = np.zeros(n)
    is_st = np.array([False, True, False, False, False])
    is_suspended = np.array([False, False, True, False, False])
    days_ipo = np.array([100, 100, 100, 100, 100])
    masked = _apply_trading_mask(
        returns=returns,
        pct_changes=pct_changes,
        is_st=is_st,
        is_suspended=is_suspended,
        days_since_ipo=days_ipo,
        respect_dynamic_price_limits=False,
    )
    assert masked[1] == 0.0  # ST
    assert masked[2] == 0.0  # suspended
    assert masked[0] != 0.0


def test_apply_trading_mask_new_stock_protected() -> None:
    returns = np.array([0.05, 0.05])
    pct = np.zeros(2)
    is_st = np.zeros(2, dtype=bool)
    susp = np.zeros(2, dtype=bool)
    days_ipo = np.array([NEW_STOCK_PROTECT_DAYS - 1, NEW_STOCK_PROTECT_DAYS + 1])
    masked = _apply_trading_mask(
        returns, pct, is_st, susp, days_ipo, respect_dynamic_price_limits=False
    )
    assert masked[0] == 0.0
    assert masked[1] != 0.0


def test_apply_trading_mask_legacy_pct_threshold() -> None:
    returns = np.array([0.05, 0.05])
    pct = np.array([0.0, LIMIT_PCT_THRESHOLD])  # second hits legacy threshold
    is_st = np.zeros(2, dtype=bool)
    susp = np.zeros(2, dtype=bool)
    days_ipo = np.array([100, 100])
    masked = _apply_trading_mask(
        returns, pct, is_st, susp, days_ipo, respect_dynamic_price_limits=False
    )
    assert masked[0] != 0.0
    assert masked[1] == 0.0


def test_apply_trading_mask_dynamic_price_limit() -> None:
    returns = np.array([0.05, 0.05])
    pct = np.array([0.0, 0.10])  # 600xxx main board → 10% triggers limit-up
    is_st = np.zeros(2, dtype=bool)
    susp = np.zeros(2, dtype=bool)
    days_ipo = np.array([100, 100])
    masked = _apply_trading_mask(
        returns, pct, is_st, susp, days_ipo,
        stock_codes=["600000.SH", "600519.SH"],
        respect_dynamic_price_limits=True,
    )
    assert masked[1] == 0.0
    assert masked[0] != 0.0


# ---------------------------------------------------------------------------
# _apply_industry_constraint
# ---------------------------------------------------------------------------


def test_industry_constraint_caps_per_industry() -> None:
    # 10 stocks, all from industry 1 → cap at top_k * 0.30
    scores = np.linspace(1.0, 0.1, 10)
    industry_codes = np.ones(10, dtype=np.int32)
    selected = _apply_industry_constraint(scores, top_k=10, industry_codes=industry_codes)
    # All 10 will be filled because we pad after exhausting; but the first
    # max_per_industry batch is the constrained selection
    assert len(selected) == 10


def test_industry_constraint_prefers_high_scores() -> None:
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    inds = np.array([1, 2, 3, 4])
    selected = _apply_industry_constraint(scores, top_k=2, industry_codes=inds)
    assert set(selected.tolist()) == {1, 2}  # idx of the two highest scores


def test_industry_constraint_diversifies_across_industries() -> None:
    # 5 stocks: 4 in industry 1, 1 in industry 2
    scores = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
    inds = np.array([1, 1, 1, 1, 2])
    selected = _apply_industry_constraint(scores, top_k=3, industry_codes=inds)
    # max_per_industry = max(1, int(3 * 0.30)) = 1 → only one from industry 1, but pads from leftovers
    assert len(selected) == 3
    # Industry 2 (idx 4) should be included as second pick
    assert 4 in selected.tolist()


# ---------------------------------------------------------------------------
# StockPickingConfig
# ---------------------------------------------------------------------------


def test_config_validation_invalid_dates() -> None:
    with pytest.raises(ValueError):
        StockPickingConfig(
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2023, 1, 1),  # before start
        )


def test_config_validation_invalid_top_k() -> None:
    with pytest.raises(ValueError):
        StockPickingConfig(top_k=0)


def test_config_validation_invalid_n_factors() -> None:
    with pytest.raises(ValueError):
        StockPickingConfig(n_factors=0)


def test_config_validation_invalid_forward_period() -> None:
    with pytest.raises(ValueError):
        StockPickingConfig(forward_period=0)


def test_config_defaults_are_reasonable() -> None:
    cfg = StockPickingConfig()
    assert cfg.top_k >= 1
    assert cfg.n_factors >= 1
    assert cfg.cost_bps >= 0
    assert 0.0 <= cfg.turnover_penalty <= 1.0


# ---------------------------------------------------------------------------
# StockPickingEnv smoke test (only if gymnasium is installed)
# ---------------------------------------------------------------------------


def test_stock_picking_env_smoke_step() -> None:
    pytest.importorskip("gymnasium")
    from aurumq_rl.env import StockPickingEnv  # type: ignore[no-redef]

    n_dates, n_stocks, n_factors = 30, 10, 4
    factor = np.random.default_rng(0).standard_normal((n_dates, n_stocks, n_factors)).astype(np.float32)
    ret = np.random.default_rng(1).standard_normal((n_dates, n_stocks)).astype(np.float32) * 0.01

    config = StockPickingConfig(
        start_date=datetime.date(2022, 1, 1),
        end_date=datetime.date(2022, 12, 31),
        n_factors=n_factors,
        top_k=3,
        forward_period=2,
        cost_bps=10.0,
        turnover_penalty=0.0,
        respect_dynamic_price_limits=False,
    )
    env = StockPickingEnv(config=config, factor_panel=factor, return_panel=ret)
    obs, info = env.reset(seed=0)
    assert obs.shape == (n_stocks * n_factors,)
    assert info["step"] == 0

    action = np.full(n_stocks, 0.5, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "portfolio_return" in info
    assert obs.shape == (n_stocks * n_factors,)
