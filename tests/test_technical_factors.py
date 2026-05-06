"""Tests for src/aurumq_rl/technical_factors.py."""
from __future__ import annotations

import numpy as np
import pytest

from aurumq_rl.technical_factors import (
    TechnicalFactorsConfig,
    compute_technical_factors,
)


def _build_simple_panel(T=80, S=2):
    """Stock 0: rising trend. Stock 1: flat then rise."""
    close = np.zeros((T, S), dtype=np.float32)
    close[:, 0] = np.linspace(10.0, 15.0, T)   # smooth uptrend
    close[:T // 2, 1] = 20.0                    # flat
    close[T // 2:, 1] = np.linspace(20.0, 30.0, T - T // 2)
    vol = np.full((T, S), 1e7, dtype=np.float32)
    pct = np.zeros((T, S), dtype=np.float32)
    pct[1:] = (close[1:] / close[:-1]) - 1.0
    return close, vol, pct


def test_all_outputs_finite_and_correct_shape():
    close, vol, pct = _build_simple_panel()
    f = compute_technical_factors(close, vol, pct)
    T, S = close.shape
    for name, arr in f.items():
        assert arr.shape == (T, S), f"{name} shape {arr.shape} != {(T, S)}"
        assert arr.dtype == np.float32
        finite = np.isfinite(arr)
        assert finite.all(), f"{name} has non-finite at {np.argwhere(~finite)[:5]}"


def test_ma_distance_correct():
    close, vol, pct = _build_simple_panel()
    f = compute_technical_factors(close, vol, pct)
    # Stock 0 trends up, so close should be ABOVE all MAs at the end
    assert f["tech_close_vs_ma5"][-1, 0] > 0
    assert f["tech_close_vs_ma60"][-1, 0] > 0
    # Stock 0 ma5_above_ma10 most days (steady uptrend)
    assert f["tech_ma5_above_ma10"][-10:, 0].mean() == 1.0


def test_kdj_in_range():
    close, vol, pct = _build_simple_panel()
    f = compute_technical_factors(close, vol, pct)
    # KDJ K, D should be in [0, 100]
    K = f["tech_kdj_k"]
    D = f["tech_kdj_d"]
    # After warmup, in valid range
    assert (K[20:] >= 0).all() and (K[20:] <= 100).all()
    assert (D[20:] >= 0).all() and (D[20:] <= 100).all()


def test_macd_golden_cross_event():
    close, vol, pct = _build_simple_panel()
    f = compute_technical_factors(close, vol, pct)
    # MACD golden cross is a binary event flag (0 or 1)
    gc = f["tech_macd_golden_cross"]
    assert ((gc == 0) | (gc == 1)).all()
    # Stock 1 (flat then rise) should have at least one golden cross
    assert gc[:, 1].sum() >= 1


def test_bollinger_pct_b():
    close, vol, pct = _build_simple_panel()
    f = compute_technical_factors(close, vol, pct)
    pct_b = f["tech_boll_pct_b"]
    # In valid range after warmup
    assert (pct_b[20:] >= -1.0).all() and (pct_b[20:] <= 2.0).all()


def test_in_ma60_band():
    """Construct a stock that hovers ±2% around its slow MA."""
    T, S = 100, 1
    close = np.full((T, S), 10.0, dtype=np.float32)
    rng = np.random.default_rng(0)
    noise = rng.uniform(-0.02, 0.02, (T, S)).astype(np.float32)
    close[:, 0] = close[:, 0] * (1 + noise[:, 0])
    vol = np.full((T, S), 1e7, dtype=np.float32)
    pct = np.zeros((T, S), dtype=np.float32)
    pct[1:] = (close[1:] / close[:-1]) - 1.0
    f = compute_technical_factors(close, vol, pct)
    # Most late-window days should be in the ±4% MA60 band
    in_band = f["tech_in_ma60_band"][-30:, 0]
    assert in_band.mean() > 0.7   # most of the time


def test_cmf_factors_when_mf_provided():
    close, vol, pct = _build_simple_panel()
    mf = np.zeros_like(close)
    # Steady main-force inflow on stock 0
    mf[:, 0] = 1e6   # 1 million per day
    f = compute_technical_factors(close, vol, pct, mf_net_1d=mf)
    # cmf_60d on stock 0 at the end should be ~ 60 * 1e6 = 6e7
    assert f["cmf_60d"][-1, 0] == pytest.approx(60 * 1e6, rel=0.01)
    # cmf_pos_days_60d should be 60 for stock 0
    assert f["cmf_pos_days_60d"][-1, 0] == pytest.approx(60.0, abs=0.1)
    # Stock 1 has 0 mf, so cmf is 0
    assert f["cmf_60d"][-1, 1] == 0.0


def test_zt_count_replaces_broken_senti():
    """Construct pct_chg with known +0.10 days, count should match."""
    T, S = 50, 1
    close = np.full((T, S), 10.0, dtype=np.float32)
    vol = np.full((T, S), 1e7, dtype=np.float32)
    pct = np.zeros((T, S), dtype=np.float32)
    # Set days 5, 10, 15, 20 to +0.10 (limit-up)
    for d in (5, 10, 15, 20):
        pct[d, 0] = 0.10
    f = compute_technical_factors(close, vol, pct)
    # zt_count_30d at day 25 should include all 4 limit-ups
    assert f["zt_count_30d"][25, 0] == pytest.approx(4.0)


def test_vol_ratio_today_vs_ma20():
    T, S = 50, 1
    close = np.full((T, S), 10.0, dtype=np.float32)
    vol = np.full((T, S), 1e7, dtype=np.float32)
    vol[40, 0] = 5e7   # 5x volume spike
    pct = np.zeros((T, S), dtype=np.float32)
    f = compute_technical_factors(close, vol, pct)
    # vol_ratio at day 40: rolling MA20 includes today, so ma20 = (19*1e7 + 5e7)/20 = 1.2e7
    # → ratio = 5e7 / 1.2e7 ≈ 4.17. The "elevated volume" signal is preserved (>3),
    # which is what the model needs.
    ratio = f["tech_vol_ratio"][40, 0]
    assert ratio == pytest.approx(4.17, abs=0.05)
    assert ratio > 3.0   # spike day clearly stands out


def test_factor_count():
    """Sanity-check the total number of factors emitted."""
    close, vol, pct = _build_simple_panel()
    mf = np.zeros_like(close)
    mf[:, 0] = 1e6
    f = compute_technical_factors(close, vol, pct, mf_net_1d=mf)
    # Roughly: 4 close_vs_ma + 3 ma_above + 3 ma_cross + 1 in_band
    # + 5 kdj + 5 macd + 3 bollinger + 2 vol + 2 amp + 5 cmf + 3 zt
    # = 36-ish. Just check we have a healthy number.
    assert len(f) >= 30
    # All names use proper prefixes
    for name in f:
        assert name.startswith(("tech_", "cmf_", "zt_")), f"bad prefix: {name}"
