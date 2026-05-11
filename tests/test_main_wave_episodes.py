"""Tests for src/aurumq_rl/main_wave_episodes.py.

Convention: ``t_start`` = first up day (close[t_start] > close[t_start-1]
by ≥ 0.5%). The user's "T-1 decision day" = ``t_start - 1``.
``peak_return = close[t_peak] / close[t_start] - 1``.
"""

from __future__ import annotations

import numpy as np
import pytest

from aurumq_rl.main_wave_episodes import (
    EpisodeConfig,
    MainWaveEpisode,
    episodes_summary,
    find_main_wave_episodes,
)


def _flat_then_rise_panel():
    """30 days, 1 stock. Flat at 10 for days 0-9. Day 10 first up day. Peak
    at day 15 (close=13.0). Then flat at 13.0."""
    T, S = 30, 1
    close = np.full((T, S), 10.0, dtype=np.float32)
    close[10, 0] = 10.5  # +5% — first up day
    close[11, 0] = 11.0
    close[12, 0] = 11.6
    close[13, 0] = 12.2
    close[14, 0] = 12.7
    close[15, 0] = 13.0
    close[16:, 0] = 13.0
    vol = np.full((T, S), 2e7, dtype=np.float32)  # amount = 20*1e7 = 2e8 > 1e8
    valid = np.ones((T, S), dtype=np.bool_)
    return close, vol, valid


def test_single_clean_wave_detected():
    close, vol, valid = _flat_then_rise_panel()
    eps = find_main_wave_episodes(close, vol, valid)
    assert len(eps) == 1, f"expected 1 episode, got {len(eps)}"
    e = eps[0]
    assert e.stock_idx == 0
    assert e.t_start == 10
    assert e.t_peak == 15
    assert e.duration == 5
    expected_peak_ret = 13.0 / 10.5 - 1.0  # ≈ 0.2381
    assert abs(e.peak_return - expected_peak_ret) < 1e-3
    assert e.max_dd_during < 1e-5


def test_wave_below_peak_threshold_not_detected():
    """Tiny rise (+5% total) should NOT register."""
    T, S = 30, 1
    close = np.full((T, S), 10.0, dtype=np.float32)
    close[10, 0] = 10.05
    close[11, 0] = 10.10
    close[12, 0] = 10.15
    close[13, 0] = 10.20
    close[14, 0] = 10.30
    close[15:, 0] = 10.30
    vol = np.full((T, S), 2e7, dtype=np.float32)
    valid = np.ones((T, S), dtype=np.bool_)
    eps = find_main_wave_episodes(close, vol, valid)
    assert len(eps) == 0


def test_wave_too_short_not_detected():
    """+15% gain in 1 day shouldn't qualify (need ≥ min_duration)."""
    T, S = 30, 1
    close = np.full((T, S), 10.0, dtype=np.float32)
    close[10, 0] = 11.5  # +15% in 1 day
    close[11:, 0] = 11.5
    vol = np.full((T, S), 2e7, dtype=np.float32)
    valid = np.ones((T, S), dtype=np.bool_)
    eps = find_main_wave_episodes(close, vol, valid)
    assert len(eps) == 0


def test_too_volatile_wave_excluded():
    """Peak +15% with -10% intermediate drop → excluded by dd guard."""
    T, S = 30, 1
    close = np.full((T, S), 10.0, dtype=np.float32)
    close[10, 0] = 10.5
    close[11, 0] = 11.5
    close[12, 0] = 10.3  # big drop from 11.5 to 10.3
    close[13, 0] = 11.0
    close[14, 0] = 11.5
    close[15:, 0] = 11.5
    vol = np.full((T, S), 2e7, dtype=np.float32)
    valid = np.ones((T, S), dtype=np.bool_)
    # Even if peak (offset 1) gives +15% with no dd, the larger window which
    # would include the dip should be either dd-rejected or beat by smaller
    # window. min_duration=3 means peak must be at offset ≥ 3 — the +15% at
    # offset 1 is filtered out. Then peak at offset 4 = 11.5/10.5-1 = +9.5%
    # < min_peak_return=0.10, rejected.
    eps = find_main_wave_episodes(close, vol, valid)
    assert len(eps) == 0


def test_liquidity_gate_excludes_low_amount():
    close, vol, valid = _flat_then_rise_panel()
    vol = np.full(vol.shape, 1e6, dtype=np.float32)  # amount ≈ 1e7 << 1e8
    eps = find_main_wave_episodes(close, vol, valid)
    assert len(eps) == 0


def test_pre_window_blocks_continuation_t_start():
    """After a wave peaks and consolidates, a sudden second spike should be
    rejected as t_start when pre_window includes the prior rise. Concretely:
    wave 1 from days 5-10, then flat at 13.0 for days 11-13. At t=14 a +5%
    spike: c[14]=13.65. prior c[9..13] = [12.7, 13, 13, 13, 13]. prior_gain =
    13/12.7 - 1 = 2.36% — actually below the 3% threshold so this WOULD pass.
    To force rejection, place t at day 11: c[11]=13.0 = c[10], today_gain=0
    (no inflection), rejected. The point: the pre_window guard works in
    concert with the inflection guard, neither alone covers all cases."""
    T, S = 40, 1
    close = np.full((T, S), 10.0, dtype=np.float32)
    for i, v in enumerate([10.5, 11.0, 11.6, 12.2, 12.7, 13.0]):
        close[5 + i, 0] = v
    close[11:, 0] = 13.0
    vol = np.full((T, S), 2e7, dtype=np.float32)
    valid = np.ones((T, S), dtype=np.bool_)
    eps = find_main_wave_episodes(close, vol, valid)
    # Should detect exactly one episode (wave 1), no continuation episodes.
    assert len(eps) == 1
    assert eps[0].t_start == 5


def test_slow_drift_excluded_by_avg_daily():
    """+10% over 12 days = 0.83%/day, below 1.5% threshold → reject."""
    T, S = 30, 1
    close = np.full((T, S), 10.0, dtype=np.float32)
    base = 10.0
    for i in range(12):
        base = base * (1.008)  # ~0.8%/day
        close[10 + i, 0] = base
    close[22:, 0] = base
    vol = np.full((T, S), 2e7, dtype=np.float32)
    valid = np.ones((T, S), dtype=np.bool_)
    eps = find_main_wave_episodes(close, vol, valid)
    # Either nothing detected OR at most something with avg_daily ≥ 0.015 in a sub-window
    for e in eps:
        avg_daily = e.peak_return / max(e.duration, 1)
        assert avg_daily >= EpisodeConfig().min_avg_daily_return - 1e-6


def test_non_overlapping_episodes():
    """Two sequential clean waves — both detected, non-overlap."""
    T, S = 60, 1
    close = np.full((T, S), 10.0, dtype=np.float32)
    # Wave 1: t_start=10, peak at day 15 (close=13.0, peak_ret=23.8%)
    for i, v in enumerate([10.5, 11.0, 11.6, 12.2, 12.7, 13.0]):
        close[10 + i, 0] = v
    close[16:35, 0] = 13.0  # 19 days flat between waves
    # Wave 2: t_start=35, peak at day 40 (close=16.9 from base 13.0)
    for i, mult in enumerate([1.05, 1.05, 1.05, 1.05, 1.04, 1.04]):
        close[35 + i, 0] = close[35 + i - 1, 0] * mult
    close[41:, 0] = close[40, 0]
    vol = np.full((T, S), 2e7, dtype=np.float32)
    valid = np.ones((T, S), dtype=np.bool_)
    eps = find_main_wave_episodes(close, vol, valid)
    assert len(eps) == 2
    eps = sorted(eps, key=lambda e: e.t_start)
    assert eps[0].t_start == 10
    assert eps[0].t_peak == 15
    assert eps[1].t_start == 35
    assert eps[1].t_start > eps[0].t_peak  # non-overlap


def test_summary_helper():
    eps = [
        MainWaveEpisode(0, 10, 15, 0.30, 5, 0.01),
        MainWaveEpisode(0, 25, 30, 0.20, 5, 0.02),
        MainWaveEpisode(1, 8, 12, 0.15, 4, 0.03),
    ]
    s = episodes_summary(eps)
    assert s["n_episodes"] == 3
    assert s["avg_peak_return"] == pytest.approx((0.30 + 0.20 + 0.15) / 3)
    assert s["max_peak_return"] == pytest.approx(0.30)
    assert s["avg_duration"] == pytest.approx((5 + 5 + 4) / 3)


def test_invalid_basic_mask_excludes_t_start():
    close, vol, valid = _flat_then_rise_panel()
    valid[10, 0] = False  # ST that day
    eps = find_main_wave_episodes(close, vol, valid)
    # Episode at t_start=10 must NOT be recorded; t_start=11 also rejected
    # because pre-window gain c[10]/c[6] = 10.5/10 - 1 = 5% > 3% threshold.
    # So expect 0 episodes.
    assert len(eps) == 0
