"""Tests for src/aurumq_rl/main_wave_target_labels.py."""
from __future__ import annotations

import numpy as np
import pytest

from aurumq_rl.main_wave_episodes import MainWaveEpisode
from aurumq_rl.main_wave_target_labels import (
    TargetConfig,
    build_target_quality,
    quality_of_episode,
)


def test_quality_increases_with_peak_return():
    e_low = MainWaveEpisode(0, 10, 15, peak_return=0.10, duration=5, max_dd_during=0.0)
    e_high = MainWaveEpisode(0, 10, 15, peak_return=0.30, duration=5, max_dd_during=0.0)
    assert quality_of_episode(e_high) > quality_of_episode(e_low)


def test_quality_increases_with_duration():
    e_short = MainWaveEpisode(0, 10, 13, peak_return=0.20, duration=3, max_dd_during=0.0)
    e_long = MainWaveEpisode(0, 10, 18, peak_return=0.20, duration=10, max_dd_during=0.0)
    assert quality_of_episode(e_long) > quality_of_episode(e_short)


def test_quality_decreases_with_drawdown():
    e_clean = MainWaveEpisode(0, 10, 15, peak_return=0.20, duration=5, max_dd_during=0.0)
    e_volatile = MainWaveEpisode(0, 10, 15, peak_return=0.20, duration=5, max_dd_during=0.10)
    assert quality_of_episode(e_clean) > quality_of_episode(e_volatile)


def test_target_at_t_minus_1():
    eps = [MainWaveEpisode(0, 10, 15, 0.30, 5, 0.0)]
    T = MainWaveEpisode  # not used
    targets = build_target_quality(eps, n_dates=20, n_stocks=2)
    # T-1 = day 9; should have full credit
    assert targets.proximity[9, 0] == 1
    expected = quality_of_episode(eps[0]) * 1.0
    assert abs(targets.target_quality[9, 0] - expected) < 1e-5
    # T-2 = day 8; partial credit
    assert targets.proximity[8, 0] == 2
    expected2 = quality_of_episode(eps[0]) * 0.6
    assert abs(targets.target_quality[8, 0] - expected2) < 1e-5
    # T-3 = day 7; smaller partial credit
    assert targets.proximity[7, 0] == 3
    # T-4 = day 6; should be miss
    assert targets.proximity[6, 0] == 0
    assert targets.target_quality[6, 0] == 0.0
    # other stock = no credit
    assert targets.proximity[9, 1] == 0


def test_target_other_stocks_zero():
    eps = [MainWaveEpisode(stock_idx=2, t_start=10, t_peak=15,
                           peak_return=0.20, duration=5, max_dd_during=0.0)]
    targets = build_target_quality(eps, n_dates=20, n_stocks=5)
    # Only stock 2 should have credit at T-1=9, T-2=8, T-3=7
    for j in range(5):
        if j == 2:
            assert targets.proximity[9, j] == 1
        else:
            assert targets.proximity[9, j] == 0


def test_overlapping_episodes_max_wins():
    """If two episodes both give credit at decision day t, max(quality * weight) wins."""
    e_small_at_T_minus_1 = MainWaveEpisode(0, 10, 15, 0.10, 5, 0.0)
    e_huge_at_T_minus_3 = MainWaveEpisode(0, 12, 17, 0.50, 5, 0.0)
    eps = [e_small_at_T_minus_1, e_huge_at_T_minus_3]
    targets = build_target_quality(eps, n_dates=20, n_stocks=1)
    # At t=9 (decision day):
    # - e_small: T-1, weight 1.0, quality ≈ 0.10 * dur_factor(5/10=0.5) * smoothness(1) = 0.05 → weighted 0.05
    # - e_huge: T-3, weight 0.3, quality ≈ 0.50 * dur_factor(0.5) * smoothness(1) = 0.25 → weighted 0.075
    # → e_huge wins with weighted 0.075
    expected_huge = quality_of_episode(e_huge_at_T_minus_3) * 0.3
    expected_small = quality_of_episode(e_small_at_T_minus_1) * 1.0
    if expected_huge > expected_small:
        assert targets.proximity[9, 0] == 3
        assert abs(targets.target_quality[9, 0] - expected_huge) < 1e-5
    else:
        assert targets.proximity[9, 0] == 1
        assert abs(targets.target_quality[9, 0] - expected_small) < 1e-5


def test_panel_edge_dates():
    """Episodes near panel start: T-3 may be < 0, gracefully handled."""
    eps = [MainWaveEpisode(0, 2, 5, 0.20, 3, 0.0)]   # t_start=2, T-3 = -1 (invalid)
    targets = build_target_quality(eps, n_dates=10, n_stocks=1)
    assert targets.proximity[1, 0] == 1   # T-1 valid
    assert targets.proximity[0, 0] == 2   # T-2 valid
    # T-3 = -1, skipped


def test_empty_episodes():
    targets = build_target_quality([], n_dates=10, n_stocks=5)
    assert (targets.target_quality == 0).all()
    assert (targets.proximity == 0).all()
    assert (targets.episode_id_at_proximity == -1).all()
