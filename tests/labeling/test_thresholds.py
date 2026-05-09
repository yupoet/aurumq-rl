"""Threshold search tests (SPEC §5.2)."""

import numpy as np

from aurumq_rl.labeling.thresholds import search_threshold


def test_threshold_hits_target():
    # 10000 quality values; want ~ 1% pos rate over 100000 cells
    rng = np.random.default_rng(42)
    qualities = rng.normal(loc=2.0, scale=1.0, size=10000)
    qualities = np.maximum(qualities, 0)  # only positive (event qualities)
    res = search_threshold(qualities, n_decision_cells=100000,
                          target_pos_rate=0.01, pos_rate_min=0.005, pos_rate_max=0.015)
    # Should land in band
    assert 0.005 <= res.achieved_positive_rate <= 0.015
    assert abs(res.achieved_positive_rate - 0.01) < 0.005


def test_threshold_handles_empty():
    res = search_threshold(np.array([]), n_decision_cells=1000)
    assert res.threshold == float("inf")
    assert res.n_candidates == 0


def test_threshold_handles_constant():
    qualities = np.full(100, 3.0)
    res = search_threshold(qualities, n_decision_cells=10000, target_pos_rate=0.01)
    assert res.threshold == 3.0


def test_threshold_falls_back_when_band_empty():
    # All values 100x bigger than typical → no τ achieves <2% pos rate
    qualities = np.array([5.0] * 1000)
    res = search_threshold(qualities, n_decision_cells=1000, target_pos_rate=0.005,
                          pos_rate_min=0.001, pos_rate_max=0.002)
    # Falls back to nearest
    assert res.threshold > 0
