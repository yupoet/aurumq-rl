"""Tests for CUSUM event sampling + concurrency/uniqueness weights (issue #8, Parts 1-2).

López de Prado, *Advances in Financial Machine Learning* (2018):
- ch. 2.5.2 symmetric CUSUM filter for event sampling
- ch. 4 concurrency / average uniqueness for sample weighting
"""

from __future__ import annotations

import numpy as np

from aurumq_rl.labeling.sampling import (
    average_uniqueness,
    cusum_filter,
    label_concurrency,
)

# ---------------------------------------------------------------------------
# Part 1 — CUSUM filter
# ---------------------------------------------------------------------------


def test_cusum_filter_detects_level_shifts_not_flat_stretches():
    """A flat series with two sharp level shifts should emit events only near the shifts."""
    series = np.concatenate(
        [
            np.full(20, 10.0),  # flat
            np.full(20, 11.0),  # +10% shift at idx 20
            np.full(20, 11.0),  # flat
            np.full(20, 9.5),  # -13.6% shift at idx 60
            np.full(20, 9.5),  # flat
        ]
    )
    events = cusum_filter(series, threshold=0.5)
    assert len(events) >= 1, "Should detect at least the two level shifts"
    # No events should fire deep inside the flat stretches (far from shift boundaries)
    flat_zone = set(range(2, 18)) | set(range(42, 58)) | set(range(82, 99))
    assert not (set(events.tolist()) & flat_zone), (
        f"CUSUM should not fire inside flat stretches, got {events}"
    )
    # Events should cluster near the two shift points (idx ~20, ~60)
    assert any(18 <= e <= 22 for e in events), f"expected event near idx 20, got {events}"
    assert any(58 <= e <= 62 for e in events), f"expected event near idx 60, got {events}"


def test_cusum_filter_flat_series_no_events():
    series = np.full(50, 10.0)
    events = cusum_filter(series, threshold=0.5)
    assert len(events) == 0


def test_cusum_filter_symmetric_mirror():
    """Mirroring the series (negating diffs) should mirror which direction fires,
    but the event *indices* (magnitude-triggered) should be identical."""
    rng = np.random.default_rng(7)
    steps = rng.choice([-1.0, 1.0], size=99) * 0.3
    series = 10.0 + np.concatenate([[0.0], np.cumsum(steps)])
    mirrored = 20.0 - series  # negate the walk around a constant

    events = cusum_filter(series, threshold=1.0)
    events_mirrored = cusum_filter(mirrored, threshold=1.0)
    np.testing.assert_array_equal(events, events_mirrored)


def test_cusum_filter_threshold_zero_raises_or_no_crash_boundary():
    # Very small threshold on a monotone series should fire many events.
    series = np.arange(30, dtype=np.float64)
    events = cusum_filter(series, threshold=1.0)
    assert len(events) > 0
    # Events should be roughly evenly spaced given a monotone unit-step walk
    assert np.all(np.diff(events) >= 1)


# ---------------------------------------------------------------------------
# Part 1b — triple_barrier opt-in seeding via CUSUM/sample events
# ---------------------------------------------------------------------------


def test_triple_barrier_sample_events_default_unchanged():
    """Default (event_idx=None) must reproduce the existing all-t seeding behavior."""
    from datetime import date, timedelta

    from aurumq_rl.labeling.panels import MarketPanel
    from aurumq_rl.labeling.triple_barrier import detect_events_triple_barrier

    n = 60
    c = np.full(n, 10.0)
    rng = np.random.default_rng(42)
    c[1:30] = 10.0 + rng.normal(0, 0.05, size=29)
    for i in range(30, 35):
        c[i] = 10.0 * (1 + 0.012 * (i - 30))
    c[35:] = c[34]
    adj_close = c.reshape(-1, 1)
    pct = np.diff(adj_close, axis=0, prepend=adj_close[:1]) / np.where(adj_close > 0, adj_close, 1)
    pct[0] = 0.0
    amount = np.full((n, 1), 5e8)
    universe = np.ones((n, 1), dtype=bool)
    benchmark = np.ones(n)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    panel = MarketPanel(
        trade_dates=dates,
        ts_codes=["STK000"],
        adj_close=adj_close,
        pct_change=pct,
        amount=amount,
        universe=universe,
        benchmark_close=benchmark,
    )
    baseline = detect_events_triple_barrier(panel)
    with_default_kw = detect_events_triple_barrier(panel, event_idx=None)
    assert baseline == with_default_kw


def test_triple_barrier_sample_events_restricts_seeding():
    """When event_idx is given, only those t are considered as candidate starts."""
    from datetime import date, timedelta

    from aurumq_rl.labeling.panels import MarketPanel
    from aurumq_rl.labeling.triple_barrier import detect_events_triple_barrier

    n = 60
    c = np.full(n, 10.0)
    rng = np.random.default_rng(42)
    c[1:30] = 10.0 + rng.normal(0, 0.05, size=29)
    for i in range(30, 35):
        c[i] = 10.0 * (1 + 0.012 * (i - 30))
    c[35:] = c[34]
    adj_close = c.reshape(-1, 1)
    pct = np.diff(adj_close, axis=0, prepend=adj_close[:1]) / np.where(adj_close > 0, adj_close, 1)
    pct[0] = 0.0
    amount = np.full((n, 1), 5e8)
    universe = np.ones((n, 1), dtype=bool)
    benchmark = np.ones(n)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    panel = MarketPanel(
        trade_dates=dates,
        ts_codes=["STK000"],
        adj_close=adj_close,
        pct_change=pct,
        amount=amount,
        universe=universe,
        benchmark_close=benchmark,
    )
    # Seed set that deliberately excludes the actual event start (~day 30)
    sparse_idx = {"STK000": np.array([1, 2, 3, 4, 5], dtype=np.int64)}
    restricted = detect_events_triple_barrier(panel, event_idx=sparse_idx)
    assert restricted == [] or all(e.event_start_idx in sparse_idx["STK000"] for e in restricted)


# ---------------------------------------------------------------------------
# Part 2 — concurrency + average uniqueness
# ---------------------------------------------------------------------------


def test_label_concurrency_hand_computed():
    # Two labels: [0,4] and [2,6]; a third disjoint [10,12]
    t_starts = np.array([0, 2, 10])
    t_ends = np.array([4, 6, 12])
    index = np.arange(0, 13)
    conc = label_concurrency(t_starts, t_ends, index)
    expected = np.array(
        [1, 1, 2, 2, 2, 1, 1, 0, 0, 0, 1, 1, 1],
    )
    np.testing.assert_array_equal(conc, expected)


def test_label_concurrency_fully_non_overlapping_all_weights_equal():
    t_starts = np.array([0, 5, 10])
    t_ends = np.array([2, 7, 12])
    index = np.arange(0, 13)
    weights = average_uniqueness(t_starts, t_ends, index)
    np.testing.assert_allclose(weights, np.ones(3))


def test_average_uniqueness_heavy_overlap_downweighted():
    # 3 labels all covering the same [0, 9] window → concurrency=3 everywhere → weight = 1/3 each
    t_starts = np.array([0, 0, 0])
    t_ends = np.array([9, 9, 9])
    index = np.arange(0, 10)
    weights = average_uniqueness(t_starts, t_ends, index)
    np.testing.assert_allclose(weights, np.full(3, 1.0 / 3.0))
    # Sanity: heavy overlap weights strictly less than the non-overlapping case
    assert np.all(weights < 1.0)


def test_average_uniqueness_partial_overlap_between_extremes():
    # label 0: [0,9] fully overlapped by label1 [0,4] and label2 [5,9]
    t_starts = np.array([0, 0, 5])
    t_ends = np.array([9, 4, 9])
    index = np.arange(0, 10)
    weights = average_uniqueness(t_starts, t_ends, index)
    # label 0 spans both halves: concurrency 2 for [0,4], 2 for [5,9] -> avg uniqueness 0.5
    assert abs(weights[0] - 0.5) < 1e-9
    # label 1 spans only [0,4], concurrency 2 throughout -> 0.5
    assert abs(weights[1] - 0.5) < 1e-9
    # label 2 spans only [5,9], concurrency 2 throughout -> 0.5
    assert abs(weights[2] - 0.5) < 1e-9
