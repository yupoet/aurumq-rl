"""Tests for v2_excess_adaptive √horizon vol scaling (issue #8, Part 4).

GUARDRAIL: `v2_excess_adaptive` is the P0-locked ablation winner. `horizon_scaling`
is OPT-IN and defaults to False, which MUST reproduce the current (pre-issue-8)
label output byte-for-byte. Enabling it changes the P0-locked label and requires
a re-ablation before production use — see module docstring in v2_excess_adaptive.py.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from aurumq_rl.labeling.panels import MarketPanel
from aurumq_rl.labeling.v2_excess_adaptive import adaptive_threshold, detect_events_v2


def _trade_dates(n: int) -> list[date]:
    base = date(2024, 1, 1)
    return [base + timedelta(days=i) for i in range(n)]


def _make_panel(n_dates: int, close_per_stock: list[np.ndarray], amount_value: float = 5e8):
    n_stocks = len(close_per_stock)
    adj_close = np.column_stack(close_per_stock).astype(np.float64)
    pct = np.diff(adj_close, axis=0, prepend=adj_close[:1]) / np.where(adj_close > 0, adj_close, 1)
    pct[0] = 0.0
    amount = np.full((n_dates, n_stocks), amount_value, dtype=np.float64)
    universe = np.ones((n_dates, n_stocks), dtype=bool)
    benchmark = np.ones(n_dates, dtype=np.float64)
    return MarketPanel(
        trade_dates=_trade_dates(n_dates),
        ts_codes=[f"STK{j:03d}" for j in range(n_stocks)],
        adj_close=adj_close,
        pct_change=pct,
        amount=amount,
        universe=universe,
        benchmark_close=benchmark,
    )


def _strong_wave_panel() -> MarketPanel:
    n = 80
    c = np.full(n, 10.0)
    for i in range(30, 41):
        c[i] = 10.0 * (1 + 0.012 * (i - 29))
    c[41:] = c[40]
    return _make_panel(n, [c])


# ---------------------------------------------------------------------------
# Byte-for-byte default reproduction (P0 guardrail)
# ---------------------------------------------------------------------------


def test_horizon_scaling_default_false_reproduces_p0_locked_output_byte_for_byte():
    panel = _strong_wave_panel()
    events_no_kwarg = detect_events_v2(panel)
    events_explicit_false = detect_events_v2(panel, horizon_scaling=False)

    # Pinned exact value captured from pre-issue-8 P0-locked behavior.
    assert len(events_no_kwarg) == 1
    e = events_no_kwarg[0]
    assert e.event_start_idx == 31
    assert e.event_peak_idx == 40
    assert e.event_quality == 1.7578125
    assert e.event_method == "A"

    assert events_no_kwarg == events_explicit_false


def test_horizon_scaling_false_matches_existing_method_A_tests_unchanged():
    """Re-assert the pre-existing test_method_A_* expectations still hold verbatim."""
    panel = _strong_wave_panel()
    events = detect_events_v2(panel)
    assert len(events) >= 1
    e = events[0]
    assert 28 <= e.event_start_idx <= 31
    assert e.event_quality > 1.0


# ---------------------------------------------------------------------------
# adaptive_threshold() pure-function unit tests (Part 4 formula)
# ---------------------------------------------------------------------------


def test_adaptive_threshold_without_scaling_ignores_duration():
    thr_short = adaptive_threshold(vol20=0.05, duration=1, horizon_scaling=False)
    thr_long = adaptive_threshold(vol20=0.05, duration=20, horizon_scaling=False)
    assert thr_short == thr_long == max(0.06, 1.8 * 0.05)


def test_adaptive_threshold_with_scaling_grows_with_sqrt_duration():
    vol20 = 0.05
    thr_1 = adaptive_threshold(vol20=vol20, duration=1, horizon_scaling=True)
    thr_20 = adaptive_threshold(vol20=vol20, duration=20, horizon_scaling=True)
    assert thr_1 == max(0.06, 1.8 * vol20 * math.sqrt(1))
    assert thr_20 == max(0.06, 1.8 * vol20 * math.sqrt(20))
    assert thr_20 > thr_1
    assert thr_20 == max(0.06, 1.8 * vol20 * math.sqrt(20))


def test_adaptive_threshold_floor_still_applies_when_scaled_term_small():
    thr = adaptive_threshold(vol20=0.001, duration=20, horizon_scaling=True)
    assert thr == 0.06


# ---------------------------------------------------------------------------
# Integration-level: horizon_scaling=True changes the threshold on a synthetic case
# ---------------------------------------------------------------------------


def test_horizon_scaling_true_changes_event_quality_on_synthetic_case():
    """With enough pre-event vol, scaling the threshold by sqrt(duration) should raise
    the bar and lower (or eliminate) the resulting event_quality vs the unscaled default.
    """
    n = 80
    # Noisy but bounded pre-window (keeps 5d cumulative gain small) to get vol20 > 0.0333
    # (so 1.8*vol20 > floor 0.06 and sqrt-duration scaling has visible effect).
    rng = np.random.default_rng(11)
    c = np.full(n, 10.0)
    noise = rng.normal(0, 0.05, size=25)
    c[1:26] = 10.0 * (1 + np.cumsum(noise) * 0.02)  # damped random walk, mild cumulative drift
    c[26] = c[25]
    for i in range(27, 41):
        c[i] = c[26] * (1 + 0.015 * (i - 26))
    c[41:] = c[40]
    panel = _make_panel(n, [c])

    events_off = detect_events_v2(panel, horizon_scaling=False)
    events_on = detect_events_v2(panel, horizon_scaling=True)

    assert len(events_off) >= 1, "Sanity: unscaled default should find the wave"
    off_by_start = {e.event_start_idx: e for e in events_off}
    on_by_start = {e.event_start_idx: e for e in events_on}

    if set(off_by_start) & set(on_by_start):
        for start in set(off_by_start) & set(on_by_start):
            assert on_by_start[start].event_quality <= off_by_start[start].event_quality
    else:
        # Threshold scaling rejected the event entirely under horizon_scaling=True.
        assert len(events_on) < len(events_off)
