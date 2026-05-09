"""Synthetic-data tests for each labeling method.

Each test constructs a small (T, S) panel with a known pattern and verifies
that the method detects the expected event(s).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from aurumq_rl.labeling.events import Event, dedupe_events
from aurumq_rl.labeling.panels import MarketPanel


def _trade_dates(n: int) -> list[date]:
    base = date(2024, 1, 1)
    return [base + timedelta(days=i) for i in range(n)]


def _make_panel(
    n_dates: int,
    close_per_stock: list[np.ndarray],
    amount_value: float = 5e8,
) -> MarketPanel:
    n_stocks = len(close_per_stock)
    adj_close = np.column_stack(close_per_stock).astype(np.float64)
    pct = np.diff(adj_close, axis=0, prepend=adj_close[:1]) / np.where(
        adj_close > 0, adj_close, 1
    )
    pct[0] = 0.0
    amount = np.full((n_dates, n_stocks), amount_value, dtype=np.float64)
    universe = np.ones((n_dates, n_stocks), dtype=bool)
    benchmark = np.ones(n_dates, dtype=np.float64)   # flat benchmark = excess equals raw
    return MarketPanel(
        trade_dates=_trade_dates(n_dates),
        ts_codes=[f"STK{j:03d}" for j in range(n_stocks)],
        adj_close=adj_close,
        pct_change=pct,
        amount=amount,
        universe=universe,
        benchmark_close=benchmark,
    )


# ---------------------------------------------------------------------------
# Method A
# ---------------------------------------------------------------------------

def test_method_A_detects_strong_wave():
    from aurumq_rl.labeling.v2_excess_adaptive import detect_events_v2
    n = 80
    # flat 0..29, then 30..40 climb 10% over 10 days, then plateau
    c = np.full(n, 10.0)
    for i in range(30, 41):
        c[i] = 10.0 * (1 + 0.012 * (i - 29))   # +1.2% per day, ~14% peak
    c[41:] = c[40]
    panel = _make_panel(n, [c])
    events = detect_events_v2(panel)
    assert len(events) >= 1, "Method A should detect a clean +14% wave"
    e = events[0]
    assert 28 <= e.event_start_idx <= 31, f"start near 30, got {e.event_start_idx}"
    assert e.event_quality > 1.0


def test_method_A_rejects_flat():
    from aurumq_rl.labeling.v2_excess_adaptive import detect_events_v2
    panel = _make_panel(80, [np.full(80, 10.0)])
    events = detect_events_v2(panel)
    assert len(events) == 0


def test_method_A_rejects_slow_drift():
    from aurumq_rl.labeling.v2_excess_adaptive import detect_events_v2
    # 0.2% per day for 50 days = ~10% but no inflection (steady drift)
    c = 10.0 * (1.002 ** np.arange(80))
    panel = _make_panel(80, [c])
    events = detect_events_v2(panel)
    # The slow drift fails inflection (prior 5d > 1%) — should reject most
    assert len(events) <= 1


# ---------------------------------------------------------------------------
# Method B
# ---------------------------------------------------------------------------

def test_method_B_detects_log_linear_trend():
    from aurumq_rl.labeling.trend_scanning import detect_events_trend_scanning
    n = 60
    # linear log price rise from t=20 to t=40
    c = np.full(n, 10.0)
    for i in range(20, 40):
        c[i] = 10.0 * (1 + 0.008 * (i - 20))
    c[40:] = c[39]
    panel = _make_panel(n, [c])
    events = detect_events_trend_scanning(panel)
    assert len(events) >= 1
    # First event should land in the rising segment
    assert any(15 <= e.event_start_idx <= 35 for e in events)


def test_method_B_rejects_iid_noise():
    """IID returns (not random walk) should not produce strong t-stats."""
    from aurumq_rl.labeling.trend_scanning import detect_events_trend_scanning
    rng = np.random.default_rng(42)
    # IID daily returns N(0, 1%) → mean-zero, no trend
    rets = rng.normal(0, 0.01, size=80)
    c = 10.0 * np.exp(np.cumsum(rets - rets.mean()))   # subtract mean → drift-free
    panel = _make_panel(80, [c])
    events = detect_events_trend_scanning(panel)
    # With drift-free IID we still get some windows that look trendy by chance,
    # but |t-stat| > 8 should be rare; tolerate a couple of spurious events
    very_high = [e for e in events if abs(e.event_quality) > 8]
    assert len(very_high) <= 1, f"Drift-free IID should yield few extreme t-stats; got {very_high}"


# ---------------------------------------------------------------------------
# Method C
# ---------------------------------------------------------------------------

def test_method_C_detects_upper_barrier_hit():
    from aurumq_rl.labeling.triple_barrier import detect_events_triple_barrier
    n = 60
    c = np.full(n, 10.0)
    # small noise to set sigma > 0
    rng = np.random.default_rng(42)
    c[1:30] = 10.0 + rng.normal(0, 0.05, size=29)   # ~0.5% daily noise
    c[30] = 10.05
    # Upper barrier hit at day 32 with +6%
    for i in range(30, 35):
        c[i] = 10.0 * (1 + 0.012 * (i - 30))
    c[35:] = c[34]
    panel = _make_panel(n, [c])
    events = detect_events_triple_barrier(panel)
    assert len(events) >= 1


def test_method_C_no_event_when_lower_first():
    from aurumq_rl.labeling.triple_barrier import detect_events_triple_barrier
    n = 60
    c = np.full(n, 10.0)
    rng = np.random.default_rng(42)
    c[1:30] = 10.0 + rng.normal(0, 0.05, size=29)
    # Drop 10% at day 32 — lower barrier first
    for i in range(30, 35):
        c[i] = 10.0 * (1 - 0.025 * (i - 29))
    c[35:] = c[34]
    panel = _make_panel(n, [c])
    events = detect_events_triple_barrier(panel)
    # Should be 0 events for this stock
    assert all(e.event_start_idx >= 35 for e in events) or len(events) == 0


# ---------------------------------------------------------------------------
# Method D
# ---------------------------------------------------------------------------

def test_method_D_detects_3pct_swing():
    from aurumq_rl.labeling.directional_change import detect_events_directional_change
    n = 80
    c = np.full(n, 10.0)
    # Drop to 9.7 (-3%) at day 20, then rise to 10.5 (+8.2%) at day 40
    c[10:21] = np.linspace(10.0, 9.7, 11)
    c[20:41] = np.linspace(9.7, 10.5, 21)
    c[40:] = c[40]
    panel = _make_panel(n, [c])
    events = detect_events_directional_change(panel)
    assert len(events) >= 1
    # Event start should be near the local low at day 20
    assert any(15 <= e.event_start_idx <= 25 for e in events)


def test_method_D_no_event_below_theta():
    from aurumq_rl.labeling.directional_change import detect_events_directional_change
    n = 80
    c = 10.0 + np.sin(np.arange(n) * 0.3) * 0.05   # +/- 0.5% oscillation
    panel = _make_panel(n, [c])
    events = detect_events_directional_change(panel)
    assert len(events) == 0, f"Sub-θ swings should not produce events; got {len(events)}"
