"""Tests for trend_scanning t-stat squashing transform (issue #8, Part 3).

`tstat_transform` is opt-in; default "raw" must reproduce current output exactly.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from aurumq_rl.labeling.panels import MarketPanel
from aurumq_rl.labeling.trend_scanning import detect_events_trend_scanning


def _trade_dates(n: int) -> list[date]:
    base = date(2024, 1, 1)
    return [base + timedelta(days=i) for i in range(n)]


def _make_panel(n_dates: int, close: np.ndarray, amount_value: float = 5e8) -> MarketPanel:
    adj_close = close.reshape(-1, 1).astype(np.float64)
    pct = np.diff(adj_close, axis=0, prepend=adj_close[:1]) / np.where(adj_close > 0, adj_close, 1)
    pct[0] = 0.0
    amount = np.full((n_dates, 1), amount_value, dtype=np.float64)
    universe = np.ones((n_dates, 1), dtype=bool)
    benchmark = np.ones(n_dates, dtype=np.float64)
    return MarketPanel(
        trade_dates=_trade_dates(n_dates),
        ts_codes=["STK000"],
        adj_close=adj_close,
        pct_change=pct,
        amount=amount,
        universe=universe,
        benchmark_close=benchmark,
    )


def _low_vol_microdrift_panel() -> MarketPanel:
    # Extremely smooth, tiny daily drift -> huge |t-stat| (near-perfect linear fit)
    n = 60
    c = 10.0 * (1.0 + 0.0005 * np.arange(n))
    return _make_panel(n, c)


def test_tstat_transform_raw_reproduces_current_quality_exactly():
    panel = _low_vol_microdrift_panel()
    baseline = detect_events_trend_scanning(panel)
    explicit_raw = detect_events_trend_scanning(panel, tstat_transform="raw")
    assert len(baseline) == len(explicit_raw)
    for e_base, e_raw in zip(baseline, explicit_raw, strict=True):
        assert e_base.event_quality == e_raw.event_quality
        assert e_base.event_start_idx == e_raw.event_start_idx


def test_tstat_transform_tanh_compresses_high_tstat():
    panel = _low_vol_microdrift_panel()
    raw_events = detect_events_trend_scanning(panel, tstat_transform="raw")
    tanh_events = detect_events_trend_scanning(panel, tstat_transform="tanh")
    assert len(raw_events) >= 1 and len(tanh_events) >= 1
    raw_by_start = {e.event_start_idx: e.event_quality for e in raw_events}
    tanh_by_start = {e.event_start_idx: e.event_quality for e in tanh_events}
    shared = set(raw_by_start) & set(tanh_by_start)
    assert shared, "raw and tanh should fire at the same event starts"
    for start in shared:
        raw_q = raw_by_start[start]
        tanh_q = tanh_by_start[start]
        assert raw_q > 50, f"expected a very high raw t-stat on smooth micro-drift, got {raw_q}"
        assert tanh_q < raw_q, "tanh transform must compress a large raw t-stat"
        assert tanh_q > 0, "sign should be preserved for an up-trend"


def test_tstat_transform_clip_caps_at_bound():
    panel = _low_vol_microdrift_panel()
    clipped = detect_events_trend_scanning(panel, tstat_transform="clip")
    assert len(clipped) >= 1
    for e in clipped:
        assert abs(e.event_quality) <= 10.0 + 1e-9


def test_tstat_transform_invalid_mode_raises():
    panel = _low_vol_microdrift_panel()
    try:
        detect_events_trend_scanning(panel, tstat_transform="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown tstat_transform")


def test_tstat_transform_invalid_mode_raises_even_with_zero_events():
    # Eager validation: a flat panel fires no events, so a lazy per-event
    # check would silently accept a typo'd mode. Must raise regardless.
    flat = _make_panel(60, np.full(60, 10.0))
    assert detect_events_trend_scanning(flat) == []
    try:
        detect_events_trend_scanning(flat, tstat_transform="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError even when no events fire")
