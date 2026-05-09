"""Method D — Directional Change events (Glattfelder & Tsang, 2011).

State machine over adj_close per stock with multiple thresholds θ.
On each step, track a running extreme (high or low) per direction; an event
fires when the price reverses by ≥ θ from the running extreme.

For our use: we want UP-events. Each UP-event corresponds to a confirmed
upward swing of magnitude ≥ θ.
- event_start = the prior local LOW (extreme low before reversal)
- event_peak  = the new local HIGH (extreme high reached)
- event_quality = (peak - low) / low / θ_min   (multiples of base threshold)

We use multi-θ {0.03, 0.05, 0.08} and dedupe overlapping events keeping
the highest-quality one.

See SPEC §2.D.
"""

from __future__ import annotations

import numpy as np

from .events import Event
from .panels import MarketPanel
from ._common import rolling_mean_1d


__all__ = ["detect_events_directional_change"]


THETA_GRID = (0.03, 0.05, 0.08)


def _dc_events_at_theta(
    adj_close: np.ndarray,
    universe: np.ndarray,
    theta: float,
) -> list[tuple[int, int, float]]:
    """Run DC state machine at a single θ; return up-events as (low_idx, peak_idx, magnitude).

    State: direction in {'up', 'down'}, running extreme. Start scanning from
    first valid index. When direction == 'down' (tracking lows), if price >=
    extreme_low * (1+θ): emit up-event (low_idx, current_idx, peak_return),
    flip direction to 'up', reset extreme to current price.
    """
    n = len(adj_close)
    if n < 2:
        return []
    out: list[tuple[int, int, float]] = []

    # Start: find first valid in-universe price
    start_t = -1
    for t in range(n):
        if universe[t] and np.isfinite(adj_close[t]) and adj_close[t] > 0:
            start_t = t
            break
    if start_t < 0:
        return []
    direction = "down"   # initially track for first up-move (look for low first)
    extreme_low_idx = start_t
    extreme_low_val = adj_close[start_t]
    extreme_high_idx = start_t
    extreme_high_val = adj_close[start_t]

    for t in range(start_t + 1, n):
        c = adj_close[t]
        if not np.isfinite(c) or c <= 0:
            continue
        if direction == "down":
            # update running low
            if c < extreme_low_val:
                extreme_low_val = c
                extreme_low_idx = t
            # check for up-reversal
            if c >= extreme_low_val * (1.0 + theta):
                magnitude = (c - extreme_low_val) / extreme_low_val
                out.append((extreme_low_idx, t, magnitude))
                # Flip to up direction; current point is new high anchor
                direction = "up"
                extreme_high_val = c
                extreme_high_idx = t
        else:  # direction == "up"
            if c > extreme_high_val:
                extreme_high_val = c
                extreme_high_idx = t
            # check for down-reversal
            if c <= extreme_high_val * (1.0 - theta):
                # Flip to down direction; this becomes the new low anchor
                direction = "down"
                extreme_low_val = c
                extreme_low_idx = t
    return out


def _detect_events_one_stock(
    adj_close: np.ndarray,
    universe: np.ndarray,
    amount: np.ndarray,
    ts_code: str,
    *,
    theta_grid: tuple[float, ...] = THETA_GRID,
    amt_ma_window: int = 20,
    amt_min: float = 1e8,
    min_duration: int = 3,
    max_duration: int = 30,
) -> list[Event]:
    if len(theta_grid) == 0:
        return []
    theta_min = min(theta_grid)
    amt_ma = rolling_mean_1d(amount, window=amt_ma_window)

    raw_events: list[tuple[int, int, float, float]] = []
    for theta in theta_grid:
        events_theta = _dc_events_at_theta(adj_close, universe, theta)
        for low_idx, peak_idx, magnitude in events_theta:
            quality = magnitude / theta_min
            raw_events.append((low_idx, peak_idx, quality, magnitude))

    # Filter: liquidity + duration + universe at start
    filtered: list[Event] = []
    for low_idx, peak_idx, quality, magnitude in raw_events:
        dur = peak_idx - low_idx
        if dur < min_duration or dur > max_duration:
            continue
        if not universe[low_idx]:
            continue
        if not (np.isfinite(amt_ma[low_idx]) and amt_ma[low_idx] >= amt_min):
            continue
        filtered.append(Event(
            ts_code=ts_code,
            event_start_idx=low_idx,
            event_peak_idx=peak_idx,
            event_quality=float(quality),
            event_method="D",
        ))
    return filtered


def detect_events_directional_change(panel: MarketPanel) -> list[Event]:
    events: list[Event] = []
    for j, ts_code in enumerate(panel.ts_codes):
        evs = _detect_events_one_stock(
            adj_close=panel.adj_close[:, j],
            universe=panel.universe[:, j],
            amount=panel.amount[:, j],
            ts_code=ts_code,
        )
        events.extend(evs)
    return events
