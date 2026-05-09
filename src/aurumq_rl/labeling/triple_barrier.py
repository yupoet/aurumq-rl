"""Method C — Triple-Barrier labels (López de Prado, 2018).

For each (t, j):
    upper = adj_close[t] * (1 + h * sigma_t)
    lower = adj_close[t] * (1 - h * sigma_t)
    vert  = t + T

Walk forward t+1..t+T; emit event at t iff upper barrier hits FIRST.
event_start = t, event_peak = first day k where adj_close[t+k] >= upper.
event_quality = (peak_close - adj_close[t]) / (adj_close[t] * sigma_t)

See SPEC §2.C.
"""

from __future__ import annotations

import numpy as np

from .events import Event
from .panels import MarketPanel
from ._common import ewm_std_1d, rolling_mean_1d


__all__ = ["detect_events_triple_barrier"]


DEFAULT_H = 2.0
DEFAULT_VERT_T = 20
DEFAULT_VOL_HALFLIFE = 10
DEFAULT_AMT_MIN = 1e8
DEFAULT_AMT_MA_WINDOW = 20


def _detect_events_one_stock(
    adj_close: np.ndarray,
    pct_change: np.ndarray,
    universe: np.ndarray,
    amount: np.ndarray,
    ts_code: str,
    *,
    h: float = DEFAULT_H,
    vert_T: int = DEFAULT_VERT_T,
    vol_halflife: float = DEFAULT_VOL_HALFLIFE,
    amt_ma_window: int = DEFAULT_AMT_MA_WINDOW,
    amt_min: float = DEFAULT_AMT_MIN,
) -> list[Event]:
    n = len(adj_close)
    if n < vert_T + 2:
        return []
    sigma = ewm_std_1d(pct_change, halflife=vol_halflife)
    amt_ma = rolling_mean_1d(amount, window=amt_ma_window)

    events: list[Event] = []
    last_peak = -1
    t = 1
    while t < n - vert_T:
        if t <= last_peak:
            t += 1
            continue
        if not universe[t]:
            t += 1
            continue
        c0 = adj_close[t]
        if not (np.isfinite(c0) and c0 > 0):
            t += 1
            continue
        sig = sigma[t - 1] if t >= 1 else np.nan
        if not (np.isfinite(sig) and sig > 0):
            t += 1
            continue
        if not (np.isfinite(amt_ma[t]) and amt_ma[t] >= amt_min):
            t += 1
            continue
        upper = c0 * (1 + h * sig)
        lower = c0 * (1 - h * sig)
        # Walk forward
        upper_idx = -1
        lower_idx = -1
        for k in range(1, vert_T + 1):
            ck = adj_close[t + k]
            if not np.isfinite(ck):
                continue
            if ck >= upper and upper_idx < 0:
                upper_idx = k
                break
            if ck <= lower and lower_idx < 0:
                lower_idx = k
                break
        # event only if upper first
        if upper_idx > 0 and (lower_idx < 0 or upper_idx < lower_idx):
            event_quality = float((adj_close[t + upper_idx] - c0) / (c0 * sig))
            events.append(Event(
                ts_code=ts_code,
                event_start_idx=t,
                event_peak_idx=t + upper_idx,
                event_quality=event_quality,
                event_method="C",
            ))
            last_peak = t + upper_idx
            t = last_peak + 1
        else:
            t += 1
    return events


def detect_events_triple_barrier(panel: MarketPanel) -> list[Event]:
    events: list[Event] = []
    for j, ts_code in enumerate(panel.ts_codes):
        evs = _detect_events_one_stock(
            adj_close=panel.adj_close[:, j],
            pct_change=panel.pct_change[:, j],
            universe=panel.universe[:, j],
            amount=panel.amount[:, j],
            ts_code=ts_code,
        )
        events.extend(evs)
    return events
