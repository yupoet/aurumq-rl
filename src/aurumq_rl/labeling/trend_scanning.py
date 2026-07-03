"""Method B — Trend-Scanning labels (López de Prado, 2020).

For each (t, j), fit OLS on log(adj_close[t : t+L]) vs k for L in {5,10,15,20}.
event_quality = max_L |t_stat(L)|, signed by slope.
event = first up-trending t with quality >= τ_B (post-dedupe).

Vectorized OLS: for window L, k = arange(L) is fixed → analytic slope/SE in
closed form via cumulative sums.

See SPEC §2.B.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .events import Event
from .panels import MarketPanel

__all__ = ["detect_events_trend_scanning"]


L_GRID = (5, 10, 15, 20)

# Part 3 (issue #8): opt-in t-stat quality squashing. Default "raw" reproduces
# current (unbounded) event_quality exactly — see `_transform_tstat`.
TstatTransform = Literal["raw", "tanh", "clip"]
_TANH_SCALE = 10.0  # tanh(t / _TANH_SCALE) stays ~linear for |t| << _TANH_SCALE
_TSTAT_CAP = 10.0  # bound used by both "tanh" (asymptote) and "clip" (hard cap)


def _transform_tstat(t_stat: float, mode: TstatTransform) -> float:
    """Squash/cap a raw t-stat quality score. Default "raw" is the identity.

    Low-vol micro-drifts can produce raw t-stats in the hundreds/thousands
    that dominate `search_threshold`-based selection, favoring smooth drifts
    over genuine main waves. "tanh" and "clip" bound the score while
    preserving sign (direction) and relative ordering for |t_stat| << cap.
    """
    if mode == "raw":
        return t_stat
    if mode == "tanh":
        return float(np.tanh(t_stat / _TANH_SCALE) * _TSTAT_CAP)
    if mode == "clip":
        return float(np.clip(t_stat, -_TSTAT_CAP, _TSTAT_CAP))
    raise ValueError(f"Unknown tstat_transform {mode!r}; expected 'raw', 'tanh', or 'clip'")


def _vectorized_ols_tstat(y: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized OLS slope and t-stat over forward windows of size L.

    For each t, fits y[t:t+L] ~ alpha + beta * k where k = 0..L-1.
    Returns (slope[t], t_stat[t]) for each t. The fit covers t..t+L-1, so
    slope[t] is well-defined for t in [0, n-L].

    Closed-form for fixed regressor k = 0..L-1:
        beta = sum((k - k_bar)(y - y_bar)) / sum((k - k_bar)^2)
        residual variance s^2 = SSR / (L - 2)
        SE(beta) = sqrt(s^2 / sum((k - k_bar)^2))
        t_stat = beta / SE(beta)
    """
    n = len(y)
    slope = np.full(n, np.nan)
    tstat = np.full(n, np.nan)
    if n < L or L < 3:
        return slope, tstat

    k = np.arange(L, dtype=np.float64)
    k_bar = (L - 1) / 2.0
    k_dev = k - k_bar
    sk2 = (k_dev * k_dev).sum()  # constant
    # Slide windows
    for t in range(n - L + 1):
        seg = y[t : t + L]
        if not np.isfinite(seg).all():
            continue
        y_bar = seg.mean()
        y_dev = seg - y_bar
        beta = (k_dev * y_dev).sum() / sk2
        # residuals
        pred = beta * k_dev  # centered
        ss_res = ((y_dev - pred) ** 2).sum()
        if L <= 2:
            continue
        s2 = ss_res / (L - 2)
        if s2 <= 0:
            continue
        se_beta = np.sqrt(s2 / sk2)
        if se_beta <= 0:
            continue
        slope[t] = beta
        tstat[t] = beta / se_beta
    return slope, tstat


def _detect_events_one_stock(
    adj_close: np.ndarray,
    universe: np.ndarray,
    amount: np.ndarray,
    ts_code: str,
    *,
    L_grid: tuple[int, ...] = L_GRID,
    amt_ma_window: int = 20,
    amt_min: float = 1e8,
    tstat_transform: TstatTransform = "raw",
) -> list[Event]:
    n = len(adj_close)
    if n < max(L_grid) + 2:
        return []
    log_close = np.log(np.where(adj_close > 0, adj_close, np.nan))

    # Per-L slope/tstat arrays, then take argmax |tstat| per t
    best_t = np.full(n, np.nan)
    best_slope = np.full(n, np.nan)
    best_L = np.full(n, -1, dtype=np.int32)
    for L in L_grid:
        slope_L, tstat_L = _vectorized_ols_tstat(log_close, L)
        for t in range(n):
            if not np.isfinite(tstat_L[t]):
                continue
            if not np.isfinite(best_t[t]) or abs(tstat_L[t]) > abs(best_t[t]):
                best_t[t] = tstat_L[t]
                best_slope[t] = slope_L[t]
                best_L[t] = L

    # event detection: walk forward, find first t with universe & best_slope > 0 & best_t > 0,
    # event_start = t (i.e., the trend starts at t and is observed in [t, t+L])
    # event_peak = t + best_L[t]
    # event_quality = best_t[t] (continuous)
    from ._common import rolling_mean_1d

    amt_ma = rolling_mean_1d(amount, window=amt_ma_window)

    events: list[Event] = []
    last_peak = -1
    for t in range(n):
        if t <= last_peak:
            continue
        if not universe[t]:
            continue
        if not (
            np.isfinite(best_t[t])
            and best_t[t] > 0
            and np.isfinite(best_slope[t])
            and best_slope[t] > 0
        ):
            continue
        if not (np.isfinite(amt_ma[t]) and amt_ma[t] >= amt_min):
            continue
        L_used = int(best_L[t])
        if L_used <= 0:
            continue
        peak_idx = min(n - 1, t + L_used)
        # Selection/gating above always uses the raw t-stat (best_t); the
        # transform only affects the reported event_quality (Part 3, issue #8).
        events.append(
            Event(
                ts_code=ts_code,
                event_start_idx=t,
                event_peak_idx=peak_idx,
                event_quality=_transform_tstat(float(best_t[t]), tstat_transform),
                event_method="B",
            )
        )
        last_peak = peak_idx
    return events


def detect_events_trend_scanning(
    panel: MarketPanel,
    L_grid: tuple[int, ...] = L_GRID,
    *,
    tstat_transform: TstatTransform = "raw",
) -> list[Event]:
    """Detect Method B (trend-scanning) events.

    Parameters
    ----------
    panel
        Market panel (see `MarketPanel`).
    L_grid
        Forward-window lengths to fit OLS trends over.
    tstat_transform
        OPT-IN quality squashing (Part 3, issue #8). "raw" (default) reproduces
        the current unbounded `event_quality = t_stat` exactly. "tanh" and
        "clip" bound the reported quality so low-vol micro-drift t-stats in
        the hundreds/thousands no longer dominate `search_threshold`-based
        selection; "tanh" is the recommended setting (smoothly bounded,
        sign-preserving, near-linear for small |t|). Selection of which
        forward window `L` wins per `t` (and the up/down direction gate)
        always uses the raw t-stat — the transform only affects the reported
        `event_quality`.
    """
    events: list[Event] = []
    for j, ts_code in enumerate(panel.ts_codes):
        evs = _detect_events_one_stock(
            adj_close=panel.adj_close[:, j],
            universe=panel.universe[:, j],
            amount=panel.amount[:, j],
            ts_code=ts_code,
            L_grid=L_grid,
            tstat_transform=tstat_transform,
        )
        events.extend(evs)
    return events
