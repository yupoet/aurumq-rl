"""Method A — User's v2 (excess return + adaptive vol).

Per-stock per-day forward scan:
    inflection at t (today_gain ≥ 0.5%, prior 5d cum ≤ 3%)
    fwd peak in [t+3, t+20]
    fwd_max_excess >= max(0.06, 1.8 * vol20)
    max_dd <= 0.02 + 0.5 * fwd_max_excess
    amount_ma20 >= 1e8

event_quality = fwd_max_excess / adaptive_thr  (continuous, ≥1 means above threshold)

See SPEC §2.A.
"""

from __future__ import annotations

import math

import numpy as np

from ._common import ewm_std_1d, rolling_mean_1d
from .events import Event
from .panels import MarketPanel

__all__ = ["detect_events_v2", "adaptive_threshold"]


# Defaults from user's original v2 spec
DEFAULT_MIN_TODAY_GAIN = 0.005
DEFAULT_PRE_WINDOW_GAIN = 0.03
DEFAULT_PRE_WINDOW = 5
DEFAULT_MIN_DURATION = 3
DEFAULT_MAX_DURATION = 20
DEFAULT_VOL_HALFLIFE = 10
DEFAULT_VOL_LOOKBACK = 20
DEFAULT_AMT_MA_WINDOW = 20
DEFAULT_AMT_MIN = 1e8


_ewm_std = ewm_std_1d
_rolling_mean = rolling_mean_1d


def adaptive_threshold(vol20: float, duration: int, *, horizon_scaling: bool = False) -> float:
    """Compute the P0 adaptive excess-return threshold `max(0.06, 1.8 * vol20)`.

    GUARDRAIL (P0-locked): `horizon_scaling=False` (the default) is the exact
    formula the P0-locked `v2_excess_adaptive` label was calibrated with — it
    ignores `duration` entirely. This is what `detect_events_v2` uses unless
    the caller explicitly opts in.

    `horizon_scaling=True` is OPT-IN (issue #8, Part 4): it scales the vol
    term by `sqrt(duration)`, since the vol input (`vol20`) is a *daily* EWM
    std while the excess return it is compared against accumulates over up to
    20 trading days with no horizon scaling — so the 0.06 floor binds almost
    always and the label is effectively non-adaptive. Enabling this changes
    the P0-locked label output and requires a re-ablation before production
    use; do not flip the default.

    Parameters
    ----------
    vol20
        Daily EWM volatility (halflife 10) as of the decision day.
    duration
        Number of trading days from event start to the forward peak used to
        realize `fwd_max_excess` (only used when `horizon_scaling=True`).
    horizon_scaling
        OPT-IN. Default False reproduces the P0-locked formula byte-for-byte.
    """
    if horizon_scaling:
        return max(0.06, 1.8 * vol20 * math.sqrt(max(duration, 0)))
    return max(0.06, 1.8 * vol20)


def _detect_events_one_stock(
    adj_close: np.ndarray,
    pct_change: np.ndarray,
    amount: np.ndarray,
    universe: np.ndarray,
    benchmark_close: np.ndarray,
    ts_code: str,
    *,
    min_today_gain: float = DEFAULT_MIN_TODAY_GAIN,
    pre_window_gain: float = DEFAULT_PRE_WINDOW_GAIN,
    pre_window: int = DEFAULT_PRE_WINDOW,
    min_duration: int = DEFAULT_MIN_DURATION,
    max_duration: int = DEFAULT_MAX_DURATION,
    vol_halflife: float = DEFAULT_VOL_HALFLIFE,
    amt_ma_window: int = DEFAULT_AMT_MA_WINDOW,
    amt_min: float = DEFAULT_AMT_MIN,
    horizon_scaling: bool = False,
) -> list[Event]:
    n = len(adj_close)
    if n < pre_window + max_duration + 2:
        return []
    vol = _ewm_std(pct_change, halflife=vol_halflife)
    amt_ma = _rolling_mean(amount, window=amt_ma_window)

    events: list[Event] = []
    last_peak_idx = -1  # for non-overlapping detection within this stock
    t = pre_window + 1
    while t <= n - min_duration - 1:
        if t <= last_peak_idx:
            t += 1
            continue
        # Universe gate at t
        if not universe[t]:
            t += 1
            continue
        # Need adj_close[t-1] valid
        if not (
            np.isfinite(adj_close[t]) and np.isfinite(adj_close[t - 1]) and adj_close[t - 1] > 0
        ):
            t += 1
            continue
        # Inflection: today_gain >= min_today_gain
        today_gain = adj_close[t] / adj_close[t - 1] - 1.0
        if today_gain < min_today_gain:
            t += 1
            continue
        # Prior 5d cumulative gain <= pre_window_gain
        prior_lo = max(0, t - pre_window - 1)
        if not (np.isfinite(adj_close[prior_lo]) and adj_close[prior_lo] > 0):
            t += 1
            continue
        prior_gain = adj_close[t - 1] / adj_close[prior_lo] - 1.0
        if prior_gain > pre_window_gain:
            t += 1
            continue
        # Liquidity at t
        if not (np.isfinite(amt_ma[t]) and amt_ma[t] >= amt_min):
            t += 1
            continue
        # Vol20 at t (use t-1 to avoid lookahead)
        vol20 = vol[t - 1] if t >= 1 else np.nan
        if not np.isfinite(vol20) or vol20 <= 0:
            t += 1
            continue
        # Non-scaled threshold, computed here for parity with the pre-issue-8
        # (P0-locked) code path — this is exactly what default (horizon_scaling
        # =False) uses below, unaffected by the eventual peak duration.
        adaptive_thr = adaptive_threshold(vol20, duration=0, horizon_scaling=False)
        # Forward [t, t+max_duration] peak search
        end = min(n, t + max_duration + 1)
        sub = adj_close[t:end]
        if len(sub) <= min_duration:
            t += 1
            continue
        # Build returns vs t (use adj_close[t] as base; fwd peak vs benchmark)
        if not np.isfinite(sub).all() or (sub <= 0).any():
            t += 1
            continue
        rel = sub / adj_close[t] - 1.0  # returns since t (offset 0..len-1)
        # benchmark forward returns since t
        bench_sub = benchmark_close[t:end]
        if not np.isfinite(bench_sub).all() or bench_sub[0] <= 0:
            t += 1
            continue
        bench_rel = bench_sub / bench_sub[0] - 1.0
        excess = rel - bench_rel

        # Peak must be at offset >= min_duration
        search_region = excess[min_duration:]
        if len(search_region) == 0:
            t += 1
            continue
        peak_offset_in_search = int(np.argmax(search_region))
        peak_offset = min_duration + peak_offset_in_search
        if peak_offset > max_duration:
            t += 1
            continue
        if horizon_scaling:
            # OPT-IN (Part 4, issue #8): rescale the vol term by sqrt(duration)
            # now that the realized peak duration is known. Changes the
            # P0-locked threshold — see `adaptive_threshold` docstring.
            adaptive_thr = adaptive_threshold(vol20, duration=peak_offset, horizon_scaling=True)
        fwd_max_excess = float(excess[peak_offset])
        if fwd_max_excess < adaptive_thr:
            t += 1
            continue
        # Drawdown during [0, peak_offset]
        running_max = np.maximum.accumulate(rel[: peak_offset + 1])
        max_dd = float((running_max - rel[: peak_offset + 1]).max())
        if max_dd > 0.02 + 0.5 * fwd_max_excess:
            t += 1
            continue
        # Average daily rate (sanity, allow small to be permissive)
        if peak_offset == 0 or fwd_max_excess / peak_offset < 0.005:
            t += 1
            continue
        # All gates passed
        event_quality = float(fwd_max_excess / adaptive_thr)
        events.append(
            Event(
                ts_code=ts_code,
                event_start_idx=t,
                event_peak_idx=t + peak_offset,
                event_quality=event_quality,
                event_method="A",
            )
        )
        last_peak_idx = t + peak_offset
        t = last_peak_idx + 1
    return events


def detect_events_v2(panel: MarketPanel, *, horizon_scaling: bool = False) -> list[Event]:
    """Run method A on a full MarketPanel; returns flat event list.

    Parameters
    ----------
    panel
        Market panel (see `MarketPanel`).
    horizon_scaling
        OPT-IN (Part 4, issue #8), default False. GUARDRAIL: `v2_excess_adaptive`
        is the P0-locked ablation winner — `horizon_scaling=False` reproduces
        the current P0-locked label output byte-for-byte. Setting `True`
        scales the adaptive vol threshold by `sqrt(duration)` (see
        `adaptive_threshold`), which changes the label and requires a
        re-ablation before production use. Do not flip this default.
    """
    events: list[Event] = []
    for j, ts_code in enumerate(panel.ts_codes):
        evs = _detect_events_one_stock(
            adj_close=panel.adj_close[:, j],
            pct_change=panel.pct_change[:, j],
            amount=panel.amount[:, j],
            universe=panel.universe[:, j],
            benchmark_close=panel.benchmark_close,
            ts_code=ts_code,
            horizon_scaling=horizon_scaling,
        )
        events.extend(evs)
    return events
