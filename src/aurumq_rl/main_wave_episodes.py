"""Phase 23 main-wave episode scanner.

Scans (close, vol) panel and identifies non-overlapping "real main wave"
episodes per stock. Each episode has:

- ``t_start``: first up day of the wave. ``close[t_start] > close[t_start-1]``
  by at least ``min_first_day_return`` AND the prior ``lookback_for_inflection``
  days were not already trending up (cumulative gain ≤ ``pre_window_max_gain``).
- ``t_peak``: day cumulative return from t_start peaks within the look-ahead
  window ``[min_duration, max_duration]``.
- ``peak_return = close[t_peak] / close[t_start] - 1``
- ``duration = t_peak - t_start`` (so duration days of upward movement)

Quality gates (configurable):

- ``peak_return ≥ min_peak_return`` (default 0.10)
- ``duration ∈ [min_duration, max_duration]`` (default [3, 20])
- ``max_dd_during_rally ≤ base_dd_allowance + dd_per_peak * peak_return``
- ``avg_daily_return = peak_return / duration ≥ min_avg_daily_return``
  (default 0.015 → rejects slow drifts that happen to accumulate ≥10%)
- amount_ma20 at t_start > ``amount_ma_min``

By convention, **the user's "T-1 decision day" = ``t_start - 1``**: deciding
to buy on the day BEFORE first up move. Entry price proxy = ``close[t_start]``
(no open data available).

Pure numpy. No torch / SB3 dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Config + container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeConfig:
    """All knobs for episode detection."""

    # Quality thresholds
    min_peak_return: float = 0.10
    min_duration: int = 3
    max_duration: int = 20
    min_avg_daily_return: float = 0.015   # peak_return / duration must exceed

    # Drawdown allowance: linear in peak_return so a +30% wave can have larger
    # pullbacks than a +10% wave. allowance = base + per_peak * peak_return
    base_dd_allowance: float = 0.02
    dd_per_peak: float = 0.50

    # Inflection criteria — t_start must be a clear "first up day"
    min_first_day_return: float = 0.005   # today's close > yesterday's by ≥ 0.5%
    pre_window_max_gain: float = 0.03      # prior 5d cumulative gain ≤ 3%
    lookback_for_inflection: int = 5

    # Liquidity at t_start
    amount_ma_min: float = 1e8
    amount_ma_window: int = 20


@dataclass
class MainWaveEpisode:
    """A detected main-wave episode for a single stock.

    Convention: ``t_start`` is the first up day. ``t_peak`` is the cumulative-
    return peak day within the look-ahead window. The user's "T-1 decision
    day" is ``t_start - 1``.
    """

    stock_idx: int
    t_start: int
    t_peak: int
    peak_return: float
    duration: int          # = t_peak - t_start (days of upward movement)
    max_dd_during: float   # absolute, ≥ 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rolling_mean_2d(a: np.ndarray, w: int) -> np.ndarray:
    T, S = a.shape
    out = np.zeros_like(a, dtype=np.float64)
    cs = np.cumsum(a, axis=0, dtype=np.float64)
    for t in range(T):
        lo = max(0, t - w + 1)
        seg = cs[t] - (cs[lo - 1] if lo > 0 else 0.0)
        out[t] = seg / float(t - lo + 1)
    return out.astype(np.float32)


def _max_dd_allowance(peak_return: float, cfg: EpisodeConfig) -> float:
    return cfg.base_dd_allowance + cfg.dd_per_peak * peak_return


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------


def find_main_wave_episodes(
    close: np.ndarray,
    vol: np.ndarray,
    valid_basic: np.ndarray,
    cfg: EpisodeConfig | None = None,
) -> list[MainWaveEpisode]:
    """Scan the panel for main-wave episodes per stock.

    Episodes are non-overlapping within a stock; the next candidate
    ``t_start`` must be > previous ``t_peak``.

    Parameters
    ----------
    close, vol : (T, S) float
    valid_basic : (T, S) bool
    cfg : EpisodeConfig

    Returns
    -------
    Flat list of MainWaveEpisode records, ordered by stock then t_start.
    """
    cfg = cfg or EpisodeConfig()
    T, S = close.shape
    assert vol.shape == (T, S)
    assert valid_basic.shape == (T, S)

    amount = close * vol
    amount_ma = _rolling_mean_2d(amount, cfg.amount_ma_window)
    liquid_mask = amount_ma > cfg.amount_ma_min

    episodes: list[MainWaveEpisode] = []

    for j in range(S):
        c = close[:, j]
        liquid_j = liquid_mask[:, j]
        valid_j = valid_basic[:, j]
        # t needs at least one prior day for today_gain. The pre_window
        # check below uses whatever lookback is available.
        t = 1
        while t <= T - cfg.min_duration - 1:
            # 1. Liquidity / basic mask gates
            if not liquid_j[t] or not valid_j[t]:
                t += 1
                continue
            # 2. Need positive prices for ratio math
            if c[t] <= 0 or c[t - 1] <= 0:
                t += 1
                continue
            # 3. Inflection: today must be a clear up day
            today_gain = float(c[t] / c[t - 1]) - 1.0
            if today_gain < cfg.min_first_day_return:
                t += 1
                continue
            # 4. Prior window must not already be trending up
            lb = cfg.lookback_for_inflection
            prior_start = max(0, t - lb)
            prior = c[prior_start:t]   # excludes today
            if len(prior) >= 2 and prior[0] > 0:
                prior_gain = float(prior[-1] / prior[0]) - 1.0
                if prior_gain > cfg.pre_window_max_gain:
                    t += 1
                    continue
            # 5. Look-ahead window for peak
            end = min(T, t + cfg.max_duration + 1)
            sub = c[t:end]
            if not np.isfinite(sub).all() or (sub <= 0).any():
                t += 1
                continue
            rel = sub.astype(np.float64) / float(c[t]) - 1.0   # offset 0..len-1
            # Peak must be at offset >= min_duration (require ≥ min_duration days of rise)
            if len(rel) <= cfg.min_duration:
                t += 1
                continue
            search_region = rel[cfg.min_duration:]
            if len(search_region) == 0:
                t += 1
                continue
            peak_offset_in_search = int(np.argmax(search_region))
            peak_offset = cfg.min_duration + peak_offset_in_search
            peak_ret = float(rel[peak_offset])
            # 6. Peak threshold
            if peak_ret < cfg.min_peak_return:
                t += 1
                continue
            # 7. DD during [0, peak_offset]
            running_peak = np.maximum.accumulate(rel[:peak_offset + 1])
            max_dd = float((running_peak - rel[:peak_offset + 1]).max())
            if max_dd > _max_dd_allowance(peak_ret, cfg):
                t += 1
                continue
            # 8. Avg daily rate
            if peak_offset == 0 or peak_ret / peak_offset < cfg.min_avg_daily_return:
                t += 1
                continue
            # All gates passed — record episode
            episodes.append(MainWaveEpisode(
                stock_idx=j,
                t_start=t,
                t_peak=t + peak_offset,
                peak_return=peak_ret,
                duration=peak_offset,
                max_dd_during=max_dd,
            ))
            # Advance past peak to avoid overlapping episodes from the same
            # rising region.
            t = t + peak_offset + 1

    return episodes


# ---------------------------------------------------------------------------
# Helpers exposed for inspection / aggregation
# ---------------------------------------------------------------------------


def episodes_summary(episodes: Iterable[MainWaveEpisode]) -> dict:
    eps = list(episodes)
    if not eps:
        return {
            "n_episodes": 0,
            "avg_peak_return": 0.0,
            "median_peak_return": 0.0,
            "avg_duration": 0.0,
            "avg_max_dd": 0.0,
            "max_peak_return": 0.0,
        }
    pr = np.asarray([e.peak_return for e in eps])
    dur = np.asarray([e.duration for e in eps])
    dd = np.asarray([e.max_dd_during for e in eps])
    return {
        "n_episodes": len(eps),
        "avg_peak_return": float(pr.mean()),
        "median_peak_return": float(np.median(pr)),
        "p90_peak_return": float(np.percentile(pr, 90)),
        "avg_duration": float(dur.mean()),
        "median_duration": float(np.median(dur)),
        "avg_max_dd": float(dd.mean()),
        "max_peak_return": float(pr.max()),
    }


__all__ = [
    "EpisodeConfig",
    "MainWaveEpisode",
    "find_main_wave_episodes",
    "episodes_summary",
]
