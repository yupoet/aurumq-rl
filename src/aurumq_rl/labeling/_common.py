"""Shared internal helpers."""

from __future__ import annotations

import numpy as np


def rolling_mean_1d(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean of 1-D array, NaN where window not full or has NaN."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < window:
        return out
    valid = np.isfinite(x).astype(np.int32)
    x_safe = np.where(np.isfinite(x), x, 0.0)
    cs_x = np.concatenate([[0.0], np.cumsum(x_safe)])
    cs_v = np.concatenate([[0], np.cumsum(valid)])
    for i in range(window - 1, n):
        lo = i - window + 1
        v_count = cs_v[i + 1] - cs_v[lo]
        if v_count == window:
            out[i] = (cs_x[i + 1] - cs_x[lo]) / window
    return out


def ewm_std_1d(x: np.ndarray, halflife: float) -> np.ndarray:
    """EWM std of 1-D, NaN-aware."""
    alpha = 1 - np.exp(np.log(0.5) / halflife)
    n = len(x)
    out = np.full(n, np.nan)
    mean = np.nan
    var = np.nan
    for i in range(n):
        v = x[i]
        if not np.isfinite(v):
            out[i] = np.nan if np.isnan(var) else np.sqrt(var)
            continue
        if np.isnan(mean):
            mean = v
            var = 0.0
        else:
            d = v - mean
            mean = mean + alpha * d
            var = (1 - alpha) * (var + alpha * d * d)
        out[i] = np.sqrt(var) if i >= 1 else np.nan
    return out
