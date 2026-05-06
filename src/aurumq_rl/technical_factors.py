"""Phase 24 — technical & cumulative-position factors.

Computes 30+ technical/structural factors on the panel (close, vol) at load
time and injects them as additional channels in ``factor_array``. Pure
numpy. Used by the data loader when ``--add-technical-factors`` is set.

Categories
----------

**MA & cross**:
- ma5, ma10, ma20, ma60 (rolling means)
- close_vs_ma5/10/20/60 (close/ma - 1, distance %)
- ma5_above_ma10 / ma10_above_ma20 / ma20_above_ma60 (state)
- ma5_cross_ma10 / ma10_cross_ma20 / ma20_cross_ma60 (golden cross event)
- in_ma60_band_4pct (close within ±4% of MA60)

**KDJ** (close-only approximation):
- kdj_k, kdj_d, kdj_j
- kdj_golden_cross (K crosses above D)
- kdj_oversold (J < 20), kdj_overbought (J > 80)

**MACD**:
- macd_dif, macd_dea, macd_hist
- macd_golden_cross (DIF crosses above DEA)

**Bollinger**:
- boll_pct_b (position in band, 0=lower, 1=upper)
- boll_band_width (width / middle)
- boll_squeeze (band width is at 20d low — about to expand)

**Volume / amplitude**:
- vol_ma20, vol_ratio (today / vol_ma20)
- vol_decay_5d (vol_5d / vol_20d, low = volume drying up)
- amplitude_5d, amplitude_20d (max/min - 1 in window)

**Cumulative main force** (computed from mf_net_1d):
- cmf_60d, cmf_120d (rolling sum)
- cmf_60d_pct (cmf_60d / amount_60d_total — accumulated position ratio)
- cmf_60d_pos_days (count of days with mf_net_1d > 0 in last 60)

**Computed 涨停 count** (replaces broken senti_zt_count_30d):
- zt_count_30d (computed from pct_chg ≥ 0.099)

All factors are emitted in float32. Cross-section z-scoring is done downstream
by data_loader._cross_section_zscore, NOT here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rolling_mean_2d(a: np.ndarray, w: int) -> np.ndarray:
    """(T, S) → (T, S) rolling mean down time axis. Partial windows allowed."""
    T, S = a.shape
    out = np.zeros_like(a, dtype=np.float64)
    cs = np.cumsum(a, axis=0, dtype=np.float64)
    for t in range(T):
        lo = max(0, t - w + 1)
        seg = cs[t] - (cs[lo - 1] if lo > 0 else 0.0)
        out[t] = seg / float(t - lo + 1)
    return out.astype(np.float32)


def _rolling_sum_2d(a: np.ndarray, w: int) -> np.ndarray:
    T, S = a.shape
    out = np.zeros_like(a, dtype=np.float64)
    cs = np.cumsum(a, axis=0, dtype=np.float64)
    for t in range(T):
        lo = max(0, t - w + 1)
        out[t] = cs[t] - (cs[lo - 1] if lo > 0 else 0.0)
    return out.astype(np.float32)


def _rolling_std_2d(a: np.ndarray, w: int) -> np.ndarray:
    T, S = a.shape
    out = np.zeros_like(a, dtype=np.float64)
    cs = np.cumsum(a, axis=0, dtype=np.float64)
    cs2 = np.cumsum(a.astype(np.float64) ** 2, axis=0)
    for t in range(T):
        lo = max(0, t - w + 1)
        n = float(t - lo + 1)
        s1 = cs[t] - (cs[lo - 1] if lo > 0 else 0.0)
        s2 = cs2[t] - (cs2[lo - 1] if lo > 0 else 0.0)
        if n < 2:
            out[t] = 0.0
        else:
            var = np.maximum(s2 / n - (s1 / n) ** 2, 0.0)
            out[t] = np.sqrt(var)
    return out.astype(np.float32)


def _rolling_max_2d(a: np.ndarray, w: int) -> np.ndarray:
    T, S = a.shape
    out = np.zeros_like(a, dtype=np.float32)
    for t in range(T):
        lo = max(0, t - w + 1)
        out[t] = a[lo:t + 1].max(axis=0)
    return out


def _rolling_min_2d(a: np.ndarray, w: int) -> np.ndarray:
    T, S = a.shape
    out = np.zeros_like(a, dtype=np.float32)
    for t in range(T):
        lo = max(0, t - w + 1)
        # Skip non-positive values (panel zeros from missing data)
        seg = a[lo:t + 1].copy()
        seg[seg <= 0] = np.inf
        m = seg.min(axis=0)
        m = np.where(np.isfinite(m), m, a[t])
        out[t] = m
    return out


def _ema_2d(a: np.ndarray, period: int) -> np.ndarray:
    """(T, S) EMA along time axis, period N → smoothing factor 2/(N+1)."""
    T, S = a.shape
    out = np.zeros_like(a, dtype=np.float64)
    alpha = 2.0 / (period + 1.0)
    out[0] = a[0]
    for t in range(1, T):
        out[t] = alpha * a[t] + (1.0 - alpha) * out[t - 1]
    return out.astype(np.float32)


def _cross_event(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 on day t if a[t-1] <= b[t-1] and a[t] > b[t] (golden cross). 0 else."""
    T, S = a.shape
    out = np.zeros_like(a, dtype=np.float32)
    out[1:] = ((a[:-1] <= b[:-1]) & (a[1:] > b[1:])).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TechnicalFactorsConfig:
    """Knobs (mostly defaults, periodically tuned)."""

    ma_periods: tuple[int, ...] = (5, 10, 20, 60)
    kdj_period: int = 9
    kdj_k_period: int = 3
    kdj_d_period: int = 3
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    boll_period: int = 20
    boll_std_mult: float = 2.0
    vol_ma_period: int = 20
    cmf_periods: tuple[int, ...] = (60, 120)
    zt_threshold: float = 0.099   # +9.9% counts as 涨停 (decimal pct)
    ma60_band_pct: float = 0.04   # ±4% band around MA60


def compute_technical_factors(
    close: np.ndarray,
    vol: np.ndarray,
    pct_chg: np.ndarray,
    mf_net_1d: np.ndarray | None = None,
    cfg: TechnicalFactorsConfig | None = None,
) -> dict[str, np.ndarray]:
    """Compute all technical factors.

    Parameters
    ----------
    close, vol : (T, S) float32
    pct_chg : (T, S) decimal returns
    mf_net_1d : (T, S) optional, daily main-force net inflow (元). When
        provided, cumulative main-force factors (cmf_*) are computed.
    cfg : TechnicalFactorsConfig

    Returns
    -------
    Ordered dict of {factor_name: (T, S) float32 array}. Factor names use
    the prefix conventions: ``tech_*``, ``cmf_*``, ``zt_*``.
    """
    cfg = cfg or TechnicalFactorsConfig()
    T, S = close.shape
    assert vol.shape == (T, S)
    assert pct_chg.shape == (T, S)

    out: dict[str, np.ndarray] = {}

    # ---- MA & cross ----
    mas: dict[int, np.ndarray] = {}
    for p in cfg.ma_periods:
        ma = _rolling_mean_2d(close, p)
        mas[p] = ma
        # close vs MA distance — use safe division
        dist = np.where(ma > 0, close / ma - 1.0, 0.0).astype(np.float32)
        out[f"tech_close_vs_ma{p}"] = dist

    # MA crosses (golden cross events) and state
    if 5 in mas and 10 in mas:
        out["tech_ma5_above_ma10"] = (mas[5] > mas[10]).astype(np.float32)
        out["tech_ma5_cross_ma10"] = _cross_event(mas[5], mas[10])
    if 10 in mas and 20 in mas:
        out["tech_ma10_above_ma20"] = (mas[10] > mas[20]).astype(np.float32)
        out["tech_ma10_cross_ma20"] = _cross_event(mas[10], mas[20])
    if 20 in mas and 60 in mas:
        out["tech_ma20_above_ma60"] = (mas[20] > mas[60]).astype(np.float32)
        out["tech_ma20_cross_ma60"] = _cross_event(mas[20], mas[60])

    # MA60 band proximity
    if 60 in mas:
        ma60 = mas[60]
        with np.errstate(invalid="ignore", divide="ignore"):
            dist60 = np.where(ma60 > 0, close / ma60 - 1.0, 0.0)
        in_band = (np.abs(dist60) <= cfg.ma60_band_pct).astype(np.float32)
        out["tech_in_ma60_band"] = in_band

    # ---- KDJ (close-only proxy) ----
    rolling_max = _rolling_max_2d(close, cfg.kdj_period)
    rolling_min = _rolling_min_2d(close, cfg.kdj_period)
    rng = rolling_max - rolling_min
    rng_safe = np.where(rng > 1e-9, rng, 1.0)
    rsv = ((close - rolling_min) / rng_safe * 100.0).astype(np.float32)
    rsv = np.clip(rsv, 0.0, 100.0)

    # KDJ K (smooth RSV with simple recursion: K = (2/3)*K_prev + (1/3)*RSV)
    K = np.zeros_like(rsv)
    K[0] = 50.0
    for t in range(1, T):
        K[t] = (2.0 / 3.0) * K[t - 1] + (1.0 / 3.0) * rsv[t]
    K = np.clip(K, 0.0, 100.0).astype(np.float32)
    D = np.zeros_like(rsv)
    D[0] = 50.0
    for t in range(1, T):
        D[t] = (2.0 / 3.0) * D[t - 1] + (1.0 / 3.0) * K[t]
    D = np.clip(D, 0.0, 100.0).astype(np.float32)
    J = np.clip(3.0 * K - 2.0 * D, -50.0, 150.0).astype(np.float32)
    out["tech_kdj_k"] = K.astype(np.float32)
    out["tech_kdj_d"] = D.astype(np.float32)
    out["tech_kdj_j"] = J
    out["tech_kdj_golden_cross"] = _cross_event(K, D)
    out["tech_kdj_oversold"] = (J < 20).astype(np.float32)

    # ---- MACD ----
    ema_fast = _ema_2d(close, cfg.macd_fast)
    ema_slow = _ema_2d(close, cfg.macd_slow)
    dif = (ema_fast - ema_slow).astype(np.float32)
    dea = _ema_2d(dif, cfg.macd_signal)
    macd_hist = (2.0 * (dif - dea)).astype(np.float32)
    # Normalize MACD by close to make cross-stock comparable
    safe_close = np.where(close > 0, close, 1.0)
    out["tech_macd_dif_norm"] = (dif / safe_close).astype(np.float32)
    out["tech_macd_dea_norm"] = (dea / safe_close).astype(np.float32)
    out["tech_macd_hist_norm"] = (macd_hist / safe_close).astype(np.float32)
    out["tech_macd_golden_cross"] = _cross_event(dif, dea)
    out["tech_macd_dif_above_dea"] = (dif > dea).astype(np.float32)

    # ---- Bollinger ----
    boll_mid = _rolling_mean_2d(close, cfg.boll_period)
    boll_std = _rolling_std_2d(close, cfg.boll_period)
    boll_upper = boll_mid + cfg.boll_std_mult * boll_std
    boll_lower = boll_mid - cfg.boll_std_mult * boll_std
    band_width = (boll_upper - boll_lower)
    band_width_safe = np.where(band_width > 1e-9, band_width, 1.0)
    pct_b = ((close - boll_lower) / band_width_safe).astype(np.float32)
    pct_b = np.clip(pct_b, -1.0, 2.0)
    bw_relative = np.where(boll_mid > 0, band_width / boll_mid, 0.0).astype(np.float32)
    # squeeze: current bw_relative at 20d minimum within last 60 days
    bw_min60 = _rolling_min_2d(np.where(bw_relative > 0, bw_relative, 1e6), 60)
    squeeze = np.where(
        bw_relative > 1e-9,
        (bw_relative <= bw_min60 * 1.05).astype(np.float32),  # within 5% of 60d min
        0.0,
    ).astype(np.float32)
    out["tech_boll_pct_b"] = pct_b
    out["tech_boll_band_width"] = bw_relative.astype(np.float32)
    out["tech_boll_squeeze"] = squeeze

    # ---- Volume ----
    vol_ma20 = _rolling_mean_2d(vol, cfg.vol_ma_period)
    vol_ma20_safe = np.where(vol_ma20 > 1.0, vol_ma20, 1.0)
    vol_ratio = (vol / vol_ma20_safe).astype(np.float32)
    vol_5d = _rolling_mean_2d(vol, 5)
    vol_decay_5d = (vol_5d / vol_ma20_safe).astype(np.float32)
    out["tech_vol_ratio"] = vol_ratio
    out["tech_vol_decay_5d"] = vol_decay_5d

    # ---- Amplitude (close-based) ----
    high_5d = _rolling_max_2d(close, 5)
    low_5d = _rolling_min_2d(close, 5)
    high_20d = _rolling_max_2d(close, 20)
    low_20d = _rolling_min_2d(close, 20)
    safe_low_5d = np.where(low_5d > 1e-9, low_5d, 1.0)
    safe_low_20d = np.where(low_20d > 1e-9, low_20d, 1.0)
    out["tech_amplitude_5d"] = (high_5d / safe_low_5d - 1.0).astype(np.float32)
    out["tech_amplitude_20d"] = (high_20d / safe_low_20d - 1.0).astype(np.float32)

    # ---- Cumulative main force (if mf_net_1d available) ----
    if mf_net_1d is not None:
        for w in cfg.cmf_periods:
            cmf = _rolling_sum_2d(mf_net_1d, w)
            out[f"cmf_{w}d"] = cmf
            # Normalize by 60d avg amount (close * vol)
            amount = (close * vol).astype(np.float32)
            amount_60d = _rolling_sum_2d(amount, 60)
            amount_60d_safe = np.where(amount_60d > 1.0, amount_60d, 1.0)
            out[f"cmf_{w}d_pct"] = (cmf / amount_60d_safe).astype(np.float32)
        # Days with positive net inflow in last 60 days
        pos_days = (mf_net_1d > 0).astype(np.float32)
        out["cmf_pos_days_60d"] = _rolling_sum_2d(pos_days, 60)

    # ---- Computed 涨停 count (replaces broken senti_zt_count_30d) ----
    is_zt = (pct_chg >= cfg.zt_threshold).astype(np.float32)
    out["zt_count_30d"] = _rolling_sum_2d(is_zt, 30)
    out["zt_count_60d"] = _rolling_sum_2d(is_zt, 60)
    is_dt = (pct_chg <= -cfg.zt_threshold).astype(np.float32)
    out["zt_dt_imbalance_60d"] = (
        _rolling_sum_2d(is_zt, 60) - _rolling_sum_2d(is_dt, 60)
    )

    return out


__all__ = [
    "TechnicalFactorsConfig",
    "compute_technical_factors",
]
