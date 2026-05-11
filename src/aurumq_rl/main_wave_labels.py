"""Phase 22 main-wave labels: precompute the eval/reward labels that capture
"主升浪前夕" (rally-eve) buying decisions.

Pure numpy module, no torch / SB3 dependency. Consumed by both the new
``_eval_main_wave.py`` CLI and (in a later phase) by the env's reward
function.

Field availability constraints
------------------------------
The current parquet schema has ``close``, ``pct_chg``, ``vol`` per
(date, stock), but does NOT have ``open`` or ``high`` / ``low``. Two
necessary approximations follow from this:

1.  **entry_price = close[t+1]** instead of ``open[t+1]``. The decision is
    assumed to be made after close on day t; the position is opened at
    next day's close. ``picks.jsonl`` records ``entry_price_proxy="next_close"``.
2.  **path-of-prices uses daily close only.** Max cumulative return and
    max drawdown during the hold window therefore underestimate the true
    intraday extremes, but the relative ordering across stocks remains
    meaningful. ``picks.jsonl`` records ``path_uses_close_only=True``.

Amount is derived as ``close * vol`` (元). The 20-day rolling mean is the
liquidity filter: ``amount_ma20 > amount_ma_min`` (default 1e8).

Death cross is the **event-form** crossing:
``MA5[t-1] >= MA10[t-1] AND MA5[t] < MA10[t]``. State-form (``MA5 < MA10``
on any single day) is also exposed via ``below_ma_state`` for the
"don't buy if already in death cross" entry filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MainWaveConfig:
    """All knobs for label / score computation. Fixed default values per the
    user spec; override in ``_eval_main_wave.py`` CLI as needed.
    """

    hold_window: int = 5  # max hold days (entry_day .. entry_day+H-1)
    vol_window: int = 20  # past-vol window for threshold
    sigma_multiplier: float = 2.0  # threshold = sigma_mult * past_vol * sqrt(H)
    absolute_threshold: float = 0.06  # threshold floor
    max_adverse_limit: float = 0.05  # 5% — max allowed adverse excursion for hit
    amount_ma_window: int = 20  # rolling window for amount MA
    amount_ma_min: float = 1e8  # 20d avg amount > 1 亿 to be tradeable
    ma5_window: int = 5
    ma10_window: int = 10

    # Score weights (sum of positive weights = 1.0; drawdown is penalty)
    w_height: float = 0.35
    w_duration: float = 0.20
    w_smoothness: float = 0.20
    w_win: float = 0.15
    w_entry_timing: float = 0.10
    w_drawdown_penalty: float = 0.20

    # Eval-score weights (per Phase 22 spec, used to rank checkpoints)
    es_w_hit: float = 0.30
    es_w_avg_score: float = 0.25
    es_w_avg_hold_return: float = 0.20
    es_w_basic_win: float = 0.15
    es_w_payoff: float = 0.10
    es_w_drawdown: float = 0.20  # subtracted


# ---------------------------------------------------------------------------
# Public container
# ---------------------------------------------------------------------------


@dataclass
class MainWaveLabels:
    """Precomputed per-(date, stock) labels and masks. All arrays have shape
    (n_dates, n_stocks); all returns are decimals (0.05 == +5%)."""

    # Liquidity / eligibility
    amount_ma20: np.ndarray  # rolling mean of (close * vol)
    liquid_mask: np.ndarray  # bool, amount_ma20 > min
    below_ma_state: np.ndarray  # bool, MA5 < MA10 on day t
    death_cross_event: np.ndarray  # bool, MA5 first crosses below MA10 at t
    entry_eligible_mask: np.ndarray  # bool, OK to buy at entry_day = t+1

    # Path metrics over the hold window starting at t+1
    entry_day_idx: np.ndarray  # int, t+1 (-1 if no valid entry)
    exit_day_idx: np.ndarray  # int, in [entry_day, entry_day+H]
    holding_days: np.ndarray  # int, exit_day_idx - entry_day_idx
    entry_price: np.ndarray  # close[entry_day_idx]
    exit_price: np.ndarray
    hold_return: np.ndarray  # exit/entry - 1
    final_return: np.ndarray  # alias of hold_return for clarity
    max_cum_return_5d: np.ndarray  # max of (close[entry+i]/entry - 1) over hold path
    peak_day_offset: np.ndarray  # int, days from entry to peak
    max_drawdown_during_hold: np.ndarray  # min of running drawdown from running peak
    max_adverse_excursion: np.ndarray  # min of (close[entry+i]/entry - 1) (≤ 0)
    rise_duration: np.ndarray  # peak_day_offset (days from entry to peak)

    # Threshold and event labels
    past_vol: np.ndarray  # std of pct_chg over vol_window (per (t, j))
    threshold: np.ndarray  # max(sigma * past_vol * sqrt(H), abs_threshold)
    hit_main_wave: np.ndarray  # bool

    # Score (0..100)
    height_score: np.ndarray
    duration_score: np.ndarray
    smoothness_score: np.ndarray
    win_score: np.ndarray
    entry_timing_score: np.ndarray
    drawdown_penalty: np.ndarray
    main_wave_score: np.ndarray  # final composite, can be negative

    # Validity (False if path could not be evaluated, e.g. ran past panel end)
    label_valid_mask: np.ndarray  # bool

    config: MainWaveConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rolling_mean_2d(a: np.ndarray, w: int) -> np.ndarray:
    """(T, S) -> (T, S) rolling mean down the time axis. Edge: forward-fill
    over partial windows (use the available observations)."""
    T, S = a.shape
    out = np.zeros_like(a, dtype=np.float64)
    cs = np.cumsum(a, axis=0, dtype=np.float64)
    for t in range(T):
        lo = max(0, t - w + 1)
        seg_sum = cs[t] - (cs[lo - 1] if lo > 0 else 0.0)
        out[t] = seg_sum / float(t - lo + 1)
    return out.astype(np.float32)


def _rolling_std_2d(a: np.ndarray, w: int) -> np.ndarray:
    """(T, S) -> (T, S) rolling sample std (ddof=0) along time axis. Edge:
    partial windows; std=0 if only one obs."""
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


def _compute_ma_arrays(close: np.ndarray, ma5_w: int, ma10_w: int):
    """Return (ma5, ma10, below_ma_state, death_cross_event) all (T, S)."""
    ma5 = _rolling_mean_2d(close, ma5_w)
    ma10 = _rolling_mean_2d(close, ma10_w)
    below_state = ma5 < ma10
    # Event: ma5 was >= ma10 yesterday and is < ma10 today.
    prev_above_or_equal = np.zeros_like(below_state, dtype=np.bool_)
    prev_above_or_equal[1:] = ~below_state[:-1]  # yesterday MA5 >= MA10
    death_cross_event = below_state & prev_above_or_equal
    return ma5, ma10, below_state, death_cross_event


# ---------------------------------------------------------------------------
# Main label-builder
# ---------------------------------------------------------------------------


def compute_main_wave_labels(
    close: np.ndarray,
    pct_chg: np.ndarray,
    vol: np.ndarray,
    valid_mask_basic: np.ndarray,  # ~is_st & ~is_suspended & days_since_ipo>=60
    cfg: MainWaveConfig | None = None,
) -> MainWaveLabels:
    """Compute every label needed for main-wave eval.

    Parameters
    ----------
    close : (T, S) float
        Daily close, in 元.
    pct_chg : (T, S) float
        Daily pct change, decimal (0.10 == +10%).
    vol : (T, S) float
        Daily volume, in number of shares.
    valid_mask_basic : (T, S) bool
        The env's existing valid mask — ~is_st & ~is_suspended &
        days_since_ipo>=60. We AND in the liquidity filter on top.
    cfg : MainWaveConfig
        See dataclass. Default mirrors the user's Phase 22 spec.

    Returns
    -------
    MainWaveLabels
    """
    if cfg is None:
        cfg = MainWaveConfig()

    T, S = close.shape
    assert pct_chg.shape == (T, S)
    assert vol.shape == (T, S)
    assert valid_mask_basic.shape == (T, S)

    # ---------- Liquidity ----------
    amount = close * vol  # (T, S), 元
    amount_ma20 = _rolling_mean_2d(amount, cfg.amount_ma_window)
    liquid_mask = amount_ma20 > cfg.amount_ma_min

    # ---------- MA5 / MA10 ----------
    ma5, ma10, below_state, death_cross_event = _compute_ma_arrays(
        close, cfg.ma5_window, cfg.ma10_window
    )

    # ---------- Past vol ----------
    past_vol = _rolling_std_2d(pct_chg, cfg.vol_window)

    # ---------- Entry eligibility ----------
    # Buy at entry_day = t+1. Entry is allowed if:
    #   - basic valid AND liquid at t (decision day)
    #   - not currently in below-MA state at t  (per user spec: don't buy
    #     if MA5 already < MA10)
    #   - t+1 < T (can compute a path)
    decision_eligible = valid_mask_basic & liquid_mask & (~below_state)
    entry_eligible_mask = np.zeros_like(decision_eligible, dtype=np.bool_)
    entry_eligible_mask[: T - 1] = decision_eligible[: T - 1]  # need t+1 to exist

    # ---------- Path metrics ----------
    H = cfg.hold_window

    entry_day_idx = np.full((T, S), -1, dtype=np.int32)
    exit_day_idx = np.full((T, S), -1, dtype=np.int32)
    holding_days = np.zeros((T, S), dtype=np.int32)
    entry_price = np.zeros((T, S), dtype=np.float32)
    exit_price = np.zeros((T, S), dtype=np.float32)
    hold_return = np.zeros((T, S), dtype=np.float32)
    max_cum_return_5d = np.zeros((T, S), dtype=np.float32)
    peak_day_offset = np.zeros((T, S), dtype=np.int32)
    max_drawdown_during_hold = np.zeros((T, S), dtype=np.float32)
    max_adverse_excursion = np.zeros((T, S), dtype=np.float32)
    label_valid = np.zeros((T, S), dtype=np.bool_)

    for t in range(T - 1):
        entry = t + 1
        path_max_t = min(entry + H, T)  # exclusive upper bound on path indices
        if path_max_t - entry < 1:
            continue
        # For each stock, traverse path days [entry .. entry + H - 1].
        # Exit_day = first death-cross-event in the path, OR entry+H-1 if none.
        eligible_now = entry_eligible_mask[t]  # (S,)
        if not eligible_now.any():
            continue

        # close at entry; same denominator for all path days.
        ep = close[entry]  # (S,)
        # Avoid division by zero / negative; treat ep==0 as ineligible.
        ep_safe = np.where(eligible_now & (ep > 0), ep, np.nan)

        # Build the path for the H days starting at entry. We iterate over the
        # path days and update running peak / running min for each stock.
        # Initial state: at day = entry (offset 0), cum_ret = 0.
        running_max = np.zeros(S, dtype=np.float32)
        running_max_offset = np.zeros(S, dtype=np.int32)
        running_min = np.zeros(S, dtype=np.float32)
        running_min_during_path = np.zeros(S, dtype=np.float32)
        # exit offset (0..H-1); H means "no death cross within window"
        exit_offset = np.full(S, H - 1, dtype=np.int32)
        exit_found = np.zeros(S, dtype=np.bool_)

        for offset in range(0, min(H, T - entry)):
            day = entry + offset
            cur_close = close[day]
            cur_ret = (cur_close / ep_safe) - 1.0  # (S,) NaN where ineligible
            # update running max
            new_max_mask = cur_ret > running_max
            running_max = np.where(new_max_mask, cur_ret, running_max)
            running_max_offset = np.where(
                new_max_mask, np.full(S, offset, dtype=np.int32), running_max_offset
            )
            # update min adverse
            running_min = np.minimum(running_min, cur_ret)
            # drawdown from running max
            running_min_during_path = np.minimum(running_min_during_path, cur_ret - running_max)
            # Death cross at this day → exit AT this day if not already exited.
            dc_today = death_cross_event[day]
            new_exit = dc_today & (~exit_found) & eligible_now
            exit_offset = np.where(new_exit, offset, exit_offset)
            exit_found = exit_found | new_exit

        # Build outputs for stocks with eligible_now.
        # Days available may be < H near the end of the panel; clip exit.
        available_days = min(H, T - entry)
        clipped_exit_offset = np.minimum(exit_offset, available_days - 1)
        e_idx = entry
        x_idx = entry + clipped_exit_offset  # (S,)

        # Mask of (t, j) cells whose path has at least 1 day (always 1 here).
        ok = eligible_now & (ep > 0)

        # Need 5-day path (or path-end) for "label valid" — i.e. exit_offset
        # equals what we computed and we did not run off panel for the death-
        # cross detection. Mark valid only when we had at least available_days
        # >= 1 (which is enforced) and exited cleanly.
        label_valid[t] = ok

        entry_day_idx[t, ok] = e_idx
        exit_day_idx[t, ok] = x_idx[ok]
        holding_days[t, ok] = clipped_exit_offset[ok] + 1  # 1..H
        entry_price[t, ok] = ep[ok]
        # exit price = close[x_idx[j], j]
        # gather along axis 0
        exit_price_row = np.take_along_axis(close, x_idx[None, :], axis=0)[0]  # (S,)
        exit_price[t, ok] = exit_price_row[ok]
        hold_return[t, ok] = (exit_price_row[ok] / ep[ok]) - 1.0

        max_cum_return_5d[t, ok] = running_max[ok]
        peak_day_offset[t, ok] = running_max_offset[ok]
        max_drawdown_during_hold[t, ok] = running_min_during_path[ok]
        max_adverse_excursion[t, ok] = running_min[ok]

    # Corrupted-path filter: in the parquet, days with no data become close=0
    # (data_loader fills missing rows with zeros). A path day with close=0
    # gives cur_ret = -1.0, which stays in max_adverse_excursion and
    # hold_return. We exclude any (t, j) whose exit_price <= 0 OR whose
    # hold_return is implausibly large-negative (< -50% in 5 days).
    corrupted = (exit_price <= 0.0) | (hold_return < -0.5)
    label_valid &= ~corrupted

    final_return = hold_return  # alias
    rise_duration = peak_day_offset.copy()  # days from entry to peak (0..H-1)

    # ---------- Threshold & hit ----------
    # threshold = max(sigma_mult * past_vol[t,j] * sqrt(H), abs_threshold)
    threshold = np.maximum(
        cfg.sigma_multiplier * past_vol * float(np.sqrt(cfg.hold_window)),
        cfg.absolute_threshold,
    ).astype(np.float32)

    hit_main_wave = (
        (max_cum_return_5d >= threshold)
        & (final_return > 0)
        & (max_adverse_excursion > -cfg.max_adverse_limit)
        & label_valid
    )

    # ---------- Sub-scores ----------
    eps = 1e-6
    height_score = np.clip(max_cum_return_5d / np.maximum(threshold, eps), 0.0, 2.0) * 0.5
    duration_score = np.clip(rise_duration.astype(np.float32) / float(H), 0.0, 1.0)
    smoothness_score = 1.0 - np.clip(
        np.abs(max_drawdown_during_hold) / np.maximum(max_cum_return_5d, eps),
        0.0,
        1.0,
    )
    win_score = (final_return > 0).astype(np.float32)
    entry_timing_score = 0.5 * np.clip(
        rise_duration.astype(np.float32) / float(H), 0.0, 1.0
    ) + 0.5 * (1.0 - np.clip(np.abs(max_adverse_excursion) / cfg.max_adverse_limit, 0.0, 1.0))
    drawdown_penalty = np.clip(np.abs(max_adverse_excursion) / cfg.max_adverse_limit, 0.0, 1.0)

    main_wave_score = (
        100.0
        * (
            cfg.w_height * height_score
            + cfg.w_duration * duration_score
            + cfg.w_smoothness * smoothness_score
            + cfg.w_win * win_score
            + cfg.w_entry_timing * entry_timing_score
        )
        - 100.0 * cfg.w_drawdown_penalty * drawdown_penalty
    )

    # Cells that are not label_valid get 0 score and never count as hits.
    main_wave_score = np.where(label_valid, main_wave_score, 0.0).astype(np.float32)

    return MainWaveLabels(
        amount_ma20=amount_ma20,
        liquid_mask=liquid_mask,
        below_ma_state=below_state,
        death_cross_event=death_cross_event,
        entry_eligible_mask=entry_eligible_mask,
        entry_day_idx=entry_day_idx,
        exit_day_idx=exit_day_idx,
        holding_days=holding_days,
        entry_price=entry_price,
        exit_price=exit_price,
        hold_return=hold_return,
        final_return=final_return,
        max_cum_return_5d=max_cum_return_5d,
        peak_day_offset=peak_day_offset,
        max_drawdown_during_hold=max_drawdown_during_hold,
        max_adverse_excursion=max_adverse_excursion,
        rise_duration=rise_duration,
        past_vol=past_vol,
        threshold=threshold,
        hit_main_wave=hit_main_wave,
        height_score=height_score.astype(np.float32),
        duration_score=duration_score.astype(np.float32),
        smoothness_score=smoothness_score.astype(np.float32),
        win_score=win_score,
        entry_timing_score=entry_timing_score.astype(np.float32),
        drawdown_penalty=drawdown_penalty.astype(np.float32),
        main_wave_score=main_wave_score,
        label_valid_mask=label_valid,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Eval-score aggregation across (date, top-K) selections
# ---------------------------------------------------------------------------


def aggregate_eval_metrics(
    labels: MainWaveLabels,
    selected_idx_per_date: list[np.ndarray],
    cfg: MainWaveConfig | None = None,
) -> dict[str, float]:
    """Given per-date selected stock indices, compute the eval metrics.

    Parameters
    ----------
    labels : MainWaveLabels
        From ``compute_main_wave_labels``.
    selected_idx_per_date : list of np.ndarray
        Length T_eval. Each element is the (variable-length, ≤ top_k) array of
        selected stock indices for that date. Empty arrays are allowed (no
        eligible stocks that day).

    Returns
    -------
    Dict of scalar metrics + composite eval_score.
    """
    cfg = cfg or labels.config

    holds: list[float] = []
    hits: list[bool] = []
    wins: list[bool] = []
    scores: list[float] = []
    drawdowns: list[float] = []
    max_cums: list[float] = []
    holding_days_all: list[int] = []
    daily_precision_hits: list[bool] = []

    n_dates_with_picks = 0

    for t, idx in enumerate(selected_idx_per_date):
        if idx is None or len(idx) == 0:
            continue
        n_dates_with_picks += 1
        # Filter to label-valid only (path was evaluable)
        valid = labels.label_valid_mask[t, idx]
        if not valid.any():
            continue
        idx_v = idx[valid]
        holds.extend(labels.hold_return[t, idx_v].tolist())
        hits.extend(labels.hit_main_wave[t, idx_v].tolist())
        wins.extend((labels.final_return[t, idx_v] > 0).tolist())
        scores.extend(labels.main_wave_score[t, idx_v].tolist())
        drawdowns.extend(np.abs(labels.max_drawdown_during_hold[t, idx_v]).tolist())
        max_cums.extend(labels.max_cum_return_5d[t, idx_v].tolist())
        holding_days_all.extend(labels.holding_days[t, idx_v].tolist())
        daily_precision_hits.append(bool(labels.hit_main_wave[t, idx_v].any()))

    if not holds:
        return {
            "n_picks": 0,
            "n_dates_with_picks": n_dates_with_picks,
            "basic_win_rate": 0.0,
            "main_wave_hit_rate": 0.0,
            "avg_hold_return": 0.0,
            "median_hold_return": 0.0,
            "avg_max_cum_return_5d": 0.0,
            "avg_main_wave_score": 0.0,
            "avg_max_drawdown": 0.0,
            "payoff_ratio": 0.0,
            "avg_holding_days": 0.0,
            "top_k_daily_precision": 0.0,
            "eval_score": 0.0,
        }

    holds_a = np.asarray(holds, dtype=np.float64)
    hits_a = np.asarray(hits, dtype=np.bool_)
    wins_a = np.asarray(wins, dtype=np.bool_)
    pos = holds_a[holds_a > 0]
    neg = holds_a[holds_a < 0]
    payoff = (
        float(pos.mean()) / max(float(np.abs(neg).mean()), 1e-9)
        if len(pos) > 0 and len(neg) > 0
        else (float(pos.mean()) if len(pos) > 0 else 0.0)
    )

    avg_hold_return = float(holds_a.mean())
    avg_main_wave_score = float(np.mean(scores)) if scores else 0.0
    avg_max_dd = float(np.mean(drawdowns)) if drawdowns else 0.0
    basic_win_rate = float(wins_a.mean())
    main_wave_hit_rate = float(hits_a.mean())

    eval_score = (
        cfg.es_w_hit * main_wave_hit_rate
        + cfg.es_w_avg_score * (avg_main_wave_score / 100.0)
        + cfg.es_w_avg_hold_return * float(np.tanh(avg_hold_return / 0.05))
        + cfg.es_w_basic_win * basic_win_rate
        + cfg.es_w_payoff * float(np.tanh(payoff - 1.0))
        - cfg.es_w_drawdown * float(np.clip(avg_max_dd / 0.05, 0.0, 1.0))
    )

    return {
        "n_picks": int(len(holds_a)),
        "n_dates_with_picks": int(n_dates_with_picks),
        "basic_win_rate": basic_win_rate,
        "main_wave_hit_rate": main_wave_hit_rate,
        "avg_hold_return": avg_hold_return,
        "median_hold_return": float(np.median(holds_a)),
        "avg_max_cum_return_5d": float(np.mean(max_cums)),
        "avg_main_wave_score": avg_main_wave_score,
        "avg_max_drawdown": avg_max_dd,
        "payoff_ratio": payoff,
        "avg_holding_days": float(np.mean(holding_days_all)) if holding_days_all else 0.0,
        "top_k_daily_precision": (
            float(np.mean(daily_precision_hits)) if daily_precision_hits else 0.0
        ),
        "eval_score": eval_score,
    }


__all__ = [
    "MainWaveConfig",
    "MainWaveLabels",
    "compute_main_wave_labels",
    "aggregate_eval_metrics",
]
