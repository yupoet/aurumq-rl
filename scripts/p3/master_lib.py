"""Pure helpers for the MASTER-lite cross-sectional experiment (2026-07 course correction).

Shared by ``master_train.py`` (GPU) and ``master_ensemble_eval.py`` (CPU), and by the
pre-registered kill-criteria policy that now also governs the Kronos embedding track
(README §12.7). Only numpy / polars / scipy are imported here; importing this module
creates no directories, reads no files, and needs no torch — so the defect-prone logic
(sequence indexing, embargo split, rank blending, kill verdict) is unit-testable on
boxes without a GPU (same pattern as ``kronos_matrix_v13_lib``).
"""

from __future__ import annotations

import datetime as dt
import math
import warnings
from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy.stats import spearmanr

# Cross-sectional z-scores beyond this are clipped. Prevents single-name data wounds
# (README §7 Finding 6: most infs/outliers are upstream data trauma, not signal).
ZSCORE_CLIP = 5.0

# Std floor below which a (date, factor) cross-section is treated as constant
# (z-score forced to 0 instead of exploding).
STD_FLOOR = 1e-8


# =============================================================================
# Panel normalization
# =============================================================================


def cs_zscore_panel(x: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score a dense panel per (date, factor), NaN-aware.

    Args:
        x: float array of shape [D, N, F] (dates x stocks x factors). NaN marks
            "stock absent / factor missing on this date".

    Returns:
        float32 array of the same shape. Per (date, factor): subtract the
        cross-sectional nanmean over stocks, divide by nanstd (floored at
        STD_FLOOR), clip to +-ZSCORE_CLIP. NaN cells become 0.0 — i.e. exactly
        the cross-section mean, which is the neutral value for a model that
        consumes the z-scored panel.
    """
    if x.ndim != 3:
        raise ValueError(f"expected [D, N, F] panel, got shape {x.shape}")
    x = x.astype(np.float32, copy=True)
    # All-NaN cross-sections are an expected, handled case (delisted names,
    # pre-listing dates) — silence numpy's warning for them, keep the NaN result.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
        mean = np.nanmean(x, axis=1, keepdims=True)
        std = np.nanstd(x, axis=1, keepdims=True)
    # All-NaN cross-sections produce NaN mean/std; make them inert.
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=0.0)
    std = np.maximum(std, STD_FLOOR)
    z = (x - mean) / std
    np.clip(z, -ZSCORE_CLIP, ZSCORE_CLIP, out=z)
    return np.nan_to_num(z, nan=0.0)


# =============================================================================
# Sequence windows + embargoed split
# =============================================================================


def build_sequence_windows(n_dates: int, seq_len: int) -> np.ndarray:
    """Gather-index matrix for lookback sequences.

    Returns an int array of shape [n_valid, seq_len] where row i holds the date
    indices [anchor - seq_len + 1, ..., anchor] for anchor = seq_len - 1 + i.
    Anchors with insufficient history are excluded (no padding: a padded warmup
    window would mix a different effective lookback into the same batch shape).
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if n_dates < seq_len:
        return np.empty((0, seq_len), dtype=np.int64)
    anchors = np.arange(seq_len - 1, n_dates, dtype=np.int64)
    offsets = np.arange(-(seq_len - 1), 1, dtype=np.int64)
    return anchors[:, None] + offsets[None, :]


def train_val_split_with_embargo(
    dates: Sequence[dt.date],
    val_frac: float = 0.15,
    embargo_days: int = 30,
) -> tuple[list[dt.date], list[dt.date]]:
    """Split sorted unique trading dates into (train, val) with a leakage embargo.

    The last ``ceil(val_frac * n)`` dates become validation. The ``embargo_days``
    trading dates immediately before the validation block are dropped from train
    entirely — wave-style labels look forward ~25-30 trading days, so a train row
    just before the val block shares its label's event window with val dates
    (same rationale as ``kronos_matrix_v13_lib.EMBARGO_DAYS``).
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    if embargo_days < 0:
        raise ValueError(f"embargo_days must be >= 0, got {embargo_days}")
    uniq = sorted(set(dates))
    n = len(uniq)
    n_val = math.ceil(val_frac * n)
    n_train = n - n_val - embargo_days
    if n_train < 1:
        raise ValueError(
            f"not enough dates: n={n}, n_val={n_val}, embargo={embargo_days} "
            f"leaves {n_train} train dates"
        )
    return uniq[:n_train], uniq[n - n_val :]


# =============================================================================
# Metrics + rank blending
# =============================================================================


def daily_rank_ic(
    df: pl.DataFrame,
    score_col: str = "score",
    y_col: str = "actual_y",
    min_names_per_day: int = 3,
) -> float:
    """Mean over days of the per-date Spearman IC between score and realized y.

    Days with fewer than ``min_names_per_day`` non-null rows are skipped.
    Returns 0.0 when no day qualifies (matches path1_eval's degenerate-input
    convention).
    """
    df = df.drop_nulls([score_col, y_col])
    ics: list[float] = []
    for _, day in df.group_by("trade_date"):
        if len(day) < min_names_per_day:
            continue
        rho, _ = spearmanr(day[score_col].to_numpy(), day[y_col].to_numpy())
        if np.isfinite(rho):
            ics.append(float(rho))
    return float(np.mean(ics)) if ics else 0.0


def blend_rank_scores(
    preds: Sequence[pl.DataFrame],
    weights: Sequence[float],
) -> pl.DataFrame:
    """Per-date rank-percentile blend of several prediction frames.

    Each frame must have (trade_date, ts_code, score). Scores are converted to
    per-date rank percentiles (average ties, in (0, 1]) so that models with
    different score scales (LGBM proximity vs MASTER logits) blend fairly —
    same trick as ``hybrid_v3_rank_pct``. Frames are inner-joined: only
    (date, stock) pairs covered by ALL models are blended, because filling a
    missing model with a neutral 0.5 would silently dilute exactly the names
    where models disagree on coverage.

    Weights are renormalized to sum to 1.
    """
    if len(preds) != len(weights):
        raise ValueError(f"{len(preds)} frames but {len(weights)} weights")
    if len(preds) == 0:
        raise ValueError("need at least one prediction frame")
    w = np.asarray(weights, dtype=np.float64)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"weights must be non-negative with positive sum, got {list(weights)}")
    w = w / w.sum()

    blended: pl.DataFrame | None = None
    for i, (frame, wi) in enumerate(zip(preds, w, strict=True)):
        rank_pct = pl.col("score").rank(method="average").over("trade_date") / pl.col(
            "score"
        ).count().over("trade_date")
        ranked = frame.select(
            "trade_date",
            "ts_code",
            (rank_pct * wi).alias(f"_rp_{i}"),
        )
        blended = (
            ranked
            if blended is None
            else blended.join(ranked, on=["trade_date", "ts_code"], how="inner")
        )
    assert blended is not None
    rp_cols = [c for c in blended.columns if c.startswith("_rp_")]
    return blended.select("trade_date", "ts_code", pl.sum_horizontal(rp_cols).alias("score"))


# =============================================================================
# Cost model (README §12.7d — cost as a first-class citizen)
# =============================================================================

# Pre-registered cost spec v1 for daily top-K rebalancing on the A-share main
# board. Dividend tax note: the exact per-lot FIFO holding-period tiering
# (rqalpha `_pay_dividend_tax`) needs dividend-event data our bundles do not
# carry; v1 pre-registers the approximation "~2 % average yield x 20 % tier
# (holding <= 1 month, which daily top-K churn guarantees) = 40 bps/yr" applied
# pro-rata per trading day. Changing any number here after looking at results
# voids the verdict — bump to a COST_SPEC_V2 and rerun instead.
COST_SPEC_V1: dict = {
    "commission_bps_per_side": 2.5,
    "stamp_duty_bps_sell": 5.0,
    "slippage_bps_per_side": 10.0,
    "dividend_tax_drag_bps_annual": 40.0,
    "trading_days_per_year": 243,
}


def daily_topk_replacement(preds: pl.DataFrame, top_k: int) -> float:
    """Mean fraction of the top-k basket replaced between consecutive dates.

    ``preds`` needs (trade_date, ts_code, score). For each adjacent date pair
    the replaced fraction is ``1 - |A ∩ B| / min(|A|, |B|)`` (the min guards
    days that have fewer than ``top_k`` scored names). Ties on score break by
    ``ts_code`` so basket membership is deterministic across input row order
    and polars versions. Returns 0.0 with fewer than two dates
    (degenerate-input convention, same as ``daily_rank_ic``).
    """
    topk = (
        preds.sort(["trade_date", "score", "ts_code"], descending=[False, True, False])
        .group_by("trade_date", maintain_order=True)
        .head(top_k)
    )
    days = topk.partition_by("trade_date", maintain_order=True)
    if len(days) < 2:
        return 0.0
    fracs: list[float] = []
    prev = set(days[0]["ts_code"].to_list())
    for day in days[1:]:
        cur = set(day["ts_code"].to_list())
        denom = max(1, min(len(prev), len(cur)))
        fracs.append(1.0 - len(prev & cur) / denom)
        prev = cur
    return float(np.mean(fracs))


def cost_drag_daily(replaced_frac: float, spec: dict = COST_SPEC_V1) -> float:
    """Daily return drag (fraction) of running the top-k basket.

    A replaced slot pays a full round trip: sell the outgoing name
    (commission + stamp duty + slippage) and buy the incoming one
    (commission + slippage). The dividend-tax drag accrues pro-rata daily
    regardless of churn (holding at all incurs it under the v1 approximation).
    """
    round_trip = (
        2 * spec["commission_bps_per_side"]
        + spec["stamp_duty_bps_sell"]
        + 2 * spec["slippage_bps_per_side"]
    ) / 1e4
    dividend_daily = spec["dividend_tax_drag_bps_annual"] / 1e4 / spec["trading_days_per_year"]
    return replaced_frac * round_trip + dividend_daily


def cost_metric_block(
    preds: pl.DataFrame,
    realized: pl.DataFrame,
    market: pl.DataFrame,
    window: tuple,
    top_k: int = 50,
    spec: dict = COST_SPEC_V1,
) -> dict:
    """Turnover + cost-adjusted top-k excess block (README §12.7d).

    Args mirror ``path1_eval.evaluate``: ``realized`` has
    (trade_date, ts_code, pct_chg_t_plus_1) where the row at the anchor date
    already carries that anchor's T+1 return; ``market`` has
    (trade_date, eq_weight_pct_chg_t_plus_1). Per-date top-k mean T+1 excess
    is reported gross and net of ``cost_drag_daily``. Turnover is computed on
    the same joined frame the gross metric uses, so basket membership and
    return attribution agree.

    Registered v1 simplification: turnover is pairwise between consecutive
    dates, so the initial basket entry on day 1 of the window is not charged
    (understates drag slightly on short windows; bump the spec version if
    this is ever changed).
    """
    lo, hi = window
    joined = (
        preds.filter((pl.col("trade_date") >= lo) & (pl.col("trade_date") <= hi))
        .join(realized, on=["trade_date", "ts_code"], how="inner")
        .join(market, on="trade_date", how="inner")
        .with_columns(
            (pl.col("pct_chg_t_plus_1") - pl.col("eq_weight_pct_chg_t_plus_1")).alias("e1")
        )
        .drop_nulls(["score", "e1"])
    )
    n_dates = joined["trade_date"].n_unique() if len(joined) else 0
    if n_dates == 0:
        return {
            "n_dates": 0,
            "topk_daily_replaced_frac": 0.0,
            "annualized_two_sided_turnover": 0.0,
            "gross_mean_topk_excess_t1": 0.0,
            "net_mean_topk_excess_t1": 0.0,
            "cost_spec": dict(spec),
        }
    topk = (
        joined.sort(["trade_date", "score", "ts_code"], descending=[False, True, False])
        .group_by("trade_date", maintain_order=True)
        .head(top_k)
    )
    gross = float(topk.group_by("trade_date").agg(pl.col("e1").mean().alias("m"))["m"].mean())
    replaced = daily_topk_replacement(joined, top_k)
    return {
        "n_dates": n_dates,
        "topk_daily_replaced_frac": replaced,
        "annualized_two_sided_turnover": replaced * 2 * spec["trading_days_per_year"],
        "gross_mean_topk_excess_t1": gross,
        "net_mean_topk_excess_t1": gross - cost_drag_daily(replaced, spec),
        "cost_spec": dict(spec),
    }


# =============================================================================
# Pre-registered kill criteria (README §12.7)
# =============================================================================


def kill_criteria_verdict(
    base: dict[str, dict],
    treatment: dict[str, dict],
    ic_key: str = "spearman",
    primary_key: str = "primary_mean_top50_proximity_excess",
    required_win_frac: float = 2 / 3,
    cost_key: str | None = None,
) -> dict:
    """Pre-registered KEEP/KILL verdict for a treatment vs its baseline.

    Both args map window name -> metric block (the dicts produced by
    ``p3.path1_eval.evaluate``, optionally merged with ``cost_metric_block``).
    A window is a WIN when the treatment beats the base on ``ic_key`` AND is
    not worse on ``primary_key`` AND — when ``cost_key`` is given — is not
    worse on the cost-adjusted metric either (§12.7d: an IC win that loses
    net of trading costs is not a win). Verdict is KEEP iff
    wins >= ceil(required_win_frac * n_windows), else KILL.

    The criteria are registered BEFORE the experiment runs (this function is the
    registration). Post-hoc window cherry-picking, metric swaps or threshold
    tweaks void the verdict — rerun as a new pre-registered experiment instead.
    """
    windows = sorted(set(base) & set(treatment))
    if not windows:
        raise ValueError("no common eval windows between base and treatment")
    detail = {}
    wins = 0
    for name in windows:
        b, t = base[name], treatment[name]
        ic_win = t[ic_key] > b[ic_key]
        primary_ok = t[primary_key] >= b[primary_key]
        win = bool(ic_win and primary_ok)
        entry = {
            "ic_base": b[ic_key],
            "ic_treatment": t[ic_key],
            "primary_base": b[primary_key],
            "primary_treatment": t[primary_key],
        }
        if cost_key is not None:
            cost_ok = t[cost_key] >= b[cost_key]
            win = win and cost_ok
            entry["cost_base"] = b[cost_key]
            entry["cost_treatment"] = t[cost_key]
            entry["cost_ok"] = cost_ok
        entry["win"] = win
        wins += win
        detail[name] = entry
    required = math.ceil(required_win_frac * len(windows))
    return {
        "verdict": "KEEP" if wins >= required else "KILL",
        "wins": wins,
        "required_wins": required,
        "n_windows": len(windows),
        "ic_key": ic_key,
        "primary_key": primary_key,
        "cost_key": cost_key,
        "windows": detail,
    }
