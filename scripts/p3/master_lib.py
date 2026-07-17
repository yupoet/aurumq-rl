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
# Pre-registered kill criteria (README §12.7)
# =============================================================================


def kill_criteria_verdict(
    base: dict[str, dict],
    treatment: dict[str, dict],
    ic_key: str = "spearman",
    primary_key: str = "primary_mean_top50_proximity_excess",
    required_win_frac: float = 2 / 3,
) -> dict:
    """Pre-registered KEEP/KILL verdict for a treatment vs its baseline.

    Both args map window name -> metric block (the dicts produced by
    ``p3.path1_eval.evaluate``). A window is a WIN when the treatment beats the
    base on ``ic_key`` AND is not worse on ``primary_key``. Verdict is KEEP iff
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
        wins += win
        detail[name] = {
            "win": win,
            "ic_base": b[ic_key],
            "ic_treatment": t[ic_key],
            "primary_base": b[primary_key],
            "primary_treatment": t[primary_key],
        }
    required = math.ceil(required_win_frac * len(windows))
    return {
        "verdict": "KEEP" if wins >= required else "KILL",
        "wins": wins,
        "required_wins": required,
        "n_windows": len(windows),
        "ic_key": ic_key,
        "primary_key": primary_key,
        "windows": detail,
    }
