"""Pure helpers for matrix v13 (paradigm 3 Kronos) — no heavy deps, no side effects.

Extracted from ``kronos_matrix_v13.py`` so the defect fixes for C9 / M4 / M19 / M20
are unit-testable with synthetic frames on boxes without lightgbm, GPU checkpoints
or data parquets. Only numpy + pandas are imported here; importing this module
creates no directories and reads no files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

# M4: early-stopping validation embargo, in TRADING days (unique sorted trade_dates
# act as the trading calendar). Anchor labels look forward by the anchor horizon
# (T1/T3/T5 ≤ 5 trading days) plus the wave-event extension (~20-25 trading days),
# so a train row up to ~25-30 days before a val date can share its label's event
# window with that val date. 30 rounds that up.
EMBARGO_DAYS = 30


# =============================================================================
# C9 — which embeddings frame does a cell train on / score on
# =============================================================================

def select_cell_embeddings(cell_spec: Mapping[str, Any],
                           available_frames: Mapping[str, Any]) -> tuple[str, str]:
    """Pick the (train_key, eval_key) embedding frames for one v13 cell.

    Args:
        cell_spec: one entry of ``CELL_SPEC`` (uses ``is_base``; null cells train
            on the fine-tuned label frame like regular cells — their embedding
            columns are randomized afterwards).
        available_frames: mapping of loaded embedding frames, keyed by
            ``"label"`` (fine-tuned label pairs), ``"eval"`` (fine-tuned
            eval-window pairs), ``"base"`` (base-model label pairs) and
            ``"base_eval"`` (base-model eval-window pairs). Only presence of the
            keys matters here.

    Returns:
        ``(train_key, eval_key)`` into ``available_frames``. For fine-tuned and
        null cells the eval key is returned even when absent — callers preserve
        the existing "train but skip scoring" behavior. For ``is_base`` cells
        both frames are REQUIRED: base cells must never be scored on fine-tuned
        eval embeddings (that was defect C9, which invalidated the 3-way
        base-vs-finetuned control).

    Raises:
        FileNotFoundError: base cell without base-model embeddings — raised
            BEFORE any LGB training so hours of fitting are not wasted. The
            message names the phase-2 command that generates the parquet.
    """
    if not cell_spec.get("is_base"):
        return "label", "eval"

    if "base_eval" not in available_frames:
        raise FileNotFoundError(
            "base-model control cell requires base-model eval-window embeddings "
            "(embeddings_*_base_eval*.parquet), which are missing. Generate them "
            "with: python scripts/p3/kronos_matrix_v13.py --phase 2 --base-model "
            "--eval-window (add --smoke for smoke runs). Scoring *_BASE cells on "
            "fine-tuned eval embeddings would invalidate the 3-way control (C9)."
        )
    # A --base-model --eval-window phase-2 run writes ONE parquet containing both
    # the label pairs and the eval-window pairs, so it doubles as training source
    # when the label-only "_base" parquet was never extracted separately.
    train_key = "base" if "base" in available_frames else "base_eval"
    return train_key, "base_eval"


# =============================================================================
# M4 — date-sorted, embargoed early-stopping split
# =============================================================================

def date_embargo_split(df: pd.DataFrame, val_frac: float = 0.10,
                       embargo_days: int = EMBARGO_DAYS,
                       date_col: str = "trade_date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by DATE (not row position) with a trading-day embargo gap.

    Replaces the defective ``train.tail(0.1)`` split on an unsorted frame (M4):
    row-position tails could span the same dates as the fit rows (cross-sectional
    twin leakage) and, even date-ordered, share forward-looking label event
    windows across the boundary.

    Args:
        df: training frame; row order is irrelevant (output is date-sorted).
        val_frac: fraction of UNIQUE dates (last ones) assigned to validation.
        embargo_days: number of trading days (unique sorted ``date_col`` values
            act as the trading calendar) dropped from the end of the fit block,
            immediately before the first validation date.
        date_col: date column name.

    Returns:
        ``(train_fit, val)`` — date-sorted, index-reset. ``train_fit`` may be
        empty when the embargo swallows every pre-validation date; callers
        should skip such cells.

    Raises:
        ValueError: ``val_frac`` outside (0, 1), or too few unique dates to
            carve out a non-empty validation block strictly after training.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    if embargo_days < 0:
        raise ValueError(f"embargo_days must be >= 0, got {embargo_days}")

    dates = np.array(sorted(pd.unique(df[date_col])))
    n_val_dates = max(1, int(round(len(dates) * val_frac)))
    if n_val_dates >= len(dates):
        raise ValueError(
            f"cannot split {len(dates)} unique dates into train + {n_val_dates} "
            f"val dates (val_frac={val_frac}); need more history"
        )
    val_dates = dates[-n_val_dates:]
    cut = len(dates) - n_val_dates  # calendar position of the first val date
    train_dates = dates[: max(0, cut - embargo_days)]

    df_sorted = df.sort_values(date_col, kind="mergesort")
    val = df_sorted[df_sorted[date_col].isin(val_dates)].reset_index(drop=True)
    train_fit = df_sorted[df_sorted[date_col].isin(train_dates)].reset_index(drop=True)
    return train_fit, val


# =============================================================================
# M19 — smoke-suffixed run artifacts + retryable "skipped" checkpoints
# =============================================================================

def v13_artifact_path(kind: str, smoke: bool, out_dir: Path, results_dir: Path,
                      cid: str | None = None) -> Path:
    """Single source of truth for v13 run-artifact paths.

    Smoke runs get a ``_smoke`` suffix on EVERY artifact (phase-3 checkpoint,
    per-cell pred parquets, phase-4 results json) so a smoke run can never
    poison a full run's resume/eval, and vice versa (M19).

    Args:
        kind: ``"checkpoint"`` | ``"pred"`` | ``"results"``.
        smoke: whether this is a ``--smoke`` run.
        out_dir: matrix-v13 output dir (pred parquets live here).
        results_dir: parent outputs dir (checkpoint + results json live here).
        cid: cell id, required for ``kind="pred"``.
    """
    sfx = "_smoke" if smoke else ""
    if kind == "checkpoint":
        return results_dir / f"matrix_v13_phase3_checkpoint{sfx}.json"
    if kind == "results":
        return results_dir / f"matrix_v13_results{sfx}.json"
    if kind == "pred":
        if cid is None:
            raise ValueError("kind='pred' requires cid")
        return out_dir / f"pred_{cid}{sfx}.parquet"
    raise ValueError(f"unknown artifact kind: {kind!r}")


def is_cell_done(entry: Mapping[str, Any] | None) -> bool:
    """Checkpoint-resume predicate: only genuinely trained cells are done.

    ``{"skipped": ...}`` entries (e.g. missing embeddings, sparse labels) stay
    in the checkpoint for observability but are retried on the next run (M19 —
    previously a cell checkpointed as skipped was skipped forever, even after
    its embeddings appeared).
    """
    if not entry:
        return False
    return "skipped" not in entry


# =============================================================================
# M20 — the REAL lookback-window slice, unit-testable
# =============================================================================

def build_lookback_window(values: np.ndarray, dates: np.ndarray, idx: int,
                          seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Strict D-1 lookback window used by the phase-2 extraction path.

    ``embedding(D) = encoder(OHLCV[D - seq_len : D - 1])``: the window is rows
    ``[idx - seq_len, idx)`` — exactly ``seq_len`` rows ending at ``idx - 1``,
    NEVER including the anchor row ``idx``.

    ``idx == seq_len`` is the first eligible anchor (window = rows
    ``[0, seq_len)``). The old eligibility check ``idx - 1 < seq_len`` demanded
    one extra bar and silently dropped each stock's first eligible anchor date
    (M20 boundary off-by-one).

    Args:
        values: per-stock feature rows, shape ``[n, k]`` (date-ascending).
        dates: matching per-stock dates, shape ``[n]``.
        idx: positional index of the anchor date within ``values``/``dates``.
        seq_len: window length in rows.

    Returns:
        ``(window, window_dates)`` with ``window.shape[0] == seq_len`` and
        ``max(window_dates) < dates[idx]``.

    Raises:
        ValueError: non-positive ``seq_len``, insufficient history
            (``idx < seq_len``), or ``idx`` beyond the series.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if idx < seq_len:
        raise ValueError(
            f"insufficient history: anchor idx {idx} needs {seq_len} prior rows "
            f"(idx >= seq_len required)"
        )
    if idx > len(values) or len(values) != len(dates):
        raise ValueError(
            f"anchor idx {idx} out of range for series of length "
            f"{len(values)} (dates: {len(dates)})"
        )
    sl = slice(idx - seq_len, idx)
    return values[sl], dates[sl]


def d1_leakage_selfcheck(seq_len: int) -> None:
    """Runtime D-1 leakage guard asserting on the REAL extraction slice.

    Replaces the vacuous hand-built two-window comparison (M20): builds a
    strictly increasing synthetic series where any off-by-one is detectable and
    asserts, via :func:`build_lookback_window` (the function the extraction
    loop actually uses), the strict D-1 property and exact window bounds.

    Raises:
        AssertionError: if the extraction slice ever includes the anchor day,
            returns the wrong length, or rejects the first eligible anchor.
    """
    n = seq_len + 10
    dates = pd.bdate_range("2024-01-02", periods=n).values
    values = np.arange(n, dtype=np.float64).reshape(-1, 1)  # strictly increasing

    for idx in (seq_len, seq_len + 5, n - 1):  # first eligible, interior, last
        window, window_dates = build_lookback_window(values, dates, idx, seq_len)
        assert window.shape[0] == seq_len, (
            f"window length {window.shape[0]} != seq_len {seq_len} at idx {idx}"
        )
        assert window_dates.max() < dates[idx], (
            f"D-1 leakage: window includes anchor day at idx {idx}"
        )
        assert window_dates[-1] == dates[idx - 1], (
            f"window must end exactly at D-1 (idx {idx})"
        )
        assert float(window[-1, 0]) == float(idx - 1), (
            f"window values misaligned with dates at idx {idx}"
        )

    # idx == seq_len - 1 has only seq_len - 1 prior rows → must be rejected
    try:
        build_lookback_window(values, dates, seq_len - 1, seq_len)
    except ValueError:
        pass
    else:
        raise AssertionError("idx == seq_len - 1 must raise (insufficient history)")
