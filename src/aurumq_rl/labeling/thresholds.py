"""Threshold search by target positive rate.

See SPEC §5.2. Goal: find τ_M such that mask(score >= τ) achieves a target
positive rate on the train_eff window.

We do NOT use quantile(score, 0.99) because score distributions can be
heavily zero-inflated; a quantile may be undefined or unstable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = ["search_threshold", "ThresholdResult"]


@dataclass(frozen=True)
class ThresholdResult:
    """Result of a threshold search."""
    threshold: float
    achieved_positive_rate: float
    target_positive_rate: float
    n_candidates: int


def search_threshold(
    quality_values: np.ndarray,
    n_decision_cells: int,
    *,
    target_pos_rate: float = 0.008,
    pos_rate_min: float = 0.005,
    pos_rate_max: float = 0.012,
    n_grid: int = 50,
) -> ThresholdResult:
    """Find τ minimizing |achieved_pos_rate - target_pos_rate|.

    Parameters
    ----------
    quality_values
        1-D array of event_quality scores from train_eff events.
    n_decision_cells
        Total number of (t, j) decision cells in train_eff (denominator).
        positive_rate = sum(quality >= τ) / n_decision_cells.
    target_pos_rate, pos_rate_min, pos_rate_max
        Target and acceptable band.
    n_grid
        How many τ candidates to try.

    Returns
    -------
    ThresholdResult with chosen τ.
    """
    quality_values = np.asarray(quality_values, dtype=np.float64)
    quality_values = quality_values[np.isfinite(quality_values)]
    if quality_values.size == 0:
        return ThresholdResult(
            threshold=float("inf"),
            achieved_positive_rate=0.0,
            target_positive_rate=target_pos_rate,
            n_candidates=0,
        )

    # Direct approach: target n_pos = target_pos_rate * n_decision_cells.
    # Sort scores descending; τ = sorted[target_n - 1] places exactly target_n
    # (or slightly more if ties) above τ.
    target_n = int(round(target_pos_rate * n_decision_cells))
    if target_n <= 0:
        target_n = 1
    if target_n >= quality_values.size:
        # Want more positives than we have events; pick lowest score
        tau = float(quality_values.min())
        n_pos = int((quality_values >= tau).sum())
        return ThresholdResult(
            threshold=tau,
            achieved_positive_rate=n_pos / max(n_decision_cells, 1),
            target_positive_rate=target_pos_rate,
            n_candidates=n_pos,
        )
    sorted_desc = np.sort(quality_values)[::-1]
    # We want τ such that (scores >= τ).count == target_n.
    # If sorted_desc[target_n - 1] = x, scores >= x are at least target_n.
    # If many ties at x, we may overshoot; binary-search refines.
    tau = float(sorted_desc[target_n - 1])
    n_pos = int((quality_values >= tau).sum())
    achieved = n_pos / max(n_decision_cells, 1)

    # If overshoot (too many ties at τ), bump τ up slightly to drop ties
    if achieved > pos_rate_max and target_n - 1 > 0:
        # try τ = next distinct value upward
        for k in range(target_n - 2, -1, -1):
            new_tau = float(sorted_desc[k])
            if new_tau > tau:
                tau = new_tau
                n_pos = int((quality_values >= tau).sum())
                achieved = n_pos / max(n_decision_cells, 1)
                if pos_rate_min <= achieved <= pos_rate_max:
                    break

    return ThresholdResult(
        threshold=tau,
        achieved_positive_rate=achieved,
        target_positive_rate=target_pos_rate,
        n_candidates=n_pos,
    )
