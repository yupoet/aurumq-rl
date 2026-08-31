"""Event sampling and sample-weighting utilities (López de Prado, 2018).

These are additive, opt-in utilities layered on top of the existing labeling
methods in this package. Nothing here changes any existing label emitter's
default output — see `triple_barrier.detect_events_triple_barrier`'s
`event_idx` parameter for the opt-in wiring point (ch. 2.5.2), and
`v2_excess_adaptive` / training code for where `label_concurrency` /
`average_uniqueness` sample weights (ch. 4) can be consumed downstream.

Part 1 — CUSUM filter
----------------------
`cusum_filter` implements the symmetric CUSUM event-sampling filter: it
accumulates positive/negative run-sums of consecutive diffs and emits an
event (resetting the run-sum) whenever either crosses `+threshold` /
`-threshold`. This concentrates event seeds at genuine level shifts instead
of sampling every bar, which is what `triple_barrier`'s default (all-t)
seeding does.

Part 2 — concurrency / average uniqueness
------------------------------------------
`label_concurrency` counts, for each bar, how many labels' outcome windows
`[t_start, t_end]` are "live" at that bar. `average_uniqueness` turns that
into a per-label weight (`sample_weight`) equal to the mean of `1/concurrency`
over the label's own window — labels whose outcome windows heavily overlap
other labels get down-weighted.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cusum_filter", "label_concurrency", "average_uniqueness"]


def cusum_filter(series: np.ndarray, threshold: float) -> np.ndarray:
    """Symmetric CUSUM event filter (LdP, *Advances in Financial ML*, ch. 2.5.2).

    Accumulates positive and negative run-sums of consecutive first differences
    of `series`. Whenever either run-sum crosses `+threshold` / `-threshold`,
    an event is emitted at that index and both run-sums reset to zero.

    Parameters
    ----------
    series
        1-D array of a level series (e.g. price or cumulative log-return).
    threshold
        Symmetric CUSUM trigger level (same units as `series`); must be > 0.

    Returns
    -------
    np.ndarray
        Sorted, strictly increasing int64 array of event indices into `series`
        (index `i+1` for a diff at position `i`, i.e. the index of the
        observation that caused the crossing).
    """
    if threshold <= 0:
        raise ValueError(f"cusum_filter threshold must be > 0, got {threshold}")
    series = np.asarray(series, dtype=np.float64)
    diffs = np.diff(series)
    events: list[int] = []
    s_pos = 0.0
    s_neg = 0.0
    for i, d in enumerate(diffs):
        if not np.isfinite(d):
            continue
        s_pos = max(0.0, s_pos + d)
        s_neg = min(0.0, s_neg + d)
        if s_neg < -threshold:
            s_neg = 0.0
            events.append(i + 1)
        elif s_pos > threshold:
            s_pos = 0.0
            events.append(i + 1)
    return np.array(events, dtype=np.int64)


def label_concurrency(t_starts: np.ndarray, t_ends: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Count how many labels' outcome windows cover each bar in `index`.

    Parameters
    ----------
    t_starts, t_ends
        1-D integer arrays (same length), the closed `[t_start_i, t_end_i]`
        outcome window of each label, in the same integer bar-index space as
        `index`.
    index
        1-D sorted-ascending integer array of bar positions to evaluate
        concurrency at (typically all bars in the sample, e.g. `0..T-1`).

    Returns
    -------
    np.ndarray
        int64 array, same length as `index`: `concurrency[k]` = number of
        labels whose `[t_start, t_end]` window contains `index[k]`.
    """
    t_starts = np.asarray(t_starts, dtype=np.int64)
    t_ends = np.asarray(t_ends, dtype=np.int64)
    index = np.asarray(index, dtype=np.int64)
    if t_starts.shape != t_ends.shape:
        raise ValueError("t_starts and t_ends must have the same shape")
    if index.size == 0:
        return np.zeros(0, dtype=np.int64)

    lo = int(index.min())
    hi = int(index.max())
    # Sweep-line diff array over the contiguous bar range [lo, hi].
    diff = np.zeros(hi - lo + 2, dtype=np.int64)
    for s, e in zip(t_starts, t_ends, strict=True):
        s_c = max(int(s), lo)
        e_c = min(int(e), hi)
        if s_c > e_c:
            continue
        diff[s_c - lo] += 1
        diff[e_c - lo + 1] -= 1
    counts_full = np.cumsum(diff)[:-1]  # counts_full[p] = concurrency at bar (lo + p)
    return counts_full[index - lo]


def average_uniqueness(t_starts: np.ndarray, t_ends: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Per-label average uniqueness → sample weight vector (LdP ch. 4).

    For each label `i` with outcome window `[t_start_i, t_end_i]`, computes
    the mean of `1 / concurrency[b]` over all bars `b` in `index` that fall
    within that window. A label whose window never overlaps any other
    label's window gets weight 1.0; heavy overlap pulls the weight toward 0.

    Parameters
    ----------
    t_starts, t_ends
        1-D integer arrays (same length, one entry per label).
    index
        1-D sorted-ascending integer array of bar positions (the universe of
        bars over which concurrency is computed).

    Returns
    -------
    np.ndarray
        float64 array, one weight per label (same length as `t_starts`).
    """
    t_starts = np.asarray(t_starts, dtype=np.int64)
    t_ends = np.asarray(t_ends, dtype=np.int64)
    index = np.asarray(index, dtype=np.int64)
    concurrency = label_concurrency(t_starts, t_ends, index)

    weights = np.zeros(len(t_starts), dtype=np.float64)
    for i, (s, e) in enumerate(zip(t_starts, t_ends, strict=True)):
        lo_pos = int(np.searchsorted(index, s, side="left"))
        hi_pos = int(np.searchsorted(index, e, side="right"))
        seg = concurrency[lo_pos:hi_pos]
        seg = seg[seg > 0]
        weights[i] = float(np.mean(1.0 / seg)) if seg.size > 0 else 0.0
    return weights
