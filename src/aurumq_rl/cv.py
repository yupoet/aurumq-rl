"""Purged + embargoed walk-forward cross-validation (López de Prado, ch. 7).

Forward-constructed labels (anchor / triple-barrier / trend-scanning) in this
repo embed information from ``[d+1, d+K]`` at date ``d`` (``K`` =
``label_horizon_days``). Any CV split that lets a validation sample sit
within ``K`` days of a training sample leaks label information backward into
training. :class:`PurgedWalkForwardCV` generalizes the single ad-hoc split in
``scripts/p3/kronos_matrix_v13_lib.py::date_embargo_split`` into a reusable,
multi-fold, sklearn-compatible splitter.

Units — trading days, not calendar days
----------------------------------------
``label_horizon_days`` and ``embargo_days`` are counted as **positions in the
sorted unique-date sequence** (i.e. "trading days" — the same convention as
``date_embargo_split``'s ``EMBARGO_DAYS``), NOT calendar-day deltas. This
keeps the splitter agnostic to the concrete type of the date column (it may
be an actual date, a ``trade_date`` integer code, or any other sortable
value) and matches the precedent this class generalizes.

API — ``groups=``, not a constructor ``dates=``
-------------------------------------------------
Per-row dates are passed to :meth:`split` / :meth:`get_n_splits` via the
standard scikit-learn ``groups`` parameter (as in ``GroupKFold``), not a
constructor argument. This keeps the splitter instance stateless/reusable
across differently-shaped panels and is the literal shape LightGBM/sklearn
expect when they call ``cv.split(X, y, groups)``.

No lookahead
------------
For every fold, every training row's date position ``p`` satisfies
``p + label_horizon_days < test_start`` (strict), so a train sample's label
horizon never reaches into the test fold. Embargo additionally excludes
training rows within ``embargo_days`` immediately after any *earlier* fold's
test window — relevant once an earlier fold's test period is absorbed into a
later fold's (expanding or rolling) training window.

Empty-train behavior
---------------------
If purge + embargo remove every candidate training row for a fold, that fold
is still yielded (so ``len(list(cv.split(...))) == cv.get_n_splits()``
always holds) with an empty ``train_idx`` array rather than being skipped.
Callers that cannot tolerate an empty training fold should check
``train_idx.size`` themselves.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

import numpy as np

__all__ = ["PurgedWalkForwardCV"]

_Mode = Literal["expanding", "rolling"]


def _num_samples(x: Any) -> int:
    """Row count of an array-like, without requiring sklearn/pandas."""
    if hasattr(x, "shape"):
        return int(x.shape[0])
    return len(x)


class PurgedWalkForwardCV:
    """Purged + embargoed walk-forward CV splitter (scikit-learn compatible).

    Splits rows into ``n_splits`` forward-advancing (train, test) folds over
    the sorted **unique** dates supplied via ``groups``. All rows sharing a
    date always land on the same side of every split. Training rows whose
    label horizon would overlap the test fold are purged, and rows within
    ``embargo_days`` after any earlier fold's test window are additionally
    embargoed (see module docstring for exact semantics and units).

    Drops into scikit-learn / LightGBM ``cv=`` slots: implements
    ``get_n_splits(X, y, groups)`` and ``split(X, y, groups) ->
    Iterator[(train_idx, test_idx)]``.

    Args:
        n_splits: number of test folds.
        label_horizon_days: ``K`` — a training sample at date position ``p``
            is purged from a fold if ``p + K >= test_start`` (its label
            window would overlap the test fold). Trading-day units (see
            module docstring). ``0`` disables purging.
        embargo_days: number of trading days immediately after an *earlier*
            fold's test window that are excluded from training in later
            folds. ``0`` disables embargoing.
        mode: ``"expanding"`` (default) — training uses every date before
            the test fold's purge cutoff. ``"rolling"`` — training uses a
            fixed-size window of ``max_train_size`` (or ``test_size`` if
            unset) dates immediately preceding the purge cutoff.
        test_size: number of unique dates per test fold. Defaults to
            ``n_unique_dates // (n_splits + 1)`` (mirrors
            ``sklearn.model_selection.TimeSeriesSplit``: the earliest chunk
            of dates is reserved purely for initial training).
        max_train_size: rolling-mode only; maximum number of unique dates in
            the training window. Ignored in expanding mode.

    Raises:
        ValueError: invalid constructor arguments (checked eagerly), or
            (raised lazily, on iteration) too few unique dates for the
            requested ``n_splits`` / ``test_size``.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        label_horizon_days: int = 0,
        embargo_days: int = 0,
        mode: _Mode = "expanding",
        test_size: int | None = None,
        max_train_size: int | None = None,
    ) -> None:
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if label_horizon_days < 0:
            raise ValueError(f"label_horizon_days must be >= 0, got {label_horizon_days}")
        if embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {embargo_days}")
        if mode not in ("expanding", "rolling"):
            raise ValueError(f"mode must be 'expanding' or 'rolling', got {mode!r}")
        if test_size is not None and test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {test_size}")
        if max_train_size is not None and max_train_size < 1:
            raise ValueError(f"max_train_size must be >= 1, got {max_train_size}")

        self.n_splits = n_splits
        self.label_horizon_days = label_horizon_days
        self.embargo_days = embargo_days
        self.mode = mode
        self.test_size = test_size
        self.max_train_size = max_train_size

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Number of folds (sklearn splitter interface; args are unused)."""
        del X, y, groups  # static value, kept for interface compatibility
        return self.n_splits

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` row-index arrays for each fold.

        Args:
            X: array-like used only to validate row count against ``groups``
                (its values are never inspected). May be ``None`` to skip
                that check.
            y: unused; accepted for sklearn interface compatibility.
            groups: required — per-row date array (any sortable dtype),
                length == number of rows.

        Yields:
            ``(train_idx, test_idx)``: sorted ``int64`` row-index arrays into
            the original (unpermuted) ``groups`` array. Exactly
            ``get_n_splits()`` pairs are yielded; ``train_idx`` may be empty
            (see module docstring "Empty-train behavior").

        Raises:
            ValueError: ``groups`` is ``None``/empty, its length mismatches
                ``X``, or there are too few unique dates for the requested
                ``n_splits`` / ``test_size``.
        """
        del y  # unused; accepted for sklearn interface compatibility
        if groups is None:
            raise ValueError(
                "PurgedWalkForwardCV.split() requires groups=<per-row date array>; "
                "pass the date column via the standard sklearn `groups` kwarg "
                "(this splitter holds no data of its own)."
            )
        dates = np.asarray(groups)
        if dates.size == 0:
            raise ValueError("groups must be non-empty")
        if X is not None:
            n_rows = _num_samples(X)
            if len(dates) != n_rows:
                raise ValueError(f"groups length {len(dates)} != X length {n_rows}")

        unique_dates = np.unique(dates)
        n_dates = len(unique_dates)

        test_size = self.test_size or n_dates // (self.n_splits + 1)
        if test_size < 1:
            raise ValueError(
                f"cannot derive test_size from {n_dates} unique dates and "
                f"n_splits={self.n_splits}; need >= {self.n_splits + 1} unique "
                f"dates, or pass test_size explicitly"
            )

        # Test windows are precomputed up front (as [start, end) positions
        # into unique_dates) so later folds can look back at earlier folds'
        # test boundaries for the embargo rule.
        test_windows: list[tuple[int, int]] = []
        for i in range(self.n_splits):
            test_start = n_dates - (self.n_splits - i) * test_size
            test_end = test_start + test_size
            if test_start < 0:
                raise ValueError(
                    f"not enough unique dates ({n_dates}) for n_splits="
                    f"{self.n_splits} with test_size={test_size}; reduce "
                    f"n_splits/test_size or supply more history"
                )
            test_windows.append((test_start, test_end))

        for i, (test_start, test_end) in enumerate(test_windows):
            if self.mode == "expanding":
                train_start = 0
            else:  # rolling
                span = self.max_train_size or test_size
                train_start = max(0, test_start - span)

            # Purge: keep only train positions whose label horizon ends
            # strictly before the test fold starts (p + K < test_start).
            purge_cutoff = max(train_start, test_start - self.label_horizon_days)
            train_positions = np.arange(train_start, purge_cutoff)

            # Embargo: drop positions within embargo_days after any EARLIER
            # fold's test window (matters once that fold's test period is
            # absorbed into a later training window; see module docstring).
            if self.embargo_days > 0 and train_positions.size > 0:
                keep = np.ones(train_positions.shape[0], dtype=bool)
                for _prev_start, prev_end in test_windows[:i]:
                    embargo_lo = prev_end
                    embargo_hi = prev_end + self.embargo_days
                    keep &= ~((train_positions >= embargo_lo) & (train_positions < embargo_hi))
                train_positions = train_positions[keep]

            train_dates = unique_dates[train_positions]
            test_dates = unique_dates[test_start:test_end]

            train_idx = np.flatnonzero(np.isin(dates, train_dates)).astype(np.int64)
            test_idx = np.flatnonzero(np.isin(dates, test_dates)).astype(np.int64)

            yield train_idx, test_idx
