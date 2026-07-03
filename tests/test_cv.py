"""Unit tests for src/aurumq_rl/cv.py — purged + embargoed walk-forward CV.

All synthetic, no real data. Dates are consecutive integers ``0..n_dates-1``
so "trading-day index" and "date value" coincide, which sidesteps any
ambiguity about whether ``label_horizon_days`` / ``embargo_days`` are counted
as calendar-day deltas or unique-date positions (see module docstring in
``aurumq_rl/cv.py`` — they are unique-date positions, matching
``kronos_matrix_v13_lib.date_embargo_split``'s "trading calendar" convention).
"""

from __future__ import annotations

import numpy as np
import pytest

from aurumq_rl.cv import PurgedWalkForwardCV


def _synthetic_dates(n_dates: int = 60, n_symbols: int = 5, seed: int = 0) -> np.ndarray:
    """Row-per-(date, symbol) date array, e.g. 60 dates x 5 symbols = 300 rows."""
    rng = np.random.default_rng(seed)
    dates = np.repeat(np.arange(n_dates), n_symbols)
    order = rng.permutation(len(dates))
    return dates[order]


# ---------------------------------------------------------------------------
# 1. Purge correctness
# ---------------------------------------------------------------------------


def test_purge_removes_label_horizon_overlap():
    dates = _synthetic_dates(n_dates=60, n_symbols=4, seed=1)
    K = 5
    cv = PurgedWalkForwardCV(n_splits=4, label_horizon_days=K, embargo_days=0)
    folds = list(cv.split(X=None, groups=dates))
    assert len(folds) == 4
    for train_idx, test_idx in folds:
        if train_idx.size == 0:
            continue
        first_test_date = dates[test_idx].min()
        train_dates = dates[train_idx]
        assert (train_dates + K < first_test_date).all(), (
            f"purge violated: some train date + {K} >= {first_test_date}"
        )


# ---------------------------------------------------------------------------
# 2. Embargo correctness
# ---------------------------------------------------------------------------


def test_embargo_removes_post_test_window():
    # n_splits=3, n_dates=60 -> test_size=15, contiguous folds [15,30) [30,45) [45,60).
    # Fold 0's test ends at date 30; embargo=4 must keep dates [30, 34) out of
    # training for LATER folds that would otherwise absorb them (fold 2 here,
    # since fold 1's own test occupies [30, 45) already).
    dates = _synthetic_dates(n_dates=60, n_symbols=3, seed=2)
    embargo_days = 4

    cv_embargo = PurgedWalkForwardCV(n_splits=3, label_horizon_days=0, embargo_days=embargo_days)
    folds_embargo = list(cv_embargo.split(X=None, groups=dates))

    cv_baseline = PurgedWalkForwardCV(n_splits=3, label_horizon_days=0, embargo_days=0)
    folds_baseline = list(cv_baseline.split(X=None, groups=dates))

    train_idx_fold2, test_idx_fold2 = folds_embargo[2]
    train_dates_fold2 = set(np.unique(dates[train_idx_fold2]).tolist())
    embargo_zone = set(range(30, 30 + embargo_days))
    assert not (train_dates_fold2 & embargo_zone), (
        f"embargoed dates {embargo_zone} leaked into fold-2 train: "
        f"{train_dates_fold2 & embargo_zone}"
    )

    # Non-vacuous: without embargo, that zone WOULD have been train data.
    base_train_idx_fold2, _ = folds_baseline[2]
    base_train_dates_fold2 = set(np.unique(dates[base_train_idx_fold2]).tolist())
    assert base_train_dates_fold2 & embargo_zone, (
        "test setup invalid: baseline (embargo_days=0) fold-2 train does not "
        "even contain the zone under test, so the embargo assertion is vacuous"
    )

    # General property: no fold's train ever contains a date within
    # embargo_days after any EARLIER fold's test window.
    test_windows = []
    for _, test_idx in folds_embargo:
        if test_idx.size:
            test_windows.append((dates[test_idx].min(), dates[test_idx].max()))
    for i, (train_idx, _) in enumerate(folds_embargo):
        if train_idx.size == 0:
            continue
        train_dates = set(np.unique(dates[train_idx]).tolist())
        for j in range(i):
            _, prev_test_max = test_windows[j]
            forbidden = set(range(prev_test_max + 1, prev_test_max + 1 + embargo_days))
            assert not (train_dates & forbidden), (
                f"fold {i} train contains dates embargoed after fold {j}'s test"
            )


# ---------------------------------------------------------------------------
# 3. Walk-forward direction
# ---------------------------------------------------------------------------


def test_walk_forward_direction_strictly_forward():
    dates = _synthetic_dates(n_dates=80, n_symbols=3, seed=3)
    cv = PurgedWalkForwardCV(n_splits=5, label_horizon_days=2, embargo_days=1)
    folds = list(cv.split(X=None, groups=dates))
    assert len(folds) == 5
    prior_test_max = -np.inf
    for train_idx, test_idx in folds:
        assert test_idx.size > 0
        test_min = dates[test_idx].min()
        test_max = dates[test_idx].max()
        if train_idx.size:
            assert dates[train_idx].max() < test_min, "train must precede test strictly"
        # folds advance forward: this fold's test starts after the previous one's ended.
        assert test_min > prior_test_max
        prior_test_max = test_max


# ---------------------------------------------------------------------------
# 4. Determinism under row permutation
# ---------------------------------------------------------------------------


def test_determinism_under_row_permutation():
    base_dates = _synthetic_dates(n_dates=50, n_symbols=4, seed=4)
    rng = np.random.default_rng(99)
    shuffled_order = rng.permutation(len(base_dates))
    shuffled_dates = base_dates[shuffled_order]

    cv1 = PurgedWalkForwardCV(n_splits=4, label_horizon_days=3, embargo_days=2)
    cv2 = PurgedWalkForwardCV(n_splits=4, label_horizon_days=3, embargo_days=2)

    folds1 = list(cv1.split(X=None, groups=base_dates))
    folds2 = list(cv2.split(X=None, groups=shuffled_dates))

    assert len(folds1) == len(folds2)
    for (train1, test1), (train2, test2) in zip(folds1, folds2, strict=True):
        train_dates1 = set(np.unique(base_dates[train1]).tolist())
        train_dates2 = set(np.unique(shuffled_dates[train2]).tolist())
        test_dates1 = set(np.unique(base_dates[test1]).tolist())
        test_dates2 = set(np.unique(shuffled_dates[test2]).tolist())
        assert train_dates1 == train_dates2
        assert test_dates1 == test_dates2


# ---------------------------------------------------------------------------
# 5. Whole-date grouping
# ---------------------------------------------------------------------------


def test_whole_date_grouping_never_splits_a_date():
    dates = _synthetic_dates(n_dates=50, n_symbols=6, seed=5)
    cv = PurgedWalkForwardCV(n_splits=4, label_horizon_days=2, embargo_days=1)
    for train_idx, test_idx in cv.split(X=None, groups=dates):
        train_dates = set(dates[train_idx].tolist())
        test_dates = set(dates[test_idx].tolist())
        assert not (train_dates & test_dates), "a date must never appear on both sides"
        # every row for a given date must land in the same set as every other
        # row sharing that date.
        for d in train_dates:
            rows_for_d = np.flatnonzero(dates == d)
            assert set(rows_for_d.tolist()).issubset(set(train_idx.tolist()))
        for d in test_dates:
            rows_for_d = np.flatnonzero(dates == d)
            assert set(rows_for_d.tolist()).issubset(set(test_idx.tolist()))


# ---------------------------------------------------------------------------
# 6. sklearn interface smoke
# ---------------------------------------------------------------------------


def test_sklearn_interface_smoke():
    dates = _synthetic_dates(n_dates=50, n_symbols=5, seed=6)
    n_rows = len(dates)
    X = np.zeros((n_rows, 3))
    cv = PurgedWalkForwardCV(n_splits=5, label_horizon_days=1, embargo_days=1)

    assert cv.get_n_splits(X, groups=dates) == 5

    folds = list(cv.split(X, groups=dates))
    assert len(folds) == cv.get_n_splits(X, groups=dates)

    for train_idx, test_idx in folds:
        assert np.issubdtype(train_idx.dtype, np.integer)
        assert np.issubdtype(test_idx.dtype, np.integer)
        if train_idx.size:
            assert train_idx.min() >= 0
            assert train_idx.max() < n_rows
        assert test_idx.min() >= 0
        assert test_idx.max() < n_rows
        assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0


def test_groups_required():
    dates = _synthetic_dates(n_dates=20, n_symbols=2, seed=7)
    cv = PurgedWalkForwardCV(n_splits=2)
    with pytest.raises(ValueError, match="groups"):
        list(cv.split(X=np.zeros((len(dates), 1)), groups=None))


def test_groups_length_mismatch_raises():
    dates = _synthetic_dates(n_dates=20, n_symbols=2, seed=8)
    cv = PurgedWalkForwardCV(n_splits=2)
    with pytest.raises(ValueError, match="length"):
        list(cv.split(X=np.zeros((len(dates) - 1, 1)), groups=dates))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_splits": 0},
        {"n_splits": 2, "label_horizon_days": -1},
        {"n_splits": 2, "embargo_days": -1},
        {"n_splits": 2, "mode": "sideways"},
        {"n_splits": 2, "test_size": 0},
        {"n_splits": 2, "max_train_size": 0},
    ],
)
def test_invalid_constructor_args_raise(kwargs):
    with pytest.raises(ValueError):
        PurgedWalkForwardCV(**kwargs)


# ---------------------------------------------------------------------------
# 7. Edge case: purge/embargo large enough to empty a train fold
# ---------------------------------------------------------------------------


def test_large_horizon_yields_empty_train_not_skipped_fold():
    dates = _synthetic_dates(n_dates=40, n_symbols=3, seed=9)
    # label_horizon_days larger than the entire pre-test span empties fold 0's
    # train under the documented behavior: the fold is still yielded (matching
    # get_n_splits()), with an empty (but valid-dtype) train index array.
    cv = PurgedWalkForwardCV(n_splits=4, label_horizon_days=1000, embargo_days=0)
    folds = list(cv.split(X=None, groups=dates))
    assert len(folds) == cv.get_n_splits() == 4

    train_idx0, test_idx0 = folds[0]
    assert train_idx0.size == 0
    assert np.issubdtype(train_idx0.dtype, np.integer)
    assert test_idx0.size > 0


# ---------------------------------------------------------------------------
# Rolling mode
# ---------------------------------------------------------------------------


def test_rolling_mode_bounds_train_span():
    dates = _synthetic_dates(n_dates=80, n_symbols=3, seed=10)
    max_train_size = 10
    cv = PurgedWalkForwardCV(
        n_splits=4,
        label_horizon_days=1,
        embargo_days=0,
        mode="rolling",
        max_train_size=max_train_size,
    )
    for train_idx, test_idx in cv.split(X=None, groups=dates):
        if train_idx.size == 0:
            continue
        n_unique_train_dates = len(np.unique(dates[train_idx]))
        assert n_unique_train_dates <= max_train_size
        assert dates[train_idx].max() < dates[test_idx].min()


# ---------------------------------------------------------------------------
# Public export
# ---------------------------------------------------------------------------


def test_exported_from_package_root():
    from aurumq_rl import PurgedWalkForwardCV as ExportedCV

    assert ExportedCV is PurgedWalkForwardCV
