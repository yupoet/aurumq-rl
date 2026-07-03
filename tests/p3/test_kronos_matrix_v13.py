"""Tests for the v13 Kronos matrix pure helpers (C9 / M4 / M19 / M20).

The full script ``p3.kronos_matrix_v13`` needs lightgbm + GPU checkpoints, so the
defect fixes live in the side-effect-free module ``p3.kronos_matrix_v13_lib``
which is unit-testable with synthetic frames only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from p3.kronos_matrix_v13_lib import (
    build_lookback_window,
    d1_leakage_selfcheck,
    date_embargo_split,
    is_cell_done,
    select_cell_embeddings,
    v13_artifact_path,
)

# =============================================================================
# C9 — which embeddings frame does cell X train/score on
# =============================================================================


def _frames(*keys: str) -> dict:
    """Fake available-frames mapping (values irrelevant to key selection)."""
    return {k: object() for k in keys}


class TestSelectCellEmbeddings:
    FT_CELL = {
        "univ": "MAIN_BOARD",
        "anchor": "T3",
        "spec": "alpha",
        "is_null": False,
        "is_base": False,
    }
    NULL_CELL = {
        "univ": "MAIN_BOARD",
        "anchor": "T3",
        "spec": "alpha",
        "is_null": True,
        "is_base": False,
    }
    BASE_CELL = {
        "univ": "MAIN_BOARD",
        "anchor": "T3",
        "spec": "alpha",
        "is_null": False,
        "is_base": True,
    }

    def test_finetuned_cell_uses_label_and_eval(self):
        assert select_cell_embeddings(
            self.FT_CELL, _frames("label", "eval", "base", "base_eval")
        ) == (
            "label",
            "eval",
        )

    def test_null_cell_uses_label_and_eval(self):
        assert select_cell_embeddings(self.NULL_CELL, _frames("label", "eval")) == ("label", "eval")

    def test_finetuned_cell_missing_eval_still_selects_eval_key(self):
        # ft cells without eval embeddings train but skip scoring (existing behavior)
        assert select_cell_embeddings(self.FT_CELL, _frames("label")) == ("label", "eval")

    def test_base_cell_uses_base_and_base_eval(self):
        assert select_cell_embeddings(
            self.BASE_CELL, _frames("label", "eval", "base", "base_eval")
        ) == (
            "base",
            "base_eval",
        )

    def test_base_cell_falls_back_to_base_eval_for_training(self):
        # --phase 2 --base-model --eval-window writes ONE parquet containing
        # both label pairs and eval pairs; it is a valid training source.
        assert select_cell_embeddings(self.BASE_CELL, _frames("label", "eval", "base_eval")) == (
            "base_eval",
            "base_eval",
        )

    def test_base_cell_missing_base_eval_raises(self):
        with pytest.raises(FileNotFoundError, match=r"--phase 2 --base-model --eval-window"):
            select_cell_embeddings(self.BASE_CELL, _frames("label", "eval", "base"))

    def test_base_cell_missing_everything_raises(self):
        with pytest.raises(FileNotFoundError):
            select_cell_embeddings(self.BASE_CELL, _frames("label", "eval"))


# =============================================================================
# M4 — date-sorted, embargoed early-stopping split
# =============================================================================


def _make_shuffled_frame(n_dates: int = 100, n_stocks: int = 5, seed: int = 7) -> pd.DataFrame:
    """Synthetic (ts_code, trade_date, y) frame with per-year-concat-like shuffled rows."""
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    rows = [
        {"ts_code": f"SYN{s:03d}", "trade_date": d, "y": (s + i) % 2}
        for i, d in enumerate(dates)
        for s in range(n_stocks)
    ]
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


class TestDateEmbargoSplit:
    def test_val_is_max_dates(self):
        df = _make_shuffled_frame(n_dates=100, n_stocks=5)
        train_fit, val = date_embargo_split(df, val_frac=0.10, embargo_days=30)
        all_dates = np.array(sorted(df["trade_date"].unique()))
        assert set(val["trade_date"].unique()) == set(all_dates[-10:])

    def test_embargo_gap_in_trading_days(self):
        df = _make_shuffled_frame(n_dates=100, n_stocks=5)
        train_fit, val = date_embargo_split(df, val_frac=0.10, embargo_days=30)
        all_dates = np.array(sorted(df["trade_date"].unique()))
        pos = {d: i for i, d in enumerate(all_dates)}
        gap = pos[val["trade_date"].min()] - pos[train_fit["trade_date"].max()]
        # train_fit's max date must be >= embargo_days trading days before val's min date
        assert gap >= 30

    def test_no_row_loss_besides_embargo_gap(self):
        n_stocks = 5
        df = _make_shuffled_frame(n_dates=100, n_stocks=n_stocks)
        train_fit, val = date_embargo_split(df, val_frac=0.10, embargo_days=30)
        # 100 dates: 10 val, 30 embargoed, 60 train
        assert len(val) == 10 * n_stocks
        assert len(train_fit) == 60 * n_stocks
        assert len(train_fit) + len(val) == len(df) - 30 * n_stocks
        # no overlap
        assert set(train_fit["trade_date"]).isdisjoint(set(val["trade_date"]))

    def test_zero_embargo_is_contiguous(self):
        df = _make_shuffled_frame(n_dates=50, n_stocks=3)
        train_fit, val = date_embargo_split(df, val_frac=0.10, embargo_days=0)
        assert len(train_fit) + len(val) == len(df)
        all_dates = np.array(sorted(df["trade_date"].unique()))
        pos = {d: i for i, d in enumerate(all_dates)}
        assert pos[val["trade_date"].min()] - pos[train_fit["trade_date"].max()] == 1

    def test_row_order_invariant(self):
        df = _make_shuffled_frame(n_dates=60, n_stocks=4, seed=1)
        df_sorted = df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        a_fit, a_val = date_embargo_split(df, val_frac=0.10, embargo_days=10)
        b_fit, b_val = date_embargo_split(df_sorted, val_frac=0.10, embargo_days=10)
        for a, b in ((a_fit, b_fit), (a_val, b_val)):
            pd.testing.assert_frame_equal(
                a.sort_values(["trade_date", "ts_code"]).reset_index(drop=True),
                b.sort_values(["trade_date", "ts_code"]).reset_index(drop=True),
            )

    def test_output_sorted_by_date(self):
        df = _make_shuffled_frame(n_dates=40, n_stocks=3)
        train_fit, val = date_embargo_split(df, val_frac=0.10, embargo_days=5)
        assert train_fit["trade_date"].is_monotonic_increasing
        assert val["trade_date"].is_monotonic_increasing

    def test_embargo_swallowing_all_train_dates_gives_empty_train(self):
        df = _make_shuffled_frame(n_dates=20, n_stocks=2)
        train_fit, val = date_embargo_split(df, val_frac=0.10, embargo_days=30)
        assert len(train_fit) == 0
        assert len(val) == 2 * 2  # last 2 of 20 dates

    def test_val_frac_out_of_range_raises(self):
        df = _make_shuffled_frame(n_dates=20, n_stocks=2)
        with pytest.raises(ValueError):
            date_embargo_split(df, val_frac=0.0, embargo_days=0)
        with pytest.raises(ValueError):
            date_embargo_split(df, val_frac=1.0, embargo_days=0)

    def test_single_date_raises(self):
        df = _make_shuffled_frame(n_dates=1, n_stocks=3)
        with pytest.raises(ValueError):
            date_embargo_split(df, val_frac=0.10, embargo_days=0)


# =============================================================================
# M19 — smoke-suffixed run artifacts + retryable skipped checkpoints
# =============================================================================


class TestArtifactPaths:
    OUT = Path("data/kronos/outputs/matrix_v13")
    RES = Path("data/kronos/outputs")

    def test_checkpoint_full_vs_smoke(self):
        full = v13_artifact_path("checkpoint", smoke=False, out_dir=self.OUT, results_dir=self.RES)
        smoke = v13_artifact_path("checkpoint", smoke=True, out_dir=self.OUT, results_dir=self.RES)
        assert full == self.RES / "matrix_v13_phase3_checkpoint.json"
        assert smoke == self.RES / "matrix_v13_phase3_checkpoint_smoke.json"

    def test_pred_full_vs_smoke(self):
        cid = "alpha_T3_MAIN_BOARD_BASE"
        full = v13_artifact_path(
            "pred", smoke=False, out_dir=self.OUT, results_dir=self.RES, cid=cid
        )
        smoke = v13_artifact_path(
            "pred", smoke=True, out_dir=self.OUT, results_dir=self.RES, cid=cid
        )
        assert full == self.OUT / f"pred_{cid}.parquet"
        assert smoke == self.OUT / f"pred_{cid}_smoke.parquet"

    def test_results_full_vs_smoke(self):
        full = v13_artifact_path("results", smoke=False, out_dir=self.OUT, results_dir=self.RES)
        smoke = v13_artifact_path("results", smoke=True, out_dir=self.OUT, results_dir=self.RES)
        assert full == self.RES / "matrix_v13_results.json"
        assert smoke == self.RES / "matrix_v13_results_smoke.json"

    def test_pred_requires_cid(self):
        with pytest.raises(ValueError):
            v13_artifact_path("pred", smoke=False, out_dir=self.OUT, results_dir=self.RES)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            v13_artifact_path("bogus", smoke=False, out_dir=self.OUT, results_dir=self.RES)


class TestIsCellDone:
    def test_absent_runs(self):
        assert is_cell_done(None) is False

    def test_done_entry_skips(self):
        assert is_cell_done({"n_train": 1000, "best_iter": 42}) is True
        assert is_cell_done({"done": True}) is True

    def test_skipped_entry_retries(self):
        assert is_cell_done({"skipped": "no_base_embeddings"}) is False
        assert is_cell_done({"skipped": True, "n_train": 3}) is False


# =============================================================================
# M20 — real D-1 leakage guard on the actual extraction slice
# =============================================================================


def _increasing_series(n: int = 130):
    """Strictly increasing dates + values so any off-by-one is detectable."""
    dates = pd.bdate_range("2024-01-02", periods=n).values  # datetime64[ns]
    values = np.arange(n, dtype=np.float64).reshape(-1, 1) * np.ones((1, 6))
    values += np.arange(6) * 0.1  # distinct per column, still strictly increasing per row
    return values, dates


class TestBuildLookbackWindow:
    def test_strict_d1_no_anchor_day_leak(self):
        values, dates = _increasing_series()
        idx, seq_len = 80, 60
        window, window_dates = build_lookback_window(values, dates, idx, seq_len)
        assert window.shape == (seq_len, 6)
        assert len(window_dates) == seq_len
        assert window_dates.max() < dates[idx]  # strict D-1
        assert window_dates[-1] == dates[idx - 1]  # ends exactly at D-1 (catches idx+1 slices)
        assert window_dates[0] == dates[idx - seq_len]
        np.testing.assert_array_equal(window, values[idx - seq_len : idx])

    def test_first_eligible_anchor_is_idx_equals_seq_len(self):
        # idx == seq_len needs exactly rows [0, seq_len) — all strictly before idx.
        values, dates = _increasing_series()
        seq_len = 120
        window, window_dates = build_lookback_window(values, dates, seq_len, seq_len)
        assert window.shape == (seq_len, 6)
        assert window_dates.max() < dates[seq_len]
        np.testing.assert_array_equal(window, values[:seq_len])

    def test_insufficient_history_raises(self):
        values, dates = _increasing_series()
        with pytest.raises(ValueError):
            build_lookback_window(values, dates, 119, 120)

    def test_idx_beyond_series_raises(self):
        values, dates = _increasing_series(n=130)
        with pytest.raises(ValueError):
            build_lookback_window(values, dates, 131, 60)

    def test_bad_seq_len_raises(self):
        values, dates = _increasing_series()
        with pytest.raises(ValueError):
            build_lookback_window(values, dates, 80, 0)

    def test_selfcheck_passes_for_both_seq_lens(self):
        d1_leakage_selfcheck(60)
        d1_leakage_selfcheck(120)


# =============================================================================
# Full script import (needs lightgbm; skipped on boxes without it)
# =============================================================================


def test_v13_script_importable_and_wired():
    pytest.importorskip("lightgbm")
    import p3.kronos_matrix_v13 as v13

    # the script must consume the fixed helpers, not re-implement them
    from p3 import kronos_matrix_v13_lib as lib

    assert v13.date_embargo_split is lib.date_embargo_split
    assert v13.build_lookback_window is lib.build_lookback_window
    assert v13.select_cell_embeddings is lib.select_cell_embeddings
    assert v13.EMBARGO_DAYS == 30
