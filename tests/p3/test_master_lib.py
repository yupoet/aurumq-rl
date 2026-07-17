"""Tests for the MASTER-lite pure helpers (scripts/p3/master_lib.py).

The GPU training script needs torch + a real bundle, so all defect-prone logic
(z-scoring, sequence indexing, embargoed split, rank blending, kill verdict)
lives in the side-effect-free ``p3.master_lib`` and is tested here with
synthetic frames only — same pattern as ``test_kronos_matrix_v13_lib``.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from p3.master_lib import (
    ZSCORE_CLIP,
    blend_rank_scores,
    build_sequence_windows,
    cs_zscore_panel,
    daily_rank_ic,
    kill_criteria_verdict,
    train_val_split_with_embargo,
)

# =============================================================================
# cs_zscore_panel
# =============================================================================


def test_zscore_centers_and_scales_cross_section():
    x = np.zeros((1, 4, 1), dtype=np.float32)
    x[0, :, 0] = [1.0, 2.0, 3.0, 4.0]
    z = cs_zscore_panel(x)
    np.testing.assert_allclose(z[0, :, 0].mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(z[0, :, 0].std(), 1.0, atol=1e-5)


def test_zscore_nan_becomes_neutral_zero_and_ignores_nan_in_stats():
    x = np.zeros((1, 4, 1), dtype=np.float32)
    x[0, :, 0] = [1.0, 3.0, np.nan, np.nan]
    z = cs_zscore_panel(x)
    # NaN cells -> 0.0 (the cross-section mean, neutral for the model)
    assert z[0, 2, 0] == 0.0 and z[0, 3, 0] == 0.0
    # stats computed over the 2 valid values only: they z-score to -1/+1
    np.testing.assert_allclose(z[0, :2, 0], [-1.0, 1.0], atol=1e-5)


def test_zscore_constant_and_all_nan_sections_are_inert():
    x = np.zeros((2, 3, 1), dtype=np.float32)
    x[0, :, 0] = 7.0  # constant cross-section: std floor, z = 0
    x[1, :, 0] = np.nan  # all-NaN cross-section
    z = cs_zscore_panel(x)
    assert np.all(z == 0.0)
    assert np.all(np.abs(z) <= ZSCORE_CLIP)


def test_zscore_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\[D, N, F\]"):
        cs_zscore_panel(np.zeros((3, 4)))


# =============================================================================
# build_sequence_windows
# =============================================================================


def test_sequence_windows_shape_and_content():
    w = build_sequence_windows(n_dates=5, seq_len=3)
    assert w.shape == (3, 3)
    np.testing.assert_array_equal(w[0], [0, 1, 2])
    np.testing.assert_array_equal(w[-1], [2, 3, 4])


def test_sequence_windows_no_padding_short_history():
    assert build_sequence_windows(n_dates=2, seq_len=3).shape == (0, 3)


def test_sequence_windows_seq_len_one_is_identity():
    w = build_sequence_windows(n_dates=4, seq_len=1)
    np.testing.assert_array_equal(w.ravel(), [0, 1, 2, 3])


# =============================================================================
# train_val_split_with_embargo
# =============================================================================


def _dates(n: int) -> list[dt.date]:
    return [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(n)]


def test_embargo_gap_between_train_and_val():
    dates = _dates(100)
    train, val = train_val_split_with_embargo(dates, val_frac=0.10, embargo_days=30)
    assert len(val) == 10
    assert val == dates[90:]
    # last train date must be >= 30 trading dates before the first val date
    assert train[-1] == dates[90 - 30 - 1]
    assert not (set(train) & set(val))


def test_embargo_split_raises_when_nothing_left_for_train():
    with pytest.raises(ValueError, match="not enough dates"):
        train_val_split_with_embargo(_dates(20), val_frac=0.5, embargo_days=30)


def test_embargo_split_validates_args():
    with pytest.raises(ValueError, match="val_frac"):
        train_val_split_with_embargo(_dates(50), val_frac=1.5)
    with pytest.raises(ValueError, match="embargo_days"):
        train_val_split_with_embargo(_dates(50), embargo_days=-1)


# =============================================================================
# daily_rank_ic
# =============================================================================


def _pred_frame(rows: list[tuple[dt.date, str, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [r[0] for r in rows],
            "ts_code": [r[1] for r in rows],
            "score": [r[2] for r in rows],
            "actual_y": [r[3] for r in rows],
        }
    )


def test_daily_rank_ic_perfect_and_inverted():
    d = dt.date(2025, 6, 2)
    perfect = _pred_frame([(d, f"c{i}", float(i), float(i) * 2) for i in range(5)])
    assert daily_rank_ic(perfect) == pytest.approx(1.0)
    inverted = _pred_frame([(d, f"c{i}", float(i), -float(i)) for i in range(5)])
    assert daily_rank_ic(inverted) == pytest.approx(-1.0)


def test_daily_rank_ic_averages_across_days_and_skips_tiny_days():
    d1, d2, d3 = dt.date(2025, 6, 2), dt.date(2025, 6, 3), dt.date(2025, 6, 4)
    rows = [(d1, f"c{i}", float(i), float(i)) for i in range(5)]  # IC +1
    rows += [(d2, f"c{i}", float(i), -float(i)) for i in range(5)]  # IC -1
    rows += [(d3, "c0", 1.0, 1.0), (d3, "c1", 2.0, 2.0)]  # 2 names: skipped
    assert daily_rank_ic(_pred_frame(rows)) == pytest.approx(0.0)


def test_daily_rank_ic_empty_returns_zero():
    assert daily_rank_ic(_pred_frame([])) == 0.0


# =============================================================================
# blend_rank_scores
# =============================================================================


def _score_frame(day: dt.date, scores: dict[str, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [day] * len(scores),
            "ts_code": list(scores),
            "score": list(scores.values()),
        }
    )


def test_blend_is_scale_invariant_via_ranks():
    d = dt.date(2025, 6, 2)
    a = _score_frame(d, {"c1": 0.1, "c2": 0.2, "c3": 0.3})
    b = _score_frame(d, {"c1": 100.0, "c2": 200.0, "c3": 300.0})  # same order, wild scale
    out = blend_rank_scores([a, b], [0.5, 0.5]).sort("ts_code")
    # identical orderings -> blended score equals either model's rank pct
    np.testing.assert_allclose(out["score"].to_numpy(), [1 / 3, 2 / 3, 1.0])


def test_blend_weights_tilt_disagreement():
    d = dt.date(2025, 6, 2)
    a = _score_frame(d, {"c1": 1.0, "c2": 2.0})  # a prefers c2
    b = _score_frame(d, {"c1": 2.0, "c2": 1.0})  # b prefers c1
    heavy_a = blend_rank_scores([a, b], [0.9, 0.1]).sort("ts_code")
    assert heavy_a["score"][1] > heavy_a["score"][0]  # c2 wins under a-heavy blend
    heavy_b = blend_rank_scores([a, b], [0.1, 0.9]).sort("ts_code")
    assert heavy_b["score"][0] > heavy_b["score"][1]


def test_blend_inner_joins_coverage():
    d = dt.date(2025, 6, 2)
    a = _score_frame(d, {"c1": 1.0, "c2": 2.0, "c3": 3.0})
    b = _score_frame(d, {"c1": 1.0, "c2": 2.0})  # b never scored c3
    out = blend_rank_scores([a, b], [0.5, 0.5])
    assert set(out["ts_code"].to_list()) == {"c1", "c2"}


def test_blend_validates_inputs():
    d = dt.date(2025, 6, 2)
    a = _score_frame(d, {"c1": 1.0})
    with pytest.raises(ValueError, match="weights"):
        blend_rank_scores([a], [0.5, 0.5])
    with pytest.raises(ValueError, match="at least one"):
        blend_rank_scores([], [])
    with pytest.raises(ValueError, match="non-negative"):
        blend_rank_scores([a, a], [-1.0, 2.0])


# =============================================================================
# kill_criteria_verdict
# =============================================================================


def _metrics(ic: float, primary: float) -> dict:
    return {"spearman": ic, "primary_mean_top50_proximity_excess": primary}


def test_kill_verdict_keep_when_treatment_wins_two_of_three():
    base = {"H1": _metrics(0.03, 0.010), "H2": _metrics(0.02, 0.008), "Q1": _metrics(0.04, 0.012)}
    treat = {"H1": _metrics(0.05, 0.011), "H2": _metrics(0.04, 0.009), "Q1": _metrics(0.01, 0.005)}
    v = kill_criteria_verdict(base, treat)
    assert v["verdict"] == "KEEP"
    assert v["wins"] == 2 and v["required_wins"] == 2
    assert v["windows"]["Q1"]["win"] is False


def test_kill_verdict_ic_win_alone_is_not_enough():
    # IC up but primary metric degrades -> that window is NOT a win
    base = {"H1": _metrics(0.03, 0.010), "H2": _metrics(0.02, 0.008)}
    treat = {"H1": _metrics(0.05, 0.005), "H2": _metrics(0.04, 0.002)}
    v = kill_criteria_verdict(base, treat)
    assert v["verdict"] == "KILL"
    assert v["wins"] == 0


def test_kill_verdict_requires_common_windows():
    with pytest.raises(ValueError, match="common eval windows"):
        kill_criteria_verdict({"H1": _metrics(0, 0)}, {"H2": _metrics(0, 0)})


def test_kill_verdict_two_windows_requires_both():
    # ceil(2/3 * 2) = 2 -> both windows must be wins
    base = {"H1": _metrics(0.03, 0.010), "H2": _metrics(0.02, 0.008)}
    treat = {"H1": _metrics(0.05, 0.011), "H2": _metrics(0.01, 0.009)}
    assert kill_criteria_verdict(base, treat)["verdict"] == "KILL"
