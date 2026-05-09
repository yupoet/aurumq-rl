"""Tests for Path 1 eval metrics module (spec §3)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from p3.path1_eval import (
    compute_ece_10bin,
    compute_mean_top50_proximity_excess,
    compute_spearman,
    compute_top50_hit_rates,
)


def _mk_eval_frame(rows):
    """rows = list of (date_iso, ts_code, score, actual_y) tuples."""
    return pl.DataFrame(
        [(dt.date.fromisoformat(d), c, s, y) for d, c, s, y in rows],
        schema=["trade_date", "ts_code", "score", "actual_y"],
        orient="row",
    )


def test_mean_top50_proximity_excess_perfect_predictor():
    """Score == actual_y → top-50 each day are exactly the highest-y stocks.

    Build 2 dates × 100 stocks. y = uniform(0, 0.1). score = y exactly.
    Expected: mean_top50_proximity_excess = mean of top-50 actual_y per day,
              averaged across the 2 dates.
    """
    rng = np.random.default_rng(0)
    rows = []
    for d_iso in ("2025-08-01", "2025-08-02"):
        for j in range(100):
            y = float(rng.uniform(0, 0.1))
            rows.append((d_iso, f"S{j:03}.SH", y, y))
    df = _mk_eval_frame(rows)

    result = compute_mean_top50_proximity_excess(df, top_k=50)
    expected = float(
        np.mean([
            df.filter(pl.col("trade_date") == dt.date.fromisoformat(d))
              .sort("score", descending=True).head(50)["actual_y"].mean()
            for d in ("2025-08-01", "2025-08-02")
        ])
    )
    assert result == pytest.approx(expected, abs=1e-9)


def test_mean_top50_proximity_excess_random_predictor():
    """Random score → top-50 daily averages should be close to overall mean of y."""
    rng = np.random.default_rng(1)
    rows = []
    for d_iso in ("2025-08-01",):
        ys = rng.uniform(0, 0.1, size=200)
        scores = rng.uniform(0, 1, size=200)
        for j in range(200):
            rows.append((d_iso, f"S{j:03}.SH", float(scores[j]), float(ys[j])))
    df = _mk_eval_frame(rows)

    result = compute_mean_top50_proximity_excess(df, top_k=50)
    overall_mean = df["actual_y"].mean()
    assert abs(result - overall_mean) < 0.3 * overall_mean


def test_spearman_perfect_correlation():
    """score = actual_y exactly → spearman = 1.0."""
    rows = [
        ("2025-08-01", "A.SH", 0.01, 0.01),
        ("2025-08-01", "B.SH", 0.02, 0.02),
        ("2025-08-01", "C.SH", 0.03, 0.03),
        ("2025-08-01", "D.SH", 0.04, 0.04),
    ]
    df = _mk_eval_frame(rows)
    rho = compute_spearman(df)
    assert rho == pytest.approx(1.0, abs=1e-9)


def test_spearman_anti_correlation():
    """score = -actual_y → spearman = -1.0."""
    rows = [
        ("2025-08-01", "A.SH", 0.04, 0.01),
        ("2025-08-01", "B.SH", 0.03, 0.02),
        ("2025-08-01", "C.SH", 0.02, 0.03),
        ("2025-08-01", "D.SH", 0.01, 0.04),
    ]
    df = _mk_eval_frame(rows)
    rho = compute_spearman(df)
    assert rho == pytest.approx(-1.0, abs=1e-9)


def test_top50_hit_rates_returns_three_values():
    """Hit-rate function returns (T1_hit, T13_hit, T1_avg_excess) tuple of floats."""
    rng = np.random.default_rng(2)
    rows = []
    for d_iso in ("2025-08-01",):
        for j in range(100):
            score = float(rng.uniform(0, 1))
            e1 = float(rng.uniform(-0.05, 0.05))
            e2 = float(rng.uniform(-0.05, 0.05))
            e3 = float(rng.uniform(-0.05, 0.05))
            rows.append((d_iso, f"S{j:03}.SH", score, e1, e2, e3))
    df = pl.DataFrame(
        [(dt.date.fromisoformat(d), c, s, e1, e2, e3) for d, c, s, e1, e2, e3 in rows],
        schema=["trade_date", "ts_code", "score", "e1", "e2", "e3"],
        orient="row",
    )
    out = compute_top50_hit_rates(df, top_k=50)
    assert isinstance(out, dict)
    assert set(out.keys()) >= {"top50_T1_hit_rate", "top50_T13_hit_rate", "top50_T1_avg_excess"}
    assert 0.0 <= out["top50_T1_hit_rate"] <= 1.0
    assert 0.0 <= out["top50_T13_hit_rate"] <= 1.0


def test_ece_perfect_calibration():
    """If predicted = actual, ECE on 10-bin should be ~0."""
    rng = np.random.default_rng(3)
    n = 5000
    actual = rng.uniform(0, 0.1, size=n)
    pred = actual.copy()
    ece = compute_ece_10bin(pred, actual)
    assert ece < 1e-9


def test_ece_constant_prediction_far_from_actual_mean_high_error():
    """Constant prediction far from actual.mean() → all rows in one bin, large |pred-actual|.

    Standard ECE bins by PREDICTION quantile. With a constant prediction, all
    rows fall in one bin and bin's |mean(pred) - mean(actual)| is the gap
    between the constant and the actual mean. Use pred = 0.5 vs actual ~ 0.05
    → ECE ≈ 0.45.
    """
    rng = np.random.default_rng(4)
    n = 5000
    actual = rng.uniform(0, 0.1, size=n)  # mean ≈ 0.05
    pred = np.full(n, 0.5)  # constant, far from actual mean
    ece = compute_ece_10bin(pred, actual)
    assert ece > 0.4  # single bin, |0.5 - 0.05| ≈ 0.45
