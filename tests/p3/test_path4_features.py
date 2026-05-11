"""Tests for Path 4 cross-sectional rank-z feature transform."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from p3.path4_features import cross_sectional_rank_z


def _mk_panel(rows):
    """rows = list of (date, ts_code, alpha_001, alpha_002) tuples."""
    return pl.DataFrame(
        [(dt.date.fromisoformat(d), c, a1, a2) for d, c, a1, a2 in rows],
        schema=["trade_date", "ts_code", "alpha_001", "alpha_002"],
        orient="row",
    )


def _mk_universe(rows):
    """rows = list of (date, ts_code, in_universe) tuples."""
    return pl.DataFrame(
        [(dt.date.fromisoformat(d), c, u) for d, c, u in rows],
        schema=["trade_date", "ts_code", "in_universe"],
        orient="row",
    )


def test_rank_z_strict_monotone_input():
    """5 in-universe stocks, alpha_001 = 1, 2, 3, 4, 5 → rank-z = -1, -0.5, 0, 0.5, 1.

    rank_z = (rank - 1) / (N - 1) * 2 - 1, with N=5 and rank ∈ [1, 5]:
        rank=1 → 0/4 * 2 - 1 = -1
        rank=2 → 1/4 * 2 - 1 = -0.5
        rank=3 → 2/4 * 2 - 1 =  0
        rank=4 → 3/4 * 2 - 1 = +0.5
        rank=5 → 4/4 * 2 - 1 = +1
    """
    panel = _mk_panel(
        [
            ("2024-01-02", "S1.SH", 1.0, 0.0),
            ("2024-01-02", "S2.SH", 2.0, 0.0),
            ("2024-01-02", "S3.SH", 3.0, 0.0),
            ("2024-01-02", "S4.SH", 4.0, 0.0),
            ("2024-01-02", "S5.SH", 5.0, 0.0),
        ]
    )
    uni = _mk_universe(
        [
            ("2024-01-02", "S1.SH", True),
            ("2024-01-02", "S2.SH", True),
            ("2024-01-02", "S3.SH", True),
            ("2024-01-02", "S4.SH", True),
            ("2024-01-02", "S5.SH", True),
        ]
    )
    out = cross_sectional_rank_z(panel, uni, ["alpha_001", "alpha_002"])
    out = out.sort("ts_code")
    assert out["alpha_001"].to_list() == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0])


def test_rank_z_independent_per_day():
    """Two days with same input ordering → same rank-z output per day."""
    panel = _mk_panel(
        [
            ("2024-01-02", "S1.SH", 1.0, 0.0),
            ("2024-01-02", "S2.SH", 2.0, 0.0),
            ("2024-01-03", "S1.SH", 100.0, 0.0),  # Different absolute scale, same ordering
            ("2024-01-03", "S2.SH", 200.0, 0.0),
        ]
    )
    uni = _mk_universe(
        [
            ("2024-01-02", "S1.SH", True),
            ("2024-01-02", "S2.SH", True),
            ("2024-01-03", "S1.SH", True),
            ("2024-01-03", "S2.SH", True),
        ]
    )
    out = cross_sectional_rank_z(panel, uni, ["alpha_001"]).sort(["trade_date", "ts_code"])
    # Each day: 2 stocks, ranks 1 and 2 → rank_z = -1 and +1
    assert out["alpha_001"].to_list() == pytest.approx([-1.0, 1.0, -1.0, 1.0])


def test_rank_z_excludes_out_of_universe():
    """Out-of-universe stock: not ranked, value = 0.0 (neutral). Doesn't affect in-uni ranks."""
    panel = _mk_panel(
        [
            ("2024-01-02", "S1.SH", 1.0, 0.0),
            ("2024-01-02", "S2.SH", 2.0, 0.0),
            ("2024-01-02", "S3.SH", 999.0, 0.0),  # OUT of universe; should NOT shift S1/S2 ranks
        ]
    )
    uni = _mk_universe(
        [
            ("2024-01-02", "S1.SH", True),
            ("2024-01-02", "S2.SH", True),
            ("2024-01-02", "S3.SH", False),
        ]
    )
    out = cross_sectional_rank_z(panel, uni, ["alpha_001"]).sort("ts_code")
    # S1/S2 should be -1, +1 (only 2 in-uni stocks).
    # S3 should be 0.0 (out-of-universe sentinel).
    rows = out.to_dicts()
    by_code = {r["ts_code"]: r for r in rows}
    assert by_code["S1.SH"]["alpha_001"] == pytest.approx(-1.0)
    assert by_code["S2.SH"]["alpha_001"] == pytest.approx(1.0)
    assert by_code["S3.SH"]["alpha_001"] == pytest.approx(0.0)


def test_rank_z_handles_extreme_outliers():
    """alpha_001 = 1, 2, 3, 1e38 (overflow) → rank-z still -1, -1/3, +1/3, +1.

    The whole point of rank transform is robustness to extreme values.
    """
    panel = _mk_panel(
        [
            ("2024-01-02", "S1.SH", 1.0, 0.0),
            ("2024-01-02", "S2.SH", 2.0, 0.0),
            ("2024-01-02", "S3.SH", 3.0, 0.0),
            ("2024-01-02", "S4.SH", 1e38, 0.0),
        ]
    )
    uni = _mk_universe(
        [
            ("2024-01-02", "S1.SH", True),
            ("2024-01-02", "S2.SH", True),
            ("2024-01-02", "S3.SH", True),
            ("2024-01-02", "S4.SH", True),
        ]
    )
    out = cross_sectional_rank_z(panel, uni, ["alpha_001"]).sort("ts_code")
    # ranks 1,2,3,4 over N=4: rank_z = (rank-1)/3 * 2 - 1
    # rank=1 → -1, rank=2 → -1/3, rank=3 → +1/3, rank=4 → +1
    assert out["alpha_001"].to_list() == pytest.approx([-1.0, -1 / 3, 1 / 3, 1.0], abs=1e-6)
