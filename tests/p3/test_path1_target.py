"""Tests for proximity-weighted target_y formula (spec §2)."""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from p3.path1_target import compute_target_y


def _mk_realized(rows):
    """Helper: rows = list of (date_iso, ts_code, pct_chg_t_plus_1) tuples."""
    return pl.DataFrame(
        [(dt.date.fromisoformat(d), c, r) for d, c, r in rows],
        schema=["trade_date", "ts_code", "pct_chg_t_plus_1"],
        orient="row",
    )


def _mk_market(rows):
    """Helper: rows = list of (date_iso, eq_weight_pct_chg_t_plus_1) tuples."""
    return pl.DataFrame(
        [(dt.date.fromisoformat(d), r) for d, r in rows],
        schema=["trade_date", "eq_weight_pct_chg_t_plus_1"],
        orient="row",
    )


def test_y_zero_when_excess_all_negative():
    """All forward excess returns negative → max(0,·) clips to 0 → y=0."""
    realized = _mk_realized([
        ("2024-01-02", "600001.SH", -0.01),  # T+1 excess = -0.01 - 0.005 = -0.015
        ("2024-01-03", "600001.SH", -0.02),  # T+2 (= 2024-01-03's T+1)
        ("2024-01-04", "600001.SH", -0.03),  # T+3
    ])
    market = _mk_market([
        ("2024-01-02", 0.005),
        ("2024-01-03", 0.005),
        ("2024-01-04", 0.005),
    ])
    out = compute_target_y(realized, market)
    row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 2)).row(0, named=True)
    assert row["y"] == pytest.approx(0.0)


def test_y_proximity_weighted_when_only_t_plus_1_positive():
    """T+1 excess = +0.05, T+2/T+3 excess = 0 (after market subtraction).

    Expected: y = (1.0 * 0.05 + 0.6 * 0 + 0.3 * 0) / 1.9 = 0.05 / 1.9 ≈ 0.02632
    """
    realized = _mk_realized([
        ("2024-01-02", "600001.SH", 0.05),
        ("2024-01-03", "600001.SH", 0.0),
        ("2024-01-04", "600001.SH", 0.0),
    ])
    market = _mk_market([
        ("2024-01-02", 0.0),
        ("2024-01-03", 0.0),
        ("2024-01-04", 0.0),
    ])
    out = compute_target_y(realized, market)
    row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 2)).row(0, named=True)
    assert row["y"] == pytest.approx(0.05 / 1.9, abs=1e-6)


def test_y_full_proximity_pattern():
    """T+1 = +0.04, T+2 = +0.02, T+3 = +0.01 (all excess after market).

    Expected: y = (1.0*0.04 + 0.6*0.02 + 0.3*0.01) / 1.9 = 0.055 / 1.9 ≈ 0.02895
    """
    realized = _mk_realized([
        ("2024-01-02", "600001.SH", 0.04),
        ("2024-01-03", "600001.SH", 0.02),
        ("2024-01-04", "600001.SH", 0.01),
    ])
    market = _mk_market([
        ("2024-01-02", 0.0),
        ("2024-01-03", 0.0),
        ("2024-01-04", 0.0),
    ])
    out = compute_target_y(realized, market)
    row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 2)).row(0, named=True)
    assert row["y"] == pytest.approx(0.055 / 1.9, abs=1e-6)


def test_y_at_panel_boundary_drops_when_t_plus_3_missing():
    """For trade_dates near the end of the panel, T+3 doesn't exist → drop them.

    With a 3-date fixture (Jan2, Jan3, Jan4) and the convention that
    T+1 == anchor's own row, T+2 == anchor+1's row, T+3 == anchor+2's row:
      - Jan2: T+1=Jan2 ✓  T+2=Jan3 ✓  T+3=Jan4 ✓  → keep
      - Jan3: T+1=Jan3 ✓  T+2=Jan4 ✓  T+3=Jan5 (missing) → drop
      - Jan4: T+2=Jan5 (missing) → drop

    Output should contain ONLY Jan2.
    """
    realized = _mk_realized([
        ("2024-01-02", "600001.SH", 0.01),
        ("2024-01-03", "600001.SH", 0.01),
        ("2024-01-04", "600001.SH", 0.01),
    ])
    market = _mk_market([
        ("2024-01-02", 0.0),
        ("2024-01-03", 0.0),
        ("2024-01-04", 0.0),
    ])
    out = compute_target_y(realized, market)
    out_dates = sorted(out["trade_date"].unique().to_list())
    assert out_dates == [dt.date(2024, 1, 2)]


def test_y_max_zero_clipping_per_horizon():
    """T+1 = +0.05, T+2 = -0.03, T+3 = +0.02.

    Each horizon clipped at 0 BEFORE weighting: T+2 contributes 0 not -0.03.
    Expected: y = (1.0*0.05 + 0.6*0 + 0.3*0.02) / 1.9 ≈ 0.02947
    """
    realized = _mk_realized([
        ("2024-01-02", "600001.SH", 0.05),
        ("2024-01-03", "600001.SH", -0.03),
        ("2024-01-04", "600001.SH", 0.02),
    ])
    market = _mk_market([
        ("2024-01-02", 0.0),
        ("2024-01-03", 0.0),
        ("2024-01-04", 0.0),
    ])
    out = compute_target_y(realized, market)
    row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 2)).row(0, named=True)
    assert row["y"] == pytest.approx((0.05 + 0.006) / 1.9, abs=1e-6)


def test_y_subtracts_market_per_horizon():
    """T+1 = +0.05, market T+1 = +0.02 → excess T+1 = +0.03.

    Per-horizon market subtraction (not constant market across the window).
    Expected: y = (1.0 * 0.03 + 0 + 0) / 1.9 ≈ 0.01579
    """
    realized = _mk_realized([
        ("2024-01-02", "600001.SH", 0.05),
        ("2024-01-03", "600001.SH", 0.0),
        ("2024-01-04", "600001.SH", 0.0),
    ])
    market = _mk_market([
        ("2024-01-02", 0.02),
        ("2024-01-03", 0.0),
        ("2024-01-04", 0.0),
    ])
    out = compute_target_y(realized, market)
    row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 2)).row(0, named=True)
    assert row["y"] == pytest.approx(0.03 / 1.9, abs=1e-6)


def test_y_two_stocks_independent():
    """Two stocks on same dates → each stock's y depends only on its own returns.

    Pins the self-join key including ts_code: if ts_code were dropped from
    the join key, stock A's T+2 could pick up stock B's pct_chg, contaminating y.
    Stock A has all positive returns, stock B all zero — they should produce
    distinctly different y values.
    """
    realized = _mk_realized([
        ("2024-01-02", "600001.SH", 0.05),
        ("2024-01-02", "600002.SH", 0.0),
        ("2024-01-03", "600001.SH", 0.05),
        ("2024-01-03", "600002.SH", 0.0),
        ("2024-01-04", "600001.SH", 0.05),
        ("2024-01-04", "600002.SH", 0.0),
    ])
    market = _mk_market([
        ("2024-01-02", 0.0),
        ("2024-01-03", 0.0),
        ("2024-01-04", 0.0),
    ])
    out = compute_target_y(realized, market)
    a = out.filter(
        (pl.col("trade_date") == dt.date(2024, 1, 2)) & (pl.col("ts_code") == "600001.SH")
    ).row(0, named=True)
    b = out.filter(
        (pl.col("trade_date") == dt.date(2024, 1, 2)) & (pl.col("ts_code") == "600002.SH")
    ).row(0, named=True)
    # Stock A: all 0.05 → y = (1.0 + 0.6 + 0.3) * 0.05 / 1.9 = 0.05 (note: weights sum to 1.9)
    assert a["y"] == pytest.approx(0.05, abs=1e-6)
    # Stock B: all 0.0 → y = 0
    assert b["y"] == pytest.approx(0.0, abs=1e-6)
