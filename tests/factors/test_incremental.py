"""Tests for factor registry incremental protocol."""

from __future__ import annotations

import polars as pl
import pytest

from aurumq_rl.factor_registry import FactorImpl, run_incremental
from aurumq_rl.factors.alpha_momentum import Momentum5dImpl


class MockNonIncrementalFactor(FactorImpl):
    def impl(self, df: pl.DataFrame) -> pl.Series:
        df_sorted = df.sort("ts_code", "trade_date")
        return df_sorted.select(pl.lit(0.0)).to_series()


def test_incremental_dispatch_calls_impl_incremental() -> None:
    factor = Momentum5dImpl()
    df_new = pl.DataFrame({
        "ts_code": ["S1"],
        "trade_date": [1, 2],
        "close": [10.0, 11.0]
    })
    buf = {}
    res, new_buf = run_incremental(factor, df_new, buf, max_window=10)
    assert len(res) == 2
    assert len(new_buf) == 1


def test_incremental_matches_full_panel() -> None:
    # Build a 10-day history for S1 and S2
    dates = list(range(1, 11))
    data = []
    for s in ["S1", "S2"]:
        for d in dates:
            data.append({"ts_code": s, "trade_date": d, "close": float(d + 100)})
    full_df = pl.DataFrame(data)

    # Incremental: only last 3 days
    new_df = full_df.filter(pl.col("trade_date").is_in([8, 9, 10]))
    buf = {
        "S1": full_df.filter(pl.col("ts_code") == "S1").filter(pl.col("trade_date").is_in([1, 2, 3, 4, 5, 6, 7])),
        "S2": full_df.filter(pl.col("ts_code") == "S2").filter(pl.col("trade_date").is_in([1, 2, 3, 4, 5, 6, 7]))
    }

    inc_res, _ = run_incremental(Momentum5dImpl(), new_df, buf, max_window=10)

    # Momentum at d=8 for S1: close(8)/close(3) - 1 = 108/103 - 1
    # Incremental should yield same.
    assert abs(inc_res[0] - 0.04854368932) < 1e-5
    assert abs(inc_res[1] - 0.04807692308) < 1e-5
    assert abs(inc_res[2] - 0.04761904762) < 1e-5
