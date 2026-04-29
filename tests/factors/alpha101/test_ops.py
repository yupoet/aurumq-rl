"""Contract tests for ``aurumq_rl.factors.alpha101._ops``.

Each operator gets at least one test that covers:

* output dtype (Float64 for numeric ops)
* output length matches panel height
* edge case: rolling windows produce null for the first ``window-1`` rows
  per stock partition
* numerical correctness: a hand-computed expected value where feasible

Tests are grouped by operator class (rolling / element-wise / cs / cond).
The synthetic_panel fixture (10 stocks × 60 days, seed=42) lives in
``tests/factors/conftest.py``.
"""
from __future__ import annotations

import math

import polars as pl
import pytest

from aurumq_rl.factors.alpha101._ops import (
    CS_PART,
    TS_PART,
    abs_,
    clip_,
    count_,
    cs_rank,
    cs_scale,
    delay,
    delta,
    if_then_else,
    ind_neutralize,
    log1p,
    log_,
    power,
    sign_,
    signed_power,
    sma,
    sumif,
    ts_argmax,
    ts_argmin,
    ts_corr,
    ts_cov,
    ts_decay_linear,
    ts_kurt,
    ts_max,
    ts_mean,
    ts_median,
    ts_min,
    ts_product,
    ts_rank,
    ts_skew,
    ts_std,
    ts_sum,
    ts_zscore,
    wma,
)

# ---------------------------------------------------------------------------
# Tiny deterministic panel for hand-computed numerical expectations.
# Two stocks (A, B), 5 trading days each — ordered ascending.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "stock_code": ["A"] * 5 + ["B"] * 5,
            "trade_date": list(range(5)) * 2,
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "industry": ["T"] * 5 + ["F"] * 5,
        }
    )


def _apply(df: pl.DataFrame, expr: pl.Expr, name: str = "out") -> list:
    """Helper: materialise an expression and return its values as a list."""
    return df.with_columns(expr.alias(name))[name].to_list()


# ===========================================================================
# Rolling — fast path
# ===========================================================================


class TestTsMean:
    def test_dtype_and_length(self, synthetic_panel):
        out = synthetic_panel.with_columns(
            ts_mean(pl.col("close"), 5).alias("o")
        )
        assert out["o"].dtype == pl.Float64
        assert out.height == synthetic_panel.height

    def test_first_window_minus_one_is_null(self, synthetic_panel):
        df = synthetic_panel.with_columns(
            ts_mean(pl.col("close"), 5).alias("o")
        ).sort([TS_PART, "trade_date"])
        first_per_stock = df.group_by(TS_PART, maintain_order=True).head(4)
        assert first_per_stock["o"].null_count() == first_per_stock.height

    def test_known_values(self, tiny_panel):
        out = _apply(tiny_panel, ts_mean(pl.col("x"), 3))
        # A: NaN, NaN, mean(1,2,3)=2, mean(2,3,4)=3, mean(3,4,5)=4
        # B: NaN, NaN, mean(5,4,3)=4, mean(4,3,2)=3, mean(3,2,1)=2
        assert out[:5] == [None, None, 2.0, 3.0, 4.0]
        assert out[5:] == [None, None, 4.0, 3.0, 2.0]


class TestTsSum:
    def test_known_values(self, tiny_panel):
        out = _apply(tiny_panel, ts_sum(pl.col("x"), 3))
        assert out[:5] == [None, None, 6.0, 9.0, 12.0]
        assert out[5:] == [None, None, 12.0, 9.0, 6.0]


class TestTsStd:
    def test_dtype(self, synthetic_panel):
        out = synthetic_panel.with_columns(ts_std(pl.col("close"), 5).alias("o"))
        assert out["o"].dtype == pl.Float64

    def test_known_value(self, tiny_panel):
        out = _apply(tiny_panel, ts_std(pl.col("x"), 3))
        # std(1,2,3) sample = 1.0
        assert out[2] == pytest.approx(1.0)
        assert out[3] == pytest.approx(1.0)


class TestTsMin:
    def test_known_values(self, tiny_panel):
        out = _apply(tiny_panel, ts_min(pl.col("x"), 3))
        assert out[2:5] == [1.0, 2.0, 3.0]
        assert out[7:] == [3.0, 2.0, 1.0]


class TestTsMax:
    def test_known_values(self, tiny_panel):
        out = _apply(tiny_panel, ts_max(pl.col("x"), 3))
        assert out[2:5] == [3.0, 4.0, 5.0]
        assert out[7:] == [5.0, 4.0, 3.0]


class TestTsMedian:
    def test_known_values(self, tiny_panel):
        out = _apply(tiny_panel, ts_median(pl.col("x"), 3))
        assert out[2:5] == [2.0, 3.0, 4.0]


class TestTsSkew:
    def test_dtype_and_nulls(self, synthetic_panel):
        out = synthetic_panel.with_columns(ts_skew(pl.col("close"), 10).alias("o"))
        assert out["o"].dtype == pl.Float64
        # First 9 per stock should be null.
        first = out.sort([TS_PART, "trade_date"]).group_by(TS_PART, maintain_order=True).head(9)
        assert first["o"].null_count() == first.height


class TestTsKurt:
    def test_dtype(self, synthetic_panel):
        out = synthetic_panel.with_columns(ts_kurt(pl.col("close"), 10).alias("o"))
        assert out["o"].dtype == pl.Float64


class TestTsZscore:
    def test_known_value(self, tiny_panel):
        out = _apply(tiny_panel, ts_zscore(pl.col("x"), 3))
        # window (1,2,3) -> mean=2, std=1, last value=3 -> z=(3-2)/1=1
        assert out[2] == pytest.approx(1.0)
        # window (2,3,4) -> z=1
        assert out[3] == pytest.approx(1.0)


# ===========================================================================
# Rolling — slow path (rolling_map fallbacks)
# ===========================================================================


class TestTsArgmax:
    def test_known_values(self, tiny_panel):
        out = _apply(tiny_panel, ts_argmax(pl.col("x"), 3))
        # For stock A, x is monotone increasing — max is always last (index 2).
        assert out[2:5] == [2.0, 2.0, 2.0]
        # For stock B, x is monotone decreasing — max is always first (index 0).
        assert out[7:] == [0.0, 0.0, 0.0]

    def test_first_max_on_tie(self):
        df = pl.DataFrame(
            {
                "stock_code": ["A"] * 5,
                "trade_date": [1, 2, 3, 4, 5],
                "x": [1.0, 2.0, 2.0, 2.0, 2.0],
            }
        )
        # Window at idx 4 = [2,2,2] → first index = 0 (polars convention)
        out = _apply(df, ts_argmax(pl.col("x"), 3))
        assert out[4] == 0.0

    def test_dtype_float64(self, synthetic_panel):
        out = synthetic_panel.with_columns(ts_argmax(pl.col("close"), 5).alias("o"))
        assert out["o"].dtype == pl.Float64


class TestTsArgmin:
    def test_known_values(self, tiny_panel):
        out = _apply(tiny_panel, ts_argmin(pl.col("x"), 3))
        # Stock A monotone increasing — min is always first (index 0)
        assert out[2:5] == [0.0, 0.0, 0.0]
        # Stock B monotone decreasing — min is last (index 2)
        assert out[7:] == [2.0, 2.0, 2.0]


class TestTsRank:
    def test_known_values(self, tiny_panel):
        out = _apply(tiny_panel, ts_rank(pl.col("x"), 3))
        # Stock A: window (1,2,3), last=3 → rank 3 → (3-1)/2 = 1.0
        assert out[2] == pytest.approx(1.0)
        # Stock B: window (5,4,3), last=3 → rank 1 → (1-1)/2 = 0.0
        assert out[7] == pytest.approx(0.0)

    def test_range_in_unit_interval(self, synthetic_panel):
        df = synthetic_panel.with_columns(ts_rank(pl.col("close"), 10).alias("o"))
        non_null = df["o"].drop_nulls()
        assert non_null.min() >= 0.0
        assert non_null.max() <= 1.0


class TestTsDecayLinear:
    def test_known_value(self, tiny_panel):
        out = _apply(tiny_panel, ts_decay_linear(pl.col("x"), 3))
        # Weights: latest=3, mid=2, oldest=1, sum=6
        # Stock A window (1,2,3) at idx2: (3*3 + 2*2 + 1*1)/6 = 14/6
        assert out[2] == pytest.approx(14 / 6)
        # idx 3: window (2,3,4): (3*4 + 2*3 + 1*2)/6 = 20/6
        assert out[3] == pytest.approx(20 / 6)

    def test_first_window_minus_one_null(self, synthetic_panel):
        df = synthetic_panel.with_columns(
            ts_decay_linear(pl.col("close"), 5).alias("o")
        ).sort([TS_PART, "trade_date"])
        first = df.group_by(TS_PART, maintain_order=True).head(4)
        assert first["o"].null_count() == first.height


class TestTsProduct:
    def test_known_value(self, tiny_panel):
        out = _apply(tiny_panel, ts_product(pl.col("x"), 3))
        # Stock A window (1,2,3) → 6
        assert out[2] == pytest.approx(6.0, rel=1e-9)
        # Stock A window (2,3,4) → 24
        assert out[3] == pytest.approx(24.0, rel=1e-9)


class TestTsCorrCov:
    def test_corr_perfect_positive(self, tiny_panel):
        # Stock A: x=(1..5), y=(2..10) — linearly perfectly correlated
        out = _apply(tiny_panel, ts_corr(pl.col("x"), pl.col("y"), 3))
        assert out[2] == pytest.approx(1.0)
        assert out[3] == pytest.approx(1.0)

    def test_cov_known_value(self, tiny_panel):
        # cov(x,y) for window (1,2,3),(2,4,6) sample ddof=1 = sum((x-2)(y-4))/2
        # = ((-1)(-2) + 0 + (1)(2))/2 = 4/2 = 2
        out = _apply(tiny_panel, ts_cov(pl.col("x"), pl.col("y"), 3))
        assert out[2] == pytest.approx(2.0)


class TestCount:
    def test_count_full_window(self, tiny_panel):
        out = _apply(tiny_panel, count_(pl.col("x"), 3))
        # Partial windows return partial counts (min_samples=1)
        assert out[0] == 1.0
        assert out[1] == 2.0
        assert out[2] == 3.0
        assert out[3] == 3.0


class TestSumif:
    def test_known_value(self, tiny_panel):
        out = _apply(
            tiny_panel,
            sumif(pl.col("x"), pl.col("x") > 1.0, 3),
        )
        # Stock A window (1,2,3): sum where >1 is 2+3=5 at idx 2
        assert out[2] == 5.0
        # Window (2,3,4) -> 9
        assert out[3] == 9.0


class TestSmaWma:
    def test_sma_alias_of_ts_mean(self, tiny_panel):
        a = _apply(tiny_panel, sma(pl.col("x"), 3))
        b = _apply(tiny_panel, ts_mean(pl.col("x"), 3))
        assert a == b

    def test_wma_alias_of_decay_linear(self, tiny_panel):
        a = _apply(tiny_panel, wma(pl.col("x"), 3))
        b = _apply(tiny_panel, ts_decay_linear(pl.col("x"), 3))
        # Both should match exactly
        for av, bv in zip(a, b, strict=True):
            if av is None and bv is None:
                continue
            assert av == pytest.approx(bv)


# ===========================================================================
# Delay / Delta
# ===========================================================================


class TestDelay:
    def test_known_values(self, tiny_panel):
        out = _apply(tiny_panel, delay(pl.col("x"), 1))
        # Stock A: shift 1 → [None, 1, 2, 3, 4]
        assert out[:5] == [None, 1.0, 2.0, 3.0, 4.0]
        assert out[5:] == [None, 5.0, 4.0, 3.0, 2.0]

    def test_partition_isolation(self, tiny_panel):
        # First row of stock B must be None (no leak from stock A's last row)
        out = _apply(tiny_panel, delay(pl.col("x"), 1))
        assert out[5] is None


class TestDelta:
    def test_known_values(self, tiny_panel):
        out = _apply(tiny_panel, delta(pl.col("x"), 1))
        # Stock A: x diffs → [None, 1, 1, 1, 1]
        assert out[:5] == [None, 1.0, 1.0, 1.0, 1.0]
        assert out[5:] == [None, -1.0, -1.0, -1.0, -1.0]


# ===========================================================================
# Element-wise scalar transforms
# ===========================================================================


class TestElementwise:
    def test_abs(self, tiny_panel):
        df = tiny_panel.with_columns((pl.col("x") - 3.0).alias("z"))
        out = _apply(df, abs_(pl.col("z")))
        assert out == [2.0, 1.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 1.0, 2.0]

    def test_log(self, tiny_panel):
        out = _apply(tiny_panel, log_(pl.col("x")))
        assert out[0] == pytest.approx(math.log(1.0))
        assert out[4] == pytest.approx(math.log(5.0))

    def test_log1p(self, tiny_panel):
        out = _apply(tiny_panel, log1p(pl.col("x")))
        assert out[0] == pytest.approx(math.log(2.0))

    def test_sign(self, tiny_panel):
        df = tiny_panel.with_columns((pl.col("x") - 3.0).alias("z"))
        out = _apply(df, sign_(pl.col("z")))
        # x=[1,2,3,4,5] → z-3 = [-2,-1,0,1,2] → signs [-1,-1,0,1,1]
        assert out[:5] == [-1.0, -1.0, 0.0, 1.0, 1.0]

    def test_signed_power(self, tiny_panel):
        df = tiny_panel.with_columns((pl.col("x") - 3.0).alias("z"))
        out = _apply(df, signed_power(pl.col("z"), 2.0))
        # z=[-2,-1,0,1,2] → signed_power = [-4,-1,0,1,4]
        assert out[:5] == [-4.0, -1.0, 0.0, 1.0, 4.0]

    def test_power(self, tiny_panel):
        out = _apply(tiny_panel, power(pl.col("x"), pl.lit(2.0)))
        assert out[:5] == [1.0, 4.0, 9.0, 16.0, 25.0]

    def test_clip(self, tiny_panel):
        out = _apply(tiny_panel, clip_(pl.col("x"), 2.0, 4.0))
        assert out[:5] == [2.0, 2.0, 3.0, 4.0, 4.0]


# ===========================================================================
# Cross-section
# ===========================================================================


class TestCsRank:
    def test_dtype_and_range(self, synthetic_panel):
        df = synthetic_panel.with_columns(cs_rank(pl.col("close")).alias("r"))
        assert df["r"].dtype == pl.Float64
        non_null = df["r"].drop_nulls()
        assert non_null.min() > 0.0
        assert non_null.max() <= 1.0

    def test_known_values(self):
        # Single trade_date, three stocks with x=[10, 20, 30]
        df = pl.DataFrame(
            {
                "stock_code": ["A", "B", "C"],
                "trade_date": [1, 1, 1],
                "x": [10.0, 20.0, 30.0],
            }
        )
        out = _apply(df, cs_rank(pl.col("x")))
        # avg ranks 1,2,3 / 3 = 0.333, 0.666, 1.0
        assert out == pytest.approx([1 / 3, 2 / 3, 1.0])

    def test_uses_pl_len_not_count_deprecated(self):
        # Smoke test: simply running the operator must not emit pl.count
        # deprecation warnings on polars 1.40+.
        df = pl.DataFrame(
            {
                "stock_code": ["A", "B"],
                "trade_date": [1, 1],
                "x": [1.0, 2.0],
            }
        )
        # If pl.count were used, polars 1.40 emits a DeprecationWarning. This
        # captures any such warning into the test record.
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df.with_columns(cs_rank(pl.col("x")).alias("r"))
        for w in caught:
            assert "pl.count" not in str(w.message)


class TestCsScale:
    def test_unit_gross_per_day(self, synthetic_panel):
        df = synthetic_panel.with_columns(
            cs_scale(pl.col("close"), scale=1.0).alias("s")
        )
        # Per trade_date, sum(|s|) should equal 1.0
        per_day = df.group_by(CS_PART).agg(pl.col("s").abs().sum().alias("gross"))
        assert (per_day["gross"] - 1.0).abs().max() < 1e-9

    def test_known_values(self):
        df = pl.DataFrame(
            {
                "stock_code": ["A", "B", "C"],
                "trade_date": [1, 1, 1],
                "x": [1.0, 2.0, 3.0],
            }
        )
        out = _apply(df, cs_scale(pl.col("x"), scale=1.0))
        # sum(|x|) = 6 → x/6
        assert out == pytest.approx([1 / 6, 2 / 6, 3 / 6])


class TestIndNeutralize:
    def test_zero_mean_per_industry_per_day(self):
        df = pl.DataFrame(
            {
                "stock_code": ["A", "B", "C", "D"] * 2,
                "trade_date": [1, 1, 1, 1, 2, 2, 2, 2],
                "x": [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0],
                "industry": ["T", "T", "F", "F", "T", "T", "F", "F"],
            }
        )
        out = df.with_columns(ind_neutralize(pl.col("x"), "industry").alias("n"))
        per_cell = out.group_by([CS_PART, "industry"]).agg(pl.col("n").sum())
        assert per_cell["n"].abs().max() < 1e-9

    def test_accepts_expr_group(self):
        # Group provided as Expr instead of column name.
        df = pl.DataFrame(
            {
                "stock_code": ["A", "B"],
                "trade_date": [1, 1],
                "x": [2.0, 4.0],
                "industry": ["T", "T"],
            }
        )
        out = _apply(df, ind_neutralize(pl.col("x"), pl.col("industry")))
        # Both in same cell, mean=3 → [-1, 1]
        assert out == [-1.0, 1.0]


# ===========================================================================
# Conditional helper
# ===========================================================================


class TestIfThenElse:
    def test_with_expr_branches(self, tiny_panel):
        out = _apply(
            tiny_panel,
            if_then_else(pl.col("x") > 3.0, pl.col("x"), pl.lit(0.0)),
        )
        assert out[:5] == [0.0, 0.0, 0.0, 4.0, 5.0]

    def test_with_scalar_branches(self, tiny_panel):
        out = _apply(
            tiny_panel,
            if_then_else(pl.col("x") > 3.0, 1.0, -1.0),
        )
        assert out[:5] == [-1.0, -1.0, -1.0, 1.0, 1.0]


# ===========================================================================
# Synthetic-panel sanity — ensure ops compose on the real fixture without err
# ===========================================================================


def test_ops_compose_on_synthetic_panel(synthetic_panel):
    """Smoke test: chain a few operators end-to-end on the real fixture.

    Mimics how alpha101 factors actually compose. If this raises, alpha
    sub-agents will hit the same failure when wiring up new factors.
    """
    out = synthetic_panel.with_columns(
        signed_power(
            if_then_else(
                pl.col("returns") < 0,
                ts_std(pl.col("returns"), 20),
                pl.col("close"),
            ),
            2.0,
        )
        .alias("composite")
    )
    assert out["composite"].dtype == pl.Float64
    assert out.height == synthetic_panel.height


def test_partition_keys_constants():
    assert TS_PART == "stock_code"
    assert CS_PART == "trade_date"
