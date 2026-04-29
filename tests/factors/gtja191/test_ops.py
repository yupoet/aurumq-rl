"""Unit tests for the GTJA-191 polars operator library.

These tests pin the numerical semantics of every operator in
:mod:`aurumq_rl.factors.gtja191._ops` against an independent, hand-rolled
numpy/pandas computation. The reference behaviour deliberately mirrors
the Daic115/alpha191 oracle so that downstream factor implementations
have a stable foundation to test against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from aurumq_rl.factors.gtja191 import _ops

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _per_stock_array(df: pl.DataFrame, col: str) -> np.ndarray:
    """Return the values of ``col`` for the first stock as a numpy array."""
    first_stock = df["stock_code"][0]
    return df.filter(pl.col("stock_code") == first_stock)[col].to_numpy()


# ---------------------------------------------------------------------------
# Cross-section operators
# ---------------------------------------------------------------------------


class TestRank:
    def test_dtype_is_float64(self, synthetic_panel):
        out = synthetic_panel.with_columns(_ops.rank(pl.col("close")).alias("r"))
        assert out["r"].dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        out = synthetic_panel.with_columns(_ops.rank(pl.col("close")).alias("r"))
        assert len(out) == synthetic_panel.height

    def test_range_is_zero_to_one(self, synthetic_panel):
        out = synthetic_panel.with_columns(_ops.rank(pl.col("close")).alias("r"))
        non_null = out["r"].drop_nulls()
        assert non_null.min() > 0.0
        assert non_null.max() <= 1.0

    def test_matches_pandas_pct_rank(self, synthetic_panel):
        """``rank`` must equal ``pandas.rank(axis=1, pct=True)`` row-by-row."""
        wide = synthetic_panel.to_pandas().pivot(
            index="trade_date", columns="stock_code", values="close"
        )
        expected = wide.rank(axis=1, pct=True)
        ours = (
            synthetic_panel.with_columns(_ops.rank(pl.col("close")).alias("r"))
            .to_pandas()
            .pivot(index="trade_date", columns="stock_code", values="r")
        )
        diff = (expected - ours).abs().max().max()
        assert diff < 1e-12

    def test_handcomputed_three_stocks(self):
        """Hand-computed: 3 stocks on 1 day with values [10, 30, 20]."""
        df = pl.DataFrame(
            {
                "stock_code": ["A", "B", "C"],
                "trade_date": [pd.Timestamp("2024-01-02").date()] * 3,
                "x": [10.0, 30.0, 20.0],
            }
        )
        out = df.with_columns(_ops.rank(pl.col("x")).alias("r"))
        # ranks are 1, 3, 2 → / 3 → 0.333, 1.0, 0.667
        np.testing.assert_allclose(
            out["r"].to_numpy(),
            np.array([1, 3, 2]) / 3.0,
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Vectorised rolling operators
# ---------------------------------------------------------------------------


class TestMean:
    def test_matches_pandas_rolling_mean(self, synthetic_panel):
        ours = synthetic_panel.with_columns(_ops.mean(pl.col("close"), 5).alias("m"))
        y = _per_stock_array(synthetic_panel, "close")
        expected = pd.Series(y).rolling(5).mean().to_numpy()
        ours_arr = _per_stock_array(ours, "m")
        mask = ~np.isnan(expected) & ~np.isnan(ours_arr)
        np.testing.assert_allclose(ours_arr[mask], expected[mask], rtol=1e-12)

    def test_handcomputed(self):
        df = pl.DataFrame(
            {
                "stock_code": ["A"] * 5,
                "trade_date": pd.date_range("2024-01-02", periods=5).date,
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        out = df.with_columns(_ops.mean(pl.col("x"), 3).alias("m"))
        # Window of 3: rolling means are nan, nan, 2.0, 3.0, 4.0
        ours = out["m"].to_list()
        assert ours[0] is None and ours[1] is None
        assert ours[2:] == [2.0, 3.0, 4.0]


class TestSum:
    def test_matches_pandas_rolling_sum(self, synthetic_panel):
        ours = synthetic_panel.with_columns(_ops.sum_(pl.col("volume"), 5).alias("s"))
        v = _per_stock_array(synthetic_panel, "volume")
        expected = pd.Series(v).rolling(5).sum().to_numpy()
        ours_arr = _per_stock_array(ours, "s")
        mask = ~np.isnan(expected) & ~np.isnan(ours_arr)
        np.testing.assert_allclose(ours_arr[mask], expected[mask], rtol=1e-12)


class TestStd:
    def test_matches_pandas_rolling_std(self, synthetic_panel):
        ours = synthetic_panel.with_columns(_ops.std_(pl.col("close"), 5).alias("s"))
        y = _per_stock_array(synthetic_panel, "close")
        expected = pd.Series(y).rolling(5).std().to_numpy()
        ours_arr = _per_stock_array(ours, "s")
        mask = ~np.isnan(expected) & ~np.isnan(ours_arr)
        np.testing.assert_allclose(ours_arr[mask], expected[mask], rtol=1e-10)


class TestTsMin:
    def test_matches_pandas_rolling_min(self, synthetic_panel):
        ours = synthetic_panel.with_columns(_ops.ts_min(pl.col("close"), 5).alias("m"))
        y = _per_stock_array(synthetic_panel, "close")
        expected = pd.Series(y).rolling(5).min().to_numpy()
        ours_arr = _per_stock_array(ours, "m")
        mask = ~np.isnan(expected) & ~np.isnan(ours_arr)
        np.testing.assert_allclose(ours_arr[mask], expected[mask], rtol=1e-12)


class TestTsMax:
    def test_matches_pandas_rolling_max(self, synthetic_panel):
        ours = synthetic_panel.with_columns(_ops.ts_max(pl.col("close"), 5).alias("m"))
        y = _per_stock_array(synthetic_panel, "close")
        expected = pd.Series(y).rolling(5).max().to_numpy()
        ours_arr = _per_stock_array(ours, "m")
        mask = ~np.isnan(expected) & ~np.isnan(ours_arr)
        np.testing.assert_allclose(ours_arr[mask], expected[mask], rtol=1e-12)


class TestDelta:
    def test_matches_pandas_diff(self, synthetic_panel):
        ours = synthetic_panel.with_columns(_ops.delta(pl.col("close"), 1).alias("d"))
        y = _per_stock_array(synthetic_panel, "close")
        expected = pd.Series(y).diff(1).to_numpy()
        ours_arr = _per_stock_array(ours, "d")
        mask = ~np.isnan(expected) & ~np.isnan(ours_arr)
        np.testing.assert_allclose(ours_arr[mask], expected[mask], rtol=1e-12)

    def test_first_n_rows_null(self, synthetic_panel):
        ours = synthetic_panel.with_columns(_ops.delta(pl.col("close"), 3).alias("d"))
        first_stock = ours["stock_code"][0]
        rows = ours.filter(pl.col("stock_code") == first_stock)["d"][:3].to_list()
        assert all(r is None for r in rows)


class TestDelay:
    def test_matches_pandas_shift(self, synthetic_panel):
        ours = synthetic_panel.with_columns(_ops.delay(pl.col("close"), 2).alias("d"))
        y = _per_stock_array(synthetic_panel, "close")
        expected = pd.Series(y).shift(2).to_numpy()
        ours_arr = _per_stock_array(ours, "d")
        mask = ~np.isnan(expected) & ~np.isnan(ours_arr)
        np.testing.assert_allclose(ours_arr[mask], expected[mask], rtol=1e-12)


class TestCorr:
    def test_matches_pandas_rolling_corr(self, synthetic_panel):
        ours = synthetic_panel.with_columns(
            _ops.corr(pl.col("close"), pl.col("volume"), 5).alias("c")
        )
        y = _per_stock_array(synthetic_panel, "close")
        v = _per_stock_array(synthetic_panel, "volume")
        expected = pd.Series(y).rolling(5).corr(pd.Series(v)).to_numpy()
        ours_arr = _per_stock_array(ours, "c")
        mask = ~np.isnan(expected) & ~np.isnan(ours_arr)
        np.testing.assert_allclose(ours_arr[mask], expected[mask], rtol=1e-10)


class TestCovariance:
    def test_matches_pandas_rolling_cov(self, synthetic_panel):
        ours = synthetic_panel.with_columns(
            _ops.covariance(pl.col("close"), pl.col("volume"), 5).alias("c")
        )
        y = _per_stock_array(synthetic_panel, "close")
        v = _per_stock_array(synthetic_panel, "volume")
        expected = pd.Series(y).rolling(5).cov(pd.Series(v)).to_numpy()
        ours_arr = _per_stock_array(ours, "c")
        mask = ~np.isnan(expected) & ~np.isnan(ours_arr)
        np.testing.assert_allclose(ours_arr[mask], expected[mask], rtol=1e-10)


class TestTsRank:
    def test_matches_pandas_rolling_pct_rank(self, synthetic_panel):
        ours = synthetic_panel.with_columns(_ops.ts_rank(pl.col("close"), 5).alias("r"))
        y = _per_stock_array(synthetic_panel, "close")
        expected = pd.Series(y).rolling(5).rank(pct=True).to_numpy()
        ours_arr = _per_stock_array(ours, "r")
        mask = ~np.isnan(expected) & ~np.isnan(ours_arr)
        np.testing.assert_allclose(ours_arr[mask], expected[mask], rtol=1e-10)

    def test_range_is_zero_to_one(self, synthetic_panel):
        ours = synthetic_panel.with_columns(_ops.ts_rank(pl.col("close"), 5).alias("r"))
        non_null = ours["r"].drop_nulls()
        assert non_null.min() > 0.0
        assert non_null.max() <= 1.0


# ---------------------------------------------------------------------------
# Weighted moving averages
# ---------------------------------------------------------------------------


class TestSma:
    def test_matches_pandas_ewm(self, synthetic_panel):
        """GTJA SMA(X, N, M) ≡ pandas.ewm(alpha=M/N).mean()."""
        ours = synthetic_panel.with_columns(_ops.sma(pl.col("close"), 7, 2).alias("s"))
        y = _per_stock_array(synthetic_panel, "close")
        expected = pd.Series(y).ewm(alpha=2 / 7).mean().to_numpy()
        ours_arr = _per_stock_array(ours, "s")
        np.testing.assert_allclose(ours_arr, expected, rtol=1e-10)

    def test_first_value_equals_input(self, synthetic_panel):
        """EWMA initial condition: SMA[0] == X[0]."""
        ours = synthetic_panel.with_columns(_ops.sma(pl.col("close"), 7, 2).alias("s"))
        first_stock = ours["stock_code"][0]
        first_row = ours.filter(pl.col("stock_code") == first_stock).head(1)
        assert first_row["s"][0] == pytest.approx(first_row["close"][0])

    def test_handcomputed_three_points(self):
        """Hand-compute SMA([1, 2, 3], n=3, m=1). alpha = 1/3."""
        df = pl.DataFrame(
            {
                "stock_code": ["A"] * 3,
                "trade_date": pd.date_range("2024-01-02", periods=3).date,
                "x": [1.0, 2.0, 3.0],
            }
        )
        ours = df.with_columns(_ops.sma(pl.col("x"), 3, 1).alias("s"))["s"].to_list()
        # pandas semantics: alpha=1/3
        # s[0] = 1
        # s[1] = (alpha * 2 + (1-alpha) * 1) / (alpha + (1-alpha)) — pandas adjust=True default
        expected = pd.Series([1.0, 2.0, 3.0]).ewm(alpha=1 / 3).mean().to_list()
        np.testing.assert_allclose(ours, expected, rtol=1e-12)

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            _ops.sma(pl.col("x"), 5, 5)  # m == n
        with pytest.raises(ValueError):
            _ops.sma(pl.col("x"), 5, 0)  # m == 0
        with pytest.raises(ValueError):
            _ops.sma(pl.col("x"), 2, 5)  # m > n


class TestWma:
    def test_matches_increasing_linear_weights(self, synthetic_panel):
        """Weights [1, 2, …, n] / sum applied in time order, newest → highest."""
        n = 5
        weights = np.arange(1, n + 1, dtype=float)
        weights /= weights.sum()
        ours = synthetic_panel.with_columns(_ops.wma(pl.col("close"), n).alias("w"))
        y = _per_stock_array(synthetic_panel, "close")
        expected = np.full_like(y, np.nan, dtype=float)
        for i in range(n - 1, len(y)):
            expected[i] = np.sum(y[i - n + 1 : i + 1] * weights)
        ours_arr = _per_stock_array(ours, "w")
        mask = ~np.isnan(expected) & ~np.isnan(ours_arr)
        np.testing.assert_allclose(ours_arr[mask], expected[mask], rtol=1e-10)

    def test_handcomputed(self):
        """wma([1, 2, 3, 4], n=3) == sum([2, 3, 4] * [1, 2, 3]) / 6."""
        df = pl.DataFrame(
            {
                "stock_code": ["A"] * 4,
                "trade_date": pd.date_range("2024-01-02", periods=4).date,
                "x": [1.0, 2.0, 3.0, 4.0],
            }
        )
        out = df.with_columns(_ops.wma(pl.col("x"), 3).alias("w"))["w"].to_list()
        # Weights 1/6, 2/6, 3/6 (for oldest, mid, newest)
        # row 0, 1: null
        # row 2: (1*1 + 2*2 + 3*3)/6 = 14/6
        # row 3: (1*2 + 2*3 + 3*4)/6 = 20/6
        assert out[0] is None and out[1] is None
        np.testing.assert_allclose(out[2:], [14 / 6, 20 / 6], rtol=1e-12)


class TestDecayLinear:
    def test_matches_daic115_weights(self, synthetic_panel):
        """Daic115 decay weights: [2*i/(n*(n+1)) for i in 1..n]."""
        n = 5
        weights = np.array([2 * i / (n * (n + 1)) for i in range(1, n + 1)])
        ours = synthetic_panel.with_columns(_ops.decay_linear(pl.col("close"), n).alias("d"))
        y = _per_stock_array(synthetic_panel, "close")
        expected = np.full_like(y, np.nan, dtype=float)
        for i in range(n - 1, len(y)):
            expected[i] = np.sum(y[i - n + 1 : i + 1] * weights)
        ours_arr = _per_stock_array(ours, "d")
        mask = ~np.isnan(expected) & ~np.isnan(ours_arr)
        np.testing.assert_allclose(ours_arr[mask], expected[mask], rtol=1e-10)

    def test_equivalent_to_wma(self, synthetic_panel):
        """decay_linear and wma must agree because the weights normalise the same way."""
        a = synthetic_panel.with_columns(
            _ops.decay_linear(pl.col("close"), 7).alias("a")
        )["a"].to_numpy()
        b = synthetic_panel.with_columns(_ops.wma(pl.col("close"), 7).alias("b"))[
            "b"
        ].to_numpy()
        mask = ~np.isnan(a) & ~np.isnan(b)
        np.testing.assert_allclose(a[mask], b[mask], rtol=1e-12)


# ---------------------------------------------------------------------------
# Conditional / count / sumif
# ---------------------------------------------------------------------------


class TestIfelse:
    def test_basic_branches(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        cond = pl.col("a") > 1
        out = df.with_columns(_ops.ifelse(cond, pl.col("a"), pl.col("b")).alias("o"))
        assert out["o"].to_list() == [10.0, 2.0, 3.0]


class TestCount:
    def test_rolling_count_of_true(self):
        df = pl.DataFrame(
            {
                "stock_code": ["A"] * 6,
                "trade_date": pd.date_range("2024-01-02", periods=6).date,
                "x": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            }
        )
        out = df.with_columns(_ops.count_(pl.col("x") > 0, 3).alias("c"))["c"].to_list()
        # window of 3:
        #   row 2 windows over [T, F, T] → 2
        #   row 3 windows over [F, T, T] → 2
        #   row 4 windows over [T, T, F] → 2
        #   row 5 windows over [T, F, T] → 2
        assert out[0] is None and out[1] is None
        assert out[2:] == [2.0, 2.0, 2.0, 2.0]


class TestSumif:
    def test_rolling_sum_when_cond_true(self):
        df = pl.DataFrame(
            {
                "stock_code": ["A"] * 5,
                "trade_date": pd.date_range("2024-01-02", periods=5).date,
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                "g": [True, False, True, True, False],
            }
        )
        out = df.with_columns(_ops.sumif(pl.col("x"), 3, pl.col("g")).alias("s"))[
            "s"
        ].to_list()
        # row 2: x[0]+x[2] where g→ true: 1+3 = 4
        # row 3: x[2]+x[3] (g=True for both) = 3+4 = 7
        # row 4: x[2]+x[3] = 3+4 = 7
        assert out[0] is None and out[1] is None
        assert out[2:] == [4.0, 7.0, 7.0]


# ---------------------------------------------------------------------------
# Slow path — argmax/argmin position
# ---------------------------------------------------------------------------


class TestLowday:
    def test_matches_argmin_distance(self, synthetic_panel):
        """LOWDAY = n - argmin(window)."""
        n = 5
        ours = synthetic_panel.with_columns(_ops.lowday(pl.col("close"), n).alias("l"))
        y = _per_stock_array(synthetic_panel, "close")
        expected = np.full_like(y, np.nan, dtype=float)
        for i in range(n - 1, len(y)):
            expected[i] = float(n - int(np.argmin(y[i - n + 1 : i + 1])))
        ours_arr = _per_stock_array(ours, "l")
        mask = ~np.isnan(expected) & ~np.isnan(ours_arr)
        np.testing.assert_array_equal(ours_arr[mask], expected[mask])

    def test_today_is_min_returns_one(self):
        """If the latest value is the unique min, LOWDAY returns n."""
        df = pl.DataFrame(
            {
                "stock_code": ["A"] * 5,
                "trade_date": pd.date_range("2024-01-02", periods=5).date,
                "x": [5.0, 4.0, 3.0, 2.0, 1.0],
            }
        )
        # Window=5: argmin position is 4 (last). Distance = 5 - 4 = 1.
        out = df.with_columns(_ops.lowday(pl.col("x"), 5).alias("l"))["l"].to_list()
        assert out[-1] == 1.0


class TestHighday:
    def test_matches_argmax_distance(self, synthetic_panel):
        n = 5
        ours = synthetic_panel.with_columns(_ops.highday(pl.col("close"), n).alias("h"))
        y = _per_stock_array(synthetic_panel, "close")
        expected = np.full_like(y, np.nan, dtype=float)
        for i in range(n - 1, len(y)):
            expected[i] = float(n - int(np.argmax(y[i - n + 1 : i + 1])))
        ours_arr = _per_stock_array(ours, "h")
        mask = ~np.isnan(expected) & ~np.isnan(ours_arr)
        np.testing.assert_array_equal(ours_arr[mask], expected[mask])

    def test_today_is_max_returns_one(self):
        df = pl.DataFrame(
            {
                "stock_code": ["A"] * 5,
                "trade_date": pd.date_range("2024-01-02", periods=5).date,
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        out = df.with_columns(_ops.highday(pl.col("x"), 5).alias("h"))["h"].to_list()
        assert out[-1] == 1.0


# ---------------------------------------------------------------------------
# Rolling regression
# ---------------------------------------------------------------------------


class TestRegbeta:
    def test_matches_numpy_lstsq_slope(self, synthetic_panel):
        n = 10
        ours = synthetic_panel.with_columns(
            _ops.regbeta(pl.col("close"), pl.col("volume"), n).alias("b")
        )
        y = _per_stock_array(synthetic_panel, "close")
        x = _per_stock_array(synthetic_panel, "volume")
        expected = np.full_like(y, np.nan, dtype=float)
        for i in range(n - 1, len(y)):
            yw = y[i - n + 1 : i + 1]
            xw = x[i - n + 1 : i + 1]
            A = np.column_stack([np.ones_like(xw), xw])
            coef, *_ = np.linalg.lstsq(A, yw, rcond=None)
            expected[i] = coef[1]
        ours_arr = _per_stock_array(ours, "b")
        mask = ~np.isnan(expected) & ~np.isnan(ours_arr)
        np.testing.assert_allclose(ours_arr[mask], expected[mask], rtol=1e-8, atol=1e-10)

    def test_handcomputed_pure_linear(self):
        """If y = 2x + 3 exactly, slope should be 2.0."""
        df = pl.DataFrame(
            {
                "stock_code": ["A"] * 6,
                "trade_date": pd.date_range("2024-01-02", periods=6).date,
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
            }
        )
        out = df.with_columns(
            _ops.regbeta(pl.col("y"), pl.col("x"), 5).alias("b")
        )
        non_null = out["b"].drop_nulls().to_numpy()
        np.testing.assert_allclose(non_null, [2.0, 2.0], rtol=1e-10)


class TestRegresi:
    def test_matches_numpy_lstsq_residual(self, synthetic_panel):
        n = 10
        ours = synthetic_panel.with_columns(
            _ops.regresi(pl.col("close"), pl.col("volume"), n).alias("r")
        )
        y = _per_stock_array(synthetic_panel, "close")
        x = _per_stock_array(synthetic_panel, "volume")
        expected = np.full_like(y, np.nan, dtype=float)
        for i in range(n - 1, len(y)):
            yw = y[i - n + 1 : i + 1]
            xw = x[i - n + 1 : i + 1]
            A = np.column_stack([np.ones_like(xw), xw])
            coef, *_ = np.linalg.lstsq(A, yw, rcond=None)
            a, b = coef[0], coef[1]
            expected[i] = yw[-1] - (a + b * xw[-1])
        ours_arr = _per_stock_array(ours, "r")
        mask = ~np.isnan(expected) & ~np.isnan(ours_arr)
        np.testing.assert_allclose(ours_arr[mask], expected[mask], rtol=1e-8, atol=1e-10)

    def test_pure_linear_residual_is_zero(self):
        """Residual of a perfect linear fit is zero."""
        df = pl.DataFrame(
            {
                "stock_code": ["A"] * 6,
                "trade_date": pd.date_range("2024-01-02", periods=6).date,
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
            }
        )
        out = df.with_columns(_ops.regresi(pl.col("y"), pl.col("x"), 5).alias("r"))
        non_null = out["r"].drop_nulls().to_numpy()
        np.testing.assert_allclose(non_null, [0.0, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# Element-wise transforms
# ---------------------------------------------------------------------------


class TestSign:
    def test_basic_signs(self):
        df = pl.DataFrame({"x": [-3.0, 0.0, 7.0]})
        out = df.with_columns(_ops.sign_(pl.col("x")).alias("s"))["s"].to_list()
        assert out == [-1.0, 0.0, 1.0]


class TestAbs:
    def test_basic_absolute(self):
        df = pl.DataFrame({"x": [-3.0, 0.0, 7.0]})
        out = df.with_columns(_ops.abs_(pl.col("x")).alias("a"))["a"].to_list()
        assert out == [3.0, 0.0, 7.0]


class TestLog:
    def test_natural_log(self):
        df = pl.DataFrame({"x": [1.0, np.e, np.e**2]})
        out = df.with_columns(_ops.log_(pl.col("x")).alias("l"))["l"].to_numpy()
        np.testing.assert_allclose(out, [0.0, 1.0, 2.0], rtol=1e-12)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestSequence:
    def test_returns_one_to_n(self):
        out = pl.select(_ops.sequence(5)).to_series().to_list()
        assert out == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_dtype_is_float64(self):
        out = pl.select(_ops.sequence(5))
        assert out.dtypes[0] == pl.Float64

    def test_invalid_arg_raises(self):
        with pytest.raises(ValueError):
            _ops.sequence(0)


# ---------------------------------------------------------------------------
# Module-level smoke tests
# ---------------------------------------------------------------------------


def test_partition_constants_match_alpha101():
    """gtja191 must use the same partition keys as alpha101 for compatibility."""
    from aurumq_rl.factors.alpha101 import _ops as alpha_ops

    assert _ops.TS_PART == alpha_ops.TS_PART
    assert _ops.CS_PART == alpha_ops.CS_PART


def test_all_exported_callables_exist():
    """Every name in __all__ must resolve to a callable or string constant."""
    for name in _ops.__all__:
        assert hasattr(_ops, name), f"missing export: {name}"
