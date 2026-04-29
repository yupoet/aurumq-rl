"""Tests for alpha101.cap_weighted (alpha024, alpha056)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from aurumq_rl.factors.alpha101.cap_weighted import alpha024, alpha056

_REF_STATUS: dict[str, str] = {
    # alpha024 IS in the STHSF reference (paper-accurate, no cap term)
    "alpha024": "match",
    # alpha056 is NOT in STHSF (their reference omits it because cap was unavailable)
    "alpha056": "missing",
}


def _parity_check(
    panel: pl.DataFrame,
    reference: pl.DataFrame,
    name: str,
    fn,
) -> None:
    status = _REF_STATUS.get(name, "missing")
    if status == "missing" or name not in reference.columns:
        pytest.skip(f"{name} not in STHSF reference parquet")
        return

    ours = panel.with_columns(fn(panel).alias("ours"))
    joined = ours.join(
        reference.select(["stock_code", "trade_date", name]),
        on=["stock_code", "trade_date"],
        how="inner",
    ).drop_nulls(subset=["ours", name])
    if joined.height == 0:
        pytest.skip(f"no overlapping non-null pairs for {name}")
        return

    ours_arr = joined["ours"].to_numpy()
    ref_arr = joined[name].to_numpy()
    both_finite = ~(np.isnan(ours_arr) | np.isnan(ref_arr))

    if both_finite.sum() == 0:
        pytest.skip(f"{name}: zero finite-pair overlap with reference")
        return

    if status == "drift":
        pytest.skip(
            f"{name}: STHSF reference parquet drift — finite output verified by steady-state"
        )
        return

    np.testing.assert_allclose(
        ours_arr[both_finite],
        ref_arr[both_finite],
        rtol=1e-3,
        atol=1e-3,
        err_msg=f"{name}: drift on {both_finite.sum()} finite pairs",
    )


# ---------------------------------------------------------------------------
# alpha024 — long-horizon close-mean acceleration switch
# ---------------------------------------------------------------------------


class TestAlpha024:
    def test_returns_correct_dtype(self, synthetic_panel):
        assert alpha024(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha024(synthetic_panel)) == synthetic_panel.height

    def test_steady_state(self, synthetic_panel):
        # 100-day windows on 60-day synthetic — first branch can't activate;
        # second branch (-delta(close,3)) will produce values from day 3.
        s = alpha024(synthetic_panel)
        finite = s.drop_nulls().drop_nans()
        assert finite.shape[0] > 0, "no finite values"

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        _parity_check(synthetic_panel, alpha101_reference, "alpha024", alpha024)

    def test_null_cap_does_not_break(self, synthetic_panel):
        # alpha024 doesn't actually reference cap; null cap should still work.
        nulled = synthetic_panel.with_columns(
            pl.when(pl.col("stock_code") == "000001.SZ")
            .then(pl.lit(None))
            .otherwise(pl.col("cap"))
            .alias("cap")
        )
        s = alpha024(nulled)
        assert s.dtype == pl.Float64
        assert len(s) == nulled.height


# ---------------------------------------------------------------------------
# alpha056 — return-ratio rank scaled by rank(returns * cap)
# ---------------------------------------------------------------------------


class TestAlpha056:
    def test_returns_correct_dtype(self, synthetic_panel):
        assert alpha056(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha056(synthetic_panel)) == synthetic_panel.height

    def test_steady_state(self, synthetic_panel):
        s = alpha056(synthetic_panel)
        finite = s.drop_nulls().drop_nans()
        assert finite.shape[0] > 0, "no finite values"

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        _parity_check(synthetic_panel, alpha101_reference, "alpha056", alpha056)

    def test_null_cap_propagates_nan(self, synthetic_panel):
        # When cap is NULL for a stock, ``returns * cap`` is null which
        # means CS rank slot is null and the product is null too.
        nulled = synthetic_panel.with_columns(
            pl.when(pl.col("stock_code") == "000001.SZ")
            .then(pl.lit(None))
            .otherwise(pl.col("cap"))
            .alias("cap")
        )
        df = nulled.with_columns(alpha056(nulled).alias("a"))
        affected = df.filter(pl.col("stock_code") == "000001.SZ")
        # All rows for the null-cap stock should be null/NaN.
        finite_rows = affected["a"].drop_nulls().drop_nans().shape[0]
        assert finite_rows == 0, (
            f"alpha056: expected all-null for null-cap stock, got {finite_rows} finite rows"
        )
