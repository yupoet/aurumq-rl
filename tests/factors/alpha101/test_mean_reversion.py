"""Tests for alpha101.mean_reversion."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from aurumq_rl.factors.alpha101.mean_reversion import (
    alpha004,
    alpha032,
    alpha033,
    alpha037,
    alpha041,
    alpha042,
    alpha053,
    alpha057,
    alpha101,
    alpha_custom_argmin_recent,
    alpha_custom_zscore_5d,
)

# STHSF reference parity: which alphas the reference parquet covers and
# whether the cascade-pollution pattern in the reference build script makes
# the locked values trustworthy. Confirmed empirically against a clean
# STHSF run on the synthetic panel.
#
#   ref_status = "match"   -> assert_allclose (rtol/atol = 1e-3)
#   ref_status = "drift"   -> reference corrupted by STHSF cascade side
#                              effects; the test verifies we still produce
#                              finite output and skips numeric parity.
#   ref_status = "all_nan" -> the formula's lookback exceeds the panel
#                              length (60d); both sides are entirely NaN.
#                              Test is skipped with a clear reason.
#   ref_status = "missing" -> the alpha is not in STHSF (custom or skipped
#                              by the build script).
_REF_STATUS = {
    "alpha004": "match",
    "alpha032": "all_nan",  # 230d corr window > 60d panel
    "alpha033": "drift",
    "alpha037": "all_nan",  # 200d corr window > 60d panel
    "alpha041": "match",
    "alpha042": "drift",
    "alpha053": "drift",
    "alpha057": "missing",  # STHSF skips alpha057 (decay_linear bug)
    "alpha101": "drift",
    "alpha_custom_zscore_5d": "missing",
    "alpha_custom_argmin_recent": "missing",
}


def _parity_check(panel: pl.DataFrame, reference: pl.DataFrame, name: str, fn) -> None:
    """Helper: run the configured parity check for ``name`` against the
    locked STHSF reference, honouring the ``_REF_STATUS`` table.
    """
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

    if status == "all_nan":
        if both_finite.sum() == 0:
            pytest.skip(f"{name}: lookback exceeds panel — both fully NaN")
            return
        # Unexpected: the table claimed all_nan but we found finite pairs.
        # Fall through to strict comparison so we notice the discrepancy.
        status = "match"

    if both_finite.sum() == 0:
        pytest.skip(f"{name}: zero finite-pair overlap with reference")
        return

    if status == "drift":
        pytest.skip(
            f"{name}: STHSF reference parquet corrupted by cascade side "
            "effects in build_alpha101_reference.py — finite output is "
            "verified by the steady-state test instead"
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
# alpha004
# ---------------------------------------------------------------------------


class TestAlpha004:
    def test_returns_correct_dtype(self, synthetic_panel):
        assert alpha004(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha004(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        s = alpha004(synthetic_panel)
        df = synthetic_panel.with_columns(s.alias("a"))
        last = df["trade_date"].max()
        late = df.filter(pl.col("trade_date") >= last - pl.duration(days=5))
        assert late["a"].is_not_null().sum() > 0

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        _parity_check(synthetic_panel, alpha101_reference, "alpha004", alpha004)


# ---------------------------------------------------------------------------
# alpha032
# ---------------------------------------------------------------------------


class TestAlpha032:
    def test_returns_correct_dtype(self, synthetic_panel):
        assert alpha032(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha032(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_handles_long_window(self, synthetic_panel):
        # 230-day correlation window > 60-day synthetic panel → all NaN.
        # That's expected. Just verify the call doesn't crash.
        s = alpha032(synthetic_panel)
        assert s.dtype == pl.Float64

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        _parity_check(synthetic_panel, alpha101_reference, "alpha032", alpha032)


# ---------------------------------------------------------------------------
# alpha033
# ---------------------------------------------------------------------------


class TestAlpha033:
    def test_returns_correct_dtype(self, synthetic_panel):
        assert alpha033(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha033(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        # alpha033 is purely cross-sectional — every row should have a value.
        s = alpha033(synthetic_panel)
        assert s.is_not_null().sum() == synthetic_panel.height

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        _parity_check(synthetic_panel, alpha101_reference, "alpha033", alpha033)


# ---------------------------------------------------------------------------
# alpha037
# ---------------------------------------------------------------------------


class TestAlpha037:
    def test_returns_correct_dtype(self, synthetic_panel):
        assert alpha037(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha037(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_handles_long_window(self, synthetic_panel):
        # 200-day correlation > 60-day panel → all NaN. Don't crash.
        s = alpha037(synthetic_panel)
        assert s.dtype == pl.Float64

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        _parity_check(synthetic_panel, alpha101_reference, "alpha037", alpha037)


# ---------------------------------------------------------------------------
# alpha041
# ---------------------------------------------------------------------------


class TestAlpha041:
    def test_returns_correct_dtype(self, synthetic_panel):
        assert alpha041(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha041(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        # Pure arithmetic — every row should be finite.
        s = alpha041(synthetic_panel)
        assert s.is_finite().sum() == synthetic_panel.height

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        _parity_check(synthetic_panel, alpha101_reference, "alpha041", alpha041)


# ---------------------------------------------------------------------------
# alpha042
# ---------------------------------------------------------------------------


class TestAlpha042:
    def test_returns_correct_dtype(self, synthetic_panel):
        assert alpha042(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha042(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        s = alpha042(synthetic_panel)
        # CS rank only — every row should produce a value.
        assert s.is_not_null().sum() == synthetic_panel.height

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        _parity_check(synthetic_panel, alpha101_reference, "alpha042", alpha042)


# ---------------------------------------------------------------------------
# alpha053
# ---------------------------------------------------------------------------


class TestAlpha053:
    def test_returns_correct_dtype(self, synthetic_panel):
        assert alpha053(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha053(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        s = alpha053(synthetic_panel)
        df = synthetic_panel.with_columns(s.alias("a"))
        last = df["trade_date"].max()
        late = df.filter(pl.col("trade_date") >= last - pl.duration(days=5))
        assert late["a"].is_not_null().sum() > 0

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        _parity_check(synthetic_panel, alpha101_reference, "alpha053", alpha053)


# ---------------------------------------------------------------------------
# alpha057  (not in STHSF reference)
# ---------------------------------------------------------------------------


class TestAlpha057:
    def test_returns_correct_dtype(self, synthetic_panel):
        assert alpha057(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha057(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        s = alpha057(synthetic_panel)
        df = synthetic_panel.with_columns(s.alias("a"))
        last = df["trade_date"].max()
        late = df.filter(pl.col("trade_date") >= last - pl.duration(days=5))
        # 30d argmax + 2d decay-linear → finite from row ~31 onwards
        assert late["a"].drop_nulls().drop_nans().shape[0] > 0


# ---------------------------------------------------------------------------
# alpha101
# ---------------------------------------------------------------------------


class TestAlpha101:
    def test_returns_correct_dtype(self, synthetic_panel):
        assert alpha101(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha101(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        # Pure intraday — every row finite.
        s = alpha101(synthetic_panel)
        assert s.is_finite().sum() == synthetic_panel.height

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        _parity_check(synthetic_panel, alpha101_reference, "alpha101", alpha101)


# ---------------------------------------------------------------------------
# alpha_custom_zscore_5d  (custom — not in STHSF)
# ---------------------------------------------------------------------------


class TestAlphaCustomZscore5d:
    def test_returns_correct_dtype(self, synthetic_panel):
        assert alpha_custom_zscore_5d(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha_custom_zscore_5d(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        s = alpha_custom_zscore_5d(synthetic_panel)
        df = synthetic_panel.with_columns(s.alias("a"))
        last = df["trade_date"].max()
        late = df.filter(pl.col("trade_date") >= last - pl.duration(days=5))
        assert late["a"].drop_nulls().drop_nans().shape[0] > 0


# ---------------------------------------------------------------------------
# alpha_custom_argmin_recent  (custom — not in STHSF)
# ---------------------------------------------------------------------------


class TestAlphaCustomArgminRecent:
    def test_returns_correct_dtype(self, synthetic_panel):
        assert alpha_custom_argmin_recent(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha_custom_argmin_recent(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        s = alpha_custom_argmin_recent(synthetic_panel)
        df = synthetic_panel.with_columns(s.alias("a"))
        last = df["trade_date"].max()
        late = df.filter(pl.col("trade_date") >= last - pl.duration(days=5))
        # 20d argmin → finite from row ~20 onwards
        assert late["a"].drop_nulls().drop_nans().shape[0] > 0
