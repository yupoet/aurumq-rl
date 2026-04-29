"""Tests for alpha101.breakout factors (alpha023, alpha054, alpha095)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from aurumq_rl.factors.alpha101.breakout import alpha023, alpha054, alpha095

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_RTOL = 1e-3
_ATOL = 1e-3
_BAD_ROW_THRESHOLD = 0.05


def _parity_bad_fraction(
    panel: pl.DataFrame,
    impl,
    name: str,
    reference: pl.DataFrame,
) -> tuple[float, int]:
    """Return (bad_fraction, n_compared) for the impl-vs-reference parity."""
    ours = panel.with_columns(impl(panel).alias("ours"))
    joined = ours.join(
        reference.select(["stock_code", "trade_date", name]),
        on=["stock_code", "trade_date"],
        how="inner",
    )
    joined = joined.filter(pl.col("ours").is_not_null() & pl.col(name).is_not_null())
    joined = joined.with_columns(
        pl.col("ours").is_nan().alias("__ours_nan"),
        pl.col(name).is_nan().alias("__ref_nan"),
    ).filter(~pl.col("__ours_nan") & ~pl.col("__ref_nan"))
    if joined.height == 0:
        return 0.0, 0
    a = joined["ours"].to_numpy()
    b = joined[name].to_numpy()
    diff = np.abs(a - b)
    bad = int((diff > _RTOL * np.abs(b) + _ATOL).sum())
    return bad / joined.height, joined.height


# ---------------------------------------------------------------------------
# Per-alpha tests
# ---------------------------------------------------------------------------


class TestAlpha023:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha023(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha023(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        result = alpha023(synthetic_panel)
        assert result.tail(50).is_not_null().sum() > 0

    def test_zero_otherwise_branch(self, synthetic_panel):
        # When the high never breaks the 20d MA, the alpha is exactly 0.
        # We at least confirm zero is among the produced values for some
        # rows (most days will not be breakouts on a 60-day GBM panel).
        result = alpha023(synthetic_panel).drop_nulls()
        if result.len() == 0:
            pytest.skip("no non-null rows")
        # Either 0.0 or non-zero — check zero appears.
        assert (result == 0.0).sum() > 0

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        # alpha023 not in STHSF reference parquet; skip per spec.
        if "alpha023" not in alpha101_reference.columns:
            pytest.skip("alpha023 not in STHSF reference")
        bad, n = _parity_bad_fraction(synthetic_panel, alpha023, "alpha023", alpha101_reference)
        if n == 0:
            pytest.skip("no overlapping non-null pairs")
        assert bad <= _BAD_ROW_THRESHOLD, f"{bad:.1%} mismatched rows (n={n})"


class TestAlpha054:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha054(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha054(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        result = alpha054(synthetic_panel)
        assert result.tail(50).is_not_null().sum() > 0

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        # Restored to plain test after the STHSF cascade-pollution fix
        # (deep-copy isolates alpha001 ``self.close`` mutation; alpha054 now
        # matches the rebuilt reference parquet).
        if "alpha054" not in alpha101_reference.columns:
            pytest.skip("alpha054 not in STHSF reference")
        bad, n = _parity_bad_fraction(synthetic_panel, alpha054, "alpha054", alpha101_reference)
        if n == 0:
            pytest.skip("no overlapping non-null pairs")
        assert bad <= _BAD_ROW_THRESHOLD, f"{bad:.1%} mismatched rows (n={n})"


class TestAlpha095:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha095(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha095(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        result = alpha095(synthetic_panel)
        assert result.tail(50).is_not_null().sum() > 0

    def test_binary_zero_or_one(self, synthetic_panel):
        # alpha095 returns the WorldQuant 0/1 indicator. Confirm output is
        # restricted to {0.0, 1.0} (plus null for the warm-up window).
        result = alpha095(synthetic_panel).drop_nulls()
        if result.len() == 0:
            pytest.skip("no non-null rows")
        unique_vals = set(result.unique().to_list())
        # NaN can show up if the underlying corr produced NaN.
        unique_clean = {v for v in unique_vals if v == v}  # drop NaN
        assert unique_clean.issubset({0.0, 1.0}), f"unexpected values: {unique_clean}"

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "STHSF ts_rank scale (1..window) differs from our [0, 1] scale "
            "and the comparison branch is sensitive to that scale; "
            "reconcile in Phase D"
        ),
    )
    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        if "alpha095" not in alpha101_reference.columns:
            pytest.skip("alpha095 not in STHSF reference")
        bad, n = _parity_bad_fraction(synthetic_panel, alpha095, "alpha095", alpha101_reference)
        if n == 0:
            pytest.skip("no overlapping non-null pairs")
        assert bad <= _BAD_ROW_THRESHOLD, f"{bad:.1%} mismatched rows (n={n})"
