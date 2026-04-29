"""Tests for alpha101.technical factors (alpha092)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from aurumq_rl.factors.alpha101.technical import alpha092

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
# alpha092 tests
# ---------------------------------------------------------------------------


class TestAlpha092:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha092(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha092(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        result = alpha092(synthetic_panel)
        assert result.tail(50).is_not_null().sum() > 0

    def test_in_range_zero_one(self, synthetic_panel):
        # The output is the min of two ts_rank() values, both in [0, 1].
        result = alpha092(synthetic_panel)
        non_null = result.drop_nulls()
        if non_null.len() == 0:
            pytest.skip("no non-null rows")
        # NaN can leak through ts_corr; filter
        clean = non_null.fill_nan(None).drop_nulls()
        if clean.len() == 0:
            pytest.skip("no clean (non-NaN) values")
        assert clean.min() >= 0.0
        assert clean.max() <= 1.0

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        # alpha092 is not present in the STHSF reference parquet (the build
        # script skipped it because of pd.Panel usage in the original
        # implementation). Skip per spec.
        if "alpha092" not in alpha101_reference.columns:
            pytest.skip("alpha092 not in STHSF reference")
        bad, n = _parity_bad_fraction(synthetic_panel, alpha092, "alpha092", alpha101_reference)
        if n == 0:
            pytest.skip("no overlapping non-null pairs")
        assert bad <= _BAD_ROW_THRESHOLD, f"{bad:.1%} mismatched rows (n={n})"
