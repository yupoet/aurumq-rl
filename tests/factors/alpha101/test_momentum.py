"""Tests for alpha101.momentum factors.

For each of the fourteen momentum alphas we cover four checks:

* dtype is :class:`pl.Float64`
* output length equals the panel height
* steady-state rows produce **some** non-null values
  (alphas with very long lookbacks like 240/250 are exempt — they will
  legitimately be all-null on a 60-day synthetic panel)
* parity with the locked STHSF reference parquet within
  ``rtol=1e-3, atol=1e-3``. STHSF's ``alpha001`` mutates ``self.close``
  in place, so any alpha computed *after* alpha001 in the build script
  reads polluted closes. The diverging alphas are marked
  ``@pytest.mark.xfail(strict=False)`` per the migration plan — they
  will be reconciled in Phase D.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from aurumq_rl.factors.alpha101.momentum import (
    alpha007,
    alpha008,
    alpha009,
    alpha010,
    alpha017,
    alpha019,
    alpha038,
    alpha045,
    alpha046,
    alpha051,
    alpha052,
    alpha084,
    alpha_custom_argmax_recent,
    alpha_custom_decaylinear_mom,
)

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
    """Compute the fraction of rows where our impl disagrees with the reference.

    Returns ``(bad_fraction, n_compared)``. ``bad_fraction`` is a float in
    ``[0, 1]``; ``n_compared`` is the number of rows that had a non-null
    pair on both sides.
    """
    ours = panel.with_columns(impl(panel).alias("ours"))
    joined = ours.join(
        reference.select(["stock_code", "trade_date", name]),
        on=["stock_code", "trade_date"],
        how="inner",
    )
    joined = joined.filter(
        pl.col("ours").is_not_null() & pl.col(name).is_not_null()
    )
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


class TestAlpha007:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha007(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha007(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        result = alpha007(synthetic_panel)
        assert result.tail(50).is_not_null().sum() > 0

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        if "alpha007" not in alpha101_reference.columns:
            pytest.skip("alpha007 not in STHSF reference")
        bad, n = _parity_bad_fraction(synthetic_panel, alpha007, "alpha007", alpha101_reference)
        if n == 0:
            pytest.skip("no overlapping non-null pairs")
        assert bad <= _BAD_ROW_THRESHOLD, f"{bad:.1%} mismatched rows (n={n})"


class TestAlpha008:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha008(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha008(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        result = alpha008(synthetic_panel)
        assert result.tail(50).is_not_null().sum() > 0

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        if "alpha008" not in alpha101_reference.columns:
            pytest.skip("alpha008 not in STHSF reference")
        bad, n = _parity_bad_fraction(synthetic_panel, alpha008, "alpha008", alpha101_reference)
        if n == 0:
            pytest.skip("no overlapping non-null pairs")
        assert bad <= _BAD_ROW_THRESHOLD, f"{bad:.1%} mismatched rows (n={n})"


class TestAlpha009:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha009(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha009(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        result = alpha009(synthetic_panel)
        assert result.tail(50).is_not_null().sum() > 0

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        # Restored to plain test after STHSF cascade-pollution fix.
        if "alpha009" not in alpha101_reference.columns:
            pytest.skip("alpha009 not in STHSF reference")
        bad, n = _parity_bad_fraction(synthetic_panel, alpha009, "alpha009", alpha101_reference)
        if n == 0:
            pytest.skip("no overlapping non-null pairs")
        assert bad <= _BAD_ROW_THRESHOLD, f"{bad:.1%} mismatched rows (n={n})"


class TestAlpha010:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha010(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha010(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        result = alpha010(synthetic_panel)
        assert result.tail(50).is_not_null().sum() > 0

    @pytest.mark.xfail(
        strict=False,
        reason="STHSF reference contaminated by alpha001 self.close mutation; reconcile in Phase D",
    )
    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        if "alpha010" not in alpha101_reference.columns:
            pytest.skip("alpha010 not in STHSF reference")
        bad, n = _parity_bad_fraction(synthetic_panel, alpha010, "alpha010", alpha101_reference)
        if n == 0:
            pytest.skip("no overlapping non-null pairs")
        assert bad <= _BAD_ROW_THRESHOLD, f"{bad:.1%} mismatched rows (n={n})"


class TestAlpha017:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha017(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha017(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        result = alpha017(synthetic_panel)
        assert result.tail(50).is_not_null().sum() > 0

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        # Restored to plain test after STHSF cascade-pollution fix.
        if "alpha017" not in alpha101_reference.columns:
            pytest.skip("alpha017 not in STHSF reference")
        bad, n = _parity_bad_fraction(synthetic_panel, alpha017, "alpha017", alpha101_reference)
        if n == 0:
            pytest.skip("no overlapping non-null pairs")
        assert bad <= _BAD_ROW_THRESHOLD, f"{bad:.1%} mismatched rows (n={n})"


class TestAlpha019:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha019(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha019(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_handles_long_window(self, synthetic_panel):
        # alpha019 uses Ts_Sum(returns, 250). The 60-day synthetic panel
        # cannot reach steady state for that window — output is all-null.
        result = alpha019(synthetic_panel)
        # The output is fully null (no row reaches 250-day lookback). This
        # is the documented expected behaviour, NOT a bug.
        assert result.is_null().sum() == synthetic_panel.height

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        if "alpha019" not in alpha101_reference.columns:
            pytest.skip("alpha019 not in STHSF reference")
        # 250-day window > 60-day panel -> no overlapping non-null pairs.
        bad, n = _parity_bad_fraction(synthetic_panel, alpha019, "alpha019", alpha101_reference)
        if n == 0:
            pytest.skip("no overlapping non-null pairs (window 250 > panel 60)")
        assert bad <= _BAD_ROW_THRESHOLD, f"{bad:.1%} mismatched rows (n={n})"


class TestAlpha038:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha038(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha038(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        result = alpha038(synthetic_panel)
        assert result.tail(50).is_not_null().sum() > 0

    @pytest.mark.xfail(
        strict=False,
        reason="STHSF reference contaminated by alpha001 self.close mutation; reconcile in Phase D",
    )
    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        if "alpha038" not in alpha101_reference.columns:
            pytest.skip("alpha038 not in STHSF reference")
        bad, n = _parity_bad_fraction(synthetic_panel, alpha038, "alpha038", alpha101_reference)
        if n == 0:
            pytest.skip("no overlapping non-null pairs")
        assert bad <= _BAD_ROW_THRESHOLD, f"{bad:.1%} mismatched rows (n={n})"


class TestAlpha045:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha045(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha045(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        result = alpha045(synthetic_panel)
        assert result.tail(50).is_not_null().sum() > 0

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "STHSF reference for alpha045 is rank-tie-break-unstable across "
            "scipy/pandas versions on the 10-stock synthetic panel: rebuilding "
            "the reference locally vs the committed one already differs in 167 "
            "rows, so any platform with a different scipy.stats.rankdata "
            "tie-break order produces different alpha045 reference values. "
            "Same class as the ts_argmax FIRST-vs-LAST tie convention noted in "
            "_ops.py and the gtja191 xfails. Investigation: rebuilding ref on "
            "Windows polars 1.40.1 + scipy 1.17.1 leaves 88 row-diffs vs ours "
            "and 167 row-diffs vs the committed ref."
        ),
    )
    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        if "alpha045" not in alpha101_reference.columns:
            pytest.skip("alpha045 not in STHSF reference")
        bad, n = _parity_bad_fraction(synthetic_panel, alpha045, "alpha045", alpha101_reference)
        if n == 0:
            pytest.skip("no overlapping non-null pairs")
        assert bad <= _BAD_ROW_THRESHOLD, f"{bad:.1%} mismatched rows (n={n})"


class TestAlpha046:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha046(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha046(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        result = alpha046(synthetic_panel)
        assert result.tail(50).is_not_null().sum() > 0

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        # Restored to plain test after STHSF cascade-pollution fix.
        if "alpha046" not in alpha101_reference.columns:
            pytest.skip("alpha046 not in STHSF reference")
        bad, n = _parity_bad_fraction(synthetic_panel, alpha046, "alpha046", alpha101_reference)
        if n == 0:
            pytest.skip("no overlapping non-null pairs")
        assert bad <= _BAD_ROW_THRESHOLD, f"{bad:.1%} mismatched rows (n={n})"


class TestAlpha051:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha051(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha051(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        result = alpha051(synthetic_panel)
        assert result.tail(50).is_not_null().sum() > 0

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        # Restored to plain test after STHSF cascade-pollution fix.
        if "alpha051" not in alpha101_reference.columns:
            pytest.skip("alpha051 not in STHSF reference")
        bad, n = _parity_bad_fraction(synthetic_panel, alpha051, "alpha051", alpha101_reference)
        if n == 0:
            pytest.skip("no overlapping non-null pairs")
        assert bad <= _BAD_ROW_THRESHOLD, f"{bad:.1%} mismatched rows (n={n})"


class TestAlpha052:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha052(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha052(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_handles_long_window(self, synthetic_panel):
        # alpha052 uses Ts_Sum(returns, 240). The 60-day synthetic panel
        # cannot reach steady state for that window — output is all-null.
        result = alpha052(synthetic_panel)
        assert result.is_null().sum() == synthetic_panel.height

    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        if "alpha052" not in alpha101_reference.columns:
            pytest.skip("alpha052 not in STHSF reference")
        bad, n = _parity_bad_fraction(synthetic_panel, alpha052, "alpha052", alpha101_reference)
        if n == 0:
            pytest.skip("no overlapping non-null pairs (window 240 > panel 60)")
        assert bad <= _BAD_ROW_THRESHOLD, f"{bad:.1%} mismatched rows (n={n})"


class TestAlpha084:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha084(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha084(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        result = alpha084(synthetic_panel)
        assert result.tail(50).is_not_null().sum() > 0

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "STHSF ts_rank returns absolute rank (1..window); ours returns "
            "[0, 1] normalised. Scales differ by a factor of (window-1) — "
            "reconcile during Phase D"
        ),
    )
    def test_matches_sthsf_reference(self, synthetic_panel, alpha101_reference):
        if "alpha084" not in alpha101_reference.columns:
            pytest.skip("alpha084 not in STHSF reference")
        bad, n = _parity_bad_fraction(synthetic_panel, alpha084, "alpha084", alpha101_reference)
        if n == 0:
            pytest.skip("no overlapping non-null pairs")
        assert bad <= _BAD_ROW_THRESHOLD, f"{bad:.1%} mismatched rows (n={n})"


class TestAlphaCustomDecaylinearMom:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha_custom_decaylinear_mom(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha_custom_decaylinear_mom(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        result = alpha_custom_decaylinear_mom(synthetic_panel)
        assert result.tail(50).is_not_null().sum() > 0

    def test_in_range_zero_one(self, synthetic_panel):
        # cs_rank output is always in [0, 1] per day.
        result = alpha_custom_decaylinear_mom(synthetic_panel)
        non_null = result.drop_nulls()
        if non_null.len() == 0:
            pytest.skip("no non-null rows — too short panel?")
        assert non_null.min() >= 0.0
        assert non_null.max() <= 1.0


class TestAlphaCustomArgmaxRecent:
    def test_dtype_float64(self, synthetic_panel):
        assert alpha_custom_argmax_recent(synthetic_panel).dtype == pl.Float64

    def test_length_matches_panel(self, synthetic_panel):
        assert len(alpha_custom_argmax_recent(synthetic_panel)) == synthetic_panel.height

    def test_steady_state_has_values(self, synthetic_panel):
        result = alpha_custom_argmax_recent(synthetic_panel)
        assert result.tail(50).is_not_null().sum() > 0

    def test_in_range_zero_one(self, synthetic_panel):
        # 1 - cs_rank() output is in [0, 1].
        result = alpha_custom_argmax_recent(synthetic_panel)
        non_null = result.drop_nulls()
        if non_null.len() == 0:
            pytest.skip("no non-null rows — too short panel?")
        assert non_null.min() >= 0.0
        assert non_null.max() <= 1.0
