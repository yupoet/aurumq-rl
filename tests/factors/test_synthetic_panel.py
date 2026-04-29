"""Smoke tests for the shared synthetic factor panel fixture.

These tests guard the *contract* of :func:`tests.factors._synthetic.build_synthetic_panel`
because that contract is what makes the locked
``alpha101_reference.parquet`` valid: if the panel changes, the
reference values become stale.

If you intentionally change the panel layout, you MUST regenerate the
reference parquet via
``scripts/reference_data/build_alpha101_reference.py``.
"""

from __future__ import annotations

import polars as pl

from tests.factors._synthetic import (
    ADV_WINDOWS,
    INDUSTRIES,
    SUB_INDUSTRIES,
    build_synthetic_panel,
)

# ---------------------------------------------------------------------------
# Shape & schema
# ---------------------------------------------------------------------------


def test_panel_shape(synthetic_panel: pl.DataFrame) -> None:
    """10 stocks × 60 days = 600 rows."""
    assert synthetic_panel.shape[0] == 600
    # 14 base cols (stock_code, trade_date, OHLC, volume, amount, prev_close,
    # vwap, returns, industry, sub_industry, cap) + 12 adv windows
    assert synthetic_panel.shape[1] == 14 + len(ADV_WINDOWS)


def test_panel_columns(synthetic_panel: pl.DataFrame) -> None:
    """All required schema columns present."""
    expected_base = {
        "stock_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "prev_close",
        "vwap",
        "returns",
        "industry",
        "sub_industry",
        "cap",
    }
    expected_adv = {f"adv{w}" for w in ADV_WINDOWS}
    assert expected_base.issubset(set(synthetic_panel.columns))
    assert expected_adv.issubset(set(synthetic_panel.columns))


def test_panel_dtypes(synthetic_panel: pl.DataFrame) -> None:
    """Utf8 / Date / Float64 as specified."""
    schema = dict(zip(synthetic_panel.columns, synthetic_panel.dtypes, strict=True))
    assert schema["stock_code"] == pl.Utf8
    assert schema["trade_date"] == pl.Date
    for col in ("open", "high", "low", "close", "volume", "amount", "vwap", "returns", "cap"):
        assert schema[col] == pl.Float64, f"{col} expected Float64, got {schema[col]}"
    for w in ADV_WINDOWS:
        assert schema[f"adv{w}"] == pl.Float64
    assert schema["industry"] == pl.Utf8
    assert schema["sub_industry"] == pl.Utf8


# ---------------------------------------------------------------------------
# Returns / look-ahead invariants
# ---------------------------------------------------------------------------


def test_panel_no_lookahead_in_returns(synthetic_panel: pl.DataFrame) -> None:
    """First trading day of each stock has NaN ``returns`` (no prior close)."""
    first_day = (
        synthetic_panel.sort(["stock_code", "trade_date"])
        .group_by("stock_code", maintain_order=True)
        .head(1)
    )
    # 10 stocks → 10 first-day rows, all returns must be null
    assert first_day.shape[0] == 10
    null_count = first_day["returns"].null_count() + int(first_day["returns"].is_nan().sum() or 0)
    assert null_count == 10, (
        f"first-day returns should all be NaN/null, got {first_day['returns'].to_list()}"
    )

    # Conversely, every other row should have a finite (non-null, non-NaN) return.
    later = (
        synthetic_panel.sort(["stock_code", "trade_date"])
        .group_by("stock_code", maintain_order=True)
        .tail(59)
    )
    assert later["returns"].null_count() == 0
    assert int(later["returns"].is_nan().sum() or 0) == 0


# ---------------------------------------------------------------------------
# Industry coverage
# ---------------------------------------------------------------------------


def test_industry_coverage(synthetic_panel: pl.DataFrame) -> None:
    """Every stock has an industry; exactly two distinct industries."""
    assert synthetic_panel["industry"].null_count() == 0
    distinct = set(synthetic_panel["industry"].unique().to_list())
    assert distinct == set(INDUSTRIES)


def test_sub_industry_coverage(synthetic_panel: pl.DataFrame) -> None:
    """Every stock has a sub_industry; exactly two distinct sub_industries."""
    assert synthetic_panel["sub_industry"].null_count() == 0
    distinct = set(synthetic_panel["sub_industry"].unique().to_list())
    assert distinct == set(SUB_INDUSTRIES)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_reproducibility() -> None:
    """Two builds with seed=42 yield byte-identical DataFrames.

    This is the contract that keeps the locked alpha101 reference parquet
    valid. If this test fails, the reference values are stale.
    """
    a = build_synthetic_panel(seed=42)
    b = build_synthetic_panel(seed=42)
    assert a.equals(b), "build_synthetic_panel(seed=42) is not reproducible"


# ---------------------------------------------------------------------------
# Reference parquet — only runs when the artifact has been built
# ---------------------------------------------------------------------------


def test_alpha101_reference_alignment(
    synthetic_panel: pl.DataFrame,
    alpha101_reference: pl.DataFrame,
) -> None:
    """The reference parquet aligns 1:1 with the synthetic panel.

    Same row count, same ``(stock_code, trade_date)`` keys in the same
    order. This is the contract our polars implementations will diff
    against. If a future change to the panel breaks alignment, the
    reference parquet must be regenerated.
    """
    assert alpha101_reference.shape[0] == synthetic_panel.shape[0]

    panel_keys = synthetic_panel.select(["stock_code", "trade_date"]).sort(
        ["stock_code", "trade_date"]
    )
    ref_keys = alpha101_reference.select(["stock_code", "trade_date"])
    assert panel_keys.equals(ref_keys), (
        "reference parquet keys do not align with synthetic_panel; "
        "rebuild via scripts/reference_data/build_alpha101_reference.py"
    )

    # At least 50 alphas survived — guard against silent regressions in
    # the build pipeline.
    alpha_cols = [c for c in alpha101_reference.columns if c.startswith("alpha")]
    assert len(alpha_cols) >= 50, f"only {len(alpha_cols)} alphas in reference; expected >= 50"
