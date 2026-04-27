"""Tests for data_loader: factor discovery, universe filter, panel build."""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from aurumq_rl.data_loader import (
    FACTOR_COL_PREFIXES,
    REQUIRED_COLUMNS,
    FactorPanel,
    FactorPanelLoader,
    UniverseFilter,
    discover_factor_columns,
    filter_universe,
)

# ---------------------------------------------------------------------------
# discover_factor_columns
# ---------------------------------------------------------------------------


def test_discover_factor_columns_finds_all_prefixes() -> None:
    df = pl.DataFrame(
        {
            "ts_code": ["X"],
            "trade_date": [datetime.date(2022, 1, 1)],
            "alpha_01": [0.0],
            "mf_buy": [0.0],
            "hk_holding": [0.0],
            "fund_pe": [0.0],
            "non_factor_col": [0.0],
        }
    )
    cols = discover_factor_columns(df)
    assert "alpha_01" in cols
    assert "mf_buy" in cols
    assert "hk_holding" in cols
    assert "fund_pe" in cols
    assert "non_factor_col" not in cols


def test_discover_factor_columns_returns_sorted() -> None:
    df = pl.DataFrame(
        {
            "fund_pe": [1.0],
            "alpha_b": [1.0],
            "alpha_a": [1.0],
        }
    )
    cols = discover_factor_columns(df)
    assert cols == ["alpha_a", "alpha_b", "fund_pe"]


def test_discover_factor_columns_truncates_to_n_factors() -> None:
    df = pl.DataFrame({f"alpha_{i:02d}": [0.0] for i in range(20)})
    cols = discover_factor_columns(df, n_factors=5)
    assert len(cols) == 5
    assert cols == [f"alpha_{i:02d}" for i in range(5)]


def test_discover_factor_columns_empty_when_no_match() -> None:
    df = pl.DataFrame({"random_col": [1.0], "another": [2.0]})
    assert discover_factor_columns(df) == []


def test_factor_col_prefixes_constant_unchanged() -> None:
    """Guard the documented prefix list — changing it is a contract break."""
    expected = {
        "alpha_",
        "mf_",
        "hm_",
        "hk_",
        "inst_",
        "mg_",
        "cyq_",
        "senti_",
        "sh_",
        "fund_",
        "ind_",
        "mkt_",
    }
    assert set(FACTOR_COL_PREFIXES) == expected


# ---------------------------------------------------------------------------
# filter_universe — all 5 modes
# ---------------------------------------------------------------------------


@pytest.fixture
def universe_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_code": [
                "600519.SH",  # SH main
                "000001.SZ",  # SZ main
                "300750.SZ",  # ChiNext (filtered out)
                "688981.SH",  # STAR (filtered out)
                "830799.BJ",  # BSE (filtered out)
                "002594.SZ",  # SZ main
                "601398.SH",  # SH main
            ],
            "name": [
                "贵州茅台",
                "*ST平安",  # ST → filtered
                "宁德时代",
                "中芯国际",
                "贝特瑞",
                "比亚迪",
                "工商银行",
            ],
        }
    )


def test_filter_all_a_keeps_everything(universe_df: pl.DataFrame) -> None:
    out = filter_universe(universe_df, mode=UniverseFilter.ALL_A)
    assert len(out) == len(universe_df)


def test_filter_main_board_non_st(universe_df: pl.DataFrame) -> None:
    out = filter_universe(universe_df, mode=UniverseFilter.MAIN_BOARD_NON_ST)
    codes = set(out["ts_code"].to_list())
    # Only main-board, non-ST
    assert codes == {"600519.SH", "002594.SZ", "601398.SH"}


def test_filter_main_board_excludes_chinext_star_bj(universe_df: pl.DataFrame) -> None:
    out = filter_universe(universe_df, mode=UniverseFilter.MAIN_BOARD_NON_ST)
    codes = set(out["ts_code"].to_list())
    for excluded in ("300750.SZ", "688981.SH", "830799.BJ"):
        assert excluded not in codes


def test_filter_hs500_falls_back_to_main_board(universe_df: pl.DataFrame) -> None:
    out = filter_universe(universe_df, mode=UniverseFilter.HS300)
    for code in out["ts_code"].to_list():
        assert code.endswith((".SH", ".SZ"))
        assert not code.startswith(("300", "301", "688", "8", "4"))


def test_filter_zz500_falls_back_to_main_board(universe_df: pl.DataFrame) -> None:
    out = filter_universe(universe_df, mode=UniverseFilter.ZZ500)
    assert all(c.endswith((".SH", ".SZ")) for c in out["ts_code"].to_list())


def test_filter_zz1000_falls_back_to_main_board(universe_df: pl.DataFrame) -> None:
    out = filter_universe(universe_df, mode=UniverseFilter.ZZ1000)
    assert len(out) > 0


def test_filter_st_skipped_when_name_col_missing() -> None:
    # When the `name` column is absent, ST filter is silently skipped.
    df = pl.DataFrame({"ts_code": ["600519.SH", "000001.SZ"]})
    out = filter_universe(df, mode=UniverseFilter.MAIN_BOARD_NON_ST)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# build_synthetic
# ---------------------------------------------------------------------------


def test_build_synthetic_shapes() -> None:
    panel = FactorPanelLoader.build_synthetic(n_dates=20, n_stocks=10, n_factors=5, seed=1)
    assert isinstance(panel, FactorPanel)
    assert panel.factor_array.shape == (20, 10, 5)
    assert panel.return_array.shape == (20, 10)
    assert panel.pct_change_array.shape == (20, 10)
    assert panel.is_st_array.shape == (20, 10)
    assert len(panel.dates) == 20
    assert len(panel.stock_codes) == 10
    assert len(panel.factor_names) == 5


def test_build_synthetic_uses_synthetic_codes() -> None:
    panel = FactorPanelLoader.build_synthetic(n_dates=5, n_stocks=4, n_factors=2)
    for code in panel.stock_codes:
        assert code.startswith("SYN_")
        # Ensure no real-stock-like suffix
        assert not code.endswith((".SH", ".SZ", ".BJ"))


def test_build_synthetic_factors_zscored() -> None:
    panel = FactorPanelLoader.build_synthetic(n_dates=50, n_stocks=200, n_factors=4, seed=42)
    # Cross-section z-score: per (date, factor), mean ~ 0 and std ~ 1 across stocks
    arr = panel.factor_array
    means = np.nanmean(arr, axis=1)
    stds = np.nanstd(arr, axis=1)
    assert np.allclose(means, 0.0, atol=1e-3)
    # Std could be slightly less than 1 due to denominator stabilization
    assert np.all(stds < 1.5)


def test_build_synthetic_dates_are_weekdays() -> None:
    panel = FactorPanelLoader.build_synthetic(n_dates=20, n_stocks=4, n_factors=2)
    for d in panel.dates:
        assert d.weekday() < 5


# ---------------------------------------------------------------------------
# Loader: load_panel from real Parquet
# ---------------------------------------------------------------------------


def test_loader_missing_file_raises(tmp_path: Path) -> None:
    loader = FactorPanelLoader(parquet_path=tmp_path / "missing.parquet")
    with pytest.raises(FileNotFoundError):
        loader.load_panel(
            start_date=datetime.date(2022, 1, 1),
            end_date=datetime.date(2022, 12, 31),
        )


def test_loader_loads_tiny_panel(tiny_panel_parquet: Path) -> None:
    loader = FactorPanelLoader(parquet_path=tiny_panel_parquet)
    start, end = loader.get_date_range()
    assert start is not None and end is not None

    panel = loader.load_panel(
        start_date=start,
        end_date=end,
        n_factors=5,
        forward_period=5,
        universe_filter=UniverseFilter.ALL_A,
    )
    # Synthetic panel should populate all required arrays
    assert panel.factor_array.ndim == 3
    assert panel.factor_array.shape[2] == 5
    assert panel.return_array.shape == panel.pct_change_array.shape


def test_loader_main_board_filter_eliminates_synthetic_codes(
    tiny_panel_parquet: Path,
) -> None:
    """SYN_ codes are not main-board format → main_board filter empties the panel."""
    loader = FactorPanelLoader(parquet_path=tiny_panel_parquet)
    start, end = loader.get_date_range()
    with pytest.raises(ValueError, match="eliminated all rows"):
        loader.load_panel(
            start_date=start,
            end_date=end,
            n_factors=3,
            universe_filter=UniverseFilter.MAIN_BOARD_NON_ST,
        )


def test_loader_required_columns_constant() -> None:
    # Contract: REQUIRED_COLUMNS shape is fixed
    assert "ts_code" in REQUIRED_COLUMNS
    assert "trade_date" in REQUIRED_COLUMNS
    assert "close" in REQUIRED_COLUMNS
    assert "pct_chg" in REQUIRED_COLUMNS
    assert "vol" in REQUIRED_COLUMNS


def test_loader_get_date_range_returns_none_for_missing(tmp_path: Path) -> None:
    loader = FactorPanelLoader(parquet_path=tmp_path / "no_such.parquet")
    assert loader.get_date_range() == (None, None)
