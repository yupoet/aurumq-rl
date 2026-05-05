"""Phase 21 data_loader changes: is_suspended default-True, regime features,
schema lock, FactorPanel.regime_array."""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from aurumq_rl.data_loader import (
    FactorPanel,
    FactorPanelLoader,
    FORBIDDEN_PREFIXES,
    STOCK_FACTOR_PREFIXES,
    discover_factor_columns,
)


# ---------- Test panel: 5 dates, 3 stocks, 2 factors, hand-built ----------

def _build_panel_df() -> pl.DataFrame:
    """Hand-built panel where stock C only appears on dates 3-4 (pre-IPO on 0-2)."""
    dates = [dt.date(2024, 1, d) for d in (2, 3, 4, 5, 8)]
    rows: list[dict] = []
    for di, d in enumerate(dates):
        # A always present, vol > 0
        rows.append({"trade_date": d, "ts_code": "600001.SH", "close": 10.0 + di,
                     "pct_chg": 0.01 * (di - 2), "vol": 1000.0,
                     "alpha_001": 0.1, "alpha_002": 0.2, "is_st": False})
        # B always present but suspended (vol=0) on date 2
        rows.append({"trade_date": d, "ts_code": "000001.SZ", "close": 20.0,
                     "pct_chg": 0.0,
                     "vol": 0.0 if di == 2 else 500.0,
                     "alpha_001": -0.1, "alpha_002": 0.05, "is_st": False})
        # C MISSING entirely on dates 0,1,2 (pre-IPO); present on 3,4
        if di >= 3:
            rows.append({"trade_date": d, "ts_code": "600002.SH", "close": 5.0,
                         "pct_chg": 0.02, "vol": 200.0,
                         "alpha_001": 0.0, "alpha_002": -0.3, "is_st": False})
    return pl.DataFrame(rows)


def test_is_suspended_default_true_for_missing_rows(tmp_path):
    """Pre-IPO dates for stock C must be is_suspended=True, not False."""
    df = _build_panel_df()
    parquet_path = tmp_path / "panel.parquet"
    df.write_parquet(parquet_path)

    loader = FactorPanelLoader(parquet_path=parquet_path)
    panel = loader.load_panel(
        start_date=dt.date(2024, 1, 2), end_date=dt.date(2024, 1, 8),
        forward_period=1,
    )

    # stock_codes is sorted alphabetically: ['000001.SZ', '600001.SH', '600002.SH']
    assert panel.stock_codes == ["000001.SZ", "600001.SH", "600002.SH"]
    c = panel.stock_codes.index("600002.SH")
    # Dates 0,1,2 had no parquet row for C — must default to suspended (True)
    assert panel.is_suspended_array[0, c] == True
    assert panel.is_suspended_array[1, c] == True
    assert panel.is_suspended_array[2, c] == True
    # Dates 3,4 had C present with vol > 0 — not suspended
    assert panel.is_suspended_array[3, c] == False
    assert panel.is_suspended_array[4, c] == False
    # Stock B had vol=0 on date 2 — explicit suspension still flagged
    b = panel.stock_codes.index("000001.SZ")
    assert panel.is_suspended_array[2, b] == True
    assert panel.is_suspended_array[0, b] == False


def test_stock_factor_prefixes_excludes_mkt():
    assert "mkt_" not in STOCK_FACTOR_PREFIXES
    assert "alpha_" in STOCK_FACTOR_PREFIXES


def test_forbidden_prefixes_constant_present():
    assert FORBIDDEN_PREFIXES == ("mkt_", "index_", "regime_", "global_")


def test_discover_factor_columns_filters_forbidden():
    df = pl.DataFrame({
        "trade_date": [dt.date(2024, 1, 2)],
        "ts_code": ["A.SH"],
        "alpha_001": [0.1],
        "mf_001": [0.2],
        "mkt_congestion": [0.3],   # forbidden
        "regime_breadth_d": [0.4], # forbidden
        "global_vix": [0.5],       # forbidden
        "unknown_x": [0.6],        # not in allowlist, should also be skipped
    })
    cols = discover_factor_columns(df)
    assert cols == ["alpha_001", "mf_001"]


def test_discover_factor_columns_n_factors_limit():
    df = pl.DataFrame({
        "trade_date": [dt.date(2024, 1, 2)],
        "ts_code": ["A.SH"],
        "alpha_001": [0.1], "alpha_002": [0.2], "alpha_003": [0.3],
        "mf_001": [0.4],
    })
    cols = discover_factor_columns(df, n_factors=2)
    assert cols == ["alpha_001", "alpha_002"]
