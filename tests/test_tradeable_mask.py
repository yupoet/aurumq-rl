"""Tests for the shared tradeable-mask pipeline (C4 / M5).

``build_tradeable_mask`` is the single source of truth for entry
eligibility used by BOTH the GPU training path (scripts/train_v2.py
valid_mask) and the backtest eval scripts. Semantics must match the CPU
env's ``_apply_trading_mask``: ~suspended & ~ST & IPO gate & neither
limit-up nor limit-down at the decision date.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import polars as pl

from aurumq_rl.data_loader import (
    NEW_STOCK_PROTECT_DAYS,
    FactorPanel,
    FactorPanelLoader,
    UniverseFilter,
    align_panel_to_stock_list,
    build_tradeable_mask,
)


def _make_panel(
    n_dates: int = 4,
    codes: list[str] | None = None,
    pct: np.ndarray | None = None,
    close: np.ndarray | None = None,
) -> FactorPanel:
    codes = codes or ["600000.SH", "000001.SZ", "300750.SZ"]
    n_stocks = len(codes)
    shape = (n_dates, n_stocks)
    return FactorPanel(
        factor_array=np.zeros((n_dates, n_stocks, 2), dtype=np.float32),
        return_array=np.full(shape, 0.01, dtype=np.float32),
        pct_change_array=(
            pct if pct is not None else np.zeros(shape, dtype=np.float32)
        ),
        is_st_array=np.zeros(shape, dtype=np.bool_),
        is_suspended_array=np.zeros(shape, dtype=np.bool_),
        days_since_ipo_array=np.full(shape, 1000.0, dtype=np.float32),
        dates=[datetime.date(2024, 1, 2) + datetime.timedelta(days=i) for i in range(n_dates)],
        stock_codes=codes,
        factor_names=["alpha_a", "alpha_b"],
        close_array=close,
    )


def test_build_tradeable_mask_excludes_limit_up_cells() -> None:
    pct = np.zeros((4, 3), dtype=np.float32)
    pct[1, 0] = 0.10  # 600000.SH closes limit-up on day 1
    panel = _make_panel(pct=pct)
    mask = build_tradeable_mask(panel)
    assert mask.shape == (4, 3)
    assert not mask[1, 0]
    assert mask[0, 0] and mask[2, 0]
    assert mask[1, 1] and mask[1, 2]


def test_build_tradeable_mask_excludes_limit_down_cells() -> None:
    # Parity with env._apply_trading_mask: BOTH directions are untradeable.
    pct = np.zeros((2, 3), dtype=np.float32)
    pct[0, 1] = -0.10
    panel = _make_panel(n_dates=2, pct=pct)
    mask = build_tradeable_mask(panel)
    assert not mask[0, 1]
    assert mask[1, 1]


def test_build_tradeable_mask_uses_rounded_price_detection() -> None:
    # +9.80% on a 1.53 prev_close IS limit-up once close prices are used.
    pct = np.zeros((1, 3), dtype=np.float32)
    pct[0, 0] = (1.68 - 1.53) / 1.53
    close = np.full((1, 3), 10.0, dtype=np.float32)
    close[0, 0] = 1.68
    panel = _make_panel(n_dates=1, pct=pct, close=close)
    mask = build_tradeable_mask(panel)
    assert not mask[0, 0]
    assert mask[0, 1]


def test_build_tradeable_mask_st_suspension_ipo_gates() -> None:
    panel = _make_panel(n_dates=3)
    panel.is_st_array[0, 0] = True
    panel.is_suspended_array[1, 1] = True
    panel.days_since_ipo_array[2, 2] = NEW_STOCK_PROTECT_DAYS - 1
    mask = build_tradeable_mask(panel)
    assert not mask[0, 0]
    assert not mask[1, 1]
    assert not mask[2, 2]
    assert mask[0, 1] and mask[1, 0] and mask[2, 0]


def test_loader_populates_close_array(tmp_path: Path) -> None:
    dates = [datetime.date(2022, 1, 3) + datetime.timedelta(days=i) for i in range(3)]
    recs = []
    for i, d in enumerate(dates):
        recs.append({
            "trade_date": d, "ts_code": "600000.SH", "close": 10.0 + i,
            "pct_chg": 0.0, "vol": 1000.0, "alpha_x": float(i),
            "adj_factor": 1.0,
        })
    path = tmp_path / "panel.parquet"
    pl.DataFrame(recs).write_parquet(path)
    loader = FactorPanelLoader(parquet_path=path)
    panel = loader.load_panel(
        start_date=dates[0], end_date=dates[-1], forward_period=1,
        universe_filter=UniverseFilter.ALL_A,
    )
    assert panel.close_array is not None
    j = panel.stock_codes.index("600000.SH")
    assert panel.close_array[0, j] == 10.0
    assert panel.close_array[2, j] == 12.0


def test_align_panel_carries_close_array() -> None:
    close = np.arange(6, dtype=np.float32).reshape(2, 3) + 1.0
    panel = _make_panel(n_dates=2, close=close)
    aligned = align_panel_to_stock_list(panel, ["300750.SZ", "600000.SH", "688001.SH"])
    assert aligned.close_array is not None
    # 300750.SZ was column 2, 600000.SH column 0; 688001.SH missing → NaN.
    assert aligned.close_array[0, 0] == close[0, 2]
    assert aligned.close_array[0, 1] == close[0, 0]
    assert np.isnan(aligned.close_array[0, 2])
