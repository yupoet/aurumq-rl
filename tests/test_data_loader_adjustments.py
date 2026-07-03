"""Tests for C1/C2 data-layer fixes.

C1 — forward returns must use adjusted close (``close * adj_factor``) when the
panel parquet carries ``adj_factor``; otherwise fall back to raw close with a
loud warning (returns are corrupted around corporate actions).

C2 — missing (date, stock) cells must be un-tradeable: ``is_suspended=True``,
NaN close/forward-return, ``days_since_ipo=0``, and factor NaNs that do not
bias the cross-sectional z-score of the other stocks.
"""

from __future__ import annotations

import datetime
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from aurumq_rl.data_loader import (
    NEW_STOCK_PROTECT_DAYS,
    FactorPanelLoader,
    UniverseFilter,
    _safe_log_return,
    pivot_adjusted_close,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weekdays(start: datetime.date, n: int) -> list[datetime.date]:
    out: list[datetime.date] = []
    cur = start
    while len(out) < n:
        if cur.weekday() < 5:
            out.append(cur)
        cur += datetime.timedelta(days=1)
    return out


def _load(path: Path, dates: list[datetime.date], forward_period: int):
    loader = FactorPanelLoader(parquet_path=path)
    return loader.load_panel(
        start_date=dates[0],
        end_date=dates[-1],
        forward_period=forward_period,
        universe_filter=UniverseFilter.ALL_A,
    )


def _ex_date_df(with_adj: bool) -> tuple[pl.DataFrame, list[datetime.date]]:
    """Two stocks, six days. SYN_A has a 1:1 split on day 3.

    Raw close halves (10 → 5) while adj_factor doubles (1 → 2), so the
    ADJUSTED close is flat: the economic return across the ex-date is 0.
    """
    dates = _weekdays(datetime.date(2022, 1, 3), 6)
    close_a = [10.0, 10.0, 10.0, 5.0, 5.0, 5.0]
    adj_a = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    recs: list[dict] = []
    for i, d in enumerate(dates):
        recs.append(
            {
                "trade_date": d,
                "ts_code": "SYN_A",
                "close": close_a[i],
                "pct_chg": 0.0,
                "vol": 1000.0,
                "alpha_x": float(i),
                "adj_factor": adj_a[i],
            }
        )
        recs.append(
            {
                "trade_date": d,
                "ts_code": "SYN_B",
                "close": 20.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "alpha_x": float(-i),
                "adj_factor": 1.0,
            }
        )
    df = pl.DataFrame(recs)
    if not with_adj:
        df = df.drop("adj_factor")
    return df, dates


# ---------------------------------------------------------------------------
# C1 — adjusted-close forward returns
# ---------------------------------------------------------------------------


def test_forward_return_uses_adj_factor_across_ex_date(tmp_path: Path) -> None:
    df, dates = _ex_date_df(with_adj=True)
    path = tmp_path / "panel_adj.parquet"
    df.write_parquet(path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        panel = _load(path, dates, forward_period=1)
    assert not any("adj_factor" in str(w.message) for w in caught), (
        "no adj_factor warning expected when the column is present"
    )

    a = panel.stock_codes.index("SYN_A")
    b = panel.stock_codes.index("SYN_B")
    # Across the ex-date (t=2 → t=3): raw close halves but adj_close is flat,
    # so the economic forward return is ~0, not log(0.5).
    assert abs(float(panel.return_array[2, a])) < 1e-6
    # Flat stock stays 0 everywhere in the computed range.
    assert abs(float(panel.return_array[2, b])) < 1e-6


def test_forward_return_without_adj_factor_warns_and_uses_raw_close(
    tmp_path: Path,
) -> None:
    df, dates = _ex_date_df(with_adj=False)
    path = tmp_path / "panel_raw.parquet"
    df.write_parquet(path)

    with pytest.warns(UserWarning, match="adj_factor"):
        panel = _load(path, dates, forward_period=1)

    a = panel.stock_codes.index("SYN_A")
    # Raw-close fallback: the ex-date gap shows up as a fake -log(2) return.
    assert np.isclose(float(panel.return_array[2, a]), np.log(0.5), atol=1e-6)


# ---------------------------------------------------------------------------
# C2 — missing (date, stock) cells are un-tradeable NaNs
# ---------------------------------------------------------------------------


@pytest.fixture
def missing_cell_panel(tmp_path: Path):
    """3 stocks x 8 days; SYN_Z has NO row on day 3. adj_factor=1 everywhere."""
    dates = _weekdays(datetime.date(2022, 1, 3), 8)
    alpha = {"SYN_X": 1.0, "SYN_Y": 2.0, "SYN_Z": 4.0}
    recs: list[dict] = []
    for i, d in enumerate(dates):
        for code, val in alpha.items():
            if i == 3 and code == "SYN_Z":
                continue  # the missing cell
            recs.append(
                {
                    "trade_date": d,
                    "ts_code": code,
                    "close": 10.0,
                    "pct_chg": 0.0,
                    "vol": 1000.0,
                    "alpha_x": val,
                    "adj_factor": 1.0,
                }
            )
    path = tmp_path / "panel_missing.parquet"
    pl.DataFrame(recs).write_parquet(path)
    return _load(path, dates, forward_period=2)


def test_missing_cell_is_suspended(missing_cell_panel) -> None:
    panel = missing_cell_panel
    z = panel.stock_codes.index("SYN_Z")
    x = panel.stock_codes.index("SYN_X")
    assert panel.is_suspended_array[3, z], "missing cell must be suspended"
    assert not panel.is_suspended_array[3, x]
    assert not panel.is_suspended_array[2, z]


def test_missing_cell_forward_return_is_nan(missing_cell_panel) -> None:
    panel = missing_cell_panel
    z = panel.stock_codes.index("SYN_Z")
    x = panel.stock_codes.index("SYN_X")
    # Return AT the missing date is NaN (no valid price at t).
    assert np.isnan(panel.return_array[3, z])
    # Return whose forward window LANDS on the missing date is NaN too.
    assert np.isnan(panel.return_array[1, z])
    # Present cells stay finite.
    assert np.isfinite(panel.return_array[1, x])
    assert np.isfinite(panel.return_array[3, x])


def test_missing_cell_fails_ipo_gate(missing_cell_panel) -> None:
    panel = missing_cell_panel
    z = panel.stock_codes.index("SYN_Z")
    x = panel.stock_codes.index("SYN_X")
    # Absent cell → 0 days since IPO (fails the 60-day gate).
    assert panel.days_since_ipo_array[3, z] == 0
    # Present cells without a days_since_ipo column keep the mature default.
    assert panel.days_since_ipo_array[3, x] == NEW_STOCK_PROTECT_DAYS * 2
    assert panel.days_since_ipo_array[2, z] == NEW_STOCK_PROTECT_DAYS * 2


def test_missing_cell_does_not_bias_cross_section_zscore(missing_cell_panel) -> None:
    panel = missing_cell_panel
    x = panel.stock_codes.index("SYN_X")
    y = panel.stock_codes.index("SYN_Y")
    z = panel.stock_codes.index("SYN_Z")
    # On the missing day the cross-section is [1.0, 2.0] → z = [-1, +1],
    # exactly as if SYN_Z did not exist at all. A zero-filled raw cell would
    # instead give mean=1.0, std=0.816 → z ≈ [0, 1.22].
    assert np.allclose(panel.factor_array[3, [x, y], 0], [-1.0, 1.0], atol=1e-3)
    # The missing cell itself is neutral-0 after z-scoring.
    assert panel.factor_array[3, z, 0] == 0.0


# ---------------------------------------------------------------------------
# _safe_log_return — invalid prices → NaN, not 0.0
# ---------------------------------------------------------------------------


def test_safe_log_return_invalid_prices_are_nan() -> None:
    now = np.array([0.0, np.nan, 10.0, 10.0, -1.0], dtype=np.float32)
    fwd = np.array([10.0, 10.0, np.nan, 0.0, 10.0], dtype=np.float32)
    out = _safe_log_return(now, fwd)
    assert np.isnan(out).all(), "0/NaN/negative prices must yield NaN returns"


def test_safe_log_return_valid_prices() -> None:
    out = _safe_log_return(np.array([10.0], dtype=np.float32), np.array([20.0], dtype=np.float32))
    assert np.allclose(out, np.log(2.0), atol=1e-6)


# ---------------------------------------------------------------------------
# pivot_adjusted_close — shared helper for the wave-label scripts
# ---------------------------------------------------------------------------


def _pivot_df(with_adj: bool) -> tuple[pl.DataFrame, list[datetime.date]]:
    d0, d1 = datetime.date(2022, 1, 3), datetime.date(2022, 1, 4)
    df = pl.DataFrame(
        {
            "trade_date": [d0, d0, d1, d1],
            "ts_code": ["SYN_A", "SYN_B", "SYN_A", "SYN_B"],
            "close": [10.0, 20.0, 11.0, 21.0],
            "adj_factor": [2.0, 1.0, 2.0, 1.0],
        }
    )
    if not with_adj:
        df = df.drop("adj_factor")
    return df, [d0, d1]


def test_pivot_adjusted_close_multiplies_adj_factor() -> None:
    df, dates = _pivot_df(with_adj=True)
    arr = pivot_adjusted_close(df, ["SYN_A", "SYN_B"], dates)
    assert arr.shape == (2, 2)
    np.testing.assert_allclose(arr[:, 0], [20.0, 22.0])  # close * 2
    np.testing.assert_allclose(arr[:, 1], [20.0, 21.0])  # close * 1


def test_pivot_adjusted_close_raw_fallback_without_adj_factor() -> None:
    df, dates = _pivot_df(with_adj=False)
    arr = pivot_adjusted_close(df, ["SYN_A", "SYN_B"], dates)
    np.testing.assert_allclose(arr[:, 0], [10.0, 11.0])
    np.testing.assert_allclose(arr[:, 1], [20.0, 21.0])


def test_pivot_adjusted_close_null_adj_factor_propagates_nan() -> None:
    """A present close with a NULL adj_factor must NOT become a 0-price point.

    Mixing a raw-price fallback (or a fake 0.0) into an otherwise adjusted
    series fabricates returns / MA values. Loader semantics: the cell is
    invalid → NaN.
    """
    d0, d1 = datetime.date(2022, 1, 3), datetime.date(2022, 1, 4)
    df = pl.DataFrame(
        {
            "trade_date": [d0, d1],
            "ts_code": ["SYN_A", "SYN_A"],
            "close": [10.0, 11.0],
            "adj_factor": [2.0, None],
        }
    )
    arr = pivot_adjusted_close(df, ["SYN_A"], [d0, d1])
    assert arr[0, 0] == pytest.approx(20.0)
    assert np.isnan(arr[1, 0])


def test_pivot_adjusted_close_null_close_fills_zero() -> None:
    """A NULL close (missing quote) keeps the legacy 0.0 pivot convention."""
    d0, d1 = datetime.date(2022, 1, 3), datetime.date(2022, 1, 4)
    df = pl.DataFrame(
        {
            "trade_date": [d0, d1],
            "ts_code": ["SYN_A", "SYN_A"],
            "close": [10.0, None],
            "adj_factor": [2.0, 2.0],
        }
    )
    arr = pivot_adjusted_close(df, ["SYN_A"], [d0, d1])
    assert arr[0, 0] == pytest.approx(20.0)
    assert arr[1, 0] == 0.0


def test_pivot_adjusted_close_missing_code_zero_and_row_alignment() -> None:
    df, dates = _pivot_df(with_adj=True)
    # Unknown code → zero column; rows follow the requested date order.
    arr = pivot_adjusted_close(df, ["SYN_A", "SYN_MISSING"], dates)
    assert arr.shape == (2, 2)
    np.testing.assert_allclose(arr[:, 1], [0.0, 0.0])
    np.testing.assert_allclose(arr[:, 0], [20.0, 22.0])


# ---------------------------------------------------------------------------
# main-wave label functions — amount stays on RAW close via explicit override
# ---------------------------------------------------------------------------


def test_main_wave_labels_amount_override() -> None:
    from aurumq_rl.main_wave_labels import MainWaveConfig, compute_main_wave_labels

    T, S = 12, 2
    close = np.full((T, S), 10.0)
    vol = np.full((T, S), 1e3)  # default amount = 1e4 → illiquid
    pct = np.zeros((T, S))
    valid = np.ones((T, S), dtype=bool)
    cfg = MainWaveConfig(amount_ma_window=2)

    labels_default = compute_main_wave_labels(close, pct, vol, valid, cfg)
    assert not labels_default.liquid_mask.any()

    labels_amt = compute_main_wave_labels(close, pct, vol, valid, cfg, amount=np.full((T, S), 2e8))
    assert labels_amt.liquid_mask[cfg.amount_ma_window :].all()


def test_find_main_wave_episodes_accepts_amount_override() -> None:
    from aurumq_rl.main_wave_episodes import EpisodeConfig, find_main_wave_episodes

    T, S = 30, 2
    close = np.full((T, S), 10.0)
    vol = np.full((T, S), 1e3)
    valid = np.ones((T, S), dtype=bool)
    eps = find_main_wave_episodes(close, vol, valid, EpisodeConfig(), amount=np.full((T, S), 2e8))
    assert isinstance(eps, list)


# ---------------------------------------------------------------------------
# env._apply_trading_mask — NaN returns must not leak into the reward
# ---------------------------------------------------------------------------


def test_apply_trading_mask_zeroes_nan_returns() -> None:
    from aurumq_rl.env import _apply_trading_mask

    returns = np.array([np.nan, 0.05], dtype=np.float64)
    zeros = np.zeros(2)
    masked = _apply_trading_mask(
        returns=returns,
        pct_changes=zeros,
        is_st=np.zeros(2, dtype=bool),
        is_suspended=np.zeros(2, dtype=bool),
        days_since_ipo=np.full(2, 500.0),
        stock_codes=None,
        respect_dynamic_price_limits=False,
    )
    assert masked[0] == 0.0, "NaN forward return must be masked to 0"
    assert masked[1] == pytest.approx(0.05)
