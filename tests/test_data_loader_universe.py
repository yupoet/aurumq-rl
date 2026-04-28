"""Universe-filter tests covering both column-based and heuristic paths."""
from __future__ import annotations

import datetime as dt

import polars as pl

from aurumq_rl.data_loader import UniverseFilter, filter_universe


def _make_df(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=["ts_code", "trade_date", "is_hs300", "is_zz500", "is_st", "name"],
        orient="row",
    )


def test_hs300_uses_column_when_present():
    df = _make_df([
        ("600519.SH", dt.date(2024, 1, 2), True,  False, False, "Mao"),
        ("000001.SZ", dt.date(2024, 1, 2), False, True,  False, "Bank"),
        ("002594.SZ", dt.date(2024, 1, 2), False, False, False, "BYD"),
    ])
    out = filter_universe(df, mode=UniverseFilter.HS300)
    assert out["ts_code"].to_list() == ["600519.SH"]


def test_zz500_uses_column_when_present():
    df = _make_df([
        ("600519.SH", dt.date(2024, 1, 2), True,  False, False, "Mao"),
        ("000001.SZ", dt.date(2024, 1, 2), False, True,  False, "Bank"),
    ])
    out = filter_universe(df, mode=UniverseFilter.ZZ500)
    assert out["ts_code"].to_list() == ["000001.SZ"]


def test_hs300_falls_back_to_main_board_when_column_missing():
    df = pl.DataFrame(
        [("600519.SH",), ("8XYZ23.BJ",), ("300001.SZ",)],
        schema=["ts_code"],
        orient="row",
    )
    out = filter_universe(df, mode=UniverseFilter.HS300)
    assert out["ts_code"].to_list() == ["600519.SH"]


def test_main_board_non_st_unchanged():
    df = pl.DataFrame(
        [("600519.SH", "Mao", False), ("600000.SH", "*ST X", True)],
        schema=["ts_code", "name", "is_st"],
        orient="row",
    )
    out = filter_universe(df, mode=UniverseFilter.MAIN_BOARD_NON_ST)
    assert out["ts_code"].to_list() == ["600519.SH"]
