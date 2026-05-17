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
    df = _make_df(
        [
            ("600519.SH", dt.date(2024, 1, 2), True, False, False, "Mao"),
            ("000001.SZ", dt.date(2024, 1, 2), False, True, False, "Bank"),
            ("002594.SZ", dt.date(2024, 1, 2), False, False, False, "BYD"),
        ]
    )
    out = filter_universe(df, mode=UniverseFilter.HS300)
    assert out["ts_code"].to_list() == ["600519.SH"]


def test_zz500_uses_column_when_present():
    df = _make_df(
        [
            ("600519.SH", dt.date(2024, 1, 2), True, False, False, "Mao"),
            ("000001.SZ", dt.date(2024, 1, 2), False, True, False, "Bank"),
        ]
    )
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


# ---------------------------------------------------------------------------
# Tests for paris's 5-universe lock (2026-05-14) — require local
# data/universes/*.parquet to be present. Skip cleanly when absent so CI on a
# fresh checkout (without the membership parquets staged) still passes.
# ---------------------------------------------------------------------------


import pytest  # noqa: E402

from aurumq_rl.data_loader import _load_pit_universe, _load_static_universe  # noqa: E402


def _membership_available(name: str) -> bool:
    return len(_load_static_universe(name)) > 0


@pytest.mark.skipif(
    not _membership_available("NPF"), reason="NPF membership parquet not staged locally"
)
def test_npf_default_is_main_board_only_401():
    """NPF v2.1 default: exactly 401 stocks, all main-board (60[0135]/00[0123])."""
    codes = _load_static_universe("NPF")
    assert len(codes) == 401
    for c in codes:
        assert c.endswith(".SH") or c.endswith(".SZ")
        # main board: SH 60[0135]xxx, SZ 00[0123]xxx
        if c.endswith(".SH"):
            assert c[:3] in ("600", "601", "603", "605"), f"non-main-board SH: {c}"
        else:
            assert c[:3] in ("000", "001", "002", "003"), f"non-main-board SZ: {c}"


@pytest.mark.skipif(
    not _membership_available("NPF"), reason="NPF membership parquet not staged locally"
)
def test_npf_filter_drops_growth_and_star_boards():
    df = pl.DataFrame(
        {
            "ts_code": [
                "600519.SH",  # Mao, MAIN BOARD + NPF
                "300750.SZ",  # CATL, GROWTH (创业)
                "688001.SH",  # 华兴源创, STAR (科创)
                "999999.SZ",
            ]
        }  # synthetic non-existent
    )
    out = filter_universe(df, mode=UniverseFilter.NPF)
    # Mao may or may not be in NPF; growth/star MUST be dropped
    out_codes = set(out["ts_code"].to_list())
    assert "300750.SZ" not in out_codes
    assert "688001.SH" not in out_codes
    assert "999999.SZ" not in out_codes


@pytest.mark.skipif(
    not _membership_available("GROWTH_BOARDS"),
    reason="GROWTH_BOARDS membership parquet not staged locally",
)
def test_growth_boards_keeps_only_300_688_8xxxxx():
    df = pl.DataFrame(
        {"ts_code": ["600519.SH", "300001.SZ", "688001.SH", "836999.BJ", "000001.SZ"]}
    )
    out = filter_universe(df, mode=UniverseFilter.GROWTH_BOARDS)
    kept = set(out["ts_code"].to_list())
    assert "600519.SH" not in kept
    assert "000001.SZ" not in kept
    assert "300001.SZ" in kept
    assert "688001.SH" in kept


@pytest.mark.skipif(
    not _membership_available("MAIN_BOARD"),
    reason="MAIN_BOARD membership parquet not staged locally",
)
def test_main_board_membership_3003():
    codes = _load_static_universe("MAIN_BOARD")
    assert len(codes) == 3003


@pytest.mark.skipif(
    len(_load_pit_universe("CSI300")) == 0, reason="CSI300 PIT parquet not staged locally"
)
def test_csi300_pit_300_per_day():
    """At any single trading day, CSI300 should have exactly 300 stocks."""
    pit = _load_pit_universe("CSI300")
    # Group by trade_date and count
    counts = pit.group_by("trade_date").len()
    # Most trading days should have 300; allow ±5 buffer for rebalance edge cases
    assert (counts["len"] >= 295).all() and (counts["len"] <= 305).all()


@pytest.mark.skipif(
    len(_load_pit_universe("CSI300")) == 0, reason="CSI300 PIT parquet not staged locally"
)
def test_csi300_filter_per_date_join():
    """CSI300 filter joins per (ts_code, trade_date). Picks 600519.SH (Mao, always in)
    but drops 688001.SH (科创 not in CSI300)."""
    df = pl.DataFrame(
        {
            "ts_code": ["600519.SH", "688001.SH"],
            "trade_date": [dt.date(2025, 3, 3), dt.date(2025, 3, 3)],
        }
    )
    out = filter_universe(df, mode=UniverseFilter.CSI300)
    kept = set(out["ts_code"].to_list())
    assert "600519.SH" in kept
    assert "688001.SH" not in kept


def test_legacy_aliases_resolve():
    """HS300 → CSI300, ZZ500 → CSI500, MAIN_BOARD_NON_ST → MAIN_BOARD: same enum semantics."""
    from aurumq_rl.data_loader import _LEGACY_ALIAS

    assert _LEGACY_ALIAS[UniverseFilter.HS300] == UniverseFilter.CSI300
    assert _LEGACY_ALIAS[UniverseFilter.ZZ500] == UniverseFilter.CSI500
    assert _LEGACY_ALIAS[UniverseFilter.MAIN_BOARD_NON_ST] == UniverseFilter.MAIN_BOARD
