"""Tests for board identification and dynamic price-limit detection."""

from __future__ import annotations

import pytest

from aurumq_rl.price_limits import (
    LISTING_DAY_DOWN_MAIN_GEM,
    LISTING_DAY_UNLIMITED,
    LISTING_DAY_UP_MAIN_GEM,
    StockBoard,
    get_price_limit_pct,
    identify_board,
    is_at_limit_down,
    is_at_limit_up,
)


# ---------------------------------------------------------------------------
# Board identification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        # Shanghai main board
        ("600000.SH", StockBoard.SH_MAIN),
        ("600519.SH", StockBoard.SH_MAIN),
        ("601398.SH", StockBoard.SH_MAIN),
        ("603000.SH", StockBoard.SH_MAIN),
        ("605588.SH", StockBoard.SH_MAIN),
        # Shenzhen main board
        ("000001.SZ", StockBoard.SZ_MAIN),
        ("001979.SZ", StockBoard.SZ_MAIN),
        ("002594.SZ", StockBoard.SZ_MAIN),
        ("003816.SZ", StockBoard.SZ_MAIN),
        # ChiNext
        ("300750.SZ", StockBoard.GEM),
        ("301236.SZ", StockBoard.GEM),
        # STAR
        ("688981.SH", StockBoard.STAR),
        ("688001.SH", StockBoard.STAR),
        # BSE (8/4 prefix)
        ("830799.BJ", StockBoard.BJ),
        ("430047.BJ", StockBoard.BJ),
    ],
)
def test_identify_board_supported(code: str, expected: StockBoard) -> None:
    assert identify_board(code) is expected


def test_identify_board_handles_bare_numeric() -> None:
    assert identify_board("600519") is StockBoard.SH_MAIN
    assert identify_board("300750") is StockBoard.GEM


def test_identify_board_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        identify_board("ABCDEF.SH")
    with pytest.raises(ValueError):
        identify_board("999999.SH")  # unknown 9xx prefix


# ---------------------------------------------------------------------------
# Price-limit thresholds
# ---------------------------------------------------------------------------


def test_main_board_default_pct() -> None:
    up, down = get_price_limit_pct("600000.SH")
    assert up == pytest.approx(0.10)
    assert down == pytest.approx(-0.10)


def test_chinext_default_pct() -> None:
    up, down = get_price_limit_pct("300750.SZ")
    assert up == pytest.approx(0.20)
    assert down == pytest.approx(-0.20)


def test_star_default_pct() -> None:
    up, down = get_price_limit_pct("688981.SH")
    assert up == pytest.approx(0.20)


def test_bse_default_pct() -> None:
    up, down = get_price_limit_pct("830799.BJ")
    assert up == pytest.approx(0.30)
    assert down == pytest.approx(-0.30)


def test_st_overrides_to_5pct() -> None:
    up, down = get_price_limit_pct("600000.SH", is_st=True)
    assert up == pytest.approx(0.05)
    assert down == pytest.approx(-0.05)


def test_listing_day_main_board() -> None:
    up, down = get_price_limit_pct("600000.SH", is_listing_day=True)
    assert up == pytest.approx(LISTING_DAY_UP_MAIN_GEM)
    assert down == pytest.approx(LISTING_DAY_DOWN_MAIN_GEM)


def test_listing_day_star_unlimited() -> None:
    up, down = get_price_limit_pct("688001.SH", is_listing_day=True)
    assert up >= LISTING_DAY_UNLIMITED


def test_listing_day_bse_unlimited() -> None:
    up, down = get_price_limit_pct("830799.BJ", is_listing_day=True)
    assert up >= LISTING_DAY_UNLIMITED


# ---------------------------------------------------------------------------
# is_at_limit_up / is_at_limit_down
# ---------------------------------------------------------------------------


def test_main_board_limit_up_detection() -> None:
    assert is_at_limit_up("600519.SH", 0.10)
    assert is_at_limit_up("600519.SH", 0.099)  # within epsilon
    assert not is_at_limit_up("600519.SH", 0.05)


def test_main_board_limit_down_detection() -> None:
    assert is_at_limit_down("600519.SH", -0.10)
    assert not is_at_limit_down("600519.SH", -0.05)


def test_chinext_limit_up_at_20pct() -> None:
    assert is_at_limit_up("300750.SZ", 0.20)
    assert not is_at_limit_up("300750.SZ", 0.10)  # main-board threshold doesn't trigger
    assert is_at_limit_down("300750.SZ", -0.20)


def test_st_stock_limit_5pct() -> None:
    # 5% triggers ST limit but not normal main-board limit
    assert is_at_limit_up("600519.SH", 0.05, is_st=True)
    assert not is_at_limit_up("600519.SH", 0.05, is_st=False)


def test_listing_day_main_board_limit() -> None:
    # listing day main: +44%
    assert is_at_limit_up("600000.SH", 0.44, is_listing_day=True)
    assert not is_at_limit_up("600000.SH", 0.10, is_listing_day=True)


def test_listing_day_star_never_at_limit() -> None:
    # STAR listing day: unlimited, so any pct returns False
    assert not is_at_limit_up("688001.SH", 0.50, is_listing_day=True)
    assert not is_at_limit_down("688001.SH", -0.50, is_listing_day=True)
