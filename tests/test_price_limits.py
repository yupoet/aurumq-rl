"""Tests for board identification and dynamic price-limit detection."""

from __future__ import annotations

import numpy as np
import pytest

from aurumq_rl.price_limits import (
    LISTING_DAY_DOWN_MAIN_GEM,
    LISTING_DAY_UNLIMITED,
    LISTING_DAY_UP_MAIN_GEM,
    StockBoard,
    compute_at_limit_masks,
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


# ---------------------------------------------------------------------------
# M7 — corrected A-share rules
# ---------------------------------------------------------------------------


def test_689_star_cdr_is_star_board() -> None:
    # 689xxx.SH (STAR CDR, e.g. 689009.SH) belongs to STAR, not SH main.
    assert identify_board("689009.SH") is StockBoard.STAR


@pytest.mark.parametrize(
    ("code", "is_st", "days_since_ipo", "exp_up", "exp_down", "unlimited"),
    [
        # ST ±5% applies ONLY to SH/SZ main boards.
        ("600000.SH", True, 100, 0.05, -0.05, False),
        ("000001.SZ", True, 100, 0.05, -0.05, False),
        # Post-2020 registration reform: ST on ChiNext / STAR keeps ±20%.
        ("300750.SZ", True, 100, 0.20, -0.20, False),
        ("688001.SH", True, 100, 0.20, -0.20, False),
        # BSE keeps ±30% for ST.
        ("830799.BJ", True, 100, 0.30, -0.30, False),
        # STAR CDR (689) → ±20%, not main-board ±10%.
        ("689009.SH", False, 100, 0.20, -0.20, False),
        # STAR / ChiNext IPOs: NO price limit for the first 5 trading days.
        ("300750.SZ", False, 3, None, None, True),
        ("300750.SZ", False, 4, None, None, True),
        ("688001.SH", False, 4, None, None, True),
        # Day 5 onwards → normal ±20%.
        ("300750.SZ", False, 5, 0.20, -0.20, False),
        ("300750.SZ", False, 6, 0.20, -0.20, False),
        # Main board: 5-day IPO window NOT applied (post-2023-02-17 variant
        # is a documented TODO — listing date not derivable here).
        ("600000.SH", False, 3, 0.10, -0.10, False),
    ],
)
def test_price_limit_pct_table(
    code: str,
    is_st: bool,
    days_since_ipo: int,
    exp_up: float | None,
    exp_down: float | None,
    unlimited: bool,
) -> None:
    up, down = get_price_limit_pct(code, is_st=is_st, days_since_ipo=days_since_ipo)
    if unlimited:
        assert up >= LISTING_DAY_UNLIMITED
        assert down <= -LISTING_DAY_UNLIMITED
    else:
        assert up == pytest.approx(exp_up)
        assert down == pytest.approx(exp_down)


def test_rounded_limit_up_detected_for_low_price_stock() -> None:
    # Exchange limit prices round to 0.01 元: prev_close 1.53 → limit
    # round(1.53 * 1.10) = 1.68, only +9.80% — below the old 9.9% epsilon.
    pct = (1.68 - 1.53) / 1.53
    assert is_at_limit_up("600519.SH", pct, prev_close=1.53, close=1.68)
    # Same close derived from prev_close + pct_change (no explicit close).
    assert is_at_limit_up("600519.SH", pct, prev_close=1.53)


def test_rounded_limit_up_not_triggered_below_limit_price() -> None:
    # prev_close 10.00 → limit price 11.00; a 10.99 close is NOT limit-up.
    assert not is_at_limit_up("600519.SH", 0.099, prev_close=10.00, close=10.99)
    assert is_at_limit_up("600519.SH", 0.10, prev_close=10.00, close=11.00)


def test_rounded_limit_down_detected_for_low_price_stock() -> None:
    # prev_close 1.53 → down limit round(1.53 * 0.90) = 1.38.
    assert is_at_limit_down("600519.SH", (1.38 - 1.53) / 1.53, prev_close=1.53, close=1.38)
    assert not is_at_limit_down("600519.SH", (1.40 - 1.53) / 1.53, prev_close=1.53, close=1.40)


def test_float32_close_still_detected_at_limit() -> None:
    """Review fix: the loader stores close_array as float32. 48.51 in float32
    is 48.5099983... — a 1e-6 price tolerance would MISS this true limit-up.
    _PRICE_EPSILON is half a tick (5e-3): exact on the 0.01 元 grid (a close
    one tick below the limit can never be flagged), while absorbing float32
    storage error at any realistic A-share price."""
    prev = 44.10  # limit-up price = round(44.10 * 1.10, 2) = 48.51
    pct = np.array([[(48.51 - prev) / prev]], dtype=np.float32)  # production dtype
    close = np.array([[48.51]], dtype=np.float32)
    assert float(close[0, 0]) < 48.51 - 1e-6  # the quantization this guards against
    at_up, _ = compute_at_limit_masks(pct, ["600000.SH"], close=close)
    assert at_up[0, 0]
    # One tick below the limit must stay undetected (half-tick is exact).
    pct2 = np.array([[(48.50 - prev) / prev]], dtype=np.float32)
    close2 = np.array([[48.50]], dtype=np.float32)
    at_up2, _ = compute_at_limit_masks(pct2, ["600000.SH"], close=close2)
    assert not at_up2[0, 0]


def test_half_cent_boundary_rounds_up_not_down() -> None:
    """Review fix: 1.15 * 1.10 = 1.265 exactly in decimal but the float64
    product is 1.2649999999999997 — without the +1e-9 nudge inside the floor
    the limit price comes out 1.26 and a NON-limit 1.26 close is falsely
    flagged. The exchange publishes 1.27 (四舍五入)."""
    assert not is_at_limit_up("600000.SH", (1.26 - 1.15) / 1.15, prev_close=1.15, close=1.26)
    assert is_at_limit_up("600000.SH", (1.27 - 1.15) / 1.15, prev_close=1.15, close=1.27)
    # Same boundary through the vectorized path (prev_close reconstructed).
    pct = np.array([[(1.26 - 1.15) / 1.15]])
    close = np.array([[1.26]])
    at_up, _ = compute_at_limit_masks(pct, ["600000.SH"], close=close)
    assert not at_up[0, 0]


@pytest.mark.parametrize(
    ("prev", "non_limit_close", "limit_close"),
    [
        (1.15, 1.26, 1.27),  # 1.15 * 1.10 = 1.265 → exchange publishes 1.27
        (2.35, 2.58, 2.59),  # 2.35 * 1.10 = 2.585 → exchange publishes 2.59
    ],
)
def test_half_cent_boundary_float32_vectorized(
    prev: float, non_limit_close: float, limit_close: float
) -> None:
    """Re-review fix: under PRODUCTION dtype (float32 close/pct from the
    loader), the reconstructed prev_close carries ~1e-7 relative noise, so
    prev * (1 + limit) can land below the exact half-cent and floor the
    limit price one tick low — falsely flagging a non-limit close (the
    +1e-9 nudge only covers float64 product noise). The reconstruction must
    be snapped to the 0.01 grid (exchange prev closes are tick-quantized)
    BEFORE the limit price is computed."""
    pct = np.array(
        [[(non_limit_close - prev) / prev, (limit_close - prev) / prev]],
        dtype=np.float32,
    )
    close = np.array([[non_limit_close, limit_close]], dtype=np.float32)
    at_up, _ = compute_at_limit_masks(pct, ["600000.SH", "600000.SH"], close=close)
    assert not at_up[0, 0], f"{non_limit_close} is one tick below the true limit {limit_close}"
    assert at_up[0, 1], f"{limit_close} is the true exchange limit price"


# ---------------------------------------------------------------------------
# compute_at_limit_masks — vectorized (T, S) detection shared by training/eval
# ---------------------------------------------------------------------------


def test_compute_at_limit_masks_price_based() -> None:
    codes = ["600000.SH", "300750.SZ", "688001.SH"]
    # Row 0: 600000 at +10% limit (10→11), 300750 at +20% limit (20→24), 688 flat.
    # Row 1: 600000 low-price rounded limit (1.53→1.68), 300750 at -20% limit
    #        (15→12), 688 +5% (not at limit).
    close = np.array(
        [
            [11.00, 24.00, 22.00],
            [1.68, 12.00, 21.00],
        ],
        dtype=np.float64,
    )
    pct = np.array(
        [
            [0.10, 0.20, 0.0],
            [(1.68 - 1.53) / 1.53, -0.20, 0.05],
        ],
        dtype=np.float64,
    )
    at_up, at_down = compute_at_limit_masks(pct, codes, close=close)
    assert at_up.shape == (2, 3) and at_down.shape == (2, 3)
    assert at_up[0].tolist() == [True, True, False]
    assert at_down[0].tolist() == [False, False, False]
    assert at_up[1].tolist() == [True, False, False]
    assert at_down[1].tolist() == [False, True, False]


@pytest.mark.parametrize(
    ("prev_close", "limit_close", "below_close"),
    [
        (91.00, 100.10, 100.09),
        (68.20, 75.02, 75.01),
        (152.00, 167.20, 167.19),
    ],
)
def test_rounded_limit_detection_float32_high_price(
    prev_close: float, limit_close: float, below_close: float
) -> None:
    """Regression: the close pipeline is float32 (data_loader.close_array),
    whose half-ulp at prices >= 64 is ~3.8e-6 — a 1e-6 tolerance rejects TRUE
    limit closes like 91.00 → 100.10. Tolerance must cover float32 noise
    (half a 0.01 tick is safe: the nearest non-limit close is one tick away).
    """
    pct = np.float32((limit_close - prev_close) / prev_close)
    # Scalar API with float32 inputs.
    assert is_at_limit_up(
        "600519.SH",
        float(pct),
        prev_close=float(np.float32(prev_close)),
        close=float(np.float32(limit_close)),
    )
    assert not is_at_limit_up(
        "600519.SH",
        float(np.float32((below_close - prev_close) / prev_close)),
        prev_close=float(np.float32(prev_close)),
        close=float(np.float32(below_close)),
    )
    # Vectorized API fed float32 arrays end-to-end.
    close32 = np.array([[limit_close, below_close]], dtype=np.float32)
    pct32 = np.array(
        [[(limit_close - prev_close) / prev_close, (below_close - prev_close) / prev_close]],
        dtype=np.float32,
    )
    at_up, _ = compute_at_limit_masks(pct32, ["600519.SH", "600519.SH"], close=close32)
    assert at_up[0, 0], "true limit-up close must be detected from float32 prices"
    assert not at_up[0, 1], "one tick below the limit must NOT be flagged"


def test_rounded_limit_down_detection_float32_high_price() -> None:
    # prev 91.00 → down limit round(81.90) = 81.90; float32 81.90 ≈ 81.900002.
    prev_close, limit_close, above_close = 91.00, 81.90, 81.91
    assert is_at_limit_down(
        "600519.SH",
        float(np.float32((limit_close - prev_close) / prev_close)),
        prev_close=float(np.float32(prev_close)),
        close=float(np.float32(limit_close)),
    )
    assert not is_at_limit_down(
        "600519.SH",
        float(np.float32((above_close - prev_close) / prev_close)),
        prev_close=float(np.float32(prev_close)),
        close=float(np.float32(above_close)),
    )


def test_compute_at_limit_masks_st_per_day() -> None:
    codes = ["600000.SH", "300750.SZ"]
    # +5% closes; only the main-board stock is at limit when ST that day.
    pct = np.array([[0.05, 0.05], [0.05, 0.05]], dtype=np.float64)
    close = np.array([[10.50, 21.00], [10.50, 21.00]], dtype=np.float64)
    is_st = np.array([[True, True], [False, False]])
    at_up, _ = compute_at_limit_masks(pct, codes, is_st=is_st, close=close)
    assert at_up[0].tolist() == [True, False]  # ST day: main ±5%, GEM stays ±20%
    assert at_up[1].tolist() == [False, False]  # non-ST day: main ±10%


def test_compute_at_limit_masks_ipo_no_limit_window() -> None:
    codes = ["300750.SZ"]
    pct = np.array([[0.30], [0.30]], dtype=np.float64)
    close = np.array([[13.0], [13.0]], dtype=np.float64)
    days = np.array([[3.0], [6.0]])
    at_up, _ = compute_at_limit_masks(pct, codes, days_since_ipo=days, close=close)
    assert not at_up[0, 0]  # day 3: unlimited, +30% is a legal close
    assert at_up[1, 0]  # day 6: ±20% applies, +30% close is beyond the limit


def test_compute_at_limit_masks_pct_fallback_without_close() -> None:
    # No close available → documented pct-epsilon fallback.
    codes = ["600000.SH", "300750.SZ"]
    pct = np.array([[0.10, 0.10]], dtype=np.float64)
    at_up, at_down = compute_at_limit_masks(pct, codes)
    assert at_up[0].tolist() == [True, False]
    assert at_down[0].tolist() == [False, False]


def test_compute_at_limit_masks_unknown_board_uses_legacy_threshold() -> None:
    codes = ["SYN0000.SH"]  # non-numeric → unknown board
    pct = np.array([[0.05], [0.10]], dtype=np.float64)
    at_up, _ = compute_at_limit_masks(pct, codes)
    assert not at_up[0, 0]
    assert at_up[1, 0]  # |pct| >= legacy 9.5% threshold
