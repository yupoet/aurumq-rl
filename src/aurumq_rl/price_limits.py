"""A-share dynamic price-limit threshold detection.

Public market rules (publicly available):
  * Main board (沪/深): ±10%
  * ChiNext (300/301): ±20% (since 2020-08-24 registration system reform)
  * STAR (688/689): ±20%
  * BSE (8/4): ±30% (no limit on listing day, ±30% from day 2)
  * ST/*ST: ±5% — ONLY on SH/SZ main boards. Post-2020 registration reform,
    ChiNext/STAR ST stocks keep ±20% and BSE keeps ±30%.
  * IPO no-limit window: STAR/ChiNext have NO price limit for the first
    5 TRADING days (day index 0..4 since IPO). BSE: listing day only.
    Main-board listing day: +44%/-36%.

Stock code format (industry standard XXXXXX.SZ/SH/BJ):
  600xxx / 601xxx / 603xxx / 605xxx → SH main board (.SH)
  688xxx / 689xxx                   → STAR market (.SH; 689 = STAR CDR)
  000xxx / 001xxx / 002xxx / 003xxx → SZ main board (.SZ)
  300xxx / 301xxx                   → ChiNext (.SZ)
  8xxxxx / 4xxxxx                   → BSE (.BJ)

Limit detection reconstructs the exchange's ROUNDED limit price
(``round(prev_close * (1 ± limit_pct), 2)``, half-up to 0.01 元) whenever
``prev_close`` is available, because low-priced stocks can close exactly at
the limit price while the raw pct_change stays below any pct epsilon
(prev_close 1.53 → limit 1.68 = +9.80%). A pct-epsilon comparison is kept
ONLY as a fallback for callers without price data.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:  # pragma: no cover — Python 3.10 compat shim
    from enum import Enum

    class StrEnum(str, Enum):
        """Python 3.10 backport of :class:`enum.StrEnum`."""

        def __str__(self) -> str:  # type: ignore[override]
            return str(self.value)


class StockBoard(StrEnum):
    """A-share listing board."""

    SH_MAIN = "sh_main"  # Shanghai main board (60xxxx)
    SZ_MAIN = "sz_main"  # Shenzhen main board (00xxxx)
    GEM = "gem"  # ChiNext (300/301)
    STAR = "star"  # STAR market (688)
    BJ = "bj"  # BSE (8/4)


# Listing-day price limits (public rules)
LISTING_DAY_UP_MAIN_GEM: float = 0.44
LISTING_DAY_DOWN_MAIN_GEM: float = -0.36

# STAR / BSE listing day: unlimited (sentinel value)
LISTING_DAY_UNLIMITED: float = 1e9

# STAR / ChiNext registration-system IPOs: no price limit for the first
# 5 TRADING days (days_since_ipo 0..4).
IPO_NO_LIMIT_TRADING_DAYS: int = 5

# Tolerance when comparing a close against a reconstructed limit price
# (both quantised to 0.01 元, so 1e-6 comfortably absorbs float noise).
# Half a tick (0.01 元 grid): close and reconstructed limit price both live
# on the 0.01 grid, so ANY tolerance below one tick is exact — no false
# positives possible — while absorbing float32 storage error (the loader
# stores close_array as float32: up to ~2e-6 relative, i.e. ~6e-5 absolute
# at 1000 元 — a 1e-6 tolerance would MISS true limit closes there).
_PRICE_EPSILON: float = 5e-3


def identify_board(stock_code: str) -> StockBoard:
    """Identify listing board from stock code.

    Parameters
    ----------
    stock_code:
        Format ``XXXXXX.SH/SZ/BJ`` (e.g. ``688001.SH``, ``300750.SZ``)
        or bare 6-digit number.

    Returns
    -------
    StockBoard

    Raises
    ------
    ValueError when prefix cannot be identified.

    Examples
    --------
    >>> identify_board("600519.SH")
    <StockBoard.SH_MAIN: 'sh_main'>
    >>> identify_board("688001.SH")
    <StockBoard.STAR: 'star'>
    """
    numeric = stock_code.split(".")[0] if "." in stock_code else stock_code

    if not numeric.isdigit():
        raise ValueError(
            f"Cannot parse stock code: {stock_code!r} "
            "(expected 6-digit numeric, optionally with .SH/.SZ/.BJ suffix)"
        )

    if numeric.startswith(("8", "4")):
        return StockBoard.BJ
    # 689xxx = STAR CDRs (e.g. 689009.SH) — must NOT fall through to SH main.
    if numeric.startswith(("688", "689")):
        return StockBoard.STAR
    if numeric.startswith(("300", "301")):
        return StockBoard.GEM
    if numeric.startswith("6"):
        return StockBoard.SH_MAIN
    if numeric.startswith(("000", "001", "002", "003")):
        return StockBoard.SZ_MAIN

    raise ValueError(f"Unknown board for: {stock_code!r} (prefix={numeric[:3]!r})")


# Backward-compat alias used in some code paths
detect_board = identify_board


def get_price_limit_pct(
    stock_code: str,
    is_st: bool = False,
    is_listing_day: bool = False,
    days_since_ipo: float | None = None,
) -> tuple[float, float]:
    """Return (upper_limit_pct, lower_limit_pct) as decimals, e.g. (0.20, -0.20).

    Priority (M7 corrected):
      1. STAR/ChiNext within the first 5 trading days since IPO → unlimited
      2. Listing day (or ``days_since_ipo <= 0``): BSE unlimited,
         main board +44%/-36%
      3. ST/*ST → ±5% — ONLY on SH/SZ main boards (post-2020 registration
         reform, ChiNext/STAR ST keep ±20%, BSE keeps ±30%)
      4. Board default: main ±10%, ChiNext/STAR ±20%, BSE ±30%

    TODO(M7): main-board IPOs listed after the 2023-02-17 full-registration
    reform also have 5 no-limit trading days, but the listing DATE cannot be
    derived reliably from what this function receives (days_since_ipo alone
    cannot distinguish pre/post-reform listings). Do not guess listing dates;
    thread a listing-date signal through the panel before enabling it here.

    Parameters
    ----------
    days_since_ipo:
        Trading days since IPO (0 = listing day). Optional; when omitted only
        ``is_listing_day`` gates the IPO rules (legacy behaviour). Caveat:
        ``days_since_ipo <= 0`` is treated as the listing day, which also
        matches panel cells whose value defaults to 0 (missing data) — mask
        consumers are protected because the ≥60-day IPO gate in
        ``build_tradeable_mask`` / ``_apply_trading_mask`` excludes those
        cells regardless.

    Returns
    -------
    (up_limit, down_limit) as decimal fractions.
    """
    board = identify_board(stock_code)

    if days_since_ipo is not None and days_since_ipo <= 0:
        is_listing_day = True

    # 1. STAR / ChiNext: no price limit for the first 5 trading days.
    if board in (StockBoard.STAR, StockBoard.GEM):
        in_no_limit_window = is_listing_day or (
            days_since_ipo is not None and days_since_ipo < IPO_NO_LIMIT_TRADING_DAYS
        )
        if in_no_limit_window:
            return (LISTING_DAY_UNLIMITED, -LISTING_DAY_UNLIMITED)

    # 2. Listing day for the remaining boards.
    if is_listing_day:
        if board == StockBoard.BJ:
            return (LISTING_DAY_UNLIMITED, -LISTING_DAY_UNLIMITED)
        return (LISTING_DAY_UP_MAIN_GEM, LISTING_DAY_DOWN_MAIN_GEM)

    # 3. ST ±5% applies only to the SH/SZ main boards.
    if is_st and board in (StockBoard.SH_MAIN, StockBoard.SZ_MAIN):
        return (0.05, -0.05)

    # 4. Board defaults.
    if board in (StockBoard.GEM, StockBoard.STAR):
        return (0.20, -0.20)
    if board == StockBoard.BJ:
        return (0.30, -0.30)
    return (0.10, -0.10)


def _round_limit_price(price: float) -> float:
    """Round to 0.01 元, half away from zero (exchange 四舍五入 convention).

    Python's built-in ``round`` is banker's rounding and would map e.g.
    2.585 → 2.58 where the exchange publishes 2.59.

    The +1e-9 nudge keeps exact half-cent products from flooring one tick
    low when the float product lands within rounding error BELOW the true
    half-cent (2.35 * 1.10 = 2.5849999999999997 → must still give 2.59).
    """
    return math.floor(price * 100.0 + 0.5 + 1e-9) / 100.0


def is_at_limit_up(
    stock_code: str,
    pct_change: float,
    is_st: bool = False,
    is_listing_day: bool = False,
    epsilon: float = 1e-3,
    prev_close: float | None = None,
    close: float | None = None,
    days_since_ipo: float | None = None,
) -> bool:
    """Check whether the stock closed at its upper price limit.

    When ``prev_close`` is available the exchange's ROUNDED limit price is
    reconstructed (``round(prev_close * (1 + limit_pct), 2)``) and compared
    against ``close`` (derived from ``prev_close * (1 + pct_change)`` when
    not given). This catches low-priced stocks whose pct_change sits below
    the epsilon (e.g. prev_close 1.53 → limit 1.68 = +9.80%).

    Without ``prev_close`` a pct-epsilon fallback is used — documented as
    less precise for low-priced stocks.

    Parameters
    ----------
    pct_change : daily change in **decimal form** (e.g. +10% → 0.10).

    Examples
    --------
    >>> is_at_limit_up("000001.SZ", 0.10)   # main board +10%
    True
    >>> is_at_limit_up("300750.SZ", 0.20)   # ChiNext +20%
    True
    """
    up_limit, _ = get_price_limit_pct(
        stock_code, is_st=is_st, is_listing_day=is_listing_day, days_since_ipo=days_since_ipo
    )
    if up_limit >= LISTING_DAY_UNLIMITED:
        return False
    if prev_close is not None and prev_close > 0:
        close_val = close if close is not None else prev_close * (1.0 + pct_change)
        limit_price = _round_limit_price(prev_close * (1.0 + up_limit))
        return close_val >= limit_price - _PRICE_EPSILON
    # Fallback: pct-epsilon detection (no price data available).
    return pct_change >= up_limit - epsilon


def is_at_limit_down(
    stock_code: str,
    pct_change: float,
    is_st: bool = False,
    is_listing_day: bool = False,
    epsilon: float = 1e-3,
    prev_close: float | None = None,
    close: float | None = None,
    days_since_ipo: float | None = None,
) -> bool:
    """Check whether the stock closed at its lower price limit.

    See :func:`is_at_limit_up` for the rounded-limit-price semantics and
    the pct-epsilon fallback.
    """
    _, down_limit = get_price_limit_pct(
        stock_code, is_st=is_st, is_listing_day=is_listing_day, days_since_ipo=days_since_ipo
    )
    if down_limit <= -LISTING_DAY_UNLIMITED:
        return False
    if prev_close is not None and prev_close > 0:
        close_val = close if close is not None else prev_close * (1.0 + pct_change)
        limit_price = _round_limit_price(prev_close * (1.0 + down_limit))
        return close_val <= limit_price + _PRICE_EPSILON
    # Fallback: pct-epsilon detection (no price data available).
    return pct_change <= down_limit + epsilon


_LEGACY_THRESHOLD: float = 0.095  # legacy fallback for unidentifiable boards


@lru_cache(maxsize=8)
def _classify_boards(
    stock_codes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-stock board classification arrays, cached per universe.

    The CPU env calls :func:`compute_at_limit_masks` every ``env.step`` with
    the same stock list; string parsing 5000 codes per step would dominate
    the hot path (env.step budget < 2ms). Returned arrays are read-only.

    Returns ``(up_base, down_base, st_capable, no_limit_5d, bj, known)``.
    """
    import numpy as np

    n_stocks = len(stock_codes)
    up_base = np.empty(n_stocks, dtype=np.float64)
    down_base = np.empty(n_stocks, dtype=np.float64)
    st_capable = np.zeros(n_stocks, dtype=np.bool_)  # ST ±5% boards (SH/SZ main)
    no_limit_5d = np.zeros(n_stocks, dtype=np.bool_)  # STAR/GEM 5-day window
    bj = np.zeros(n_stocks, dtype=np.bool_)
    known = np.ones(n_stocks, dtype=np.bool_)
    for j, code in enumerate(stock_codes):
        try:
            board = identify_board(code)
        except ValueError:
            known[j] = False
            up_base[j] = _LEGACY_THRESHOLD
            down_base[j] = -_LEGACY_THRESHOLD
            continue
        if board in (StockBoard.GEM, StockBoard.STAR):
            up_base[j], down_base[j] = 0.20, -0.20
            no_limit_5d[j] = True
        elif board == StockBoard.BJ:
            up_base[j], down_base[j] = 0.30, -0.30
            bj[j] = True
        else:
            up_base[j], down_base[j] = 0.10, -0.10
            st_capable[j] = True
    for arr in (up_base, down_base, st_capable, no_limit_5d, bj, known):
        arr.flags.writeable = False
    return up_base, down_base, st_capable, no_limit_5d, bj, known


def compute_at_limit_masks(
    pct_chg: np.ndarray,
    stock_codes: Sequence[str],
    is_st: np.ndarray | None = None,
    days_since_ipo: np.ndarray | None = None,
    close: np.ndarray | None = None,
    epsilon: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized per-(date, stock) limit-up / limit-down detection.

    SINGLE SOURCE OF TRUTH for panel-wide at-limit masks: the GPU training
    path (scripts/train_v2.py valid_mask, C4) and the backtest eval scripts
    (M5) must both call this so training and eval agree exactly.

    Parameters
    ----------
    pct_chg:
        (T, S) daily pct change in decimal form (+10% = 0.10).
    stock_codes:
        length-S Tushare-format codes for board classification. Codes whose
        board cannot be identified fall back to the legacy ±9.5% pct
        threshold (mirrors ``env._apply_trading_mask``'s unknown-board path).
    is_st:
        optional (T, S) bool; per-day ST flag → ±5% on SH/SZ main boards only.
    days_since_ipo:
        optional (T, S) trading days since IPO; drives the STAR/ChiNext
        5-day no-limit window (and BSE / main-board listing-day rules).
    close:
        optional (T, S) RAW close prices. When given, prev_close is
        reconstructed as ``close / (1 + pct_chg)`` (the exchange's ex-rights
        基准价 by construction) and the rounded limit price is compared,
        catching low-price limit closes the pct epsilon misses. Cells with
        missing/invalid prices fall back to the pct-epsilon test.

    Returns
    -------
    (at_limit_up, at_limit_down) — two (T, S) bool arrays. Cells inside a
    no-limit window are always False.
    """
    import numpy as np

    pct = np.asarray(pct_chg, dtype=np.float64)
    if pct.ndim != 2:
        raise ValueError(f"pct_chg must be (T, S); got shape {pct.shape}")
    n_dates, n_stocks = pct.shape
    if len(stock_codes) != n_stocks:
        raise ValueError(f"stock_codes length {len(stock_codes)} != n_stocks {n_stocks}")

    up_base, down_base, st_capable, no_limit_5d, bj, known = _classify_boards(tuple(stock_codes))

    up_pct = np.broadcast_to(up_base, (n_dates, n_stocks)).copy()
    down_pct = np.broadcast_to(down_base, (n_dates, n_stocks)).copy()

    if is_st is not None:
        st2d = np.asarray(is_st, dtype=np.bool_)
        st_cells = st2d & st_capable  # ST ±5% only on SH/SZ main boards
        up_pct[st_cells] = 0.05
        down_pct[st_cells] = -0.05

    unlimited = np.zeros((n_dates, n_stocks), dtype=np.bool_)
    if days_since_ipo is not None:
        dsi = np.asarray(days_since_ipo, dtype=np.float64)
        unlimited |= no_limit_5d & (dsi < IPO_NO_LIMIT_TRADING_DAYS)
        unlimited |= bj & (dsi < 1)
        # Main-board listing day: +44%/-36% (TODO in get_price_limit_pct for
        # the post-2023-02-17 main-board 5-day variant applies here too).
        listing_main = st_capable & (dsi < 1)
        up_pct[listing_main] = LISTING_DAY_UP_MAIN_GEM
        down_pct[listing_main] = LISTING_DAY_DOWN_MAIN_GEM

    # pct-epsilon fallback. Unknown boards use the raw legacy threshold
    # (no epsilon), matching env._apply_trading_mask's legacy branch.
    eps2d = np.where(known, epsilon, 0.0)
    at_up = pct >= up_pct - eps2d
    at_down = pct <= down_pct + eps2d

    if close is not None:
        close64 = np.asarray(close, dtype=np.float64)
        if close64.shape != pct.shape:
            raise ValueError(f"close shape {close64.shape} != pct_chg shape {pct.shape}")
        # prev_close reconstruction: exact when pct_chg is quoted against the
        # ex-rights 前收盘基准价 (repo contract, decimal form). Vendor exports
        # that ROUND pct_chg (e.g. to 4 decimals) perturb prev_close by up to
        # ~1e-4 relative — harmless in the interior but able to flip the
        # rounded limit price by one tick right at a half-cent boundary; the
        # half-tick _PRICE_EPSILON absorbs the comparison side of that.
        with np.errstate(divide="ignore", invalid="ignore"):
            prev_close = close64 / (1.0 + pct)
        # Snap the reconstruction to the 0.01 grid: exchange prev closes are
        # always tick-quantized, but float32-origin close/pct noise (~1e-7
        # relative) can land prev * (1 + limit_pct) BELOW an exact half-cent
        # and floor the limit price one tick low — falsely flagging a close
        # one tick under the true limit (e.g. prev 1.15 → 1.26 flagged when
        # the true limit is 1.27). The +1e-9 nudge below only covers float64
        # product noise; quantizing removes reconstruction noise at the
        # source. _PRICE_EPSILON protects the comparison side, not this
        # limit-price side.
        prev_close = np.floor(prev_close * 100.0 + 0.5) / 100.0
        have_price = (
            np.isfinite(close64) & (close64 > 0) & np.isfinite(prev_close) & (prev_close > 0)
        )
        # Rounded limit price, half away from zero (exchange 四舍五入),
        # applied only for cells with valid prices on identifiable boards.
        # The +1e-9 nudge mirrors _round_limit_price: exact half-cent
        # products must not floor one tick low (2.35 * 1.10 → 2.59, not 2.58).
        have_price &= known
        limit_up_price = np.floor(prev_close * (1.0 + up_pct) * 100.0 + 0.5 + 1e-9) / 100.0
        limit_down_price = np.floor(prev_close * (1.0 + down_pct) * 100.0 + 0.5 + 1e-9) / 100.0
        at_up = np.where(have_price, close64 >= limit_up_price - _PRICE_EPSILON, at_up)
        at_down = np.where(have_price, close64 <= limit_down_price + _PRICE_EPSILON, at_down)

    at_up &= ~unlimited
    at_down &= ~unlimited
    return at_up.astype(np.bool_), at_down.astype(np.bool_)


__all__ = [
    "StockBoard",
    "LISTING_DAY_UP_MAIN_GEM",
    "LISTING_DAY_DOWN_MAIN_GEM",
    "LISTING_DAY_UNLIMITED",
    "IPO_NO_LIMIT_TRADING_DAYS",
    "identify_board",
    "detect_board",
    "get_price_limit_pct",
    "is_at_limit_up",
    "is_at_limit_down",
    "compute_at_limit_masks",
]
