"""A-share dynamic price-limit threshold detection.

Public market rules (publicly available):
  * Main board (沪/深): ±10%
  * ChiNext (300/301): ±20% (since 2020-08-24 registration system reform)
  * STAR (688): ±20%
  * BSE (8/4): ±30% (no limit on listing day, ±30% from day 2)
  * ST/*ST: ±5%
  * Listing day: main/ChiNext +44%/-36%, STAR/BSE unlimited

Stock code format (industry standard XXXXXX.SZ/SH/BJ):
  600xxx / 601xxx / 603xxx / 605xxx → SH main board (.SH)
  688xxx                            → STAR market (.SH)
  000xxx / 001xxx / 002xxx / 003xxx → SZ main board (.SZ)
  300xxx / 301xxx                   → ChiNext (.SZ)
  8xxxxx / 4xxxxx                   → BSE (.BJ)
"""

from __future__ import annotations

import sys

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
    if numeric.startswith("688"):
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
) -> tuple[float, float]:
    """Return (upper_limit_pct, lower_limit_pct) as decimals, e.g. (0.20, -0.20).

    Priority:
      1. Listing day → board-specific (STAR/BSE unlimited, main/ChiNext +44%/-36%)
      2. ST/*ST → ±5%
      3. Board default: main ±10%, ChiNext/STAR ±20%, BSE ±30%

    Returns
    -------
    (up_limit, down_limit) as decimal fractions.
    """
    board = identify_board(stock_code)

    if is_listing_day:
        if board in (StockBoard.STAR, StockBoard.BJ):
            return (LISTING_DAY_UNLIMITED, -LISTING_DAY_UNLIMITED)
        return (LISTING_DAY_UP_MAIN_GEM, LISTING_DAY_DOWN_MAIN_GEM)

    if is_st:
        return (0.05, -0.05)

    if board in (StockBoard.GEM, StockBoard.STAR):
        return (0.20, -0.20)
    if board == StockBoard.BJ:
        return (0.30, -0.30)
    return (0.10, -0.10)


def is_at_limit_up(
    stock_code: str,
    pct_change: float,
    is_st: bool = False,
    is_listing_day: bool = False,
    epsilon: float = 1e-3,
) -> bool:
    """Check if current pct_change (decimal form) hits the upper price limit.

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
    up_limit, _ = get_price_limit_pct(stock_code, is_st=is_st, is_listing_day=is_listing_day)
    if up_limit >= LISTING_DAY_UNLIMITED:
        return False
    return pct_change >= up_limit - epsilon


def is_at_limit_down(
    stock_code: str,
    pct_change: float,
    is_st: bool = False,
    is_listing_day: bool = False,
    epsilon: float = 1e-3,
) -> bool:
    """Check if current pct_change (decimal form) hits the lower price limit."""
    _, down_limit = get_price_limit_pct(stock_code, is_st=is_st, is_listing_day=is_listing_day)
    if down_limit <= -LISTING_DAY_UNLIMITED:
        return False
    return pct_change <= down_limit + epsilon


__all__ = [
    "StockBoard",
    "LISTING_DAY_UP_MAIN_GEM",
    "LISTING_DAY_DOWN_MAIN_GEM",
    "LISTING_DAY_UNLIMITED",
    "identify_board",
    "detect_board",
    "get_price_limit_pct",
    "is_at_limit_up",
    "is_at_limit_down",
]
