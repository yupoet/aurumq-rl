"""Single source of truth for the synthetic factor panel used in tests.

Both ``conftest.py`` (pytest fixture) and
``scripts/reference_data/build_alpha101_reference.py`` (STHSF/alpha101
reference builder) import :func:`build_synthetic_panel` from this module
to guarantee that the data driving the unit tests is byte-identical to
the data that produced the locked numerical reference values.

Schema mirrors AurumQ's internal panel loader:

* ``stock_code`` (Utf8) — 10 distinct codes
  (``"000001.SZ"`` ... ``"600010.SH"`` — first 5 SZ, next 5 SH)
* ``trade_date`` (Date32) — ``n_days`` consecutive business days starting
  2024-01-02
* ``open`` / ``high`` / ``low`` / ``close`` (Float64) — geometric Brownian
  motion paths with sigma=0.02 daily, anchored at random start prices
  in [20, 100]
* ``volume`` (Float64) — log-normal noise around 1M-10M shares per day
* ``amount`` (Float64) — ``volume * close * 100`` plus log-normal noise
  (units: yuan, matching A-share convention)
* ``prev_close`` (Float64) — ``close.shift(1)`` per stock
* ``vwap`` (Float64) — ``amount / (volume * 100)`` (A-share convention:
  100 shares per "lot")
* ``returns`` (Float64) — ``close / prev_close - 1``
* ``adv5`` ... ``adv180`` (Float64, 12 columns) — rolling mean of volume
* ``industry`` (Utf8) — assigned in groups of 5 stocks
  (2 industries: ``"Technology"``, ``"Finance"``)
* ``sub_industry`` (Utf8) — assigned in groups of 5 stocks
  (2 sub-industries: ``"Software"``, ``"Banking"``)
* ``cap`` (Float64) — log-normal market-cap, **static per stock**
  (same value across every date for that stock)

Reproducibility — fixed RNG seed, no system-time inputs, no I/O.
"""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    import pandas as pd

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

ADV_WINDOWS: tuple[int, ...] = (5, 10, 15, 20, 30, 40, 50, 60, 81, 120, 150, 180)
"""ADV (average daily volume) rolling windows present in the panel.

Chosen to span the supports of the WorldQuant 101 alphas. All 12 columns
exist whether or not the window has enough data — short windows fill in
quickly, long windows stay NaN until enough days accumulate.
"""

INDUSTRIES: tuple[str, str] = ("Technology", "Finance")
SUB_INDUSTRIES: tuple[str, str] = ("Software", "Banking")

PRICE_SIGMA: float = 0.02
"""Daily log-return volatility for the GBM price path."""

START_DATE: _dt.date = _dt.date(2024, 1, 2)
"""First trading date of the synthetic calendar (a Tuesday — market open)."""


# ---------------------------------------------------------------------------
# Stock-code generator
# ---------------------------------------------------------------------------


def _stock_codes(n_stocks: int) -> list[str]:
    """Return ``n_stocks`` deterministic A-share-style codes.

    First half are Shenzhen (``.SZ``, codes 000001+), second half are
    Shanghai (``.SH``, codes 600001+). Caller is expected to use small
    ``n_stocks`` (the default tests use 10).
    """
    half = n_stocks // 2
    sz = [f"{i:06d}.SZ" for i in range(1, half + 1)]
    sh = [f"{600000 + i:06d}.SH" for i in range(1, n_stocks - half + 1)]
    return sz + sh


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def _business_days(start: _dt.date, n: int) -> list[_dt.date]:
    """Return ``n`` consecutive Mon-Fri dates beginning at ``start``.

    Skips weekends. Holidays are not modelled — this is a synthetic panel,
    not a real calendar.
    """
    out: list[_dt.date] = []
    cur = start
    while len(out) < n:
        if cur.weekday() < 5:
            out.append(cur)
        cur += _dt.timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_synthetic_panel(
    seed: int = 42,
    n_stocks: int = 10,
    n_days: int = 60,
) -> pl.DataFrame:
    """Build the canonical synthetic panel.

    Parameters
    ----------
    seed:
        RNG seed. Must remain 42 for the locked ``alpha101_reference.parquet``
        to stay valid.
    n_stocks:
        Number of distinct stock codes. Default 10 matches the reference.
    n_days:
        Number of consecutive business days. Default 60 matches the reference.

    Returns
    -------
    pl.DataFrame in long format, sorted by ``(stock_code, trade_date)``.
    """
    if n_stocks < 2:
        raise ValueError(f"n_stocks={n_stocks} must be >= 2 to populate two industries")
    if n_days < 2:
        raise ValueError(f"n_days={n_days} must be >= 2")

    rng = np.random.default_rng(seed)

    codes = _stock_codes(n_stocks)
    dates = _business_days(START_DATE, n_days)

    # ----- Per-stock static draws (independent of date) -------------------
    start_prices = rng.uniform(20.0, 100.0, size=n_stocks)
    # market cap log-normal in yuan (median ~ 5e9, broad spread)
    caps = np.exp(rng.normal(loc=22.0, scale=1.0, size=n_stocks))

    # Industry assignment: groups of 5 stocks each rotate through the 2
    # industries (mirror the prompt: "assigned in groups of 5 stocks").
    industry_ix = (np.arange(n_stocks) // 5) % len(INDUSTRIES)
    sub_industry_ix = (np.arange(n_stocks) // 5) % len(SUB_INDUSTRIES)
    industries = np.array([INDUSTRIES[i] for i in industry_ix])
    sub_industries = np.array([SUB_INDUSTRIES[i] for i in sub_industry_ix])

    # ----- GBM price paths (close) ---------------------------------------
    log_returns = rng.normal(loc=0.0, scale=PRICE_SIGMA, size=(n_days, n_stocks))
    log_paths = np.cumsum(log_returns, axis=0)
    close = start_prices[None, :] * np.exp(log_paths)

    # ----- OHL — perturb close with intra-day noise ----------------------
    intraday_noise = rng.normal(loc=0.0, scale=0.5 * PRICE_SIGMA, size=(n_days, n_stocks))
    open_ = close * np.exp(intraday_noise)
    high_offset = np.abs(rng.normal(0.0, PRICE_SIGMA, size=(n_days, n_stocks)))
    low_offset = np.abs(rng.normal(0.0, PRICE_SIGMA, size=(n_days, n_stocks)))
    high = np.maximum(open_, close) * np.exp(high_offset)
    low = np.minimum(open_, close) * np.exp(-low_offset)

    # ----- Volume — log-normal around 1M-10M shares -----------------------
    log_vol = rng.normal(loc=np.log(3_000_000.0), scale=0.5, size=(n_days, n_stocks))
    volume = np.exp(log_vol)

    # ----- Amount — turnover (yuan): close * volume * 100 lot * noise ----
    amount_noise = np.exp(rng.normal(0.0, 0.05, size=(n_days, n_stocks)))
    amount = close * volume * 100.0 * amount_noise

    # ----- Long-format assembly -------------------------------------------
    n_rows = n_days * n_stocks
    # IMPORTANT: order is (stock_code outer, trade_date inner) so we can
    # easily compute prev_close as a per-stock shift.
    code_col = np.repeat(np.array(codes, dtype=object), n_days)
    date_col = np.tile(np.array(dates, dtype="datetime64[D]"), n_stocks)
    industry_col = np.repeat(industries, n_days)
    sub_industry_col = np.repeat(sub_industries, n_days)
    cap_col = np.repeat(caps, n_days)

    # Per-stock prev_close = shift by 1 along time axis. shape (n_days, n_stocks).
    prev_close = np.full_like(close, np.nan)
    prev_close[1:, :] = close[:-1, :]
    returns = close / prev_close - 1.0  # first row -> NaN, downstream code expects this

    vwap = amount / (volume * 100.0)

    # Reshape to long format. Outer loop is stock (matches code_col layout).
    def _flatten(mat: np.ndarray) -> np.ndarray:
        # mat shape: (n_days, n_stocks). We want stock outer, date inner.
        return mat.T.reshape(n_rows)

    columns: dict[str, np.ndarray] = {
        "stock_code": code_col,
        "trade_date": date_col,
        "open": _flatten(open_),
        "high": _flatten(high),
        "low": _flatten(low),
        "close": _flatten(close),
        "volume": _flatten(volume),
        "amount": _flatten(amount),
        "prev_close": _flatten(prev_close),
        "vwap": _flatten(vwap),
        "returns": _flatten(returns),
        "industry": industry_col,
        "sub_industry": sub_industry_col,
        "cap": cap_col,
    }

    df = pl.DataFrame(columns).with_columns(
        [
            pl.col("trade_date").cast(pl.Date),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.col("amount").cast(pl.Float64),
            pl.col("prev_close").cast(pl.Float64),
            pl.col("vwap").cast(pl.Float64),
            pl.col("returns").cast(pl.Float64),
            pl.col("cap").cast(pl.Float64),
        ]
    )

    # ----- ADV columns — rolling mean of volume per stock ----------------
    df = df.sort(["stock_code", "trade_date"])
    adv_exprs = [
        pl.col("volume")
        .rolling_mean(window_size=w, min_samples=1)
        .over("stock_code")
        .alias(f"adv{w}")
        for w in ADV_WINDOWS
    ]
    df = df.with_columns(adv_exprs)

    return df


# ---------------------------------------------------------------------------
# Pandas adapter for the STHSF/alpha101 reference implementation
# ---------------------------------------------------------------------------


def to_pandas_for_sthsf(panel: pl.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert the synthetic panel into the dict-of-wide-DataFrames shape
    that ``alpha101_single.Alpha101`` expects.

    STHSF's ``alpha101_single.py`` constructor reads
    ``df_data['close']`` etc. as **wide** pandas DataFrames where the
    index is ``trade_date`` and the columns are stock symbols. This
    helper performs the long → wide pivot.

    Output keys: ``open``, ``high``, ``low``, ``close``, ``volume``,
    ``vwap``, ``returns``. (The STHSF ``alpha101_single`` constructor
    reads exactly these seven; intermediate ``adv*`` columns are
    re-derived inside the alpha methods using ``sma()``.)
    """

    pdf = panel.to_pandas()
    out: dict[str, pd.DataFrame] = {}
    for col in ("open", "high", "low", "close", "volume", "vwap", "returns"):
        wide = pdf.pivot(index="trade_date", columns="stock_code", values=col)
        wide = wide.sort_index()
        out[col] = wide.astype(np.float64)
    return out
