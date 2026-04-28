"""Factor panel loader: Parquet → numpy 3D panel.

Prefix-based factor recognition
-------------------------------
Input Parquet must contain columns prefixed with one of:

* ``alpha_*``   alpha101 quant-volume factors
* ``mf_*``      main-force capital flow
* ``hm_*``      hot-money seats
* ``hk_*``      northbound capital
* ``inst_*``    institutional flow (limit-up/down list)
* ``mg_*``      margin trading
* ``cyq_*``     chip distribution
* ``senti_*``   limit-up sentiment
* ``sh_*``      shareholders
* ``fund_*``    fundamentals (PE/PB/ROE/...)
* ``ind_*``     industry relative strength
* ``mkt_*``     market regime

The loader picks **all** matching columns (sorted alphabetically) up to
``n_factors``. Missing prefixes are silently skipped — RL never errors out
because of missing factor groups; the model just sees those positions as 0.

Universe filter
---------------
Default ``UniverseFilter.MAIN_BOARD_NON_ST`` excludes:

* BSE (.BJ, codes 8/4)
* STAR market (688)
* ChiNext (300/301)
* ST/*ST stocks

Other modes: ``ALL_A`` (no filter), ``HS300``, ``ZZ500``, ``ZZ1000``.

This module has **no PyTorch / gymnasium dependency**, safe to import in any
environment.
"""

from __future__ import annotations

import datetime
import re
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PARQUET_PATH: str = "data/factor_panel.parquet"
NEW_STOCK_PROTECT_DAYS: int = 60

# Factor column prefixes (recognized by data_loader)
FACTOR_COL_PREFIXES: tuple[str, ...] = (
    "alpha_",
    "mf_",
    "hm_",
    "hk_",
    "inst_",
    "mg_",
    "cyq_",
    "senti_",
    "sh_",
    "fund_",
    "ind_",
    "mkt_",
)

# Required columns in input Parquet
REQUIRED_COLUMNS: tuple[str, ...] = ("ts_code", "trade_date", "close", "pct_chg", "vol")

# Optional columns (used if present, defaulted otherwise)
OPTIONAL_COLUMNS: tuple[str, ...] = ("is_st", "days_since_ipo", "industry_code", "name")


class UniverseFilter(StrEnum):
    """Stock universe selection mode."""

    ALL_A = "all_a"
    MAIN_BOARD_NON_ST = "main_board_non_st"
    HS300 = "hs300"
    ZZ500 = "zz500"
    ZZ1000 = "zz1000"


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


class FactorPanel(NamedTuple):
    """Container for a 3D factor panel + auxiliary arrays.

    Attributes
    ----------
    factor_array:
        shape (n_dates, n_stocks, n_factors), cross-sectionally z-scored.
    return_array:
        shape (n_dates, n_stocks), forward log-returns.
    pct_change_array:
        shape (n_dates, n_stocks), daily pct change as **decimals** (+10% = 0.10).
    is_st_array:
        shape (n_dates, n_stocks), bool.
    is_suspended_array:
        shape (n_dates, n_stocks), bool (volume == 0).
    days_since_ipo_array:
        shape (n_dates, n_stocks).
    dates:
        list[date], length n_dates.
    stock_codes:
        list[str], length n_stocks.
    factor_names:
        list[str], length n_factors.
    """

    factor_array: np.ndarray
    return_array: np.ndarray
    pct_change_array: np.ndarray
    is_st_array: np.ndarray
    is_suspended_array: np.ndarray
    days_since_ipo_array: np.ndarray
    dates: list[datetime.date]
    stock_codes: list[str]
    factor_names: list[str]


# ---------------------------------------------------------------------------
# Universe filtering
# ---------------------------------------------------------------------------

# Patterns for default "main_board_non_st" filter
_SH_MAIN_PATTERN = re.compile(r"^60[0135]\d{3}\.SH$")
_SZ_MAIN_PATTERN = re.compile(r"^00[0123]\d{3}\.SZ$")
_ST_NAME_PATTERN = re.compile(r"(\*?ST|退)")


def _is_main_board(code: str) -> bool:
    """True if code is SH/SZ main board."""
    return bool(_SH_MAIN_PATTERN.match(code) or _SZ_MAIN_PATTERN.match(code))


def filter_universe(
    df: pl.DataFrame,
    mode: UniverseFilter = UniverseFilter.MAIN_BOARD_NON_ST,
    name_col: str = "name",
) -> pl.DataFrame:
    """Filter the universe of stocks.

    Parameters
    ----------
    df:
        Input dataframe with at least ``ts_code`` column.
    mode:
        Filter mode (see :class:`UniverseFilter`).
    name_col:
        Column holding stock name (used for ST detection). If absent,
        ST filtering is skipped.

    Returns
    -------
    Filtered dataframe.
    """
    if mode == UniverseFilter.ALL_A:
        return df

    if mode == UniverseFilter.MAIN_BOARD_NON_ST:
        # Apply main-board filter
        df = df.filter(
            pl.col("ts_code").map_elements(_is_main_board, return_dtype=pl.Boolean)
        )
        # ST exclusion (only if name column exists)
        if name_col in df.columns:
            df = df.filter(
                ~pl.col(name_col).cast(pl.Utf8).str.contains(r"\*?ST|退")
            )
        return df

    # Index-component modes: prefer explicit boolean column when present,
    # fall back to main-board heuristic otherwise.
    if mode == UniverseFilter.HS300:
        if "is_hs300" in df.columns:
            return df.filter(pl.col("is_hs300") == True)  # noqa: E712
        return df.filter(
            pl.col("ts_code").map_elements(_is_main_board, return_dtype=pl.Boolean)
        )

    if mode == UniverseFilter.ZZ500:
        if "is_zz500" in df.columns:
            return df.filter(pl.col("is_zz500") == True)  # noqa: E712
        return df.filter(
            pl.col("ts_code").map_elements(_is_main_board, return_dtype=pl.Boolean)
        )

    # ZZ1000 (and any future enum value) — fall back to main-board heuristic.
    return df.filter(
        pl.col("ts_code").map_elements(_is_main_board, return_dtype=pl.Boolean)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cross_section_zscore(arr: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score normalize along axis=1 (stock dim).

    For each (date, factor), normalize across stocks:
        z = (x - mean) / (std + 1e-8)
    """
    mean = np.nanmean(arr, axis=1, keepdims=True)
    std = np.nanstd(arr, axis=1, keepdims=True)
    return (arr - mean) / (std + 1e-8)


def _safe_log_return(price_now: np.ndarray, price_fwd: np.ndarray) -> np.ndarray:
    """Compute log return with NaN/zero-price safety."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (price_now > 0) & (price_fwd > 0),
            price_fwd / price_now,
            np.nan,
        )
        log_ret = np.where(
            np.isfinite(ratio) & (ratio > 0),
            np.log(ratio),
            0.0,
        )
    return log_ret.astype(np.float32)


def discover_factor_columns(
    df: pl.DataFrame,
    n_factors: int | None = None,
    prefixes: tuple[str, ...] = FACTOR_COL_PREFIXES,
) -> list[str]:
    """Discover factor columns in a DataFrame by prefix matching.

    Parameters
    ----------
    df:
        Input DataFrame.
    n_factors:
        If given, truncate to first N columns (alphabetical order).
        If None, return all matched columns.
    prefixes:
        Recognized prefixes.

    Returns
    -------
    Sorted list of factor column names.
    """
    matched = sorted([c for c in df.columns if c.startswith(prefixes)])
    if n_factors is not None:
        return matched[:n_factors]
    return matched


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class FactorPanelLoader:
    """Load factor panels from a Parquet file.

    Parameters
    ----------
    parquet_path:
        Path to a Parquet file (single-file or hive-partitioned glob).
    """

    def __init__(self, parquet_path: str | Path = DEFAULT_PARQUET_PATH) -> None:
        self.parquet_path = Path(parquet_path)

    def load_panel(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        n_factors: int | None = None,
        forward_period: int = 10,
        universe_filter: UniverseFilter = UniverseFilter.MAIN_BOARD_NON_ST,
    ) -> FactorPanel:
        """Load a factor panel from Parquet.

        Parameters
        ----------
        start_date / end_date:
            Inclusive date range.
        n_factors:
            Number of factors to use (None = all matched).
        forward_period:
            Forward-return window in trading days.
        universe_filter:
            Stock universe filtering mode.

        Returns
        -------
        FactorPanel

        Raises
        ------
        FileNotFoundError if the Parquet file is missing.
        ValueError if no factor columns are found.
        """
        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {self.parquet_path}\n"
                "Generate one with `scripts/generate_synthetic.py` (demo) "
                "or `scripts/export_factor_panel.py` (real data)."
            )

        return self._load_from_parquet(
            start_date=start_date,
            end_date=end_date,
            n_factors=n_factors,
            forward_period=forward_period,
            universe_filter=universe_filter,
        )

    def _load_from_parquet(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        n_factors: int | None,
        forward_period: int,
        universe_filter: UniverseFilter,
    ) -> FactorPanel:
        """Internal Parquet → FactorPanel conversion."""
        # Use polars scan for memory efficiency
        df_lazy = pl.scan_parquet(str(self.parquet_path))

        # Date filter
        df = (
            df_lazy.filter(
                (pl.col("trade_date") >= start_date)
                & (pl.col("trade_date") <= end_date)
            )
            .collect()
        )

        if df.is_empty():
            raise ValueError(
                f"No data in range {start_date}..{end_date}. "
                f"Parquet covers: {self.get_date_range()}"
            )

        # Validate required columns
        missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_required:
            raise ValueError(
                f"Required columns missing from Parquet: {missing_required}\n"
                f"Parquet must contain: {REQUIRED_COLUMNS}"
            )

        # Universe filter
        df = filter_universe(df, mode=universe_filter)
        if df.is_empty():
            raise ValueError(
                f"Universe filter {universe_filter} eliminated all rows. "
                "Check ts_code format or universe selection."
            )

        return self._df_to_panel(df, n_factors=n_factors, forward_period=forward_period)

    def _df_to_panel(
        self,
        df: pl.DataFrame,
        n_factors: int | None,
        forward_period: int,
    ) -> FactorPanel:
        """Convert polars DataFrame to numpy 3D panel."""
        dates = df["trade_date"].unique().sort().to_list()
        stock_codes = df["ts_code"].unique().sort().to_list()

        n_dates = len(dates)
        n_stocks = len(stock_codes)

        # Discover factor columns
        factor_cols = discover_factor_columns(df, n_factors=n_factors)
        if not factor_cols:
            raise ValueError(
                f"No factor columns found. Expected columns prefixed with "
                f"{FACTOR_COL_PREFIXES}. Got columns: {df.columns[:20]}..."
            )

        n_factors_actual = len(factor_cols)

        date_index = {d: i for i, d in enumerate(dates)}
        stock_index = {s: j for j, s in enumerate(stock_codes)}

        factor_array = np.zeros((n_dates, n_stocks, n_factors_actual), dtype=np.float32)
        close_array = np.zeros((n_dates, n_stocks), dtype=np.float32)
        pct_change_array = np.zeros((n_dates, n_stocks), dtype=np.float32)
        is_st_array = np.zeros((n_dates, n_stocks), dtype=np.bool_)
        is_suspended_array = np.zeros((n_dates, n_stocks), dtype=np.bool_)
        days_since_ipo_array = np.full(
            (n_dates, n_stocks), NEW_STOCK_PROTECT_DAYS * 2, dtype=np.float32
        )

        has_is_st = "is_st" in df.columns
        has_days_ipo = "days_since_ipo" in df.columns

        for row in df.iter_rows(named=True):
            t = date_index.get(row["trade_date"])
            j = stock_index.get(row["ts_code"])
            if t is None or j is None:
                continue

            for fi, col in enumerate(factor_cols):
                v = row.get(col)
                if v is not None:
                    factor_array[t, j, fi] = float(v)

            close_v = row.get("close")
            if close_v is not None:
                close_array[t, j] = float(close_v)
            pct_v = row.get("pct_chg")
            if pct_v is not None:
                pct_change_array[t, j] = float(pct_v)
            vol_v = row.get("vol")
            is_suspended_array[t, j] = (vol_v is None) or (vol_v == 0)

            if has_is_st:
                is_st_array[t, j] = bool(row.get("is_st") or False)
            if has_days_ipo:
                days_v = row.get("days_since_ipo")
                if days_v is not None:
                    days_since_ipo_array[t, j] = float(days_v)

        # Forward return
        return_array = np.zeros((n_dates, n_stocks), dtype=np.float32)
        for t in range(n_dates - forward_period):
            return_array[t] = _safe_log_return(close_array[t], close_array[t + forward_period])

        # Cross-section z-score
        factor_array = _cross_section_zscore(factor_array)

        return FactorPanel(
            factor_array=factor_array,
            return_array=return_array,
            pct_change_array=pct_change_array,
            is_st_array=is_st_array,
            is_suspended_array=is_suspended_array,
            days_since_ipo_array=days_since_ipo_array,
            dates=dates,
            stock_codes=stock_codes,
            factor_names=factor_cols,
        )

    def get_date_range(self) -> tuple[datetime.date | None, datetime.date | None]:
        """Return (min_date, max_date) of the Parquet, or (None, None) if empty."""
        if not self.parquet_path.exists():
            return (None, None)
        try:
            df = (
                pl.scan_parquet(str(self.parquet_path))
                .select([pl.col("trade_date").min().alias("min"), pl.col("trade_date").max().alias("max")])
                .collect()
            )
            return (df["min"][0], df["max"][0])
        except Exception:
            return (None, None)

    @staticmethod
    def build_synthetic(
        n_dates: int = 500,
        n_stocks: int = 200,
        n_factors: int = 64,
        forward_period: int = 10,
        seed: int = 42,
        prefix: str = "alpha_",
    ) -> FactorPanel:
        """Build a synthetic panel for smoke testing — no real data needed.

        Useful for CI, demos, and smoke tests of the training pipeline.
        Stock codes are synthetic (``SYN_001`` etc.) — not real codes.
        """
        rng = np.random.default_rng(seed)

        factor_array = rng.standard_normal((n_dates, n_stocks, n_factors)).astype(np.float32)

        # Synthetic returns: factor mean + noise
        factor_mean = factor_array.mean(axis=2)
        return_array = (
            factor_mean * 0.01 + rng.standard_normal((n_dates, n_stocks)) * 0.02
        ).astype(np.float32)

        # Random pct change (decimal form)
        pct_change_array = (rng.standard_normal((n_dates, n_stocks)) * 0.02).astype(np.float32)

        # 5% ST, 2% suspended
        is_st_array = (rng.random((n_dates, n_stocks)) < 0.05).astype(np.bool_)
        is_suspended_array = (rng.random((n_dates, n_stocks)) < 0.02).astype(np.bool_)

        # All synthetic stocks are mature
        days_since_ipo_array = rng.integers(
            NEW_STOCK_PROTECT_DAYS * 2,
            NEW_STOCK_PROTECT_DAYS * 20,
            size=(n_dates, n_stocks),
        ).astype(np.float32)

        # Cross-section z-score
        factor_array = _cross_section_zscore(factor_array)

        # Synthetic dates: weekdays starting 2020-01-01
        dates: list[datetime.date] = []
        current = datetime.date(2020, 1, 1)
        while len(dates) < n_dates:
            if current.weekday() < 5:
                dates.append(current)
            current += datetime.timedelta(days=1)

        # Synthetic codes (NOT real stock codes)
        stock_codes = [f"SYN_{i:05d}" for i in range(n_stocks)]
        factor_names = [f"{prefix}{i:03d}" for i in range(n_factors)]

        return FactorPanel(
            factor_array=factor_array,
            return_array=return_array,
            pct_change_array=pct_change_array,
            is_st_array=is_st_array,
            is_suspended_array=is_suspended_array,
            days_since_ipo_array=days_since_ipo_array,
            dates=dates,
            stock_codes=stock_codes,
            factor_names=factor_names,
        )


__all__ = [
    "FactorPanel",
    "FactorPanelLoader",
    "UniverseFilter",
    "FACTOR_COL_PREFIXES",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "discover_factor_columns",
    "filter_universe",
]
