"""Factor registry with incremental update protocol.

Current contract: impl(df) -> Series (full panel recompute).
New contract: impl_incremental(df_new, tail_buffer) -> tuple[Series, tail_buffer]
Supports O(stocks * max_window) daily updates.
"""

from __future__ import annotations

from typing import Dict, Protocol, Tuple

import polars as pl


class FactorImpl(Protocol):
    """Protocol for factor implementations."""

    def impl(self, df: pl.DataFrame) -> pl.Series:
        """Full-panel computation. Returns a Series aligned with df's index."""
        ...

    def impl_incremental(
        self, df_new: pl.DataFrame, tail_buffer: Dict[str, pl.DataFrame]
    ) -> Tuple[pl.Series, Dict[str, pl.DataFrame]]:
        """Incremental update for new trading days.

        Args:
            df_new: New daily rows for all stocks, sorted by (ts_code, trade_date).
            tail_buffer: Current per-stock tail history (key: ts_code, value: DataFrame).

        Returns:
            Factor values for the new dates, and updated tail_buffer.
        """
        ...


def run_incremental(
    factor: FactorImpl,
    df_new: pl.DataFrame,
    tail_buffer: Dict[str, pl.DataFrame],
    max_window: int = 252,
) -> Tuple[pl.Series, Dict[str, pl.DataFrame]]:
    """Run factor on new data using incremental protocol if available."""
    if hasattr(factor, "impl_incremental"):
        return factor.impl_incremental(df_new, tail_buffer)
    # Fallback: recompute on concatenated data (for non-incremental factors)
    return _fallback_compute(factor, df_new, tail_buffer, max_window)


def _fallback_compute(
    factor: FactorImpl,
    df_new: pl.DataFrame,
    tail_buffer: Dict[str, pl.DataFrame],
    max_window: int,
) -> Tuple[pl.Series, Dict[str, pl.DataFrame]]:
    if not tail_buffer:
        return factor.impl(df_new), tail_buffer
    # Concatenate tail and new data, recompute, slice to new dates
    combined = pl.concat(list(tail_buffer.values()) + [df_new]).sort("ts_code", "trade_date")
    full_res = factor.impl(combined)
    new_dates = set(df_new["trade_date"].unique())
    updated_buf: Dict[str, pl.DataFrame] = {}
    res_series: list = []
    for stock in df_new["ts_code"].unique():
        stock_new = df_new.filter(pl.col("ts_code") == stock)
        stock_full = combined.filter(pl.col("ts_code") == stock).tail(max_window)
        updated_buf[stock] = stock_full
        stock_res = full_res.filter(
            (pl.col("ts_code") == stock) & (pl.col("trade_date").is_in(new_dates))
        )
        res_series.append(stock_res)
    if not res_series:
        return pl.Series(dtype=pl.Float64), updated_buf
    return pl.concat(res_series), updated_buf
