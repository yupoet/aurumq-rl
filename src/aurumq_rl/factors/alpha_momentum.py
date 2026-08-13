"""Reference factor: simple 5-day momentum with incremental support.

Demonstrates the `impl_incremental` contract by maintaining a per-stock
tail-buffer and only computing shifts over the relevant window.
"""

from __future__ import annotations

from typing import Dict, Tuple

import polars as pl

from aurumq_rl.factor_registry import FactorImpl


class Momentum5dImpl(FactorImpl):
    """5-day momentum: (close(t) - close(t-5)) / close(t-5)."""

    name: str = "alpha_momentum_5d"
    _max_window: int = 252

    def impl(self, df: pl.DataFrame) -> pl.Series:
        df_sorted = df.sort("ts_code", "trade_date")
        df_with_prev = df_sorted.with_columns(
            pl.col("close").shift(5).over("ts_code").alias("prev_close")
        )
        return ((df_with_prev["close"] - df_with_prev["prev_close"]) / df_with_prev["prev_close"]).to_series()

    def impl_incremental(
        self, df_new: pl.DataFrame, tail_buffer: Dict[str, pl.DataFrame]
    ) -> Tuple[pl.Series, Dict[str, pl.DataFrame]]:
        updated_buffer: Dict[str, pl.DataFrame] = {}
        results: list = []

        for stock in df_new["ts_code"].unique():
            stock_new = df_new.filter(pl.col("ts_code") == stock)
            history = tail_buffer.get(stock, stock_new.head(0))
            # Combine history and new data for correct shift context
            combined = pl.concat([history, stock_new]).sort("trade_date")
            # Keep only last max_window rows in buffer
            updated_buffer[stock] = combined.tail(self._max_window)

            # Compute momentum on the combined series
            mom = (
                combined
                .with_columns(pl.col("close").shift(5).alias("prev_close"))
                .filter(pl.col("trade_date").is_in(stock_new["trade_date"]))
                .select(((pl.col("close") - pl.col("prev_close")) / pl.col("prev_close")).alias(self.name))
            )
            results.append(mom)

        return pl.concat(results).select(self.name).to_series(), updated_buffer
