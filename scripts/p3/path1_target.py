"""Compute proximity-weighted target_y from realized_returns + market_returns.

Per spec §2:
    y_t = (1.0 * max(0, excess_T+1)
         + 0.6 * max(0, excess_T+2)
         + 0.3 * max(0, excess_T+3)) / 1.9

where excess_T+d = pct_chg_T+d - eq_weight_market_pct_T+d for each (date, stock).

T+1 comes directly from realized_returns. T+2 and T+3 are obtained by
self-shifting the same table on trade_date: at anchor trade_date D,
T+2 == realized_returns[trade_date == D's next trading day]'s pct_chg_t_plus_1.

Rows where T+3 (i.e. anchor date's third forward trading day) is missing
in the panel are DROPPED — no partial-window y is emitted.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl


logger = logging.getLogger(__name__)


W1, W2, W3 = 1.0, 0.6, 0.3
W_SUM = W1 + W2 + W3  # 1.9


def compute_target_y(
    realized: pl.DataFrame,
    market: pl.DataFrame,
) -> pl.DataFrame:
    """Build (trade_date, ts_code, y) frame from realized + market frames.

    Parameters
    ----------
    realized : pl.DataFrame
        Schema: trade_date, ts_code, pct_chg_t_plus_1 (float).
        Each row's ``pct_chg_t_plus_1`` is the t+1 close-to-close return for
        that stock from the next trading day after ``trade_date``.
    market : pl.DataFrame
        Schema: trade_date, eq_weight_pct_chg_t_plus_1 (float).

    Returns
    -------
    pl.DataFrame  schema: trade_date, ts_code, y (float)
        Anchored at the original trade_date. Rows missing T+2 or T+3 in
        the panel are dropped.
    """
    # 1. Build a (date → ordinal index) for self-shift via inner-join.
    all_dates = (
        realized.select("trade_date")
        .unique()
        .sort("trade_date")
        .with_row_index("date_idx")
    )
    realized = realized.join(all_dates, on="trade_date")
    market = market.join(all_dates, on="trade_date")

    # 2. Per-stock self-join at date_idx + 1 (T+2 anchor) and date_idx + 2 (T+3 anchor).
    realized_t2 = realized.select(
        pl.col("date_idx").alias("anchor_idx"),
        pl.col("ts_code"),
        pl.col("pct_chg_t_plus_1").alias("pct_t_plus_2"),
    ).with_columns(pl.col("anchor_idx") - 1)

    realized_t3 = realized.select(
        pl.col("date_idx").alias("anchor_idx"),
        pl.col("ts_code"),
        pl.col("pct_chg_t_plus_1").alias("pct_t_plus_3"),
    ).with_columns(pl.col("anchor_idx") - 2)

    market_t2 = market.select(
        pl.col("date_idx").alias("anchor_idx"),
        pl.col("eq_weight_pct_chg_t_plus_1").alias("market_t_plus_2"),
    ).with_columns(pl.col("anchor_idx") - 1)

    market_t3 = market.select(
        pl.col("date_idx").alias("anchor_idx"),
        pl.col("eq_weight_pct_chg_t_plus_1").alias("market_t_plus_3"),
    ).with_columns(pl.col("anchor_idx") - 2)

    # 3. Anchor on realized's original rows; inner-join shifts (drops boundaries).
    base = realized.select(
        pl.col("trade_date"),
        pl.col("ts_code"),
        pl.col("date_idx").alias("anchor_idx"),
        pl.col("pct_chg_t_plus_1").alias("pct_t_plus_1"),
    ).join(
        market.select(
            pl.col("date_idx").alias("anchor_idx"),
            pl.col("eq_weight_pct_chg_t_plus_1").alias("market_t_plus_1"),
        ),
        on="anchor_idx",
        how="inner",
    )
    base = base.join(realized_t2, on=["anchor_idx", "ts_code"], how="inner")
    base = base.join(realized_t3, on=["anchor_idx", "ts_code"], how="inner")
    base = base.join(market_t2, on="anchor_idx", how="inner")
    base = base.join(market_t3, on="anchor_idx", how="inner")

    # 4. Compute excess per horizon, clip max(0, ·), weight + sum.
    out = base.with_columns(
        (pl.col("pct_t_plus_1") - pl.col("market_t_plus_1")).alias("e1"),
        (pl.col("pct_t_plus_2") - pl.col("market_t_plus_2")).alias("e2"),
        (pl.col("pct_t_plus_3") - pl.col("market_t_plus_3")).alias("e3"),
    ).with_columns(
        pl.max_horizontal(pl.lit(0.0), pl.col("e1")).alias("p1"),
        pl.max_horizontal(pl.lit(0.0), pl.col("e2")).alias("p2"),
        pl.max_horizontal(pl.lit(0.0), pl.col("e3")).alias("p3"),
    ).with_columns(
        ((W1 * pl.col("p1") + W2 * pl.col("p2") + W3 * pl.col("p3")) / W_SUM).alias("y")
    ).select(["trade_date", "ts_code", "y"])

    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: read realized_returns + market_returns from a bundle dir, write target_y.parquet."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="Defaults to <bundle>/target_y.parquet")
    args = ap.parse_args(argv)

    out_path = args.out or (args.bundle / "target_y.parquet")
    realized = pl.read_parquet(args.bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    )
    market = pl.read_parquet(args.bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    )
    logger.info("realized: %d rows  market: %d rows", len(realized), len(market))

    out = compute_target_y(realized, market)
    logger.info("output: %d rows  dates=%d  stocks=%d  y_mean=%.6f  y_std=%.6f",
                len(out), out["trade_date"].n_unique(), out["ts_code"].n_unique(),
                out["y"].mean(), out["y"].std())

    out.write_parquet(out_path, compression="zstd", compression_level=10)
    logger.info("wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
