"""Path 5 — compute per-day regime features for regime-aware stacking.

Outputs `data/p3_4070/regime_features.parquet` with daily indicators that
capture market regime (volatility, breadth, factor crowding, dispersion).
The downstream meta-learner (Path 5 stacking) reads these alongside the
3-path base scores to learn regime-conditional blending.

Computed entirely from existing bundle artifacts (realized_returns +
market_returns + universe_mask + factor_panel for industry_code) — no
paris-side dependency. Schema:

  trade_date                date
  univ_vol_20d              float32  — rolling 20d std of universe pct_chg
  univ_autocorr_20d         float32  — rolling 20d lag-1 autocorrelation
  univ_skew_20d             float32  — rolling 20d skew of pct_chg
  cs_dispersion             float32  — daily cross-sectional std of pct_chg
  top_bottom_spread         float32  — daily (top_1pct_mean - bot_1pct_mean)
  market_pct                float32  — today's market eq-weight pct_chg (lag 0)
  market_pct_t_minus_1      float32  — yesterday's market pct_chg
  sector_dispersion         float32  — std across sw_l1 sectors of mean pct_chg
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--combined-panel", default="data/factor_panel_combined_short_2023_2026.parquet",
                    type=Path, help="Source for industry_code (for sector_dispersion)")
    ap.add_argument("--out", default=None, type=Path,
                    help="Defaults to <bundle>/regime_features.parquet")
    args = ap.parse_args(argv)

    out = args.out or (args.bundle / "regime_features.parquet")

    # 1. Load realized returns + universe + market
    t0 = time.time()
    realized = pl.read_parquet(args.bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    )
    market = pl.read_parquet(args.bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    )

    uni_parts = []
    for year in (2023, 2024, 2025, 2026):
        p = args.bundle / "universe_mask" / f"year={year}.parquet"
        if p.exists():
            uni_parts.append(pl.read_parquet(p).select(["trade_date", "ts_code", "in_universe"]))
    universe = pl.concat(uni_parts)
    logger.info("loaded inputs in %.1fs: realized=%d market=%d universe=%d",
                time.time() - t0, len(realized), len(market), len(universe))

    # 2. Filter to in-universe stocks
    df = realized.join(universe, on=["trade_date", "ts_code"], how="left").filter(
        pl.col("in_universe") == True  # noqa: E712
    ).drop("in_universe")
    logger.info("in-universe rows: %d", len(df))

    # 3. Daily cross-sectional aggregates
    daily = df.group_by("trade_date").agg(
        pl.col("pct_chg_t_plus_1").mean().alias("market_pct_actual"),
        pl.col("pct_chg_t_plus_1").std().alias("cs_dispersion"),
        pl.col("pct_chg_t_plus_1").skew().alias("cs_skew_daily"),
        pl.col("pct_chg_t_plus_1").quantile(0.99, "linear").alias("p99"),
        pl.col("pct_chg_t_plus_1").quantile(0.01, "linear").alias("p01"),
        pl.col("pct_chg_t_plus_1").count().alias("n_stocks_daily"),
    ).sort("trade_date").with_columns(
        (pl.col("p99") - pl.col("p01")).alias("top_bottom_spread"),
    ).drop(["p99", "p01"])

    # 4. Rolling 20d aggregates: vol, autocorr, skew
    daily = daily.with_columns(
        pl.col("market_pct_actual").rolling_std(window_size=20).alias("univ_vol_20d"),
        # Skew (rolling) — polars doesn't have rolling_skew built-in; approximate
        # by m3 / m2^1.5 over 20-day window (custom).
    )

    # Lag-1 autocorr: corr(market_pct[t], market_pct[t-1]) over rolling 20d
    daily = daily.with_columns(
        pl.col("market_pct_actual").shift(1).alias("mp_lag1"),
    )
    # rolling correlation between mp_t and mp_{t-1} over 20d window
    daily = daily.with_columns(
        pl.rolling_corr(
            pl.col("market_pct_actual"), pl.col("mp_lag1"),
            window_size=20,
        ).alias("univ_autocorr_20d")
    )

    # Approx rolling skew: (mp - rolling_mean)^3 / (rolling_std)^3 averaged over window
    daily = daily.with_columns(
        pl.col("market_pct_actual").rolling_mean(window_size=20).alias("rmean_20d"),
        pl.col("market_pct_actual").rolling_std(window_size=20).alias("rstd_20d"),
    ).with_columns(
        (
            ((pl.col("market_pct_actual") - pl.col("rmean_20d")) / (pl.col("rstd_20d") + 1e-9)) ** 3
        ).rolling_mean(window_size=20).alias("univ_skew_20d")
    ).drop(["rmean_20d", "rstd_20d", "mp_lag1"])

    # 5. Sector dispersion: std across sw_l1 sectors of daily mean pct_chg
    if args.combined_panel.exists():
        logger.info("computing sector_dispersion from %s", args.combined_panel)
        meta = pl.read_parquet(args.combined_panel).select(
            ["trade_date", "ts_code", "industry_code", "pct_chg"]
        ).filter(pl.col("industry_code").is_not_null())
        sector_means = meta.group_by(["trade_date", "industry_code"]).agg(
            pl.col("pct_chg").mean().alias("sector_mean_pct_chg")
        )
        sector_dispersion = sector_means.group_by("trade_date").agg(
            pl.col("sector_mean_pct_chg").std().alias("sector_dispersion")
        )
        daily = daily.join(sector_dispersion, on="trade_date", how="left")
    else:
        logger.warning("combined panel not found; sector_dispersion will be null")
        daily = daily.with_columns(pl.lit(None, dtype=pl.Float32).alias("sector_dispersion"))

    # 6. Lag market_pct features
    # Note: market_returns.parquet has eq_weight_pct_chg_t_plus_1 (T+1 return from T's row).
    # For "today's market regime" we want the TRAILING market move, so use t-1's row's pct_chg_t_plus_1
    # which equals "market_pct happening BETWEEN today and tomorrow". For regime, use the actually
    # realized market_pct at trade_date (not t+1 forward) — which is `market_pct_actual` we just computed.
    daily = daily.with_columns(
        pl.col("market_pct_actual").shift(1).alias("market_pct_t_minus_1"),
        pl.col("market_pct_actual").shift(5).alias("market_pct_t_minus_5"),
    )

    # 7. Cast to compact dtypes
    out_df = daily.select([
        "trade_date",
        pl.col("market_pct_actual").cast(pl.Float32).alias("market_pct"),
        pl.col("market_pct_t_minus_1").cast(pl.Float32),
        pl.col("market_pct_t_minus_5").cast(pl.Float32),
        pl.col("univ_vol_20d").cast(pl.Float32),
        pl.col("univ_autocorr_20d").cast(pl.Float32),
        pl.col("univ_skew_20d").cast(pl.Float32),
        pl.col("cs_dispersion").cast(pl.Float32),
        pl.col("cs_skew_daily").cast(pl.Float32),
        pl.col("top_bottom_spread").cast(pl.Float32),
        pl.col("sector_dispersion").cast(pl.Float32),
        pl.col("n_stocks_daily").cast(pl.Int32),
    ])

    out_df.write_parquet(out, compression="zstd", compression_level=10)
    logger.info("wrote %s: %d rows × %d cols", out, len(out_df), len(out_df.columns))

    # 8. Sanity print
    logger.info("regime features summary:")
    for c in out_df.columns[1:]:
        s = out_df[c]
        if s.dtype == pl.Float32:
            logger.info("  %-25s mean=%+.6f std=%.6f min=%+.6f max=%+.6f null_frac=%.3f",
                        c, s.mean() or 0, s.std() or 0, s.min() or 0, s.max() or 0,
                        s.is_null().sum() / len(s))

    return 0


if __name__ == "__main__":
    sys.exit(main())
