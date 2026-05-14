"""Wave label v3 — paris P4 (L2 industry) + P5 (northbound top-10) integrated.

V2 → V3 changes:
  REPLACE Industry-momentum bonus: L1 (31 categories) → L2 (127 categories) +
          per-stock momentum (top-5 in L2 or rotated_into_top5).
  ADD     Northbound top-10 bonus: per-stock in北向 top-10 today → +5%.
          rank_normalized = (11 - rank) / 10 (1.0 if rank=1, 0.1 if rank=10).

Final label (same structure as v2):
  wave_quality = 0.50*excess_20d + 0.25*excess_40d + 0.0015*Sharpe_20d + 0.10*max_dd_20d
  entry_timing_decay = min(d/5, 1.0)
  bonus = 1 + 0.10*vp_conf + 0.10*ind_l2_momentum + 0.05*lianban_norm + 0.05*north_top10
  label = max(0, wave_quality * entry_timing_decay * bonus)
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl


def build_wave_v3(bundle: Path, out_name: str = "target_y_wave_v3.parquet"):
    print(f"[wave-v3] bundle={bundle}")
    realized = pl.read_parquet(bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    ).sort(["ts_code", "trade_date"])
    market = pl.read_parquet(bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    ).sort("trade_date")
    cv = pl.read_parquet(bundle / "stock_close_volume_daily.parquet").select(
        ["trade_date", "ts_code", "volume"]
    )

    K_short = 20
    K_long = 40

    # === fwd returns ===
    r = realized
    for K in (K_short, K_long):
        r = r.with_columns(
            pl.col("pct_chg_t_plus_1").log1p()
              .rolling_sum(window_size=K)
              .shift(-(K - 1))
              .exp().sub(1.0)
              .over("ts_code")
              .alias(f"fwd_{K}d")
        )
    r = r.with_columns(
        pl.col("pct_chg_t_plus_1").log1p()
          .rolling_std(window_size=K_short)
          .shift(-(K_short - 1))
          .over("ts_code")
          .alias(f"daily_logret_std_{K_short}d")
    ).with_columns(
        (pl.col(f"fwd_{K_short}d") /
         (pl.col(f"daily_logret_std_{K_short}d") * np.sqrt(K_short) + 1e-6))
        .alias("sharpe_20d")
    )

    # Compounded path for max_dd + day_to_peak
    for k in range(1, K_short + 1):
        r = r.with_columns(
            pl.col("pct_chg_t_plus_1").shift(-(k - 1)).over("ts_code").alias(f"_r_p{k}")
        )
    r = r.with_columns((1.0 + pl.col("_r_p1")).alias("_c_p1"))
    for k in range(2, K_short + 1):
        r = r.with_columns(
            (pl.col(f"_c_p{k-1}") * (1.0 + pl.col(f"_r_p{k}"))).alias(f"_c_p{k}")
        )
    cum_cols = [f"_c_p{k}" for k in range(1, K_short + 1)]
    r = r.with_columns(
        pl.min_horizontal([pl.col(c) for c in cum_cols]).sub(1.0).alias("max_dd_20d"),
    )

    # Market fwd
    m = market
    for K in (K_short, K_long):
        m = m.with_columns(
            pl.col("eq_weight_pct_chg_t_plus_1").log1p()
              .rolling_sum(window_size=K)
              .shift(-(K - 1))
              .exp().sub(1.0)
              .alias(f"mkt_fwd_{K}d")
        )
    m = m.select(["trade_date", "mkt_fwd_20d", "mkt_fwd_40d"])

    df = r.select(["trade_date", "ts_code",
                   "fwd_20d", "fwd_40d", "sharpe_20d", "max_dd_20d"] + cum_cols)
    df = df.join(m, on="trade_date", how="left")
    df = df.with_columns([
        (pl.col("fwd_20d") - pl.col("mkt_fwd_20d")).alias("excess_20d"),
        (pl.col("fwd_40d") - pl.col("mkt_fwd_40d")).alias("excess_40d"),
    ])

    # day_to_peak via numpy
    print("  computing day_to_peak ...")
    cum_np = df.select(cum_cols).to_numpy()
    finite = np.isfinite(cum_np).all(axis=1)
    day_to_peak = np.full(len(df), np.nan, dtype=np.float32)
    if finite.any():
        day_to_peak[finite] = np.argmax(cum_np[finite], axis=1).astype(np.float32) + 1.0
    df = df.with_columns(pl.Series("day_to_peak_20d", day_to_peak))

    # === BONUS 1: volume_price_confirmation (z-score-20d > 0) ===
    cv = cv.sort(["ts_code", "trade_date"]).with_columns([
        pl.col("volume").rolling_mean(window_size=20).over("ts_code").alias("vol_mean_20d"),
        pl.col("volume").rolling_std(window_size=20).over("ts_code").alias("vol_std_20d"),
    ]).with_columns(
        ((pl.col("volume") - pl.col("vol_mean_20d")) /
         (pl.col("vol_std_20d") + 1e-6)).alias("vol_zscore_20d")
    ).select(["trade_date", "ts_code", "vol_zscore_20d"])
    df = df.join(cv, on=["trade_date", "ts_code"], how="left").with_columns(
        ((pl.col("vol_zscore_20d") > 0).cast(pl.Float64).fill_null(0.0)).alias("vp_conf")
    )

    # === BONUS 2: L2 industry-momentum (NEW v3) ===
    # Use P4 (stock_industry_l2_daily) to map (date, ts_code) → l2_code, then compute
    # rolling 20d L2-level mean_pct_chg + per-day rank, take top-25%.
    stock_l2 = pl.read_parquet(bundle / "stock_industry_l2_daily.parquet").select(
        ["trade_date", "ts_code", "l2_code"]
    ).with_columns(pl.col("trade_date").cast(pl.Date))

    # Compute L2-level daily mean_pct_chg (need pct_chg_t_plus_1 by date+ts_code, join with l2)
    # Then group by (date, l2_code) for industry mean. We approximate via shift on realized.
    print("  computing L2 industry momentum ...")
    daily_stock = realized.select(["trade_date", "ts_code", "pct_chg_t_plus_1"])
    daily_stock = daily_stock.join(stock_l2, on=["trade_date", "ts_code"], how="inner")
    l2_daily = daily_stock.group_by(["trade_date", "l2_code"]).agg(
        pl.col("pct_chg_t_plus_1").mean().alias("l2_mean_pct")
    ).sort(["l2_code", "trade_date"]).with_columns(
        pl.col("l2_mean_pct").rolling_mean(window_size=20).over("l2_code").alias("l2_mom_20d")
    )
    # Rank L2 per day (lower = better)
    l2_daily = l2_daily.sort(["trade_date", "l2_mom_20d"], descending=[False, True]).with_columns(
        pl.cum_count("trade_date").over("trade_date").alias("l2_rank_today")
    )
    n_l2 = l2_daily["l2_code"].n_unique()
    print(f"  L2 categories: {n_l2}")
    # top 25% = rank <= n_l2 * 0.25
    threshold = int(n_l2 * 0.25)
    l2_daily = l2_daily.with_columns(
        (pl.col("l2_rank_today") <= threshold).cast(pl.Float64).fill_null(0.0).alias("l2_hot")
    ).select(["trade_date", "l2_code", "l2_hot"])

    # Join into df via (date, stock → l2 → hot)
    df = df.join(stock_l2, on=["trade_date", "ts_code"], how="left")
    df = df.join(l2_daily, on=["trade_date", "l2_code"], how="left")
    df = df.with_columns(pl.col("l2_hot").fill_null(0.0).alias("ind_l2_momentum"))

    # === BONUS 3: lianban_n_ge_4 day-level ===
    lim = pl.read_parquet("data/p3_4070_slices_v2/limit_daily_agg.parquet").select(
        ["trade_date", "lianban_n_ge_4"]
    ).with_columns(pl.col("trade_date").cast(pl.Date))
    p95 = lim["lianban_n_ge_4"].quantile(0.95)
    p95 = max(float(p95) if p95 is not None else 1.0, 1.0)
    lim = lim.with_columns(
        (pl.col("lianban_n_ge_4") / p95).clip(lower_bound=0.0, upper_bound=1.0)
          .fill_null(0.0).alias("lianban_norm")
    ).select(["trade_date", "lianban_norm"])
    df = df.join(lim, on="trade_date", how="left").with_columns(
        pl.col("lianban_norm").fill_null(0.0)
    )

    # === BONUS 4: NORTHBOUND TOP-10 (NEW v3) ===
    nb = pl.read_parquet(bundle / "northbound_top10_daily.parquet").select(
        ["trade_date", "ts_code", "rank"]
    )
    # rank_normalized = (11 - rank) / 10 ∈ [0.1, 1.0]; aggregate across market_types (max if in both)
    nb = nb.with_columns(((11 - pl.col("rank")) / 10.0).alias("nb_rank_norm")).group_by(
        ["trade_date", "ts_code"]
    ).agg(pl.col("nb_rank_norm").max().alias("north_top10"))
    df = df.join(nb, on=["trade_date", "ts_code"], how="left").with_columns(
        pl.col("north_top10").fill_null(0.0)
    )

    # === FINAL LABEL ===
    df = df.with_columns([
        (
            0.50 * pl.col("excess_20d")
          + 0.25 * pl.col("excess_40d")
          + 0.15 * pl.col("sharpe_20d").clip(lower_bound=-2.0, upper_bound=2.0) * 0.01
          + 0.10 * pl.col("max_dd_20d")
        ).alias("wave_quality"),
        pl.min_horizontal([pl.col("day_to_peak_20d") / 5.0, pl.lit(1.0)])
          .alias("entry_timing_decay"),
        (
            1.0
          + 0.10 * pl.col("vp_conf")
          + 0.10 * pl.col("ind_l2_momentum")     # was L1, now L2
          + 0.05 * pl.col("lianban_norm")
          + 0.05 * pl.col("north_top10")          # NEW v3
        ).alias("bonus_v3"),
    ])
    df = df.with_columns(
        (pl.col("wave_quality") * pl.col("entry_timing_decay") * pl.col("bonus_v3"))
          .clip(lower_bound=0.0).alias("y")
    )

    out = df.select([
        "trade_date", "ts_code", "y",
        "excess_20d", "excess_40d", "sharpe_20d", "max_dd_20d",
        "day_to_peak_20d", "entry_timing_decay",
        "vp_conf", "ind_l2_momentum", "lianban_norm", "north_top10", "bonus_v3",
        "wave_quality",
    ]).drop_nulls(subset=["y"])

    out_path = bundle / out_name
    out.write_parquet(out_path, compression="zstd", compression_level=9)
    print(f"  wrote {out_path}  rows={len(out):,}")
    print(f"  y stats: mean={out['y'].mean():.5f}  std={out['y'].std():.5f}  "
          f"min={out['y'].min():.5f}  max={out['y'].max():.5f}  "
          f"pct_zero={(out['y'] == 0).sum() / len(out) * 100:.1f}%")
    # bonus stats
    print(f"  north_top10 hit rate: {(out['north_top10'] > 0).sum() / len(out) * 100:.1f}%")
    print(f"  ind_l2_momentum hit rate: {(out['ind_l2_momentum'] > 0).sum() / len(out) * 100:.1f}%")
    return out_path


def main():
    build_wave_v3(Path("data/p3_4070_long"))


if __name__ == "__main__":
    main()
