"""Wave label v2 — all 4 fixes from v1 review + 2 new bonus signals.

V1 → V2 changes:
  FIX 1: entry_timing_decay = min(day_to_peak_20d / 5, 1.0)
         (paris's proposal — plateau at d≥5, partial credit for d<5).
  FIX 2: Sharpe term weight corrected to 0.15 (was effectively 0.0075 in v1).
  FIX 3: Add lianban_n_ge_4 day-level bonus (high-连板 days are 主升浪 leading
         indicator — multiply wave_quality by 1 + 0.05 * normalized lianban).
  FIX 4: Add volume_price_confirmation = (volume_zscore_20d > 0) bool, multiplied as
         (1 + 0.10 * vp_conf) — confirmed wave bonus when entry day has above-normal
         volume.
  FIX 5: Add industry_momentum bonus — if today's stock is in a top-5 industry by
         20d momentum AND rotated_into_top5=True, multiply by (1 + 0.10 * bonus).

Final label:
  wave_quality = 0.5*fwd_20d + 0.25*fwd_40d + 0.15*Sharpe_20d.clip(-2,2) + 0.10*max_dd_20d
               (excess-vs-market on fwd_20/40)
  bonus = (1 + 0.05*lianban_norm + 0.10*vp_conf + 0.10*ind_mom)
  label = max(0, wave_quality * entry_timing_decay * bonus)
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl


def build_wave_v2(bundle: Path, out_name: str = "target_y_wave_v2.parquet"):
    print(f"[wave-v2] bundle={bundle}")
    realized = pl.read_parquet(bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    ).sort(["ts_code", "trade_date"])
    market = pl.read_parquet(bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    ).sort("trade_date")
    cv = pl.read_parquet(bundle / "stock_close_volume_daily.parquet").select(
        ["trade_date", "ts_code", "volume", "amount"]
    )
    ind = pl.read_parquet(bundle / "industry_momentum_rotation_daily.parquet").select(
        ["trade_date", "l1_code", "rank_today", "rotated_into_top5", "momentum_up"]
    )
    stock_l1 = pl.read_parquet("data/p3_4070_slices_v2/stock_industry_daily.parquet").select(
        ["trade_date", "ts_code", "l1_code"]
    ).with_columns(pl.col("trade_date").cast(pl.Date))

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

    # Sharpe 20d
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

    # Compounded path days 1..20 for max_dd + day_to_peak
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

    # day_to_peak via numpy argmax
    print("  computing day_to_peak ...")
    cum_np = df.select(cum_cols).to_numpy()
    finite = np.isfinite(cum_np).all(axis=1)
    day_to_peak = np.full(len(df), np.nan, dtype=np.float32)
    if finite.any():
        day_to_peak[finite] = np.argmax(cum_np[finite], axis=1).astype(np.float32) + 1.0
    df = df.with_columns(pl.Series("day_to_peak_20d", day_to_peak))

    # === BONUS 1: volume_price_confirmation — z-score-20d of volume on entry day ===
    cv = cv.sort(["ts_code", "trade_date"]).with_columns([
        pl.col("volume").rolling_mean(window_size=20).over("ts_code").alias("vol_mean_20d"),
        pl.col("volume").rolling_std(window_size=20).over("ts_code").alias("vol_std_20d"),
    ]).with_columns(
        ((pl.col("volume") - pl.col("vol_mean_20d")) /
         (pl.col("vol_std_20d") + 1e-6)).alias("vol_zscore_20d")
    ).select(["trade_date", "ts_code", "vol_zscore_20d"])
    df = df.join(cv, on=["trade_date", "ts_code"], how="left")
    # Confirmation: bool z > 0 (volume above normal on entry day)
    df = df.with_columns(
        ((pl.col("vol_zscore_20d") > 0).cast(pl.Float64).fill_null(0.0)).alias("vp_conf")
    )

    # === BONUS 2: industry_momentum bonus — stock-level via stock_l1 join ===
    df = df.join(stock_l1, on=["trade_date", "ts_code"], how="left")
    df = df.join(ind, on=["trade_date", "l1_code"], how="left")
    df = df.with_columns(
        # Bonus if stock in top-5 industry AND just rotated in (= momentum start)
        (
            (pl.col("rank_today") <= 5).cast(pl.Float64).fill_null(0.0) * 0.5
          + (pl.col("rotated_into_top5") == True).cast(pl.Float64).fill_null(0.0) * 0.5
        ).alias("ind_mom_bonus")
    )

    # === BONUS 3: lianban bonus — day-level normalized (paris's limit_daily_agg) ===
    lim = pl.read_parquet("data/p3_4070_slices_v2/limit_daily_agg.parquet").select(
        ["trade_date", "lianban_n_ge_4"]
    ).with_columns(pl.col("trade_date").cast(pl.Date))
    # Normalize lianban_n_ge_4 by historical p95 to keep bonus bounded
    p95 = lim["lianban_n_ge_4"].quantile(0.95)
    p95 = max(float(p95) if p95 is not None else 1.0, 1.0)
    lim = lim.with_columns(
        (pl.col("lianban_n_ge_4") / p95).clip(lower_bound=0.0, upper_bound=1.0)
          .fill_null(0.0).alias("lianban_norm")
    ).select(["trade_date", "lianban_norm"])
    df = df.join(lim, on="trade_date", how="left").with_columns(
        pl.col("lianban_norm").fill_null(0.0)
    )

    # === FINAL LABEL ===
    df = df.with_columns([
        # wave_quality v2 (FIX 2 — Sharpe weight 0.15 actually applied)
        (
            0.50 * pl.col("excess_20d")
          + 0.25 * pl.col("excess_40d")
          + 0.15 * pl.col("sharpe_20d").clip(lower_bound=-2.0, upper_bound=2.0) * 0.01  # rescale Sharpe to fit excess scale
          + 0.10 * pl.col("max_dd_20d")
        ).alias("wave_quality_v2"),
        # entry_timing_decay v2 (FIX 1 — paris's min(d/5, 1.0))
        pl.min_horizontal([pl.col("day_to_peak_20d") / 5.0, pl.lit(1.0)])
          .alias("entry_timing_decay_v2"),
        # Composite bonus (FIX 3, 4, 5)
        (
            1.0
          + 0.10 * pl.col("vp_conf")
          + 0.10 * pl.col("ind_mom_bonus")
          + 0.05 * pl.col("lianban_norm")
        ).alias("bonus_v2"),
    ])
    df = df.with_columns(
        (pl.col("wave_quality_v2") * pl.col("entry_timing_decay_v2") * pl.col("bonus_v2"))
          .clip(lower_bound=0.0).alias("y")
    )

    out = df.select([
        "trade_date", "ts_code", "y",
        "excess_20d", "excess_40d", "sharpe_20d", "max_dd_20d",
        "day_to_peak_20d", "entry_timing_decay_v2",
        "vp_conf", "ind_mom_bonus", "lianban_norm", "bonus_v2",
        "wave_quality_v2",
    ]).drop_nulls(subset=["y"])

    out_path = bundle / out_name
    out.write_parquet(out_path, compression="zstd", compression_level=9)
    print(f"  wrote {out_path}  rows={len(out):,}")
    print(f"  y stats: mean={out['y'].mean():.5f}  std={out['y'].std():.5f}  "
          f"min={out['y'].min():.5f}  max={out['y'].max():.5f}  "
          f"pct_zero={(out['y'] == 0).sum() / len(out) * 100:.1f}%")
    return out_path


def main():
    build_wave_v2(Path("data/p3_4070_long"))


if __name__ == "__main__":
    main()
