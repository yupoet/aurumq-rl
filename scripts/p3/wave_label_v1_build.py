"""Wave label v1 — wave_quality × entry_timing_decay.

Per user spec (2026-05-12):
  - wave_quality = 0.5*fwd_20d_excess + 0.25*fwd_40d_excess
                 + 0.15*trend_smoothness + 0.10*max_dd_penalty
  - entry_timing_decay = exp(-k * day_to_peak / 5), where day_to_peak is the
    forward-40d day at which compounded return peaks (entry near peak start ⇒ decay ≈ 1)
  - label = max(0, wave_quality * entry_timing_decay)

Excess vs main-board eq-weight market (same as current target_y).

Writes: data/p3_4070/target_y_wave_v1.parquet  + data/p3_4070_long/target_y_wave_v1.parquet
  schema: trade_date, ts_code, y
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl


def build_wave_label(bundle: Path, out_name: str = "target_y_wave_v1.parquet"):
    print(f"[wave label] bundle={bundle}")
    realized = pl.read_parquet(bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    ).sort(["ts_code", "trade_date"])
    market = pl.read_parquet(bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    ).sort("trade_date")

    K_short = 20
    K_long = 40

    # === Per-stock forward returns ===
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

    # Trend smoothness over 20d (Sharpe-like): cum_return / (std * sqrt(N))
    # Compute via rolling_std of log-returns × sqrt(K) ≈ approx return std at K-day horizon
    r = r.with_columns(
        pl.col("pct_chg_t_plus_1").log1p()
          .rolling_std(window_size=K_short)
          .shift(-(K_short - 1))
          .over("ts_code")
          .alias(f"daily_logret_std_{K_short}d")
    )
    r = r.with_columns(
        (pl.col(f"fwd_{K_short}d") /
         (pl.col(f"daily_logret_std_{K_short}d") * np.sqrt(K_short) + 1e-6))
        .alias("trend_smoothness_20d")
    )

    # Per-entry max drawdown within forward 20d.
    # Build compounded path: c[k] = product of (1+pct[T+1..T+k]) for k=1..20.
    # max_dd = min(c) - 1 (negative); penalty = max_dd (we'll add as +max_dd, which is ≤0)
    # We approximate via the existing eval_matrix approach but for 20d.
    for k in range(1, K_short + 1):
        r = r.with_columns(
            pl.col("pct_chg_t_plus_1").shift(-(k - 1)).over("ts_code").alias(f"_r_p{k}")
        )
    # Build compounded path
    r = r.with_columns((1.0 + pl.col("_r_p1")).alias("_c_p1"))
    for k in range(2, K_short + 1):
        r = r.with_columns(
            (pl.col(f"_c_p{k-1}") * (1.0 + pl.col(f"_r_p{k}"))).alias(f"_c_p{k}")
        )
    cum_cols = [f"_c_p{k}" for k in range(1, K_short + 1)]
    r = r.with_columns([
        pl.min_horizontal([pl.col(c) for c in cum_cols]).sub(1.0).alias("max_dd_20d"),
        pl.max_horizontal([pl.col(c) for c in cum_cols]).sub(1.0).alias("max_ret_20d"),
    ])
    # Day-of-peak within 20d window: argmax of compounded path (1..K_short)
    # Use horizontal arg_max not supported directly; approximate by horizontal max idx via
    # a small numpy step after collecting cum_cols.

    # Drop intermediate compounded cols early before merge with market (memory)
    r_keep = r.select([
        "trade_date", "ts_code",
        f"fwd_{K_short}d", f"fwd_{K_long}d", "trend_smoothness_20d",
        "max_dd_20d", "max_ret_20d",
    ] + cum_cols)

    # === Market forward (for excess) ===
    m = market
    for K in (K_short, K_long):
        m = m.with_columns(
            pl.col("eq_weight_pct_chg_t_plus_1").log1p()
              .rolling_sum(window_size=K)
              .shift(-(K - 1))
              .exp().sub(1.0)
              .alias(f"mkt_fwd_{K}d")
        )
    m = m.select(["trade_date", f"mkt_fwd_{K_short}d", f"mkt_fwd_{K_long}d"])

    df = r_keep.join(m, on="trade_date", how="left")
    df = df.with_columns([
        (pl.col(f"fwd_{K_short}d") - pl.col(f"mkt_fwd_{K_short}d")).alias(f"excess_{K_short}d"),
        (pl.col(f"fwd_{K_long}d") - pl.col(f"mkt_fwd_{K_long}d")).alias(f"excess_{K_long}d"),
    ])

    # === day_to_peak via numpy (extract cum_cols, argmax, decay) ===
    print("  computing day_to_peak via numpy argmax ...")
    cum_np = df.select(cum_cols).to_numpy()
    # valid mask: all cum values finite
    finite = np.isfinite(cum_np).all(axis=1)
    day_to_peak = np.full(len(df), np.nan, dtype=np.float32)
    if finite.any():
        # argmax along columns: 0-based idx 0..K_short-1 → +1 = day_to_peak ∈ [1, K_short]
        day_to_peak[finite] = np.argmax(cum_np[finite], axis=1).astype(np.float32) + 1.0
    df = df.with_columns(pl.Series("day_to_peak_20d", day_to_peak))

    # === wave_quality + entry_timing_decay + final label ===
    # trend_smoothness already computed.
    # max_dd_penalty: max_dd is in [-1, 0]. Penalty = max_dd (so closer to 0 = better).
    # We add max_dd directly so big drawdowns subtract weight.
    df = df.with_columns([
        (
            0.50 * pl.col(f"excess_{K_short}d")
          + 0.25 * pl.col(f"excess_{K_long}d")
          + 0.15 * pl.col("trend_smoothness_20d").clip(lower_bound=-2.0, upper_bound=2.0) * 0.05
          + 0.10 * pl.col("max_dd_20d")
        ).alias("wave_quality_raw")
    ])
    # entry_timing_decay: exp(-(day_to_peak - 1) / 5)  → 1.0 at day 1, ~0.45 at day 5, ~0.02 at day 20
    df = df.with_columns(
        (-(pl.col("day_to_peak_20d") - 1.0) / 5.0).exp().alias("entry_timing_decay")
    )

    # Final label
    df = df.with_columns(
        (pl.col("wave_quality_raw") * pl.col("entry_timing_decay")).clip(lower_bound=0.0).alias("y")
    )

    out = df.select(["trade_date", "ts_code", "y",
                     f"excess_{K_short}d", f"excess_{K_long}d",
                     "trend_smoothness_20d", "max_dd_20d", "day_to_peak_20d", "entry_timing_decay",
                     "wave_quality_raw"])
    # Drop rows with NaN y (insufficient forward window at panel tail)
    out = out.drop_nulls(subset=["y"])
    out_path = bundle / out_name
    out.write_parquet(out_path, compression="zstd", compression_level=9)
    print(f"  wrote {out_path}  rows={len(out):,}")
    print(f"  y stats: mean={out['y'].mean():.5f}  std={out['y'].std():.5f}  "
          f"min={out['y'].min():.5f}  max={out['y'].max():.5f}  "
          f"pct_zero={(out['y'] == 0).sum() / len(out) * 100:.1f}%")
    return out_path


def main():
    # Build for short bundle first (sanity check faster)
    build_wave_label(Path("data/p3_4070"))
    # Then for long bundle (used by Path 1 long retrain)
    build_wave_label(Path("data/p3_4070_long"))


if __name__ == "__main__":
    main()
