"""Hybrid v3 — rank-pct normalize cross-family blend (Path 1 long + Wave v2).

Per paris Q2 (2026-05-13 v5 reply): proximity score ~0.0018 std 0.0008, wave_v2 score
~0.049 std 0.005 — 10× scale mismatch. 50/50 average on raw calibrated scores would
be wave-dominated. Use rank-pct per day:

  rank_proxy = rank(score_path1_long) / N    # ∈ [0, 1]
  rank_wave  = rank(score_wave_v2)    / N
  hybrid_v3  = (rank_proxy + rank_wave) / 2

Saves predictions.parquet + adds to eval_matrix.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl


P1L = Path("runs/sl_path1_long/predictions.parquet")
WV2 = Path("runs/sl_path1_long_wave_v2/predictions.parquet")
OUT = Path("runs/sl_hybrid_v3_p1long_wavev2")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("[hybrid v3] reading P1L + Wave v2 calibrated scores ...")
    p1l = pl.read_parquet(P1L).select(["trade_date", "ts_code",
                                       pl.col("score_calibrated").alias("p1l_score")])
    wv2 = pl.read_parquet(WV2).select(["trade_date", "ts_code",
                                       pl.col("score_calibrated").alias("wv2_score")])
    df = p1l.join(wv2, on=["trade_date", "ts_code"], how="inner")
    print(f"  joined rows: {len(df):,}")

    # Per-day rank-pct
    df = df.with_columns([
        (pl.col("p1l_score").rank().over("trade_date") /
         pl.col("p1l_score").count().over("trade_date")).alias("rank_p1l"),
        (pl.col("wv2_score").rank().over("trade_date") /
         pl.col("wv2_score").count().over("trade_date")).alias("rank_wv2"),
    ])
    df = df.with_columns(
        ((pl.col("rank_p1l") + pl.col("rank_wv2")) / 2.0).alias("score")
    )

    out = df.select(["trade_date", "ts_code", "p1l_score", "wv2_score",
                     "rank_p1l", "rank_wv2", "score"])
    out.write_parquet(OUT / "predictions.parquet", compression="zstd", compression_level=9)
    print(f"  wrote {OUT / 'predictions.parquet'}  ({len(out):,} rows)")

    # Quick eval — top-50 realized fwd-K excess on H2 / 2026-Q1
    import datetime as dt
    fwd = build_fwd(Path("data/p3_4070"))
    joined = out.select(["trade_date", "ts_code", "score"]).join(fwd, on=["trade_date","ts_code"], how="inner")
    print("\n=== Hybrid v3 quick scoreboard ===")
    for sname, (s, e) in [
        ("VAL", (dt.date(2025,1,1), dt.date(2025,6,4))),
        ("H1",  (dt.date(2025,7,1), dt.date(2025,9,30))),
        ("H2",  (dt.date(2025,10,1), dt.date(2025,12,31))),
        ("2026-Q1", (dt.date(2026,1,1), dt.date(2026,3,31))),
        ("2026-Q2-partial", (dt.date(2026,4,1), dt.date(2026,5,11))),
    ]:
        d = joined.filter((pl.col("trade_date") >= s) & (pl.col("trade_date") <= e))
        if len(d) == 0:
            print(f"  {sname:18}  empty"); continue
        top = d.sort(["trade_date","score"], descending=[False, True]).group_by(
            "trade_date", maintain_order=True
        ).head(50)
        daily = top.group_by("trade_date").agg([
            pl.col("excess_1d").mean().alias("e1"),
            pl.col("excess_3d").mean().alias("e3"),
            pl.col("excess_5d").mean().alias("e5"),
            pl.col("excess_20d").mean().alias("e20"),
            (pl.col("excess_1d") > 0).cast(pl.Float64).mean().alias("h1"),
        ])
        print(f"  {sname:18}  n={len(daily):>3}  "
              f"fwd1d={daily['e1'].mean():+.5f}  fwd3d={daily['e3'].mean():+.5f}  "
              f"fwd5d={daily['e5'].mean():+.5f}  fwd20d={daily['e20'].mean():+.5f}  "
              f"T1={daily['h1'].mean():.3f}")


def build_fwd(bundle: Path) -> pl.DataFrame:
    realized = pl.read_parquet(bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    ).sort(["ts_code", "trade_date"])
    market = pl.read_parquet(bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    ).sort("trade_date")
    for K in (1, 3, 5, 20):
        realized = realized.with_columns(
            pl.col("pct_chg_t_plus_1").log1p().rolling_sum(window_size=K).shift(-(K-1))
              .exp().sub(1.0).over("ts_code").alias(f"fwd_{K}d")
        )
        market = market.with_columns(
            pl.col("eq_weight_pct_chg_t_plus_1").log1p().rolling_sum(window_size=K).shift(-(K-1))
              .exp().sub(1.0).alias(f"mkt_fwd_{K}d")
        )
    df = realized.join(market.select(["trade_date","mkt_fwd_1d","mkt_fwd_3d","mkt_fwd_5d","mkt_fwd_20d"]),
                       on="trade_date", how="left")
    for K in (1, 3, 5, 20):
        df = df.with_columns((pl.col(f"fwd_{K}d") - pl.col(f"mkt_fwd_{K}d")).alias(f"excess_{K}d"))
    return df.select(["trade_date","ts_code","excess_1d","excess_3d","excess_5d","excess_20d"])


if __name__ == "__main__":
    main()
