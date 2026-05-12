"""Quick fwd-20d / fwd-40d realized excess eval for wave-v1 vs other paths.

Wave label v1 was designed to capture 20-40d cumulative excess. The eval_matrix_v2
only measures fwd-1d/3d/5d. This script extends to fwd-20d and fwd-40d to test the
label on its DESIGN horizon.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl


BUNDLE = Path("data/p3_4070")
OUT = Path("runs/eval_matrix_v2_with_wave/long_horizon.md")


SLICES = {
    "VAL":             (dt.date(2025, 1, 1),  dt.date(2025, 6, 4)),
    "H1":              (dt.date(2025, 7, 1),  dt.date(2025, 9, 30)),
    "H2":              (dt.date(2025, 10, 1), dt.date(2025, 12, 31)),
    "2026-Q1":         (dt.date(2026, 1, 1),  dt.date(2026, 3, 31)),
}


PATHS = [
    ("Path 1 long",         "runs/sl_path1_long/predictions.parquet",             "score_calibrated"),
    ("Path 4 short (prod)", "runs/sl_path4/predictions.parquet",                  "score_calibrated"),
    ("Path 4 long",         "runs/sl_path_d/predictions.parquet",                 "score_calibrated"),
    ("Path 5 long",         "runs/sl_regime_stack_long/predictions.parquet",      "score_calibrated"),
    ("Hybrid p1L+p4S",      "runs/sl_hybrid_p1long_p4short/predictions.parquet",  "score"),
    ("Wave v1 (P1L retrain)", "runs/sl_path1_long_wave_v1/predictions.parquet",   "score_calibrated"),
]


def build_long_fwd_excess() -> pl.DataFrame:
    realized = pl.read_parquet(BUNDLE / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    ).sort(["ts_code", "trade_date"])
    market = pl.read_parquet(BUNDLE / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    ).sort("trade_date")
    for K in (20, 40):
        realized = realized.with_columns(
            pl.col("pct_chg_t_plus_1").log1p()
              .rolling_sum(window_size=K)
              .shift(-(K - 1))
              .exp().sub(1.0)
              .over("ts_code")
              .alias(f"fwd_{K}d")
        )
        market = market.with_columns(
            pl.col("eq_weight_pct_chg_t_plus_1").log1p()
              .rolling_sum(window_size=K)
              .shift(-(K - 1))
              .exp().sub(1.0)
              .alias(f"mkt_fwd_{K}d")
        )
    df = realized.join(market.select(["trade_date", "mkt_fwd_20d", "mkt_fwd_40d"]), on="trade_date", how="left")
    df = df.with_columns([
        (pl.col("fwd_20d") - pl.col("mkt_fwd_20d")).alias("excess_20d"),
        (pl.col("fwd_40d") - pl.col("mkt_fwd_40d")).alias("excess_40d"),
    ])
    return df.select(["trade_date", "ts_code", "excess_20d", "excess_40d"])


def main():
    print("[load] forward 20d/40d excess")
    fwd = build_long_fwd_excess()
    print(f"  rows: {len(fwd):,}  date range {fwd['trade_date'].min()} ~ {fwd['trade_date'].max()}")

    results = {}
    for name, path, score_col in PATHS:
        preds = pl.read_parquet(path)
        joined = preds.join(fwd, on=["trade_date", "ts_code"], how="inner")
        results[name] = {}
        for sname, (s, e) in SLICES.items():
            d = joined.filter((pl.col("trade_date") >= s) & (pl.col("trade_date") <= e))
            if len(d) == 0:
                results[name][sname] = {"n_days": 0}
                continue
            top = (
                d.sort(["trade_date", score_col], descending=[False, True])
                .group_by("trade_date", maintain_order=True)
                .head(50)
            )
            daily = top.group_by("trade_date").agg([
                pl.col("excess_20d").mean().alias("d20"),
                pl.col("excess_40d").mean().alias("d40"),
                (pl.col("excess_20d") > 0).cast(pl.Float64).mean().alias("h20"),
                (pl.col("excess_40d") > 0).cast(pl.Float64).mean().alias("h40"),
            ])
            results[name][sname] = {
                "n_days": len(daily),
                "fwd_20d_excess": float(daily["d20"].mean()),
                "fwd_40d_excess": float(daily["d40"].mean()),
                "T20_hit": float(daily["h20"].mean()),
                "T40_hit": float(daily["h40"].mean()),
            }
            r = results[name][sname]
            print(f"  {name:25} {sname:10} n={r['n_days']:>3} "
                  f"fwd20={r['fwd_20d_excess']:+.5f} fwd40={r['fwd_40d_excess']:+.5f} "
                  f"T20={r['T20_hit']:.3f} T40={r['T40_hit']:.3f}")

    # Markdown
    md = ["# Wave v1 long-horizon eval — fwd-20d / fwd-40d realized excess", "",
          "Wave label v1 was designed to capture 20-40d cumulative excess (per-day mean over top-50).",
          "If the label works as designed, Wave v1 should beat Path 1 long on fwd_20d / fwd_40d", ""]

    for metric, label, fmt in [
        ("fwd_20d_excess", "Top-50 fwd-20d cumulative excess", "+.5f"),
        ("fwd_40d_excess", "Top-50 fwd-40d cumulative excess", "+.5f"),
        ("T20_hit", "T20 hit (fwd-20d excess > 0)", ".3f"),
        ("T40_hit", "T40 hit (fwd-40d excess > 0)", ".3f"),
    ]:
        md.append(f"## {label}")
        md.append("")
        md.append("| Path | " + " | ".join(SLICES.keys()) + " |")
        md.append("|---" + "|---:" * len(SLICES) + "|")
        for name in results:
            row = [name]
            for s in SLICES.keys():
                v = results[name].get(s, {}).get(metric)
                row.append(format(v, fmt) if v is not None else "—")
            md.append("| " + " | ".join(row) + " |")
        md.append("")

    md.append("## Verdict")
    md.append("")
    for sname in SLICES:
        # Compare Wave v1 to Path 1 long on fwd_20d
        p1l = results.get("Path 1 long", {}).get(sname, {}).get("fwd_20d_excess")
        w1 = results.get("Wave v1 (P1L retrain)", {}).get(sname, {}).get("fwd_20d_excess")
        if p1l is not None and w1 is not None:
            d = (w1 - p1l) * 1e4
            md.append(f"- **{sname}** fwd-20d: Wave v1 - Path 1 long = {d:+.1f} bps")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
