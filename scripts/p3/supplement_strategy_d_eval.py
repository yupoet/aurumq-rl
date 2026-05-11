"""Apply Strategy D (score-weighted top-50) sizing to a path's predictions
and compute realized H1/H2 mean_y.

Usage:
    python supplement_strategy_d_eval.py \\
        --bundle data/p3_4070 \\
        --predictions runs/sl_path1_long/predictions.parquet \\
        --label "Path 1 long"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))
from p3.path1_eval import H1, H2


def evaluate_strategy_d(pred_df: pl.DataFrame, target_y: pl.DataFrame, window: tuple) -> dict:
    """Score-weighted top-50: weight_i = max(score_cal_i, 0) / sum(top50.max(score_cal, 0)).
    Returns realized mean_y across days within `window`.
    """
    df = pred_df.join(target_y, on=["trade_date", "ts_code"], how="inner").filter(
        (pl.col("trade_date") >= window[0]) & (pl.col("trade_date") <= window[1])
    )
    score_col = "score_calibrated" if "score_calibrated" in df.columns else "score"

    top50 = (
        df.sort(["trade_date", score_col], descending=[False, True])
        .group_by("trade_date", maintain_order=True)
        .head(50)
    )
    # Score-weighted: clip negative scores to 0
    top50 = top50.with_columns(
        pl.col(score_col).clip(lower_bound=0.0).alias("_w")
    ).with_columns(
        pl.col("_w").sum().over("trade_date").alias("_wsum")
    )
    # Per-day weighted return
    top50 = top50.with_columns(
        (pl.col("_w") / pl.col("_wsum") * pl.col("y")).alias("_wy")
    )
    daily = top50.group_by("trade_date").agg(pl.col("_wy").sum().alias("daily_wy"))
    return {
        "n_days": len(daily),
        "mean_top50_strategyD": float(daily["daily_wy"].mean()),
        "n_input_days": df["trade_date"].n_unique(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    target_y = pl.read_parquet(args.bundle / "target_y.parquet")
    pred = pl.read_parquet(args.predictions)
    if "score_calibrated" not in pred.columns and "score_raw" in pred.columns:
        # Already calibrated upstream — fall back to score_raw
        pred = pred.with_columns(pl.col("score_raw").alias("score_calibrated"))

    h1 = evaluate_strategy_d(pred, target_y, H1)
    h2 = evaluate_strategy_d(pred, target_y, H2)

    print(f"=== Strategy D (score-weighted top50) — {args.label} ===")
    print(f"H1 ({H1[0]} ~ {H1[1]}): mean_y={h1['mean_top50_strategyD']:+.6f} over {h1['n_days']} days")
    print(f"H2 ({H2[0]} ~ {H2[1]}): mean_y={h2['mean_top50_strategyD']:+.6f} over {h2['n_days']} days")

    out = args.predictions.parent / "strategy_d_eval.json"
    out.write_text(json.dumps({"label": args.label, "H1": h1, "H2": h2}, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
