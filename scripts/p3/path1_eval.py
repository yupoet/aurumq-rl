"""Path 1 eval metrics (spec §3).

All metrics here are pure functions over polars DataFrames or numpy arrays.
The CLI at the bottom is a thin wrapper that joins (predictions, target_y,
realized_returns) and prints the metric block.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Pure metric functions
# ------------------------------------------------------------------ #

def compute_mean_top50_proximity_excess(df: pl.DataFrame, top_k: int = 50) -> float:
    """Mean across days of the daily mean-actual-y over top-k by score.

    df: (trade_date, ts_code, score, actual_y). NaN actual_y rows excluded.
    """
    df = df.drop_nulls(["score", "actual_y"])
    if len(df) == 0:
        return 0.0
    daily_means = (
        df.sort(["trade_date", "score"], descending=[False, True])
          .group_by("trade_date", maintain_order=True)
          .head(top_k)
          .group_by("trade_date", maintain_order=True)
          .agg(pl.col("actual_y").mean().alias("topk_mean"))
    )
    return float(daily_means["topk_mean"].mean())


def compute_spearman(df: pl.DataFrame) -> float:
    """Spearman rank correlation between score and actual_y across all rows."""
    df = df.drop_nulls(["score", "actual_y"])
    if len(df) < 2:
        return 0.0
    rho, _ = spearmanr(df["score"].to_numpy(), df["actual_y"].to_numpy())
    if not np.isfinite(rho):
        return 0.0
    return float(rho)


def compute_top50_hit_rates(df: pl.DataFrame, top_k: int = 50) -> dict[str, float]:
    """Top-k hit decomposition (spec §3.3).

    df must have columns: trade_date, ts_code, score, e1 (T+1 excess), e2, e3.
    """
    df = df.drop_nulls(["score", "e1", "e2", "e3"])
    if len(df) == 0:
        return {"top50_T1_hit_rate": 0.0, "top50_T13_hit_rate": 0.0, "top50_T1_avg_excess": 0.0}
    topk = (
        df.sort(["trade_date", "score"], descending=[False, True])
          .group_by("trade_date", maintain_order=True)
          .head(top_k)
    )
    return {
        "top50_T1_hit_rate": float((topk["e1"] > 0).mean()),
        "top50_T13_hit_rate": float(((topk["e1"] > 0) | (topk["e2"] > 0) | (topk["e3"] > 0)).mean()),
        "top50_T1_avg_excess": float(topk["e1"].mean()),
    }


def compute_ece_10bin(pred: np.ndarray, actual: np.ndarray) -> float:
    """Expected Calibration Error binned to 10 quantiles of predictions.

    For each prediction-quantile bin, compare bin's mean prediction vs bin's
    mean actual. Weighted by bin size. Standard ECE definition.
    """
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[mask], actual[mask]
    if len(pred) < 10:
        return 0.0

    quantiles = np.quantile(pred, np.linspace(0, 1, 11))
    quantiles[0] -= 1e-9
    quantiles[-1] += 1e-9
    bin_idx = np.digitize(pred, quantiles[1:-1], right=False)
    n = len(pred)
    ece = 0.0
    for b in range(10):
        in_bin = bin_idx == b
        if not in_bin.any():
            continue
        ece += (in_bin.sum() / n) * abs(pred[in_bin].mean() - actual[in_bin].mean())
    return float(ece)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

H1 = (dt.date(2025, 7, 1), dt.date(2025, 9, 30))
H2 = (dt.date(2025, 10, 1), dt.date(2025, 12, 31))


def evaluate(
    predictions: pl.DataFrame,
    target_y: pl.DataFrame,
    realized: pl.DataFrame,
    market: pl.DataFrame,
    window: tuple[dt.date, dt.date],
    top_k: int = 50,
) -> dict:
    """Compute the §3 metric block on a (predictions, target_y) join over `window`.

    Joins predictions ⨝ target_y on (trade_date, ts_code) inner; restricts
    to ``window``. Computes T+1/T+2/T+3 excess returns inline for hit-rate.
    """
    lo, hi = window
    pred_w = predictions.filter((pl.col("trade_date") >= lo) & (pl.col("trade_date") <= hi))
    y_w = target_y.filter((pl.col("trade_date") >= lo) & (pl.col("trade_date") <= hi))
    df = pred_w.join(y_w, on=["trade_date", "ts_code"], how="inner")
    df = df.rename({"y": "actual_y"})

    all_dates = realized.select("trade_date").unique().sort("trade_date").with_row_index("date_idx")
    realized_w_idx = realized.join(all_dates, on="trade_date")
    market_w_idx = market.join(all_dates, on="trade_date")

    def _shift(rdf: pl.DataFrame, k: int, alias: str) -> pl.DataFrame:
        return rdf.select(
            (pl.col("date_idx") - k).alias("anchor_idx"),
            pl.col("ts_code"),
            pl.col("pct_chg_t_plus_1").alias(alias),
        )

    def _shift_market(mdf: pl.DataFrame, k: int, alias: str) -> pl.DataFrame:
        return mdf.select(
            (pl.col("date_idx") - k).alias("anchor_idx"),
            pl.col("eq_weight_pct_chg_t_plus_1").alias(alias),
        )

    df = df.join(all_dates, on="trade_date").rename({"date_idx": "anchor_idx"})
    df = df.join(_shift(realized_w_idx, 0, "pct_t_plus_1"), on=["anchor_idx", "ts_code"], how="left")
    df = df.join(_shift(realized_w_idx, 1, "pct_t_plus_2"), on=["anchor_idx", "ts_code"], how="left")
    df = df.join(_shift(realized_w_idx, 2, "pct_t_plus_3"), on=["anchor_idx", "ts_code"], how="left")
    df = df.join(_shift_market(market_w_idx, 0, "market_t_plus_1"), on="anchor_idx", how="left")
    df = df.join(_shift_market(market_w_idx, 1, "market_t_plus_2"), on="anchor_idx", how="left")
    df = df.join(_shift_market(market_w_idx, 2, "market_t_plus_3"), on="anchor_idx", how="left")
    df = df.with_columns(
        (pl.col("pct_t_plus_1") - pl.col("market_t_plus_1")).alias("e1"),
        (pl.col("pct_t_plus_2") - pl.col("market_t_plus_2")).alias("e2"),
        (pl.col("pct_t_plus_3") - pl.col("market_t_plus_3")).alias("e3"),
    )

    df = df.drop_nulls(["e1", "e2", "e3", "actual_y", "score"])

    primary = compute_mean_top50_proximity_excess(df, top_k=top_k)
    spearman = compute_spearman(df)
    hit = compute_top50_hit_rates(df, top_k=top_k)
    ece = compute_ece_10bin(df["score"].to_numpy(), df["actual_y"].to_numpy())

    return {
        "n_rows": len(df),
        "n_dates": df["trade_date"].n_unique(),
        "primary_mean_top50_proximity_excess": primary,
        "spearman": spearman,
        "ece_10bin": ece,
        **hit,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: print eval metrics for a predictions parquet on H1 and H2."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path,
                    help="Parquet with (trade_date, ts_code, score) columns")
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional JSON output of the metric blocks")
    args = ap.parse_args(argv)

    predictions = pl.read_parquet(args.predictions)
    target_y = pl.read_parquet(args.bundle / "target_y.parquet")
    realized = pl.read_parquet(args.bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    )
    market = pl.read_parquet(args.bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    )

    h1 = evaluate(predictions, target_y, realized, market, H1, args.top_k)
    h2 = evaluate(predictions, target_y, realized, market, H2, args.top_k)

    out = {"H1": h1, "H2": h2}
    print(json.dumps(out, indent=2, default=str))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, default=str))
        logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
