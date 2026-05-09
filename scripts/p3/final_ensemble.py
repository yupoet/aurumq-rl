"""Final cross-path ensemble: combine top-N runs from Path 1 + Path 4 + Path 2.

Strategy:
  1. Pick top-N runs from each path's runs/ directory by VAL_EFF primary.
  2. Per path: seed-mean within picked configs (already what path1_ensemble does).
  3. Cross-path: rank-rescale each path's per-day score (so absolute scales of
     LightGBM regression vs CatBoost vs XGBoost don't dominate), then mean.
  4. Isotonic calibrate on H1 actual_y → eval H2.
  5. Compare against Path 1 alone, Path 4 alone, paris baseline.

Outputs:
  - runs/sl_final/predictions.parquet  (trade_date, ts_code, score_raw, score_calibrated)
  - runs/sl_final/ensemble.json        (full metric blocks)
  - runs/sl_final/RESULTS.md           (headline + scoreboard)
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

# Self-contained import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p3.path1_eval import H1, H2, evaluate
from p3.path1_ensemble import calibrate_isotonic, pick_top_configs_by_val


logger = logging.getLogger(__name__)


def _load_path_runs(runs_dir: Path) -> dict[str, tuple[float, Path]]:
    """Return {run_name: (val_primary, predictions.parquet path)}."""
    out = {}
    for run_dir in sorted(runs_dir.glob("*/")):
        results = run_dir / "results.json"
        pred = run_dir / "predictions.parquet"
        if not results.exists() or not pred.exists():
            continue
        d = json.loads(results.read_text())
        if "VAL_EFF" not in d:
            continue
        out[run_dir.name] = (
            float(d["VAL_EFF"]["primary_mean_top50_proximity_excess"]),
            pred,
        )
    return out


def _seed_mean(pred_paths: list[Path]) -> pl.DataFrame:
    """Inner-join + mean(score) across multiple run prediction parquets."""
    if not pred_paths:
        raise ValueError("no prediction paths")
    base = pl.read_parquet(pred_paths[0]).select(["trade_date", "ts_code", "score"]).rename({"score": "score_0"})
    for i, p in enumerate(pred_paths[1:], start=1):
        df = pl.read_parquet(p).select(["trade_date", "ts_code", "score"]).rename({"score": f"score_{i}"})
        base = base.join(df, on=["trade_date", "ts_code"], how="inner")
    cols = [f"score_{i}" for i in range(len(pred_paths))]
    return base.with_columns(pl.mean_horizontal([pl.col(c) for c in cols]).alias("score")).select(
        ["trade_date", "ts_code", "score"]
    )


def _per_day_rank01(df: pl.DataFrame, score_col: str = "score") -> pl.DataFrame:
    """Per (trade_date), replace score with rank/N ∈ (0, 1]. Cross-path-comparable."""
    n = pl.col("trade_date").count().over("trade_date")
    rank = pl.col(score_col).rank(method="ordinal").over("trade_date").cast(pl.Float32)
    return df.with_columns((rank / n.cast(pl.Float32)).alias(score_col))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--paths", nargs="+", default=("sl_path1", "sl_path4", "sl_path2"),
                    help="Subdirectories of runs/ to harvest")
    ap.add_argument("--top-k-configs-per-path", type=int, default=3)
    ap.add_argument("--out", default=Path("runs/sl_final"), type=Path)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Per path: pick top-K configs (collapsing seeds), seed-mean predictions
    path_dfs: dict[str, pl.DataFrame] = {}
    path_chosen: dict[str, list[str]] = {}
    for p in args.paths:
        runs_dir = Path("runs") / p
        if not runs_dir.exists():
            logger.warning("path %s missing, skipping", p)
            continue
        runs = _load_path_runs(runs_dir)
        if len(runs) < args.top_k_configs_per_path:
            logger.warning("path %s has only %d runs (< top-k=%d); using all",
                           p, len(runs), args.top_k_configs_per_path)
        if not runs:
            continue
        val_scores = {n: v[0] for n, v in runs.items()}
        chosen = pick_top_configs_by_val(val_scores, top_k=args.top_k_configs_per_path)
        chosen = [c for c in chosen if c in runs]  # safety filter
        if not chosen:
            chosen = list(runs.keys())[: args.top_k_configs_per_path * 3]
        path_chosen[p] = chosen
        pred_paths = [runs[c][1] for c in chosen]
        path_dfs[p] = _seed_mean(pred_paths)
        logger.info("[%s] picked %d runs (across %d configs); ensemble %d rows",
                    p, len(chosen), len({c.rsplit('_seed', 1)[0] for c in chosen}),
                    len(path_dfs[p]))

    if not path_dfs:
        logger.error("no paths produced predictions")
        return 2

    # 2. Cross-path rank-mean
    # Per-day rank-rescale each path's score, then average across paths.
    ranked_dfs = {p: _per_day_rank01(df) for p, df in path_dfs.items()}
    base = list(ranked_dfs.values())[0].select(["trade_date", "ts_code", "score"]).rename({"score": "score_0"})
    for i, (p, df) in enumerate(list(ranked_dfs.items())[1:], start=1):
        base = base.join(
            df.select(["trade_date", "ts_code", "score"]).rename({"score": f"score_{i}"}),
            on=["trade_date", "ts_code"], how="inner",
        )
    score_cols = [f"score_{i}" for i in range(len(ranked_dfs))]
    ens = base.with_columns(pl.mean_horizontal([pl.col(c) for c in score_cols]).alias("score")).select(
        ["trade_date", "ts_code", "score"]
    )
    logger.info("cross-path ensemble: %d rows over %d paths", len(ens), len(path_dfs))

    # 3. Calibrate on H1
    target_y = pl.read_parquet(args.bundle / "target_y.parquet")
    realized = pl.read_parquet(args.bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    )
    market = pl.read_parquet(args.bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    )
    h1_join = ens.filter(
        (pl.col("trade_date") >= H1[0]) & (pl.col("trade_date") <= H1[1])
    ).join(target_y, on=["trade_date", "ts_code"], how="inner")
    calibrator = calibrate_isotonic(h1_join["score"].to_numpy(), h1_join["y"].to_numpy())
    score_calibrated = calibrator(ens["score"].to_numpy())
    ens_cal = ens.with_columns(pl.Series("score_calibrated", score_calibrated))

    # 4. Eval raw + calibrated
    eval_raw_h1 = evaluate(ens, target_y, realized, market, H1)
    eval_raw_h2 = evaluate(ens, target_y, realized, market, H2)
    eval_cal_h1 = evaluate(
        ens_cal.select(["trade_date", "ts_code", pl.col("score_calibrated").alias("score")]),
        target_y, realized, market, H1,
    )
    eval_cal_h2 = evaluate(
        ens_cal.select(["trade_date", "ts_code", pl.col("score_calibrated").alias("score")]),
        target_y, realized, market, H2,
    )

    # 5. Per-path baseline (alone, no cross-path mix) for comparison
    per_path_eval = {}
    for p, df in path_dfs.items():
        eh1 = evaluate(df, target_y, realized, market, H1)
        eh2 = evaluate(df, target_y, realized, market, H2)
        per_path_eval[p] = {"H1": eh1, "H2": eh2}

    # 6. Paris baseline
    bp = pl.read_parquet(args.bundle / "baseline_predictions.parquet")
    bp_pred = bp.select(["trade_date", "ts_code", pl.col("p_t3_baseline").alias("score")])
    bp_h1 = evaluate(bp_pred, target_y, realized, market, H1)
    bp_h2 = evaluate(bp_pred, target_y, realized, market, H2)

    summary = {
        "paths_used": list(path_dfs.keys()),
        "chosen_per_path": path_chosen,
        "n_calibration_rows_H1": len(h1_join),
        "final_ensemble_raw_H1": eval_raw_h1,
        "final_ensemble_raw_H2": eval_raw_h2,
        "final_ensemble_calibrated_H1": eval_cal_h1,
        "final_ensemble_calibrated_H2": eval_cal_h2,
        "per_path_ensemble": per_path_eval,
        "paris_baseline_H1": bp_h1,
        "paris_baseline_H2": bp_h2,
    }
    (args.out / "ensemble.json").write_text(json.dumps(summary, indent=2, default=str))
    ens_out = ens_cal.rename({"score": "score_raw"}).select(
        ["trade_date", "ts_code", "score_raw", "score_calibrated"]
    )
    ens_out.write_parquet(args.out / "predictions.parquet", compression="zstd", compression_level=10)

    # 7. Print scoreboard
    logger.info("== Final scoreboard ==")
    rows = [("paris_baseline", bp_h1, bp_h2)]
    for p, e in per_path_eval.items():
        rows.append((p, e["H1"], e["H2"]))
    rows.append(("FINAL_raw", eval_raw_h1, eval_raw_h2))
    rows.append(("FINAL_cal", eval_cal_h1, eval_cal_h2))
    logger.info("%-20s | H1 primary | H1 spear | H1 T1_hit | H2 primary | H2 spear | H2 T1_hit",
                "name")
    for name, h1, h2 in rows:
        logger.info(
            "%-20s | %+.6f | %+.4f | %.2f%% | %+.6f | %+.4f | %.2f%%",
            name,
            h1["primary_mean_top50_proximity_excess"], h1["spearman"],
            h1["top50_T1_hit_rate"] * 100,
            h2["primary_mean_top50_proximity_excess"], h2["spearman"],
            h2["top50_T1_hit_rate"] * 100,
        )
    logger.info("wrote %s and %s", args.out / "ensemble.json", args.out / "predictions.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
