"""Path 2 — train ONE XGBoost regression config on proximity-weighted target.

Mirrors path1_train.py and path2_train_catboost.py for orchestrator
compatibility. Saves xgb_model.json (XGBoost native JSON format).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

# Self-contained import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p3.path1_eval import H1, H2, evaluate
from p3.path1_train import FEATURE_PANEL_FNAME, _load_features_universe, TRAIN_EFF, VAL_EFF


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--feature-panel", default=FEATURE_PANEL_FNAME)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-iterations", type=int, default=2000)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--train-start", default=None, help="ISO date; overrides TRAIN_EFF[0]")
    ap.add_argument("--train-end", default=None, help="ISO date; overrides TRAIN_EFF[1]")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    import datetime as _dt
    train_lo, train_hi = TRAIN_EFF
    if args.train_start: train_lo = _dt.date.fromisoformat(args.train_start)
    if args.train_end:   train_hi = _dt.date.fromisoformat(args.train_end)

    # 1. Load features + target. Auto-detect pre-joined panel.
    t0 = time.time()
    panel_path = args.bundle / args.feature_panel
    schema = pl.read_parquet_schema(panel_path)
    if "y" in schema:
        df = pl.read_parquet(panel_path)
        feature_cols = [c for c in df.columns if c not in ("ts_code", "trade_date", "y")]
        target_y = df.select(["trade_date", "ts_code", "y"])
        logger.info("features (pre-joined): %d rows × %d cols (%.1fs)",
                    len(df), len(feature_cols), time.time() - t0)
    else:
        feat_df, feature_cols = _load_features_universe(args.bundle, args.feature_panel)
        logger.info("features: %d rows × %d cols (%.1fs)", len(feat_df), len(feature_cols), time.time() - t0)
        target_y = pl.read_parquet(args.bundle / "target_y.parquet")
        df = feat_df.join(target_y, on=["trade_date", "ts_code"], how="inner")
        logger.info("joined: %d rows", len(df))

    train_df = df.filter((pl.col("trade_date") >= train_lo) & (pl.col("trade_date") <= train_hi))
    val_df = df.filter((pl.col("trade_date") >= VAL_EFF[0]) & (pl.col("trade_date") <= VAL_EFF[1]))
    logger.info("TRAIN window: [%s, %s]", train_lo, train_hi)
    logger.info("splits: train=%d val=%d", len(train_df), len(val_df))

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df["y"].to_numpy().astype(np.float32)
    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df["y"].to_numpy().astype(np.float32)

    # 2. Train XGBoost
    params = {
        "objective": "reg:squarederror",
        "n_estimators": args.num_iterations,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "reg_lambda": args.reg_lambda,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
        "early_stopping_rounds": args.early_stopping_rounds,
        "verbosity": 1,
        "tree_method": "hist",
    }
    logger.info("training xgboost: max_depth=%d lr=%.3f iters=%d",
                args.max_depth, args.learning_rate, args.num_iterations)
    t1 = time.time()
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)
    train_time = time.time() - t1
    best_iter = model.best_iteration if model.best_iteration is not None else args.num_iterations
    logger.info("trained in %.1fs (best_iter=%d)", train_time, best_iter)

    # 3. Predict on full eval frame
    X_all = df.select(feature_cols).to_numpy()
    score_all = model.predict(X_all).astype(np.float32)
    pred_df = df.select(["trade_date", "ts_code"]).with_columns(pl.Series("score", score_all))

    # 4. Save artifacts
    pred_df.write_parquet(args.out / "predictions.parquet", compression="zstd", compression_level=9)
    model.save_model(str(args.out / "xgb_model.json"))
    np.savez(args.out / "predictions.npz", score=score_all)

    # 5. Eval
    realized = pl.read_parquet(args.bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    )
    market = pl.read_parquet(args.bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    )
    val_eval = evaluate(pred_df, target_y, realized, market, VAL_EFF)
    h1_eval = evaluate(pred_df, target_y, realized, market, H1)
    h2_eval = evaluate(pred_df, target_y, realized, market, H2)

    summary = {
        "model_class": "xgboost",
        "params": params,
        "best_iteration": best_iter,
        "train_time_s": train_time,
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "VAL_EFF": val_eval,
        "H1": h1_eval,
        "H2": h2_eval,
    }
    (args.out / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("VAL primary=%.6f  H1 primary=%.6f  H2 primary=%.6f",
                val_eval["primary_mean_top50_proximity_excess"],
                h1_eval["primary_mean_top50_proximity_excess"],
                h2_eval["primary_mean_top50_proximity_excess"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
