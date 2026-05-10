"""Path 1 — train ONE LightGBM regression config on proximity-weighted target.

Usage::

    python scripts/p3/path1_train.py \
        --bundle data/p3_4070 \
        --out runs/sl_path1/numleaves63_lr05_minleaf100_seed42 \
        --num-leaves 63 --learning-rate 0.05 --min-data-in-leaf 100 --seed 42
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

# Ensure scripts/ is on sys.path so `from p3.<module>` resolves when invoked
# as `python scripts/p3/path1_train.py` (without PYTHONPATH set).
sys.path.insert(0, str(Path(__file__).parent.parent))

from p3.path1_eval import H1, H2, evaluate


logger = logging.getLogger(__name__)


TRAIN_EFF = (dt.date(2023, 1, 3),  dt.date(2024, 12, 4))
VAL_EFF   = (dt.date(2025, 1, 1),  dt.date(2025, 6, 4))


FEATURE_PANEL_FNAME = "feature_panel_v3_344.parquet"


def _load_features_universe(bundle: Path, feature_panel: str = FEATURE_PANEL_FNAME) -> tuple[pl.DataFrame, list[str]]:
    df = pl.read_parquet(bundle / feature_panel)
    feature_cols = [c for c in df.columns if c not in ("ts_code", "trade_date")]
    uni_parts = []
    for year in (2023, 2024, 2025, 2026):
        p = bundle / "universe_mask" / f"year={year}.parquet"
        if p.exists():
            uni_parts.append(pl.read_parquet(p).select(["trade_date", "ts_code", "in_universe"]))
    uni = pl.concat(uni_parts)
    df = df.join(uni, on=["trade_date", "ts_code"], how="left").filter(
        pl.col("in_universe") == True  # noqa: E712
    )
    return df, feature_cols


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--feature-panel", default=FEATURE_PANEL_FNAME,
                    help="Filename within --bundle for the feature parquet")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--num-iterations", type=int, default=2000)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    # Optional regularization (used by path6 Bayesian opt)
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--lambda-l1", type=float, default=0.0)
    ap.add_argument("--lambda-l2", type=float, default=0.0)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Load features (in-universe only) and target_y
    t0 = time.time()
    feat_df, feature_cols = _load_features_universe(args.bundle, args.feature_panel)
    logger.info("features: %d rows × %d cols (%.1fs)", len(feat_df), len(feature_cols), time.time() - t0)
    target_y = pl.read_parquet(args.bundle / "target_y.parquet")
    logger.info("target_y: %d rows", len(target_y))

    df = feat_df.join(target_y, on=["trade_date", "ts_code"], how="inner")
    logger.info("joined: %d rows", len(df))

    # 2. Split by trade_date
    train_df = df.filter(
        (pl.col("trade_date") >= TRAIN_EFF[0]) & (pl.col("trade_date") <= TRAIN_EFF[1])
    )
    val_df = df.filter(
        (pl.col("trade_date") >= VAL_EFF[0]) & (pl.col("trade_date") <= VAL_EFF[1])
    )
    logger.info("splits: train=%d val=%d", len(train_df), len(val_df))

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df["y"].to_numpy().astype(np.float32)
    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df["y"].to_numpy().astype(np.float32)

    # 3. Train LightGBM regression
    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_ds = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_ds)
    params = {
        "objective": "regression_l2",
        "metric": ["l2", "l1"],
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": 5,
        "min_data_in_leaf": args.min_data_in_leaf,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "verbosity": -1,
        "seed": args.seed,
        "n_jobs": -1,
    }
    logger.info("training: %s", {k: v for k, v in params.items() if k != "metric"})
    t1 = time.time()
    model = lgb.train(
        params, train_ds,
        num_boost_round=args.num_iterations,
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    train_time = time.time() - t1
    logger.info("trained in %.1fs (best_iter=%d)", train_time, model.best_iteration)

    # 4. Predict on full eval frame
    X_all = df.select(feature_cols).to_numpy()
    score_all = model.predict(X_all, num_iteration=model.best_iteration).astype(np.float32)
    pred_df = df.select(["trade_date", "ts_code"]).with_columns(pl.Series("score", score_all))

    # 5. Save artifacts
    pred_df.write_parquet(args.out / "predictions.parquet", compression="zstd", compression_level=9)
    model.save_model(str(args.out / "lgb_model.txt"))
    np.savez(args.out / "predictions.npz", score=score_all)

    # 6. Eval on VAL_EFF, H1, H2
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
        "params": params,
        "best_iteration": model.best_iteration,
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
    logger.info("done. results: %s", args.out / "results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
