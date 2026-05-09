"""Path 3 — train ONE TabNet config on proximity-weighted target (4070 GPU).

Tabular DL via pytorch-tabnet. Different inductive bias than GBDT:
attention-based feature selection per-step. May or may not add signal.

Mirrors path1_train.py for orchestrator compatibility. NaN values in
features must be imputed (TabNet doesn't tolerate NaN); we fill with 0.0
which under cross-sectional rank-z (Path 4) corresponds to the median
input — neutral.
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
import torch
from pytorch_tabnet.tab_model import TabNetRegressor

# Self-contained import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p3.path1_eval import H1, H2, evaluate
from p3.path1_train import FEATURE_PANEL_FNAME, _load_features_universe, TRAIN_EFF, VAL_EFF


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--feature-panel", default="feature_panel_clean.parquet",
                    help="Default to clean Path 4 panel — TabNet needs no-NaN input.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-d", type=int, default=32, help="TabNet feature dimension")
    ap.add_argument("--n-a", type=int, default=32, help="TabNet attention dimension")
    ap.add_argument("--n-steps", type=int, default=5, help="TabNet decision steps")
    ap.add_argument("--gamma", type=float, default=1.5, help="TabNet sparsity gamma")
    ap.add_argument("--learning-rate", type=float, default=2e-2)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8192)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load features (clean Path 4 panel) + target
    t0 = time.time()
    feat_df, feature_cols = _load_features_universe(args.bundle, args.feature_panel)
    logger.info("features: %d rows × %d cols (%.1fs)", len(feat_df), len(feature_cols), time.time() - t0)
    target_y = pl.read_parquet(args.bundle / "target_y.parquet")
    df = feat_df.join(target_y, on=["trade_date", "ts_code"], how="inner")
    logger.info("joined: %d rows", len(df))

    train_df = df.filter((pl.col("trade_date") >= TRAIN_EFF[0]) & (pl.col("trade_date") <= TRAIN_EFF[1]))
    val_df = df.filter((pl.col("trade_date") >= VAL_EFF[0]) & (pl.col("trade_date") <= VAL_EFF[1]))
    logger.info("splits: train=%d val=%d", len(train_df), len(val_df))

    # NaN → 0.0 (median-equivalent under rank-z scaling)
    X_train = train_df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y_train = train_df["y"].to_numpy().astype(np.float32).reshape(-1, 1)
    X_val = val_df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y_val = val_df["y"].to_numpy().astype(np.float32).reshape(-1, 1)

    # NaN check (any infs would also fail tabnet)
    if not np.isfinite(X_train).all():
        logger.error("X_train has non-finite values; tabnet cannot proceed")
        return 2

    # 2. Train TabNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("training tabnet on %s: n_d=%d n_a=%d n_steps=%d gamma=%.2f lr=%.4f bs=%d",
                device, args.n_d, args.n_a, args.n_steps, args.gamma,
                args.learning_rate, args.batch_size)
    t1 = time.time()
    model = TabNetRegressor(
        n_d=args.n_d,
        n_a=args.n_a,
        n_steps=args.n_steps,
        gamma=args.gamma,
        seed=args.seed,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params={"lr": args.learning_rate, "weight_decay": 1e-5},
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params={"T_max": args.max_epochs},
        verbose=1,
        device_name=device,
    )
    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=["val"],
        eval_metric=["rmse"],
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        virtual_batch_size=min(args.batch_size, 512),
        drop_last=False,
    )
    train_time = time.time() - t1
    logger.info("trained in %.1fs", train_time)

    # 3. Predict on full eval frame
    X_all = df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    score_all = model.predict(X_all).flatten().astype(np.float32)
    pred_df = df.select(["trade_date", "ts_code"]).with_columns(pl.Series("score", score_all))

    # 4. Save artifacts
    pred_df.write_parquet(args.out / "predictions.parquet", compression="zstd", compression_level=9)
    model.save_model(str(args.out / "tabnet_model"))  # creates tabnet_model.zip
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
        "model_class": "tabnet",
        "params": {
            "n_d": args.n_d, "n_a": args.n_a, "n_steps": args.n_steps,
            "gamma": args.gamma, "learning_rate": args.learning_rate,
            "max_epochs": args.max_epochs, "batch_size": args.batch_size,
            "device": device,
        },
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
