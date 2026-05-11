"""SL baseline: train fresh LightGBM on the P3 bundle, compare to P2 v2.

Goal: establish whether SL on the same 344 features × main-wave labels can
match or beat paris's published P2 v2 ensemble (PR_AUC 0.122 lift 3.0×) on
the H1 window. If yes, this validates the pipeline and gives us a starting
point to iterate (more seeds, ensembles, model-class diversity).

Usage::

    .venv/Scripts/python.exe scripts/p3/sl_baseline.py \
        --bundle data/p3_4070 \
        --label-name labels_A_t3 \
        --out runs/sl_baseline_A_t3

Reads the cached bundle (data/p3_4070/_bundle_cache.npz, 11s load) so this
runs concurrently with PPO on the GPU without extra disk I/O.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


logger = logging.getLogger(__name__)


TRAIN_EFF = (date(2023, 1, 3),  date(2024, 12, 4))
VAL_EFF   = (date(2025, 1, 1),  date(2025, 6, 4))
H1        = (date(2025, 7, 1),  date(2025, 9, 30))
H2        = (date(2025, 10, 1), date(2025, 12, 31))


def load_features(bundle_dir: Path) -> tuple[pl.DataFrame, list[str]]:
    """Pull (trade_date, ts_code, *features) into one polars frame.

    Avoids the (T, S, F) numpy panel — for SL we want a long-format frame
    indexed by (date, stock) so we can join labels and universe mask.
    """
    feat_path = bundle_dir / "feature_panel_v3_344.parquet"
    df = pl.read_parquet(feat_path)
    feature_cols = [c for c in df.columns if c not in ("ts_code", "trade_date")]
    return df, feature_cols


def load_labels(bundle_dir: Path, label_name: str) -> pl.DataFrame:
    """Concat per-year label parquets into one (trade_date, ts_code, y) frame."""
    parts = []
    for year in (2023, 2024, 2025, 2026):
        p = bundle_dir / "labels" / f"{label_name}_year={year}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        parts.append(pl.read_parquet(p))
    return pl.concat(parts)


def load_universe(bundle_dir: Path) -> pl.DataFrame:
    """Concat per-year universe_mask into one (trade_date, ts_code, in_universe) frame."""
    parts = []
    for year in (2023, 2024, 2025, 2026):
        p = bundle_dir / "universe_mask" / f"year={year}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        parts.append(pl.read_parquet(p).select(["trade_date", "ts_code", "in_universe"]))
    return pl.concat(parts)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, top_k_pcts: list[float] = [0.01, 0.05, 0.10]):
    """Compute PR-AUC, ROC-AUC, top-K precision lift."""
    pr_auc = average_precision_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    base_rate = y_true.mean()
    out = {"pr_auc": float(pr_auc), "roc_auc": float(roc_auc), "base_rate": float(base_rate)}
    for pct in top_k_pcts:
        k = max(int(len(y_true) * pct), 1)
        top_idx = np.argsort(-y_pred)[:k]
        precision = y_true[top_idx].mean()
        out[f"top{int(pct*100)}pct_precision"] = float(precision)
        out[f"top{int(pct*100)}pct_lift"] = float(precision / base_rate) if base_rate > 0 else 0.0
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--label-name", default="labels_A_t3",
                    help="One of labels_{A,B,C,D}_{t1,t3,e20}")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--num-iterations", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Load + join
    t0 = time.time()
    logger.info("loading bundle pieces ...")
    feat_df, feature_cols = load_features(args.bundle)
    logger.info("  features: %d rows × %d cols (%.1f MB)",
                len(feat_df), len(feature_cols), feat_df.estimated_size() / 1e6)
    labels_df = load_labels(args.bundle, args.label_name)
    logger.info("  labels  : %d rows", len(labels_df))
    uni_df = load_universe(args.bundle)
    logger.info("  universe: %d rows", len(uni_df))

    # Inner-join on (trade_date, ts_code) — drops cells without label
    df = feat_df.join(labels_df, on=["trade_date", "ts_code"], how="inner")
    df = df.join(uni_df, on=["trade_date", "ts_code"], how="left")
    df = df.filter(pl.col("in_universe") == True)  # noqa: E712
    logger.info("  joined : %d rows (filtered to in-universe)", len(df))
    logger.info("  load+join: %.1fs", time.time() - t0)

    # 2. Split by trade_date
    train_mask = (df["trade_date"] >= TRAIN_EFF[0]) & (df["trade_date"] <= TRAIN_EFF[1])
    val_mask = (df["trade_date"] >= VAL_EFF[0]) & (df["trade_date"] <= VAL_EFF[1])
    h1_mask = (df["trade_date"] >= H1[0]) & (df["trade_date"] <= H1[1])
    h2_mask = (df["trade_date"] >= H2[0]) & (df["trade_date"] <= H2[1])

    train_df = df.filter(train_mask)
    val_df = df.filter(val_mask)
    h1_df = df.filter(h1_mask)
    h2_df = df.filter(h2_mask)
    logger.info("  splits : train=%d val=%d H1=%d H2=%d",
                len(train_df), len(val_df), len(h1_df), len(h2_df))

    if "y" not in df.columns:
        raise SystemExit(f"Label '{args.label_name}' didn't expose 'y' col. cols: {df.columns}")

    # NaN handling: lightgbm handles NaN natively, just need numpy arrays
    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df["y"].to_numpy().astype(np.int8)
    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df["y"].to_numpy().astype(np.int8)
    X_h1 = h1_df.select(feature_cols).to_numpy()
    y_h1 = h1_df["y"].to_numpy().astype(np.int8)
    X_h2 = h2_df.select(feature_cols).to_numpy()
    y_h2 = h2_df["y"].to_numpy().astype(np.int8)

    pos_train = int(y_train.sum())
    logger.info("  positive: train=%d (%.4f) val=%d (%.4f) H1=%d H2=%d",
                pos_train, pos_train/len(y_train),
                int(y_val.sum()), y_val.mean(),
                int(y_h1.sum()), int(y_h2.sum()))

    # 3. Train LightGBM
    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_ds = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_ds)
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "average_precision"],
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 100,
        "verbosity": -1,
        "seed": args.seed,
        "n_jobs": -1,
    }
    logger.info("training lightgbm: num_iter=%d num_leaves=%d lr=%.3f n_jobs=-1",
                args.num_iterations, args.num_leaves, args.learning_rate)
    t1 = time.time()
    model = lgb.train(
        params, train_ds,
        num_boost_round=args.num_iterations,
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    logger.info("trained in %.1fs (best iter=%d)", time.time() - t1, model.best_iteration)

    # 4. Eval on val + H1 + H2
    p_val = model.predict(X_val, num_iteration=model.best_iteration)
    p_h1 = model.predict(X_h1, num_iteration=model.best_iteration)
    p_h2 = model.predict(X_h2, num_iteration=model.best_iteration)

    def fmt(eval_d):
        return (f"PR_AUC={eval_d['pr_auc']:.4f}  ROC={eval_d['roc_auc']:.4f}  "
                f"top1%lift={eval_d['top1pct_lift']:.2f}× top5%lift={eval_d['top5pct_lift']:.2f}× "
                f"top10%lift={eval_d['top10pct_lift']:.2f}× base={eval_d['base_rate']:.4f}")

    val_eval = evaluate(y_val, p_val)
    logger.info("VAL_EFF: %s", fmt(val_eval))
    h1_eval = evaluate(y_h1, p_h1)
    logger.info("H1     : %s", fmt(h1_eval))
    h2_eval = evaluate(y_h2, p_h2)
    logger.info("H2     : %s", fmt(h2_eval))

    # 5. Save
    out = {
        "label_name": args.label_name,
        "params": params,
        "best_iteration": model.best_iteration,
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "val_eval": val_eval,
        "h1_eval": h1_eval,
        "h2_eval": h2_eval,
    }
    (args.out / "results.json").write_text(json.dumps(out, indent=2, default=str))
    model.save_model(str(args.out / "lgb_model.txt"))
    np.savez(args.out / "predictions.npz",
             p_val=p_val.astype(np.float32),
             p_h1=p_h1.astype(np.float32),
             p_h2=p_h2.astype(np.float32))

    # 6. Compare to paris's baseline_predictions.parquet on H1
    bp = pl.read_parquet(args.bundle / "baseline_predictions.parquet")
    bp_h1 = bp.filter(
        (pl.col("trade_date") >= H1[0]) & (pl.col("trade_date") <= H1[1])
    ).join(h1_df.select(["trade_date", "ts_code", "y"]), on=["trade_date", "ts_code"], how="inner")
    if len(bp_h1) > 0:
        baseline_eval = evaluate(
            bp_h1["y"].to_numpy().astype(np.int8),
            bp_h1["p_t3_baseline"].to_numpy(),
        )
        logger.info("Baseline (P2 v2) on H1 (joined): %s", fmt(baseline_eval))
        out["baseline_h1_eval"] = baseline_eval
        (args.out / "results.json").write_text(json.dumps(out, indent=2, default=str))

    logger.info("done. results: %s", args.out / "results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
