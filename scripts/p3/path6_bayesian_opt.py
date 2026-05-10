"""Path 6 — Bayesian hyperparameter optimization on pruned (post-SHAP) panel.

Two-stage:
  1. Drop the 119 features with mean(|SHAP|) < 1e-6 (per Path 4 audit) →
     panel goes from 345 to ~226 features. Saves as feature_panel_clean_pruned.parquet.
  2. Optuna 50-trial Bayesian opt on (num_leaves, lr, min_data_in_leaf,
     feature_fraction, bagging_fraction, lambda_l1, lambda_l2) using VAL_EFF
     primary as objective. Each trial = 1 LightGBM training (~50s).
  3. Pick top-3 trial configs, train each × 3 seeds = 9 final runs.
  4. Ensemble + eval on H1/H2.

Output: runs/sl_path6/ — same structure as sl_path4 (per-run results.json,
ensemble.json, predictions.parquet, RESULTS.md).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p3.path1_eval import H1, H2, evaluate
from p3.path1_train import _load_features_universe, TRAIN_EFF, VAL_EFF


logger = logging.getLogger(__name__)


def prune_panel(bundle: Path, drop_list: list[str], in_panel: str, out_panel: str) -> Path:
    """Read clean panel, drop the SHAP-zero features, write pruned panel."""
    in_path = bundle / in_panel
    out_path = bundle / out_panel
    if out_path.exists():
        logger.info("pruned panel already exists: %s", out_path)
        return out_path
    df = pl.read_parquet(in_path)
    keep = [c for c in df.columns if c not in drop_list]
    pruned = df.select(keep)
    pruned.write_parquet(out_path, compression="zstd", compression_level=9)
    logger.info("pruned %s: %d → %d cols (%.1f MB)",
                out_panel, len(df.columns), len(keep), out_path.stat().st_size / 1e6)
    return out_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--in-panel", default="feature_panel_clean.parquet")
    ap.add_argument("--out-panel", default="feature_panel_clean_pruned.parquet")
    ap.add_argument("--drop-candidates", default=Path("runs/sl_path4/feature_audit_drop_candidates.json"),
                    type=Path)
    ap.add_argument("--out-root", default=Path("runs/sl_path6"), type=Path)
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--top-k-configs", type=int, default=3)
    ap.add_argument("--final-seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--num-iterations", type=int, default=2000)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    args = ap.parse_args(argv)

    args.out_root.mkdir(parents=True, exist_ok=True)

    # 1. Build pruned panel
    drop_list = [d["feature"] for d in json.loads(args.drop_candidates.read_text())]
    logger.info("dropping %d SHAP-zero features", len(drop_list))
    pruned_panel_name = args.out_panel
    prune_panel(args.bundle, drop_list, args.in_panel, pruned_panel_name)

    # 2. Load training data ONCE for optuna trials (fast iteration)
    t0 = time.time()
    feat_df, feature_cols = _load_features_universe(args.bundle, pruned_panel_name)
    target_y = pl.read_parquet(args.bundle / "target_y.parquet")
    df = feat_df.join(target_y, on=["trade_date", "ts_code"], how="inner")

    train_df = df.filter((pl.col("trade_date") >= TRAIN_EFF[0]) & (pl.col("trade_date") <= TRAIN_EFF[1]))
    val_df = df.filter((pl.col("trade_date") >= VAL_EFF[0]) & (pl.col("trade_date") <= VAL_EFF[1]))

    X_train = train_df.select(feature_cols).to_numpy().astype(np.float32)
    y_train = train_df["y"].to_numpy().astype(np.float32)
    X_val = val_df.select(feature_cols).to_numpy().astype(np.float32)
    y_val = val_df["y"].to_numpy().astype(np.float32)
    logger.info("loaded data: train=%d val=%d feat=%d (%.1fs)",
                len(X_train), len(X_val), len(feature_cols), time.time() - t0)

    # 3. Cache realized + market for primary metric eval
    realized = pl.read_parquet(args.bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    )
    market = pl.read_parquet(args.bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    )

    val_keys = val_df.select(["trade_date", "ts_code"])

    # 4. Optuna objective: train LightGBM with sampled params, eval primary on VAL_EFF
    # feature_pre_filter=False so optuna can sample different min_data_in_leaf
    # across trials without LightGBM complaining about cached pre-filter.
    common_ds_params = {"feature_pre_filter": False}
    train_ds_master = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols,
                                   free_raw_data=False, params=common_ds_params)
    val_ds_master = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols,
                                 reference=train_ds_master, free_raw_data=False,
                                 params=common_ds_params)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression_l2",
            "metric": "l2",
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 5,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-6, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-6, 10.0, log=True),
            "verbosity": -1,
            "seed": 42,
            "n_jobs": -1,
        }
        model = lgb.train(
            params, train_ds_master,
            num_boost_round=args.num_iterations,
            valid_sets=[val_ds_master],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False)],
        )
        # Score = VAL_EFF primary metric
        pred = model.predict(X_val, num_iteration=model.best_iteration).astype(np.float32)
        pred_df = val_keys.with_columns(pl.Series("score", pred))
        eval_d = evaluate(pred_df, target_y, realized, market, VAL_EFF)
        primary = eval_d["primary_mean_top50_proximity_excess"]
        trial.set_user_attr("best_iteration", model.best_iteration)
        trial.set_user_attr("h1_primary", evaluate(pred_df, target_y, realized, market, H1)["primary_mean_top50_proximity_excess"])
        return primary

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="path6_bayesian",
    )
    logger.info("starting optuna: %d trials", args.n_trials)
    t1 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    logger.info("optuna done in %.0fs; best VAL primary=%.6f, params=%s",
                time.time() - t1, study.best_value, study.best_params)

    # 5. Save trial history
    trials_summary = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            trials_summary.append({
                "number": t.number,
                "params": t.params,
                "val_primary": t.value,
                "h1_primary": t.user_attrs.get("h1_primary"),
                "best_iteration": t.user_attrs.get("best_iteration"),
            })
    trials_summary.sort(key=lambda x: -x["val_primary"])
    (args.out_root / "trials.json").write_text(json.dumps(trials_summary, indent=2))
    logger.info("wrote %s with %d completed trials", args.out_root / "trials.json", len(trials_summary))

    # 6. Top-K configs by VAL primary, train each × seeds for ensemble
    top_configs = trials_summary[: args.top_k_configs]
    logger.info("top-%d configs by VAL primary:", args.top_k_configs)
    for i, c in enumerate(top_configs):
        logger.info("  rank %d: VAL=%.6f  H1=%.6f  params=%s",
                    i + 1, c["val_primary"], c["h1_primary"] or 0, c["params"])

    py = sys.executable
    train_script = Path(__file__).resolve().parent / "path1_train.py"

    n_ok = n_fail = 0
    for cfg_rank, cfg in enumerate(top_configs):
        for seed in args.final_seeds:
            run_name = f"opt{cfg_rank+1}_seed{seed}"
            out_dir = args.out_root / run_name
            if (out_dir / "results.json").exists():
                logger.info("[skip] %s", run_name)
                continue
            logger.info("[BEGIN] %s", run_name)
            cmd = [
                py, str(train_script),
                "--bundle", str(args.bundle),
                "--feature-panel", pruned_panel_name,
                "--out", str(out_dir),
                "--seed", str(seed),
                "--num-leaves", str(cfg["params"]["num_leaves"]),
                "--learning-rate", str(cfg["params"]["learning_rate"]),
                "--min-data-in-leaf", str(cfg["params"]["min_data_in_leaf"]),
                "--feature-fraction", str(cfg["params"]["feature_fraction"]),
                "--bagging-fraction", str(cfg["params"]["bagging_fraction"]),
                "--lambda-l1", str(cfg["params"]["lambda_l1"]),
                "--lambda-l2", str(cfg["params"]["lambda_l2"]),
                "--num-iterations", str(args.num_iterations),
                "--early-stopping-rounds", str(args.early_stopping_rounds),
            ]
            rc = subprocess.run(cmd, cwd=Path.cwd()).returncode
            if rc == 0:
                n_ok += 1
            else:
                n_fail += 1
                logger.error("FAIL %s rc=%d", run_name, rc)

    logger.info("final runs: ok=%d fail=%d", n_ok, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
