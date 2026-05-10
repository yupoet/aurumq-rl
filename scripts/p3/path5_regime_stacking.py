"""Path 5 — regime-aware stacking meta-learner.

Replaces the rank-mean cross-path ensemble with a learned blend that
conditions on regime features. The hypothesis: in high-vol regimes
(like H1/Q3-2025) the relative weight of base models should differ
from low-vol regimes — a learned meta-learner can adapt; rank-mean cannot.

Pipeline:
  1. For each base path (sl_path1, sl_path4, sl_path2): load ensemble predictions
  2. Join with target_y + regime_features
  3. Split: train_meta = VAL_EFF, eval = H1 + H2
  4. Train tiny LightGBM meta on VAL_EFF
  5. Predict on H1, H2 → score
  6. Apply isotonic on H1
  7. Eval vs rank-mean baseline + paris baseline

Outputs:
  - runs/sl_regime_stack/predictions.parquet  (trade_date, ts_code, score_raw, score_calibrated)
  - runs/sl_regime_stack/ensemble.json
  - runs/sl_regime_stack/RESULTS.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p3.path1_eval import H1, H2, evaluate
from p3.path1_ensemble import pick_top_configs_by_val, seed_mean_ensemble


logger = logging.getLogger(__name__)


VAL_EFF = (dt.date(2025, 1, 1), dt.date(2025, 6, 4))


def _build_path_ensemble(runs_root: Path, top_k: int = 3) -> pl.DataFrame:
    """Replicate path1_ensemble logic to get a single (trade_date, ts_code, score) frame."""
    runs = {}
    for run_dir in sorted(runs_root.glob("*_seed*/")):
        results = run_dir / "results.json"
        if not results.exists():
            continue
        d = json.loads(results.read_text())
        runs[run_dir.name] = (
            d["VAL_EFF"]["primary_mean_top50_proximity_excess"],
            run_dir / "predictions.parquet",
        )
    val_scores = {n: v[0] for n, v in runs.items()}
    chosen = pick_top_configs_by_val(val_scores, top_k=top_k)
    chosen = [c for c in chosen if c in runs]
    pred_dfs = [pl.read_parquet(runs[n][1]) for n in chosen]
    ens = seed_mean_ensemble(pred_dfs)
    return ens, chosen


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--paths", nargs="+", default=("sl_path1", "sl_path4", "sl_path2"))
    ap.add_argument("--top-k-configs", type=int, default=3)
    ap.add_argument("--out", default=Path("runs/sl_regime_stack"), type=Path)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Build per-path ensemble predictions
    path_ens = {}
    chosen_per_path = {}
    for p in args.paths:
        runs_dir = Path("runs") / p
        if not runs_dir.exists():
            logger.warning("path %s missing — skipping", p)
            continue
        ens, chosen = _build_path_ensemble(runs_dir, top_k=args.top_k_configs)
        path_ens[p] = ens
        chosen_per_path[p] = chosen
        logger.info("[%s] picked %d runs; %d rows", p, len(chosen), len(ens))

    if len(path_ens) < 2:
        logger.error("need at least 2 paths for stacking; got %d", len(path_ens))
        return 2

    # 2. Wide-frame: each path's score as a column
    base = list(path_ens.values())[0].select(["trade_date", "ts_code", "score"]).rename(
        {"score": f"path_{list(path_ens.keys())[0]}"}
    )
    for p, ens in list(path_ens.items())[1:]:
        base = base.join(
            ens.select(["trade_date", "ts_code", "score"]).rename({"score": f"path_{p}"}),
            on=["trade_date", "ts_code"], how="inner",
        )
    path_score_cols = [f"path_{p}" for p in path_ens.keys()]
    logger.info("wide-frame base: %d rows × %d path scores", len(base), len(path_score_cols))

    # 3. Join regime features
    regime_path = args.bundle / "regime_features.parquet"
    if not regime_path.exists():
        logger.error("regime_features.parquet missing — run path5_regime_features.py first")
        return 3
    regime = pl.read_parquet(regime_path)
    regime_cols = [c for c in regime.columns if c != "trade_date"]
    base = base.join(regime, on="trade_date", how="left")
    logger.info("after regime join: %d rows × %d total feature cols",
                len(base), len(path_score_cols) + len(regime_cols))

    # 4. Add path × regime interaction features (top-N important interactions)
    # For each path score × {univ_vol_20d, cs_dispersion, sector_dispersion}:
    interaction_cols = []
    for psc in path_score_cols:
        for rg in ("univ_vol_20d", "cs_dispersion", "sector_dispersion"):
            cname = f"{psc}__x__{rg}"
            base = base.with_columns((pl.col(psc) * pl.col(rg)).cast(pl.Float32).alias(cname))
            interaction_cols.append(cname)

    feature_cols = path_score_cols + regime_cols + interaction_cols
    logger.info("meta features: %d (path=%d, regime=%d, interaction=%d)",
                len(feature_cols), len(path_score_cols), len(regime_cols), len(interaction_cols))

    # 5. Join target_y (actual proximity-weighted excess)
    target_y = pl.read_parquet(args.bundle / "target_y.parquet")
    base = base.join(target_y, on=["trade_date", "ts_code"], how="inner")
    logger.info("after target join: %d rows", len(base))

    # 6. Split
    train_meta = base.filter(
        (pl.col("trade_date") >= VAL_EFF[0]) & (pl.col("trade_date") <= VAL_EFF[1])
    ).drop_nulls()
    eval_h1 = base.filter(
        (pl.col("trade_date") >= H1[0]) & (pl.col("trade_date") <= H1[1])
    ).drop_nulls()
    eval_h2 = base.filter(
        (pl.col("trade_date") >= H2[0]) & (pl.col("trade_date") <= H2[1])
    ).drop_nulls()
    logger.info("splits (after drop_nulls): train_meta=%d  H1=%d  H2=%d",
                len(train_meta), len(eval_h1), len(eval_h2))

    # 7. Train meta-learner (small LightGBM)
    X_train = train_meta.select(feature_cols).to_numpy().astype(np.float32)
    y_train = train_meta["y"].to_numpy().astype(np.float32)

    # Use a held-out tail of train_meta as in-sample val for early stopping
    n_train = int(len(X_train) * 0.85)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X_train))
    X_tr = X_train[perm[:n_train]]; y_tr = y_train[perm[:n_train]]
    X_vl = X_train[perm[n_train:]]; y_vl = y_train[perm[n_train:]]
    train_ds = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    val_ds = lgb.Dataset(X_vl, label=y_vl, feature_name=feature_cols, reference=train_ds)

    params = {
        "objective": "regression_l2",
        "metric": ["l2", "l1"],
        "num_leaves": 15,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "min_data_in_leaf": 200,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": -1,
    }
    logger.info("training meta-learner: %s", {k: v for k, v in params.items() if k != "metric"})
    t0 = time.time()
    meta = lgb.train(
        params, train_ds, num_boost_round=500,
        valid_sets=[val_ds], valid_names=["val"],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                   lgb.log_evaluation(period=50)],
    )
    logger.info("meta trained in %.1fs (best_iter=%d)", time.time() - t0, meta.best_iteration)

    # 8. Predict on H1, H2 + full panel for output
    def predict_window(df: pl.DataFrame) -> pl.DataFrame:
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        score = meta.predict(X, num_iteration=meta.best_iteration).astype(np.float32)
        return df.select(["trade_date", "ts_code"]).with_columns(pl.Series("score", score))

    pred_full = predict_window(base.drop_nulls())
    pred_h1 = predict_window(eval_h1)
    pred_h2 = predict_window(eval_h2)
    logger.info("predicted: full=%d H1=%d H2=%d", len(pred_full), len(pred_h1), len(pred_h2))

    # 9. Calibrate on H1
    h1_join = pred_h1.join(target_y, on=["trade_date", "ts_code"], how="inner")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(h1_join["score"].to_numpy(), h1_join["y"].to_numpy())
    pred_full_cal = pred_full.with_columns(
        pl.Series("score_calibrated", iso.transform(pred_full["score"].to_numpy()).astype(np.float32))
    )

    # 10. Eval
    realized = pl.read_parquet(args.bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    )
    market = pl.read_parquet(args.bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    )
    eval_raw_h1 = evaluate(pred_full, target_y, realized, market, H1)
    eval_raw_h2 = evaluate(pred_full, target_y, realized, market, H2)
    eval_cal_h1 = evaluate(
        pred_full_cal.select(["trade_date", "ts_code", pl.col("score_calibrated").alias("score")]),
        target_y, realized, market, H1,
    )
    eval_cal_h2 = evaluate(
        pred_full_cal.select(["trade_date", "ts_code", pl.col("score_calibrated").alias("score")]),
        target_y, realized, market, H2,
    )

    # 11. Compare against best individual path + paris baseline
    bp = pl.read_parquet(args.bundle / "baseline_predictions.parquet").select(
        ["trade_date", "ts_code", pl.col("p_t3_baseline").alias("score")]
    )
    bp_h1 = evaluate(bp, target_y, realized, market, H1)
    bp_h2 = evaluate(bp, target_y, realized, market, H2)

    per_path_eval = {}
    for p, ens in path_ens.items():
        per_path_eval[p] = {
            "H1": evaluate(ens, target_y, realized, market, H1),
            "H2": evaluate(ens, target_y, realized, market, H2),
        }

    # 12. Save outputs
    pred_out = pred_full_cal.rename({"score": "score_raw"}).select(
        ["trade_date", "ts_code", "score_raw", "score_calibrated"]
    )
    pred_out.write_parquet(args.out / "predictions.parquet", compression="zstd", compression_level=10)

    summary = {
        "paths_used": list(path_ens.keys()),
        "chosen_per_path": chosen_per_path,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "meta_params": params,
        "meta_best_iteration": meta.best_iteration,
        "stacking_raw_H1": eval_raw_h1,
        "stacking_raw_H2": eval_raw_h2,
        "stacking_calibrated_H1": eval_cal_h1,
        "stacking_calibrated_H2": eval_cal_h2,
        "per_path_ensemble": per_path_eval,
        "paris_baseline_H1": bp_h1,
        "paris_baseline_H2": bp_h2,
    }
    (args.out / "ensemble.json").write_text(json.dumps(summary, indent=2, default=str))

    # Save the meta-learner for production reuse
    meta.save_model(str(args.out / "meta_lgb_model.txt"))
    with (args.out / "meta_isotonic.pkl").open("wb") as f:
        pickle.dump(iso, f)

    # 13. Print scoreboard
    logger.info("== Regime-aware stacking scoreboard ==")
    rows = [("paris_baseline", bp_h1, bp_h2)]
    for p, e in per_path_eval.items():
        rows.append((p, e["H1"], e["H2"]))
    rows.append(("STACKING_raw", eval_raw_h1, eval_raw_h2))
    rows.append(("STACKING_cal", eval_cal_h1, eval_cal_h2))
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

    # 14. Feature importance for the meta-learner — what regime features matter?
    logger.info("== Meta-learner feature importance (top 15) ==")
    gain_imp = meta.feature_importance(importance_type="gain")
    rank = sorted(zip(feature_cols, gain_imp), key=lambda x: -x[1])
    total = sum(gain_imp)
    for c, g in rank[:15]:
        logger.info("  %-40s gain=%5.1f%%", c, g / total * 100 if total else 0)

    logger.info("done. results: %s", args.out / "ensemble.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
