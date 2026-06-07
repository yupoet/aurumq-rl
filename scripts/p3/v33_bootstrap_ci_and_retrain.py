"""v33 follow-up — bootstrap CI on 2 production candidates.

1. Re-train 7y HARD_TECH Bayesian best trial (trial #23) and save pred_full_panel
   so we have an artifact bundle to bootstrap.
2. Bootstrap CI on:
   - target_y_CSI300_v3unified (pred parquet already from v33 task 2)
   - 7y_bayesian_hard_tech_v2_no_phase_c_t5 (pred parquet from step 1)

Methodology: 1000 iter × 5-day block bootstrap on Sharpe NET K{10,50} × fwd{5,20}
Same as kronos_v11_v12_bootstrap_ci.py.
"""
from __future__ import annotations
import os, gc, json, time
from pathlib import Path
import datetime as _dt_mod

_HANDOFF_INBOX = os.environ.get("AURUMQ_HANDOFF_INBOX", "data/handoffs/inbox")
import lightgbm as lgb
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kronos_matrix_v10 import _dt, load_universes, PANEL_CLOSE, COST_ROUND_TRIP
from kronos_matrix_v11_7y_HARD_TECH import (
    PARIS_7Y_DIR, LEDASHI_3Y_PANEL_PATH, LEDASHI_3Y_LABELS_DIR,
    UNIVERSE, TRAIN_START_7Y, TRAIN_END,
)
from v33_hard_tech_7y_bayesian import load_panel_filtered, load_labels_7y

OUT_DIR = Path("data/kronos/outputs/v33_bootstrap_ci")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = OUT_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Trial 23 best params
BAYESIAN_BEST_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "n_estimators": 500,
    "min_child_samples": 249,
    "feature_fraction": 0.6185549150396007,
    "num_leaves": 144,
    "learning_rate": 0.013006405077401426,
    "bagging_fraction": 0.85, "bagging_freq": 1,
    "n_jobs": -1, "verbose": -1, "random_state": 42,
}

N_BOOTSTRAP = 1000
BLOCK_LENGTH = 5


def block_bootstrap_sharpe(daily_returns, k_horizon, n_iter=N_BOOTSTRAP, block_len=BLOCK_LENGTH, seed=42):
    arr = np.asarray(daily_returns)
    arr_net = arr - COST_ROUND_TRIP
    n = len(arr_net)
    if n < block_len * 4:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n_samples": n}
    n_blocks = (n + block_len - 1) // block_len
    ann = np.sqrt(252.0 / max(k_horizon, 1))
    rng = np.random.default_rng(seed)
    sharpes = []
    for _ in range(n_iter):
        start_idxs = rng.integers(0, n - block_len + 1, size=n_blocks)
        resampled = np.concatenate([arr_net[s:s+block_len] for s in start_idxs])[:n]
        sd = resampled.std(ddof=1)
        if sd < 1e-9: continue
        sharpes.append(resampled.mean() / sd * ann)
    arr_sh = np.asarray(sharpes)
    return {
        "mean": float(arr_sh.mean()),
        "ci95_low": float(np.percentile(arr_sh, 2.5)),
        "ci95_high": float(np.percentile(arr_sh, 97.5)),
        "n_samples": n,
        "n_bootstrap": len(sharpes),
    }


def build_realized():
    p = pd.read_parquet(PANEL_CLOSE, columns=["ts_code", "trade_date", "close", "adj_factor"])
    p = _dt(p).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    latest = p.groupby("ts_code")["adj_factor"].transform("last")
    p["adj_close"] = (p["close"] * p["adj_factor"] / latest).astype(np.float32)
    for k in (5, 10, 20):
        p[f"ret_fwd{k}"] = (p.groupby("ts_code")["adj_close"].shift(-k) / p["adj_close"] - 1.0).astype(np.float32)
    return p[["ts_code", "trade_date"] + [f"ret_fwd{k}" for k in (5, 10, 20)]]


def compute_ci(pred_df, realized, top_k, horizon, window_name):
    eval_windows = {
        "H2_2025": (_dt_mod.date(2025, 7, 1), _dt_mod.date(2025, 12, 31)),
        "Q1_2026": (_dt_mod.date(2026, 1, 1), _dt_mod.date(2026, 3, 31)),
    }
    ws, we = eval_windows[window_name]
    pred_df = pred_df.copy()
    pred_df["trade_date"] = pd.to_datetime(pred_df["trade_date"]).dt.date
    pred_df = pred_df[(pred_df["trade_date"] >= ws) & (pred_df["trade_date"] <= we)]
    eval_df = pred_df.merge(realized, on=["trade_date", "ts_code"], how="inner")
    fwd_col = f"ret_fwd{horizon}"
    daily_rets = []
    for d, g in eval_df.dropna(subset=["score", fwd_col]).groupby("trade_date"):
        if len(g) < top_k: continue
        daily_rets.append(float(g.nlargest(top_k, "score")[fwd_col].mean()))
    if len(daily_rets) < 20:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"),
                "n_samples": len(daily_rets), "skipped": "insufficient_days"}
    return block_bootstrap_sharpe(daily_rets, k_horizon=horizon)


def retrain_bayesian_best():
    """Re-train 7y HARD_TECH with trial 23 best params, save full artifact bundle."""
    print("\n=== Re-training 7y HARD_TECH Bayesian best (trial 23) for artifact bundle ===\n")
    panel = load_panel_filtered()
    print(f"  panel: {len(panel):,} rows × {panel.shape[1]} cols")
    base_cols = [c for c in panel.columns if c not in ("ts_code", "trade_date")]
    label_df = load_labels_7y()

    joined = panel.merge(label_df, on=["ts_code", "trade_date"], how="inner")
    train = joined[(joined["trade_date"] >= TRAIN_START_7Y) & (joined["trade_date"] <= TRAIN_END)]
    val_size = max(500, int(len(train) * 0.10))
    val = train.tail(val_size); train_fit = train.head(len(train) - val_size)
    del train, joined; gc.collect()

    t = time.time()
    model = lgb.LGBMClassifier(**BAYESIAN_BEST_PARAMS)
    model.fit(train_fit[base_cols], train_fit["y"],
              eval_set=[(val[base_cols], val["y"])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    best_iter = model.best_iteration_ or BAYESIAN_BEST_PARAMS["n_estimators"]
    print(f"  train {time.time()-t:.0f}s, best_iter={best_iter}")

    preds = model.predict_proba(panel[base_cols])[:, 1].astype(np.float32)
    pred_df = panel[["ts_code", "trade_date"]].copy()
    pred_df["score"] = preds

    # Save artifact bundle
    cell_id = "7y_bayesian_hard_tech_v2_no_phase_c_t5"
    bundle = ARTIFACT_DIR / cell_id
    bundle.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(bundle / "model.txt"))
    (bundle / "feature_cols.json").write_text(json.dumps(base_cols, indent=2))
    (bundle / "lgb_params.json").write_text(json.dumps(BAYESIAN_BEST_PARAMS, indent=2))
    (bundle / "train_window_spec.json").write_text(json.dumps({
        "cell_id": cell_id,
        "source": "v33 task 3 7y HARD_TECH Bayesian best trial #23",
        "universe": "HARD_TECH", "horizon": "t5", "panel": "v2_no_phase_c (7y)",
        "label_source": "labels_A_t5_HARD_TECH_year=*.parquet (paris + ledashi)",
        "train_window": [str(TRAIN_START_7Y), str(TRAIN_END)],
        "val_size": int(val_size), "best_iter": int(best_iter),
        "n_features": len(base_cols),
        "bayesian_sweep": {"composite": 6.19, "trial": 23},
    }, indent=2))
    static_sets, _ = load_universes(("HARD_TECH",))
    mdf = pd.DataFrame({"ts_code": sorted(static_sets["HARD_TECH"])})
    mdf.to_parquet(bundle / "universe_mask_HARD_TECH.parquet", compression="zstd")
    pred_df.to_parquet(bundle / "pred_full_panel.parquet", compression="zstd")
    print(f"  artifact bundle saved: {bundle}")
    del model, train_fit, val; gc.collect()
    return cell_id, pred_df


def main():
    t_total = time.time()
    print("[setup] realized returns + universes ...")
    realized = build_realized()
    realized["trade_date"] = realized["trade_date"]  # already date
    print(f"  realized: {len(realized):,} rows")

    results = {}

    # Cell 1: target_y_CSI300_v3unified (existing pred parquet from v33 task 2)
    print("\n=== Bootstrap CI: target_y_CSI300_v3unified ===")
    pred_path_csi300 = Path("data/kronos/outputs/matrix_v33_csi300_sweep/pred_target_y_CSI300_v3unified.parquet")
    pred_csi300 = pd.read_parquet(pred_path_csi300)
    csi300_ci = {}
    for window in ("H2_2025", "Q1_2026"):
        for K in (10, 50):
            for H in (5, 20):
                ci = compute_ci(pred_csi300, realized, top_k=K, horizon=H, window_name=window)
                csi300_ci[f"{window}_K{K}_fwd{H}"] = ci
                print(f"  {window} K={K} fwd{H}: mean={ci['mean']:+.2f} CI95=[{ci['ci95_low']:+.2f}, {ci['ci95_high']:+.2f}]")
    results["target_y_CSI300_v3unified"] = csi300_ci

    # Cell 2: re-train + bootstrap 7y HARD_TECH Bayesian best
    cell_id_2, pred_bayesian = retrain_bayesian_best()
    print(f"\n=== Bootstrap CI: {cell_id_2} ===")
    bayesian_ci = {}
    for window in ("H2_2025", "Q1_2026"):
        for K in (10, 50):
            for H in (5, 20):
                ci = compute_ci(pred_bayesian, realized, top_k=K, horizon=H, window_name=window)
                bayesian_ci[f"{window}_K{K}_fwd{H}"] = ci
                print(f"  {window} K={K} fwd{H}: mean={ci['mean']:+.2f} CI95=[{ci['ci95_low']:+.2f}, {ci['ci95_high']:+.2f}]")
    results[cell_id_2] = bayesian_ci

    out_path = OUT_DIR / "v33_bootstrap_ci_results.json"
    out_path.write_text(json.dumps({
        "task": "v33 bootstrap CI on 2 production candidates",
        "config": {"n_bootstrap": N_BOOTSTRAP, "block_length": BLOCK_LENGTH,
                   "cost_round_trip": COST_ROUND_TRIP,
                   "method": "block bootstrap on date axis, Sharpe NET top-K"},
        "cells": list(results.keys()),
        "results": results,
        "total_time_s": time.time() - t_total,
    }, indent=2, default=str))
    print(f"\n[saved] {out_path}")
    print(f"[done] total {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
