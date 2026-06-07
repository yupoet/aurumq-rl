"""Matrix v10de_v2 — CatBoost with paris production hyperparams.

paris v26 shipped path2_long_catboost_params.json — production-tuned hyperparams:
  depth=6, iterations=2000, lr=0.05, bootstrap_type=MVS, subsample=0.8, rsm=0.8
  early_stopping_rounds=50, RMSE loss, random_seed=42

ledashi v10de used generic params (depth=8, iter=200, lr=0.05) — paris-spec re-test
on top-priority cell `catboost_v1_CSI500_v3unified` (v10de Q1 IC +6.49% record).

If paris-spec ↑ vs generic → paris hyperparam tuning matters; ship to paris for prod.

Cell: catboost_v1_CSI500_v3unified (production candidate per paris ACK v28b Q5)
"""
from __future__ import annotations
import os, gc, json, time
from pathlib import Path

_HANDOFF_INBOX = os.environ.get("AURUMQ_HANDOFF_INBOX", "data/handoffs/inbox")
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kronos_matrix_v10 import (
    UNIVERSES, TRAIN_START, TRAIN_END,
    _dt, load_universes, filter_universe, compute_realized_and_exits, load_panel, eval_cell,
)

# paris production CatBoost params (from path2_long_catboost_params.json)
CATBOOST_PARAMS_PARIS = dict(
    depth=6,
    iterations=2000,
    learning_rate=0.05,
    bootstrap_type="MVS",
    subsample=0.8,
    rsm=0.8,
    early_stopping_rounds=50,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=0,
    thread_count=-1,
)

OUT_DIR = Path("data/kronos/outputs/matrix_v10de_v2_paris_catboost")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_PATH_BASE = Path("data/p3_4070_long")


def train_catboost_paris(X_train, y_train, X_val, y_val, X_full):
    from catboost import CatBoostRegressor
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    m = CatBoostRegressor(**CATBOOST_PARAMS_PARIS)
    m.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, early_stopping_rounds=50)
    best_iter = m.get_best_iteration()
    return m, m.predict(X_full), best_iter


def main():
    t_total = time.time()
    static_sets, pit_dfs = load_universes(UNIVERSES)
    print("[setup] realized + dyn-exit ...")
    realized, exits = compute_realized_and_exits()

    # Single-cell experiment: catboost wave_v1 CSI500 v3unified
    panel_name = "v3unified"
    univ = "CSI500"
    label_v = "v1"  # wave_v1 — v10de found CSI500 v3unified v1 Q1 +6.49% record

    panel = load_panel(panel_name)
    base_cols = [c for c in panel.columns if c not in ("ts_code", "trade_date")]
    print(f"\n[panel {panel_name}] {len(panel):,} rows × {len(base_cols)} feats")

    upanel = filter_universe(panel, univ, static_sets, pit_dfs)
    print(f"[upanel] {univ}: {len(upanel):,} rows")

    label_path = LABEL_PATH_BASE / f"target_y_wave_{label_v}.parquet"
    label_df = _dt(pd.read_parquet(label_path, columns=["trade_date", "ts_code", "y"]))
    joined = upanel.merge(label_df, on=["ts_code", "trade_date"], how="inner")
    train = joined[(joined["trade_date"] >= TRAIN_START) & (joined["trade_date"] <= TRAIN_END)]
    print(f"  train rows: {len(train):,}")

    val_size = max(1000, int(len(train) * 0.10))
    val = train.tail(val_size)
    train_fit = train.head(len(train) - val_size)

    t = time.time()
    model, preds, best_iter = train_catboost_paris(
        train_fit[base_cols].values, train_fit["y"].values,
        val[base_cols].values, val["y"].values,
        upanel[base_cols].values,
    )
    del train_fit, val, joined, train; gc.collect()
    print(f"  train {time.time()-t:.0f}s, best_iter={best_iter}")

    pred_df = upanel[["ts_code", "trade_date"]].copy()
    pred_df["score"] = preds.astype(np.float32)

    exp_id = f"catboost_v2_paris_{label_v}_{univ}_{panel_name}"
    artifact_dir = OUT_DIR / exp_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(artifact_dir / "model.cbm"))
    (artifact_dir / "feature_cols.json").write_text(json.dumps(base_cols, indent=2))
    (artifact_dir / "catboost_params.json").write_text(json.dumps(CATBOOST_PARAMS_PARIS, indent=2))
    (artifact_dir / "train_spec.json").write_text(json.dumps({
        "train_window": [str(TRAIN_START), str(TRAIN_END)],
        "best_iter": int(best_iter),
        "comment": "paris production CatBoost hyperparams (path2_long_catboost_params.json)",
    }, indent=2))
    pred_df.to_parquet(artifact_dir / "pred_full_panel.parquet", compression="zstd")

    result = eval_cell(pred_df, realized, exits, adaptive_gating=None)

    r = result["static"]["H2_2025"]
    ic20 = r["fwd20"]["ic"] * 100
    sn50 = r["fwd20"]["sizing"].get("50", {}).get("sharpe_net", float("nan"))
    q1 = result["static"]["Q1_2026"]["fwd20"]["ic"] * 100
    print(f"  {exp_id}: H2 fwd20 IC={ic20:+.3f}% Sharpe50_NET={sn50:+.2f} | Q1 IC={q1:+.3f}% best_iter={best_iter}")
    print(f"  vs v10de generic (Q1 +6.49%): paris-spec {q1:+.3f}% | gap {q1 - 6.49:+.3f}pp")

    out = {
        "config": {"tier": "v10de_v2 paris CatBoost hyperparams single-cell",
                   "exp_id": exp_id,
                   "panel": panel_name, "universe": univ, "label": f"wave_{label_v}",
                   "hyperparams": CATBOOST_PARAMS_PARIS},
        "result": result,
        "vs_generic_v10de": {"v10de_q1": 6.49, "v10de_v2_q1": q1, "delta_pp": q1 - 6.49},
        "total_time_s": time.time() - t_total,
    }
    Path("data/kronos/outputs/matrix_v10de_v2_paris_catboost_result.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\n[saved] matrix_v10de_v2_paris_catboost_result.json, {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
