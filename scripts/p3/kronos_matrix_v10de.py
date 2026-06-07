"""Matrix v10d + v10e — CatBoost + XGBoost on wave_v* labels (expanded coverage).

Per user feedback: original v10d/e with only 12 cells too few for valid conclusion.
Expanded to 2 panel × 6 universe × 4 wave_v* labels = 48 cells per algorithm.
Total: 48 (CatBoost) + 48 (XGBoost) = 96 cells.

Panels: ledashi + v3unified (production candidates)
Same wave_v1-v4 proximity-weighted labels as v10.
Same evaluation pipeline (7 sizing + 11 dyn-exit + cost-aware).
"""
from __future__ import annotations

import os
import gc
import json
import time
from pathlib import Path

_HANDOFF_INBOX = os.environ.get("AURUMQ_HANDOFF_INBOX", "data/handoffs/inbox")

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kronos_matrix_v10 import (
    PANEL_PATHS, UNIVERSES, PANEL_CLOSE, TECH_LINES, UNIVERSE_DIR,
    TRAIN_START, TRAIN_END, WINDOWS, HORIZONS, K_MAX, TRIGGERS, COST_ROUND_TRIP,
    TOP_K_LIST, ADAPTIVE_PCT, ADAPTIVE_K_CAP, ADAPTIVE_K_FLOOR, ADAPTIVE_Q_LOW, ADAPTIVE_Q_MID,
    LABELS,
    _dt, load_universes, filter_universe, compute_realized_and_exits,
    cross_sec_ic, top_k_sharpe_ann_multi, dyn_agg_multi, compute_adaptive_thresholds,
    load_panel, eval_cell,
)

LABEL_TEMPLATE = "data/p3_4070_long/target_y_wave_{v}.parquet"
OUT_DIR_CAT = Path("data/kronos/outputs/matrix_v10d")
OUT_DIR_XGB = Path("data/kronos/outputs/matrix_v10e")
OUT_DIR_CAT.mkdir(parents=True, exist_ok=True)
OUT_DIR_XGB.mkdir(parents=True, exist_ok=True)

PANELS_NARROW = ("ledashi", "v3unified")

CATBOOST_PARAMS = dict(
    iterations=200, learning_rate=0.05, depth=8,
    loss_function="RMSE",
    bagging_temperature=1.0,
    random_seed=42, verbose=0, thread_count=-1,
)
XGBOOST_PARAMS = dict(
    n_estimators=200, max_depth=8, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42, verbosity=0, n_jobs=-1, tree_method="hist",
)


def run_algorithm(algo_name, train_fn, all_labels, panels, static_sets, pit_dfs, realized, exits, out_dir):
    """Generic runner for CatBoost or XGBoost cell training."""
    matrix_results = {}
    CHECKPOINT = Path(f"data/kronos/outputs/matrix_v10{'d' if algo_name=='catboost' else 'e'}_checkpoint.json")
    if CHECKPOINT.exists():
        prev = json.loads(CHECKPOINT.read_text())
        matrix_results = prev.get("results", {})
        print(f"[{algo_name} resume] {len(matrix_results)} cells from checkpoint")

    for panel_name in panels:
        panel_cells = [f"{algo_name}_{label_v}_{u}_{panel_name}"
                       for u in UNIVERSES for label_v in LABELS]
        if all(c in matrix_results for c in panel_cells):
            print(f"\n[skip {algo_name} panel {panel_name}] all in checkpoint", flush=True); continue

        try: panel = load_panel(panel_name)
        except Exception as e:
            print(f"  [{panel_name}] LOAD FAILED: {e}"); continue
        base_cols = [c for c in panel.columns if c not in ("ts_code","trade_date")]
        print(f"\n[{algo_name} panel {panel_name}] {len(panel):,} rows × {len(base_cols)} feats")

        for univ in UNIVERSES:
            upanel = filter_universe(panel, univ, static_sets, pit_dfs)
            print(f"\n--- {algo_name} {univ} on {panel_name}: {len(upanel):,} rows ---", flush=True)

            for label_v in LABELS:
                exp_id = f"{algo_name}_{label_v}_{univ}_{panel_name}"
                if exp_id in matrix_results: continue

                label = all_labels[label_v]
                joined = upanel.merge(label, on=["ts_code","trade_date"], how="inner")
                train = joined[(joined["trade_date"]>=TRAIN_START) & (joined["trade_date"]<=TRAIN_END)]
                if len(train) < 10000:
                    matrix_results[exp_id] = {"skipped": True}; continue

                t = time.time()
                # Fill NaN with 0 for CatBoost/XGBoost (LGB handles NaN natively but these don't)
                X_train = train[base_cols].fillna(0).values
                y_train = train["y"].values
                model, preds = train_fn(X_train, y_train, upanel[base_cols].fillna(0).values)
                del train, joined, X_train, y_train; gc.collect()

                pred_df = upanel[["ts_code","trade_date"]].copy()
                pred_df["score"] = preds.astype(np.float32)
                pred_df.to_parquet(out_dir / f"pred_{exp_id}.parquet", compression="zstd")

                adaptive_gating = compute_adaptive_thresholds(pred_df)
                result = eval_cell(pred_df, realized, exits, adaptive_gating)
                if adaptive_gating:
                    result["adaptive_meta"] = {"q25": adaptive_gating["q25"], "q50": adaptive_gating["q50"]}
                matrix_results[exp_id] = result

                r = result["static"]["H2_2025"]
                ic20 = r["fwd20"]["ic"] * 100
                sn50 = r["fwd20"]["sizing"].get("50", {}).get("sharpe_net", float("nan"))
                q1 = result["static"]["Q1_2026"]["fwd20"]["ic"] * 100
                print(f"  {exp_id}: train {time.time()-t:.0f}s | H2 fwd20 IC={ic20:+.3f}% Sharpe50_NET={sn50:+.2f} | Q1 IC={q1:+.3f}%", flush=True)

                del model, preds, pred_df, result; gc.collect()
            del upanel; gc.collect()
            CHECKPOINT.write_text(json.dumps({"results": matrix_results}, indent=2, default=str))
        del panel; gc.collect()

    return matrix_results


def train_catboost(X, y, X_full):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(**CATBOOST_PARAMS)
    m.fit(X, y)
    return m, m.predict(X_full)


def train_xgboost(X, y, X_full):
    import xgboost as xgb
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    m = xgb.XGBRegressor(**XGBOOST_PARAMS)
    m.fit(X, y)
    return m, m.predict(X_full)


def main():
    t_total = time.time()
    print("[setup] universes ...")
    static_sets, pit_dfs = load_universes(UNIVERSES)

    print("\n[setup] realized + dyn-exit ...")
    realized, exits = compute_realized_and_exits()

    print("\n[setup] labels ...")
    all_labels = {}
    for v in LABELS:
        l = _dt(pd.read_parquet(LABEL_TEMPLATE.format(v=v), columns=["trade_date","ts_code","y"]))
        all_labels[v] = l
        print(f"  wave_{v}: {len(l):,} rows")

    print("\n=== v10d CatBoost ===")
    cat_results = run_algorithm("catboost", train_catboost, all_labels, PANELS_NARROW,
                                 static_sets, pit_dfs, realized, exits, OUT_DIR_CAT)

    print("\n=== v10e XGBoost ===")
    xgb_results = run_algorithm("xgboost", train_xgboost, all_labels, PANELS_NARROW,
                                 static_sets, pit_dfs, realized, exits, OUT_DIR_XGB)

    Path("data/kronos/outputs/matrix_v10d_results.json").write_text(
        json.dumps({"config": {"algorithm": "CatBoost", "panels": list(PANELS_NARROW), "labels": list(LABELS),
                               "universes": list(UNIVERSES), "params": CATBOOST_PARAMS},
                    "results": cat_results}, indent=2, default=str))
    Path("data/kronos/outputs/matrix_v10e_results.json").write_text(
        json.dumps({"config": {"algorithm": "XGBoost", "panels": list(PANELS_NARROW), "labels": list(LABELS),
                               "universes": list(UNIVERSES), "params": XGBOOST_PARAMS},
                    "results": xgb_results}, indent=2, default=str))
    print(f"\n[done] v10d+v10e {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
