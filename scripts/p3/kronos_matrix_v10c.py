"""Matrix v10c — LGB BINARY classifier on wave_v* labels (vs v10's regression).

Per user: paris production wave_v* uses BINARY classifier (objective=binary,
metric=average_precision, num_leaves=63, n=500+early_stop). v10c re-runs same
7 panel × 6 universe × 4 wave_v* labels but with BINARY classifier hyperparam.

Labels thresholded internally: y_binary = 1 if y_continuous > P{cutoff_pct} else 0.
Default cutoff_pct=75 (top quartile becomes positive).

Total: 7 × 6 × 4 = 168 cells (same scope as v10).
"""
from __future__ import annotations

import os
import gc
import json
import time
from pathlib import Path

_HANDOFF_INBOX = os.environ.get("AURUMQ_HANDOFF_INBOX", "data/handoffs/inbox")

import lightgbm as lgb
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kronos_matrix_v10 import (
    PANEL_PATHS, PANELS, UNIVERSES, PANEL_CLOSE, TECH_LINES, UNIVERSE_DIR,
    TRAIN_START, TRAIN_END, WINDOWS, HORIZONS, K_MAX, TRIGGERS, COST_ROUND_TRIP,
    TOP_K_LIST, ADAPTIVE_PCT, ADAPTIVE_K_CAP, ADAPTIVE_K_FLOOR, ADAPTIVE_Q_LOW, ADAPTIVE_Q_MID,
    FAMILY_PREFIXES, LABELS,
    _dt, load_universes, filter_universe, compute_realized_and_exits,
    cross_sec_ic, top_k_sharpe_ann_multi, dyn_agg_multi, compute_adaptive_thresholds,
    family_importance, load_panel, eval_cell,
)

LABEL_TEMPLATE = "data/p3_4070_long/target_y_wave_{v}.parquet"
OUT_DIR = Path("data/kronos/outputs/matrix_v10c")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BINARY_CUTOFF_PCT = 75  # threshold: y > P75 → binary=1

# paris production wave_binary hyperparam
LGB_BINARY_PARAMS = dict(
    objective="binary", metric="average_precision", boosting_type="gbdt",
    learning_rate=0.05, num_leaves=63,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    min_data_in_leaf=200,
    n_estimators=500,
    verbose=-1, num_threads=-1, random_state=42,
)
EARLY_STOPPING = 50


def main():
    t_total = time.time()
    print("[setup] universes ...")
    static_sets, pit_dfs = load_universes(UNIVERSES)

    print("\n[setup] realized + dyn-exit ...")
    realized, exits = compute_realized_and_exits()

    print(f"\n[setup] labels (with binary threshold P{BINARY_CUTOFF_PCT}) ...")
    all_labels = {}
    for v in LABELS:
        l = _dt(pd.read_parquet(LABEL_TEMPLATE.format(v=v), columns=["trade_date","ts_code","y"]))
        threshold = np.percentile(l["y"][l["y"] > 0], BINARY_CUTOFF_PCT) if (l["y"] > 0).sum() > 0 else 0
        l["y_binary"] = (l["y"] > threshold).astype(np.int8)
        all_labels[v] = l
        pos_rate = l["y_binary"].mean()
        print(f"  wave_{v}: threshold={threshold:.5f}, pos_rate={pos_rate:.3%}")

    matrix_results = {}
    family_importance_results = {}
    CHECKPOINT = Path("data/kronos/outputs/matrix_v10c_checkpoint.json")
    if CHECKPOINT.exists():
        prev = json.loads(CHECKPOINT.read_text())
        matrix_results = prev.get("results", {})
        family_importance_results = prev.get("family_importance", {})
        print(f"[resume] {len(matrix_results)} cells from checkpoint")

    for panel_name in PANELS:
        panel_cells = [f"binary_{label_v}_{u}_{panel_name}" for u in UNIVERSES for label_v in LABELS]
        if all(c in matrix_results for c in panel_cells):
            print(f"\n[skip panel {panel_name}] all in checkpoint", flush=True); continue

        try: panel = load_panel(panel_name)
        except Exception as e:
            print(f"  [{panel_name}] LOAD FAILED: {e}"); continue
        base_cols = [c for c in panel.columns if c not in ("ts_code","trade_date")]
        print(f"\n[panel {panel_name}] loaded, {len(panel):,} rows × {len(base_cols)} feats")

        for univ in UNIVERSES:
            upanel = filter_universe(panel, univ, static_sets, pit_dfs)
            print(f"\n--- {univ} on {panel_name}: {len(upanel):,} rows ---", flush=True)

            for label_v in LABELS:
                exp_id = f"binary_{label_v}_{univ}_{panel_name}"
                if exp_id in matrix_results:
                    print(f"  [resume skip] {exp_id}"); continue

                label = all_labels[label_v]
                joined = upanel.merge(label[["ts_code","trade_date","y_binary"]], on=["ts_code","trade_date"], how="inner")
                train_full = joined[(joined["trade_date"]>=TRAIN_START) & (joined["trade_date"]<=TRAIN_END)].copy()
                train_full = train_full.sort_values("trade_date").reset_index(drop=True)
                if len(train_full) < 10000:
                    matrix_results[exp_id] = {"skipped": True, "train_rows": len(train_full)}; continue

                # train/val split for early-stopping (last 15% by date)
                n_val = int(len(train_full) * 0.15)
                tr = train_full.iloc[:-n_val]; va = train_full.iloc[-n_val:]

                t = time.time()
                model = lgb.LGBMClassifier(**LGB_BINARY_PARAMS)
                model.fit(
                    tr[base_cols], tr["y_binary"],
                    eval_set=[(va[base_cols], va["y_binary"])],
                    eval_metric="average_precision",
                    callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
                )
                best_iter = model.best_iteration_
                del train_full, tr, va, joined; gc.collect()

                fam_imp = family_importance(model.booster_, base_cols)
                family_importance_results[exp_id] = fam_imp

                preds = model.predict_proba(upanel[base_cols])[:, 1].astype(np.float32)
                pred_df = upanel[["ts_code","trade_date"]].copy()
                pred_df["score"] = preds

                if panel_name in ("ledashi", "v3unified"):
                    pred_df.to_parquet(OUT_DIR / f"pred_{exp_id}.parquet", compression="zstd")

                adaptive_gating = compute_adaptive_thresholds(pred_df)
                result = eval_cell(pred_df, realized, exits, adaptive_gating)
                result["best_iter"] = best_iter
                if adaptive_gating:
                    result["adaptive_meta"] = {"q25": adaptive_gating["q25"], "q50": adaptive_gating["q50"]}
                matrix_results[exp_id] = result

                r = result["static"]["H2_2025"]
                ic20 = r["fwd20"]["ic"] * 100
                sn50 = r["fwd20"]["sizing"].get("50", {}).get("sharpe_net", float("nan"))
                q1 = result["static"]["Q1_2026"]["fwd20"]["ic"] * 100
                print(f"  {exp_id}: train {time.time()-t:.0f}s (best_iter={best_iter}) | H2 fwd20 IC={ic20:+.3f}% Sharpe50_NET={sn50:+.2f} | Q1 IC={q1:+.3f}%", flush=True)

                del model, preds, pred_df, result; gc.collect()
            del upanel; gc.collect()
            CHECKPOINT.write_text(json.dumps({
                "results": matrix_results, "family_importance": family_importance_results,
            }, indent=2, default=str))
            print(f"[checkpoint] {len(matrix_results)} cells", flush=True)
        del panel; gc.collect()

    out = {
        "config": {"tier": "v10c — LGB binary classifier on wave_v* labels (P75 threshold)",
                   "binary_cutoff_pct": BINARY_CUTOFF_PCT, "lgb_params": LGB_BINARY_PARAMS,
                   "panels": list(PANELS), "universes": list(UNIVERSES), "labels": list(LABELS)},
        "results": matrix_results, "family_importance": family_importance_results,
        "total_time_s": time.time()-t_total,
    }
    Path("data/kronos/outputs/matrix_v10c_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] matrix_v10c_results.json, {len(matrix_results)} cells, {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
