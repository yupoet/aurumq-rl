"""Save trained LGB artifacts for paris production integration (Option A).

Re-trains 2 production-candidate cells with model.save_model + feature_cols + lgb_params:
1. A_t5_HARD_TECH_v2_no_phase_c (v11 paradigm 1 sparse binary, paris's recommended Option A)
2. anchor_T3_HARD_TECH_v3unified (v12 paradigm 2, best v12 dual-regime cell)

Output: data/kronos/outputs/v28b_artifacts/{cell_id}/
  ├── model.txt              (LGB Booster save_model format)
  ├── feature_cols.json      (ordered column list)
  ├── lgb_params.json
  ├── train_window_spec.json
  ├── universe_mask.parquet  (ts_code list for HARD_TECH NPF v2.1 Layer 1A core)
  └── pred_full_panel.parquet (predict_proba on eval window)
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
    UNIVERSES, TRAIN_START, TRAIN_END,
    _dt, load_universes, filter_universe, load_panel,
)
from kronos_matrix_v11 import LGB_BINARY_PARAMS as V11_PARAMS, PHASE1_DIR, load_sparse_label
from kronos_matrix_v12 import LGB_BINARY_PARAMS as V12_PARAMS, ANCHOR_DIR_BETA, ANCHOR_DIR_ALPHA, load_anchor_label

OUT_BASE = Path("data/kronos/outputs/v28b_artifacts")
OUT_BASE.mkdir(parents=True, exist_ok=True)


def save_cell(cell_id, panel_name, univ, label_df, lgb_params, source_matrix, static_sets, pit_dfs):
    out_dir = OUT_BASE / cell_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {cell_id} ===")

    panel = load_panel(panel_name)
    base_cols = [c for c in panel.columns if c not in ("ts_code", "trade_date")]
    upanel = filter_universe(panel, univ, static_sets, pit_dfs)
    print(f"  panel rows: {len(panel):,} | upanel: {len(upanel):,} | feats: {len(base_cols)}")

    joined = upanel.merge(label_df, on=["ts_code", "trade_date"], how="inner")
    train = joined[(joined["trade_date"] >= TRAIN_START) & (joined["trade_date"] <= TRAIN_END)]
    print(f"  train rows: {len(train):,} pos_rate: {(train['y']>0).mean():.4f}")

    val_size = max(500, int(len(train) * 0.10))
    val = train.tail(val_size)
    train_fit = train.head(len(train) - val_size)

    t = time.time()
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        train_fit[base_cols], train_fit["y"],
        eval_set=[(val[base_cols], val["y"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best_iter = model.best_iteration_ or lgb_params["n_estimators"]
    print(f"  train {time.time()-t:.0f}s | best_iter={best_iter}")

    # Save model
    model.booster_.save_model(str(out_dir / "model.txt"))
    (out_dir / "feature_cols.json").write_text(json.dumps(base_cols, indent=2))
    (out_dir / "lgb_params.json").write_text(json.dumps(lgb_params, indent=2))
    (out_dir / "train_window_spec.json").write_text(json.dumps({
        "cell_id": cell_id,
        "source_matrix": source_matrix,
        "panel": panel_name,
        "universe": univ,
        "train_window": [str(TRAIN_START), str(TRAIN_END)],
        "best_iter": int(best_iter),
        "n_train_rows": int(len(train)),
        "pos_rate_train": float((train['y'] > 0).mean()),
        "n_features": len(base_cols),
        "lgb_params_summary": {k: v for k, v in lgb_params.items() if k not in ("verbose",)},
    }, indent=2))

    # Universe mask (ts_code list)
    if univ in static_sets:
        mask_df = pd.DataFrame({"ts_code": sorted(static_sets[univ])})
    else:
        mask_df = pit_dfs[univ][["ts_code"]].drop_duplicates().sort_values("ts_code")
    mask_df.to_parquet(out_dir / f"universe_mask_{univ}.parquet", compression="zstd")

    # Predictions on full panel for eval window
    preds = model.predict_proba(upanel[base_cols])[:, 1].astype(np.float32)
    pred_df = upanel[["ts_code", "trade_date"]].copy()
    pred_df["score"] = preds
    pred_df.to_parquet(out_dir / "pred_full_panel.parquet", compression="zstd")
    print(f"  saved to {out_dir}")

    del model, train, train_fit, val, joined, upanel, panel; gc.collect()


def main():
    static_sets, pit_dfs = load_universes(UNIVERSES)

    # 1. A_t5_HARD_TECH_v2_no_phase_c (v11 paradigm 1 sparse binary)
    label_v11 = load_sparse_label("HARD_TECH", "A", "t5")
    save_cell(
        cell_id="v11_A_t5_HARD_TECH_v2_no_phase_c",
        panel_name="v2_no_phase_c",
        univ="HARD_TECH",
        label_df=label_v11,
        lgb_params=V11_PARAMS,
        source_matrix="matrix_v11 (paris sparse 0.8% binary, Phase 1 short labels method A horizon t5)",
        static_sets=static_sets,
        pit_dfs=pit_dfs,
    )

    # 2. v12 paradigm 2 best cell — anchor_T3_HARD_TECH_v3unified (α 5-cond, dual-regime gold)
    label_v12_alpha = load_anchor_label("HARD_TECH", "T3", "alpha")
    save_cell(
        cell_id="v12_alpha_T3_HARD_TECH_v3unified",
        panel_name="v3unified",
        univ="HARD_TECH",
        label_df=label_v12_alpha,
        lgb_params=V12_PARAMS,
        source_matrix="matrix_v12 (paris Phase 2 α 5-condition draft baseline, anchor T-3 horizon)",
        static_sets=static_sets,
        pit_dfs=pit_dfs,
    )

    # 3. v11 A_t5_NPF_ledashi (Option B NPF tab)
    label_v11_npf = load_sparse_label("NPF", "A", "t5")
    save_cell(
        cell_id="v11_A_t5_NPF_ledashi",
        panel_name="ledashi",
        univ="NPF",
        label_df=label_v11_npf,
        lgb_params=V11_PARAMS,
        source_matrix="matrix_v11 (paris sparse 0.8% binary, NPF universe equi-regime cell)",
        static_sets=static_sets,
        pit_dfs=pit_dfs,
    )

    print("\n[done] all 3 artifacts saved to data/kronos/outputs/v28b_artifacts/")


if __name__ == "__main__":
    raise SystemExit(main())
