"""Matrix v10b — paris target_y label (5th label) × 7 panels × 6 universes.

After v10 completes (168 cells × 4 wave_v* labels), v10b adds 42 cells using
paris's PRIMARY proximity label `target_y.parquet`. Same panels/universes/sizing
/dyn-exit pipeline as v10 for apples-to-apples.

Total: 7 × 6 × 1 (target_y) = 42 new cells.
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

# Import everything from kronos_matrix_v10
import sys
sys.path.insert(0, str(Path(__file__).parent))
from kronos_matrix_v10 import (
    PANEL_PATHS, PANELS, UNIVERSES, PANEL_CLOSE, TECH_LINES, UNIVERSE_DIR,
    TRAIN_START, TRAIN_END, WINDOWS, HORIZONS, K_MAX, TRIGGERS, COST_ROUND_TRIP,
    TOP_K_LIST, ADAPTIVE_PCT, ADAPTIVE_K_CAP, ADAPTIVE_K_FLOOR, ADAPTIVE_Q_LOW, ADAPTIVE_Q_MID,
    FAMILY_PREFIXES, LGB_PARAMS,
    _dt, load_universes, filter_universe, compute_realized_and_exits,
    cross_sec_ic, top_k_sharpe_ann_multi, dyn_agg_multi, compute_adaptive_thresholds,
    family_importance, load_panel, eval_cell,
)

LABEL_PATH = "data/p3_4070_long/target_y.parquet"
OUT_DIR = Path("data/kronos/outputs/matrix_v10b")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t_total = time.time()
    print("[setup] universes ...")
    static_sets, pit_dfs = load_universes(UNIVERSES)

    print("\n[setup] realized + dyn-exit ...")
    realized, exits = compute_realized_and_exits()

    print(f"\n[setup] label target_y ...")
    label_df = _dt(pd.read_parquet(LABEL_PATH, columns=["trade_date", "ts_code", "y"]))
    print(f"  target_y: {len(label_df):,} rows, mean={label_df['y'].mean():.4f}, pos_rate={(label_df['y']>0).mean():.3f}")

    matrix_results = {}
    family_importance_results = {}
    CHECKPOINT = Path("data/kronos/outputs/matrix_v10b_checkpoint.json")
    if CHECKPOINT.exists():
        prev = json.loads(CHECKPOINT.read_text())
        matrix_results = prev.get("results", {})
        family_importance_results = prev.get("family_importance", {})
        print(f"[resume] {len(matrix_results)} cells from checkpoint")

    for panel_name in PANELS:
        panel_cells = [f"target_y_{u}_{panel_name}" for u in UNIVERSES]
        if all(c in matrix_results for c in panel_cells):
            print(f"\n[skip panel {panel_name}] all in checkpoint", flush=True)
            continue

        try:
            panel = load_panel(panel_name)
        except Exception as e:
            print(f"  [{panel_name}] LOAD FAILED: {e}")
            continue
        base_cols = [c for c in panel.columns if c not in ("ts_code","trade_date")]
        print(f"\n[panel {panel_name}] loaded, {len(panel):,} rows × {len(base_cols)} feats")

        for univ in UNIVERSES:
            upanel = filter_universe(panel, univ, static_sets, pit_dfs)
            exp_id = f"target_y_{univ}_{panel_name}"
            if exp_id in matrix_results:
                print(f"  [resume skip] {exp_id}", flush=True); continue

            print(f"\n--- {univ} on {panel_name}: {len(upanel):,} rows ---", flush=True)
            joined = upanel.merge(label_df, on=["ts_code","trade_date"], how="inner")
            train = joined[(joined["trade_date"]>=TRAIN_START) & (joined["trade_date"]<=TRAIN_END)]
            if len(train) < 10000:
                print(f"  [SKIP] train {len(train)} < 10K")
                matrix_results[exp_id] = {"skipped": True, "train_rows": len(train)}
                continue

            t = time.time()
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(train[base_cols], train["y"])
            del train, joined; gc.collect()

            fam_imp = family_importance(model.booster_, base_cols)
            family_importance_results[exp_id] = fam_imp

            preds = model.predict(upanel[base_cols]).astype(np.float32)
            pred_df = upanel[["ts_code","trade_date"]].copy()
            pred_df["score"] = preds

            if panel_name in ("ledashi", "v3unified"):
                pred_df.to_parquet(OUT_DIR / f"pred_{exp_id}.parquet", compression="zstd")

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
            CHECKPOINT.write_text(json.dumps({
                "results": matrix_results, "family_importance": family_importance_results,
            }, indent=2, default=str))
        del panel; gc.collect()

    out = {
        "config": {"tier": "v10b — target_y (paris primary proximity) + 7 panel × 6 universe",
                   "label": "target_y.parquet", "panels": list(PANELS), "universes": list(UNIVERSES)},
        "results": matrix_results, "family_importance": family_importance_results,
        "total_time_s": time.time()-t_total,
    }
    Path("data/kronos/outputs/matrix_v10b_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] matrix_v10b_results.json, {len(matrix_results)} cells, {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
