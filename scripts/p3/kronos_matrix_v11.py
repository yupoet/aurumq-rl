"""Matrix v11 — paris sparse binary labels (Phase 1 short_labels) apples-to-apples.

paris production wave_binary uses global static τ from train-window search with
target_pos_rate=0.008 (0.8% positive). ledashi v10c used cross-section P75 (~25%
positive per-day). 30x positive rate gap, label distributions completely different.

v11 fires: 7 panel × 6 universe × 4 method (A/B/C/D) × 3 horizon (t1/t3/t5)
= 504 cells using paris production sparse threshold labels from Phase 1.

Labels: `data/handoffs/inbox/2026-05-16-paris-reply-v24/labels_{X}_{t1|t3|t5}_{univ}_year={YYYY}.parquet`
        × 4 method × 3 horizon × 6 universe × 5 year ≈ 360 parquets, concat per (univ, method, horizon).

Train: 2022-01-01 ~ 2024-12-31 (paris production train window for short paths)
Eval: H2_2025 + Q1_2026 + Q2_2026_partial (apples-to-apples with v10/v10b/v10c)

LGB params: paris-aligned binary classifier (objective=binary, metric=average_precision,
            num_leaves=63, n=500, early_stop=50).
"""
from __future__ import annotations

import os
import gc
import json
import time
from pathlib import Path
from glob import glob

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
    FAMILY_PREFIXES,
    _dt, load_universes, filter_universe, compute_realized_and_exits,
    cross_sec_ic, top_k_sharpe_ann_multi, dyn_agg_multi, compute_adaptive_thresholds,
    family_importance, load_panel, eval_cell,
)

# paris production wave_binary hyperparams
LGB_BINARY_PARAMS = dict(
    objective="binary",
    metric="average_precision",
    num_leaves=63,
    learning_rate=0.05,
    n_estimators=500,
    min_child_samples=200,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
)

PHASE1_DIR = Path(f"{_HANDOFF_INBOX}/2026-05-16-paris-reply-v24")
METHODS = ("A", "B", "C", "D")
HORIZONS_SPARSE = ("t1", "t3", "t5")
YEARS = ("2022", "2023", "2024", "2025", "2026")

OUT_DIR = Path("data/kronos/outputs/matrix_v11")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sparse_label(univ: str, method: str, horizon: str) -> pd.DataFrame | None:
    """Concat all years of paris Phase 1 sparse binary labels for (univ, method, horizon)."""
    dfs = []
    for year in YEARS:
        f = PHASE1_DIR / f"labels_{method}_{horizon}_{univ}_year={year}.parquet"
        if f.exists():
            df = pd.read_parquet(f, columns=["trade_date", "ts_code", "y"])
            dfs.append(df)
    if not dfs: return None
    out = pd.concat(dfs, ignore_index=True)
    out = _dt(out)
    return out


def eval_binary_cell(pred_df, realized, exits):
    """Same as v10c eval_cell but for binary classifier (pred_df has 'score' column = prob)."""
    return eval_cell(pred_df, realized, exits, adaptive_gating=None)


def main():
    t_total = time.time()
    print("[setup] universes ...")
    static_sets, pit_dfs = load_universes(UNIVERSES)

    print("\n[setup] realized + dyn-exit ...")
    realized, exits = compute_realized_and_exits()

    print(f"\n[setup] verify Phase 1 labels inbox: {PHASE1_DIR}")
    sample = PHASE1_DIR / "labels_A_t3_MAIN_BOARD_year=2024.parquet"
    if not sample.exists():
        print(f"  ERROR: Phase 1 labels missing at {sample}")
        return 1
    print(f"  Phase 1 labels OK")

    matrix_results = {}
    family_importance_results = {}
    CHECKPOINT = Path("data/kronos/outputs/matrix_v11_checkpoint.json")
    if CHECKPOINT.exists():
        prev = json.loads(CHECKPOINT.read_text())
        matrix_results = prev.get("results", {})
        family_importance_results = prev.get("family_importance", {})
        print(f"[resume] {len(matrix_results)} cells from checkpoint")

    total_cells = len(PANELS) * len(UNIVERSES) * len(METHODS) * len(HORIZONS_SPARSE)
    print(f"\n[matrix] {len(PANELS)} panels × {len(UNIVERSES)} univ × {len(METHODS)} method × {len(HORIZONS_SPARSE)} horizon = {total_cells} cells")

    for panel_name in PANELS:
        panel_cells = [f"{m}_{h}_{u}_{panel_name}"
                       for u in UNIVERSES for m in METHODS for h in HORIZONS_SPARSE]
        if all(c in matrix_results for c in panel_cells):
            print(f"\n[skip panel {panel_name}] all in checkpoint"); continue

        try: panel = load_panel(panel_name)
        except Exception as e:
            print(f"  [{panel_name}] LOAD FAILED: {e}"); continue
        base_cols = [c for c in panel.columns if c not in ("ts_code","trade_date")]
        print(f"\n[panel {panel_name}] {len(panel):,} rows × {len(base_cols)} feats", flush=True)

        for univ in UNIVERSES:
            upanel = filter_universe(panel, univ, static_sets, pit_dfs)

            for method in METHODS:
                for horizon in HORIZONS_SPARSE:
                    exp_id = f"{method}_{horizon}_{univ}_{panel_name}"
                    if exp_id in matrix_results:
                        continue

                    label_df = load_sparse_label(univ, method, horizon)
                    if label_df is None or len(label_df) == 0:
                        print(f"  [SKIP] no labels for {exp_id}"); continue

                    joined = upanel.merge(label_df, on=["ts_code", "trade_date"], how="inner")
                    train = joined[(joined["trade_date"] >= TRAIN_START) & (joined["trade_date"] <= TRAIN_END)]
                    if len(train) < 10000:
                        print(f"  [SKIP] {exp_id} train {len(train)} < 10K")
                        matrix_results[exp_id] = {"skipped": True, "train_rows": len(train)}
                        continue

                    pos_rate_train = (train["y"] > 0).mean()
                    if pos_rate_train < 0.001 or pos_rate_train > 0.05:
                        print(f"  [WARN] {exp_id} train pos_rate={pos_rate_train:.4f} outside [0.001, 0.05]")

                    t = time.time()
                    val_size = max(1000, int(len(train) * 0.10))
                    val = train.tail(val_size)
                    train_fit = train.head(len(train) - val_size)

                    model = lgb.LGBMClassifier(**LGB_BINARY_PARAMS)
                    model.fit(
                        train_fit[base_cols], train_fit["y"],
                        eval_set=[(val[base_cols], val["y"])],
                        callbacks=[lgb.early_stopping(50, verbose=False)],
                    )
                    best_iter = model.best_iteration_ or LGB_BINARY_PARAMS["n_estimators"]
                    del train, train_fit, val, joined; gc.collect()

                    fam_imp = family_importance(model.booster_, base_cols)
                    family_importance_results[exp_id] = fam_imp

                    preds = model.predict_proba(upanel[base_cols])[:, 1].astype(np.float32)
                    pred_df = upanel[["ts_code", "trade_date"]].copy()
                    pred_df["score"] = preds

                    if panel_name in ("ledashi", "v3unified"):
                        pred_df.to_parquet(OUT_DIR / f"pred_{exp_id}.parquet", compression="zstd")

                    result = eval_binary_cell(pred_df, realized, exits)
                    matrix_results[exp_id] = result

                    r = result["static"]["H2_2025"]
                    ic20 = r["fwd20"]["ic"] * 100
                    sn50 = r["fwd20"]["sizing"].get("50", {}).get("sharpe_net", float("nan"))
                    q1 = result["static"]["Q1_2026"]["fwd20"]["ic"] * 100
                    print(f"  {exp_id}: train {time.time()-t:.0f}s (best_iter={best_iter}) | H2 fwd20 IC={ic20:+.3f}% Sharpe50_NET={sn50:+.2f} | Q1 IC={q1:+.3f}%", flush=True)

                    del model, preds, pred_df, result; gc.collect()
                    CHECKPOINT.write_text(json.dumps({
                        "results": matrix_results, "family_importance": family_importance_results,
                    }, indent=2, default=str))
            del upanel; gc.collect()
        del panel; gc.collect()

    out = {
        "config": {"tier": "v11 — paris sparse binary (Phase 1 short_labels)",
                   "panels": list(PANELS), "universes": list(UNIVERSES),
                   "methods": list(METHODS), "horizons": list(HORIZONS_SPARSE),
                   "lgb_params": LGB_BINARY_PARAMS},
        "results": matrix_results, "family_importance": family_importance_results,
        "total_time_s": time.time() - t_total,
    }
    Path("data/kronos/outputs/matrix_v11_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] matrix_v11_results.json, {len(matrix_results)} cells, {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
