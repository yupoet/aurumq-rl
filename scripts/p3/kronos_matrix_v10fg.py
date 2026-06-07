"""Matrix v10f + v10g — meta stacker (L1) + hybrid blend (L2) on v10 + v10b + v10c base cells.

After v10/v10b/v10c (training cells) all done, v10f trains label-specific meta
stackers; v10g produces equal-weight blends of top-3 base cells per universe.

v10f: per-label meta = 4 wave_v* × 6 universes = 24 meta cells
v10g: per-universe top-3 hybrid = 6 hybrid cells

Both consume EXISTING per-cell pred_*.parquet outputs from v10/v10b/v10c.
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
    PANELS, UNIVERSES, LABELS,
    TRAIN_START, TRAIN_END, WINDOWS, HORIZONS, K_MAX, TRIGGERS, COST_ROUND_TRIP,
    TOP_K_LIST, ADAPTIVE_PCT, ADAPTIVE_K_CAP, ADAPTIVE_K_FLOOR, ADAPTIVE_Q_LOW, ADAPTIVE_Q_MID,
    _dt, load_universes, filter_universe, compute_realized_and_exits,
    cross_sec_ic, top_k_sharpe_ann_multi, dyn_agg_multi, compute_adaptive_thresholds,
    eval_cell,
)

V10_OUT = Path("data/kronos/outputs/matrix_v10")
V10B_OUT = Path("data/kronos/outputs/matrix_v10b")
LABEL_TEMPLATE = "data/p3_4070_long/target_y_wave_{v}.parquet"
REGIME_PATH = "data/p3_4070/regime_features.parquet"
OUT_F = Path("data/kronos/outputs/matrix_v10f")
OUT_G = Path("data/kronos/outputs/matrix_v10g")
OUT_F.mkdir(parents=True, exist_ok=True)
OUT_G.mkdir(parents=True, exist_ok=True)

META_LGB_PARAMS = dict(
    objective="regression_l2", metric="l2",
    num_leaves=15, learning_rate=0.05,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    min_data_in_leaf=200,
    n_estimators=300, verbosity=-1, seed=42, n_jobs=-1,
)


def load_v10_predictions(label_v, universe):
    """Load all 7-panel × label cell predictions. Returns wide DF with cols per panel."""
    dfs = []
    for panel in PANELS:
        path = V10_OUT / f"pred_{label_v}_{universe}_{panel}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        df = df.rename(columns={"score": f"score_{panel}"})
        dfs.append(df)
    if not dfs:
        return None
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=["trade_date","ts_code"], how="outer")
    return merged


def run_v10f_meta_stacker(label_v, universe, static_sets, pit_dfs, realized, exits):
    """Train meta LGB stacker on 7-panel base scores for (label, universe)."""
    pred_df = load_v10_predictions(label_v, universe)
    if pred_df is None or len(pred_df) < 1000:
        return None
    score_cols = [c for c in pred_df.columns if c.startswith("score_")]
    if len(score_cols) < 3:
        return None

    # Load label + regime
    label = _dt(pd.read_parquet(LABEL_TEMPLATE.format(v=label_v), columns=["trade_date","ts_code","y"]))
    regime = _dt(pd.read_parquet(REGIME_PATH))

    df = pred_df.merge(label, on=["trade_date","ts_code"], how="inner")
    df = df.merge(regime, on="trade_date", how="left")
    df = df.dropna(subset=score_cols + ["y"])

    regime_cols = [c for c in regime.columns if c != "trade_date"]
    feature_cols = score_cols + regime_cols
    train = df[(df["trade_date"]>=TRAIN_START) & (df["trade_date"]<=TRAIN_END)]
    if len(train) < 1000:
        return None

    t = time.time()
    model = lgb.LGBMRegressor(**META_LGB_PARAMS)
    model.fit(train[feature_cols], train["y"])
    preds = model.predict(df[feature_cols]).astype(np.float32)
    out_df = df[["trade_date","ts_code"]].copy()
    out_df["score"] = preds
    out_df.to_parquet(OUT_F / f"meta_{label_v}_{universe}.parquet", compression="zstd")

    adaptive_gating = compute_adaptive_thresholds(out_df)
    result = eval_cell(out_df, realized, exits, adaptive_gating)
    result["train_time"] = time.time() - t
    result["n_bases"] = len(score_cols)
    return result


def run_v10g_hybrid_top3(universe, all_matrix_results, static_sets, pit_dfs, realized, exits):
    """Equal-weight blend of top-3 cells by H2 Sharpe NET for given universe."""
    # Rank all v10/v10b cells by Sharpe NET for this universe
    candidates = []
    for cell_id, r in all_matrix_results.items():
        if not cell_id.endswith(f"_{universe}_" + cell_id.rsplit("_", 1)[-1]):
            continue
        if "_" + universe + "_" not in cell_id: continue
        if r.get("skipped"): continue
        sn = r.get("static", {}).get("H2_2025", {}).get("fwd20", {}).get("sizing", {}).get("50", {}).get("sharpe_net", float("-inf"))
        if sn > float("-inf"):
            candidates.append((sn, cell_id))
    candidates.sort(reverse=True)
    top3 = candidates[:3]
    if len(top3) < 2:
        return None

    # Load top-3 cell predictions, average score
    dfs = []
    base_files = list(V10_OUT.glob("pred_*.parquet")) + list(V10B_OUT.glob("pred_*.parquet"))
    for _, cell_id in top3:
        candidate_path = None
        for p in base_files:
            if p.stem == f"pred_{cell_id}":
                candidate_path = p; break
        if candidate_path is None: continue
        df = pd.read_parquet(candidate_path)
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        dfs.append(df.rename(columns={"score": f"s_{cell_id}"}))
    if len(dfs) < 2: return None

    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on=["trade_date","ts_code"], how="outer")
    score_cols = [c for c in merged.columns if c.startswith("s_")]
    merged["score"] = merged[score_cols].mean(axis=1)
    out_df = merged[["trade_date","ts_code","score"]].dropna()
    out_df.to_parquet(OUT_G / f"hybrid_top3_{universe}.parquet", compression="zstd")

    adaptive_gating = compute_adaptive_thresholds(out_df)
    result = eval_cell(out_df, realized, exits, adaptive_gating)
    result["top3_cells"] = [c for _, c in top3]
    result["n_blend"] = len(dfs)
    return result


def main():
    t_total = time.time()
    print("[setup] universes ..."); static_sets, pit_dfs = load_universes(UNIVERSES)
    print("\n[setup] realized + dyn-exit ..."); realized, exits = compute_realized_and_exits()

    # Load v10 + v10b results for v10g top-3 ranking
    all_results = {}
    for f in [Path("data/kronos/outputs/matrix_v10_results.json"),
              Path("data/kronos/outputs/matrix_v10b_results.json")]:
        if f.exists():
            d = json.loads(f.read_text())
            all_results.update(d.get("results", {}))
    print(f"[loaded] {len(all_results)} cells from v10/v10b for hybrid ranking")

    # v10f: meta stacker per (label, universe)
    print("\n=== v10f meta stacker (per-label, per-universe) ===")
    v10f_results = {}
    for label_v in LABELS:
        for univ in UNIVERSES:
            exp_id = f"meta_{label_v}_{univ}"
            print(f"\n--- {exp_id} ---", flush=True)
            r = run_v10f_meta_stacker(label_v, univ, static_sets, pit_dfs, realized, exits)
            if r is None:
                print(f"  [SKIP] insufficient base preds")
                v10f_results[exp_id] = {"skipped": True}; continue
            v10f_results[exp_id] = r
            ic20 = r["static"]["H2_2025"]["fwd20"]["ic"] * 100
            sn50 = r["static"]["H2_2025"]["fwd20"]["sizing"].get("50", {}).get("sharpe_net", float("nan"))
            print(f"  {exp_id}: H2 fwd20 IC={ic20:+.3f}% Sharpe50_NET={sn50:+.2f} (bases={r['n_bases']})", flush=True)
            gc.collect()

    Path("data/kronos/outputs/matrix_v10f_results.json").write_text(
        json.dumps({"config": "v10f meta LGB stacker per (label, universe)",
                    "results": v10f_results, "meta_params": META_LGB_PARAMS}, indent=2, default=str))

    # v10g: hybrid top-3 blend per universe
    print("\n=== v10g hybrid (top-3 by Sharpe NET, equal-weight) ===")
    v10g_results = {}
    for univ in UNIVERSES:
        exp_id = f"hybrid_top3_{univ}"
        print(f"\n--- {exp_id} ---", flush=True)
        r = run_v10g_hybrid_top3(univ, all_results, static_sets, pit_dfs, realized, exits)
        if r is None:
            v10g_results[exp_id] = {"skipped": True}; continue
        v10g_results[exp_id] = r
        ic20 = r["static"]["H2_2025"]["fwd20"]["ic"] * 100
        sn50 = r["static"]["H2_2025"]["fwd20"]["sizing"].get("50", {}).get("sharpe_net", float("nan"))
        print(f"  {exp_id}: H2 fwd20 IC={ic20:+.3f}% Sharpe50_NET={sn50:+.2f}", flush=True)
        print(f"    top-3 cells: {r['top3_cells']}", flush=True)
        gc.collect()

    Path("data/kronos/outputs/matrix_v10g_results.json").write_text(
        json.dumps({"config": "v10g hybrid top-3 equal-weight blend",
                    "results": v10g_results}, indent=2, default=str))

    print(f"\n[done] v10f+v10g {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
