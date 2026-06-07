"""Save 2 P0 cell artifacts for paris REQUEST_v32 (Track 7 + Track 9).

Track 1 already shipped in Batch 2 at
  oss://ledashi-oss/fromsz/handoffs/2026-05-22-ledashi-5-cell-artifacts-batch2/artifacts/target_y_HARD_TECH_v2_null/
  → just re-ACK paris pointing to that bundle.

Track 7: alpha_T3_CSI500_tier4_v2_old (v12 anchor paradigm 2)
Track 9: C_t5_CSI1000_v3unified (v11 sparse binary paradigm 1)
"""
from __future__ import annotations
import os, gc, json, time
from pathlib import Path

_HANDOFF_INBOX = os.environ.get("AURUMQ_HANDOFF_INBOX", "data/handoffs/inbox")
import lightgbm as lgb
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kronos_matrix_v10 import (
    TRAIN_START, TRAIN_END,
    _dt, load_universes, filter_universe, compute_realized_and_exits,
    load_panel, eval_cell,
)
from kronos_matrix_v12 import LGB_BINARY_PARAMS, load_anchor_label
from kronos_matrix_v11 import load_sparse_label

OUT_BASE = Path("data/kronos/outputs/v32_artifacts")
OUT_BASE.mkdir(parents=True, exist_ok=True)


def save_anchor_cell(exp_id, panel_name, univ, anchor_T, spec,
                     static_sets, pit_dfs, realized, exits):
    """Track 7 — v12 anchor alpha cell."""
    print(f"\n=== {exp_id} (v12 anchor {spec} {anchor_T}) ===")
    panel = load_panel(panel_name)
    base_cols = [c for c in panel.columns if c not in ("ts_code", "trade_date")]
    upanel = filter_universe(panel, univ, static_sets, pit_dfs)
    print(f"  upanel: {len(upanel):,} rows × {len(base_cols)} cols")

    label_df = load_anchor_label(univ, anchor_T, spec)
    if label_df is None or len(label_df) == 0:
        raise RuntimeError(f"no labels for {exp_id}")

    joined = upanel.merge(label_df, on=["ts_code", "trade_date"], how="inner")
    train = joined[(joined["trade_date"] >= TRAIN_START) & (joined["trade_date"] <= TRAIN_END)]
    print(f"  train rows: {len(train):,}, pos rate: {(train['y']>0).mean():.4f}")
    if len(train) < 1000:
        raise RuntimeError(f"{exp_id} train rows {len(train)} < 1000")

    val_size = max(500, int(len(train) * 0.10))
    val = train.tail(val_size)
    train_fit = train.head(len(train) - val_size)

    t = time.time()
    model = lgb.LGBMClassifier(**LGB_BINARY_PARAMS)
    model.fit(
        train_fit[base_cols], train_fit["y"],
        eval_set=[(val[base_cols], val["y"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best_iter = model.best_iteration_ or LGB_BINARY_PARAMS["n_estimators"]
    del train, train_fit, val, joined; gc.collect()
    print(f"  train {time.time()-t:.0f}s, best_iter={best_iter}")

    preds = model.predict_proba(upanel[base_cols])[:, 1].astype(np.float32)
    pred_df = upanel[["ts_code", "trade_date"]].copy()
    pred_df["score"] = preds

    out_dir = OUT_BASE / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out_dir / "model.txt"))
    (out_dir / "feature_cols.json").write_text(json.dumps(base_cols, indent=2))
    (out_dir / "lgb_params.json").write_text(json.dumps(LGB_BINARY_PARAMS, indent=2))
    (out_dir / "train_window_spec.json").write_text(json.dumps({
        "cell_id": exp_id, "source_matrix": "v12 paradigm 2 anchor (paris α 5-cond baseline)",
        "panel": panel_name, "universe": univ, "anchor_horizon": anchor_T, "spec": spec,
        "label_source": f"inbox/2026-05-18-paris-anchor-alpha-fix/{spec}_5cond_baseline/anchor_{anchor_T}_{univ}_year=*.parquet",
        "train_window": [str(TRAIN_START), str(TRAIN_END)],
        "val_size": int(val_size), "best_iter": int(best_iter),
        "n_features": len(base_cols),
    }, indent=2))
    if univ in static_sets:
        mdf = pd.DataFrame({"ts_code": sorted(static_sets[univ])})
        mdf.to_parquet(out_dir / f"universe_mask_{univ}.parquet", compression="zstd")
    else:
        # PIT — save full membership snapshot
        pit_dfs[univ].to_parquet(out_dir / f"universe_mask_{univ}.parquet", compression="zstd")
    pred_df.to_parquet(out_dir / "pred_full_panel.parquet", compression="zstd")

    result = eval_cell(pred_df, realized, exits, adaptive_gating=None)
    r = result["static"]["H2_2025"]["fwd20"]
    q1 = result["static"]["Q1_2026"]["fwd20"]
    ic_h2 = r["ic"] * 100
    sn_h2 = r["sizing"].get("50", {}).get("sharpe_net", float("nan"))
    ic_q1 = q1["ic"] * 100
    print(f"  {exp_id}: H2 IC={ic_h2:+.3f}% Sharpe50_NET={sn_h2:+.2f} | Q1 IC={ic_q1:+.3f}%")
    del model, preds, pred_df, panel, upanel; gc.collect()
    return result


def save_sparse_binary_cell(exp_id, panel_name, univ, method, horizon,
                              static_sets, pit_dfs, realized, exits):
    """Track 9 — v11 sparse binary (paris 0.8% pos)."""
    print(f"\n=== {exp_id} (v11 sparse binary {method}_{horizon}) ===")
    panel = load_panel(panel_name)
    base_cols = [c for c in panel.columns if c not in ("ts_code", "trade_date")]
    upanel = filter_universe(panel, univ, static_sets, pit_dfs)
    print(f"  upanel: {len(upanel):,} rows × {len(base_cols)} cols")

    label_df = load_sparse_label(univ, method, horizon)
    if label_df is None or len(label_df) == 0:
        raise RuntimeError(f"no labels for {exp_id}")
    joined = upanel.merge(label_df, on=["ts_code", "trade_date"], how="inner")
    train = joined[(joined["trade_date"] >= TRAIN_START) & (joined["trade_date"] <= TRAIN_END)]
    print(f"  train rows: {len(train):,}, pos rate: {(train['y']>0).mean():.4f}")

    val_size = max(1000, int(len(train) * 0.10))
    val = train.tail(val_size)
    train_fit = train.head(len(train) - val_size)

    t = time.time()
    model = lgb.LGBMClassifier(**LGB_BINARY_PARAMS)
    model.fit(
        train_fit[base_cols], train_fit["y"],
        eval_set=[(val[base_cols], val["y"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best_iter = model.best_iteration_ or LGB_BINARY_PARAMS["n_estimators"]
    del train, train_fit, val, joined; gc.collect()
    print(f"  train {time.time()-t:.0f}s, best_iter={best_iter}")

    preds = model.predict_proba(upanel[base_cols])[:, 1].astype(np.float32)
    pred_df = upanel[["ts_code", "trade_date"]].copy()
    pred_df["score"] = preds

    out_dir = OUT_BASE / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out_dir / "model.txt"))
    (out_dir / "feature_cols.json").write_text(json.dumps(base_cols, indent=2))
    (out_dir / "lgb_params.json").write_text(json.dumps(LGB_BINARY_PARAMS, indent=2))
    (out_dir / "train_window_spec.json").write_text(json.dumps({
        "cell_id": exp_id, "source_matrix": "v11 paradigm 1 paris sparse 0.8% binary",
        "panel": panel_name, "universe": univ, "method": method, "horizon": horizon,
        "label_source": f"paris Phase 1 sparse labels labels_{method}_{horizon}_{univ}_year=*.parquet",
        "train_window": [str(TRAIN_START), str(TRAIN_END)],
        "val_size": int(val_size), "best_iter": int(best_iter),
        "n_features": len(base_cols),
    }, indent=2))
    if univ in static_sets:
        mdf = pd.DataFrame({"ts_code": sorted(static_sets[univ])})
        mdf.to_parquet(out_dir / f"universe_mask_{univ}.parquet", compression="zstd")
    else:
        pit_dfs[univ].to_parquet(out_dir / f"universe_mask_{univ}.parquet", compression="zstd")
    pred_df.to_parquet(out_dir / "pred_full_panel.parquet", compression="zstd")

    result = eval_cell(pred_df, realized, exits, adaptive_gating=None)
    r = result["static"]["H2_2025"]["fwd20"]
    q1 = result["static"]["Q1_2026"]["fwd20"]
    ic_h2 = r["ic"] * 100
    sn_h2 = r["sizing"].get("50", {}).get("sharpe_net", float("nan"))
    ic_q1 = q1["ic"] * 100
    print(f"  {exp_id}: H2 IC={ic_h2:+.3f}% Sharpe50_NET={sn_h2:+.2f} | Q1 IC={ic_q1:+.3f}%")
    del model, preds, pred_df, panel, upanel; gc.collect()
    return result


def main():
    t_total = time.time()
    print("[setup] universes ...")
    static_sets, pit_dfs = load_universes(("MAIN_BOARD", "CSI500", "CSI1000", "NPF", "NPF_FULL", "HARD_TECH"))
    print("\n[setup] realized + dyn-exit ...")
    realized, exits = compute_realized_and_exits()

    results = {}

    # Track 7: anchor_T3_CSI500 on tier4_v2_old panel (paris α 5-cond baseline)
    results["alpha_T3_CSI500_tier4_v2_old"] = save_anchor_cell(
        "alpha_T3_CSI500_tier4_v2_old",
        panel_name="tier4_v2_old",
        univ="CSI500",
        anchor_T="T3", spec="alpha",
        static_sets=static_sets, pit_dfs=pit_dfs,
        realized=realized, exits=exits,
    )

    # Track 9: C_t5_CSI1000_v3unified (sparse binary, method C, horizon t5)
    results["C_t5_CSI1000_v3unified"] = save_sparse_binary_cell(
        "C_t5_CSI1000_v3unified",
        panel_name="v3unified", univ="CSI1000",
        method="C", horizon="t5",
        static_sets=static_sets, pit_dfs=pit_dfs,
        realized=realized, exits=exits,
    )

    Path("data/kronos/outputs/v32_artifacts_results.json").write_text(json.dumps({
        "config": {"request_id": "v32", "cells": ["alpha_T3_CSI500_tier4_v2_old", "C_t5_CSI1000_v3unified"]},
        "results": results, "total_time_s": time.time() - t_total,
    }, indent=2, default=str))
    print(f"\n[done] v32 artifacts in {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
