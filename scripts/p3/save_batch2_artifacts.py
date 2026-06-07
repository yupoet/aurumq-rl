"""Save 5 cell artifacts + bootstrap CI for paris Batch 2 ask.

5 cells (Track 1/4/5/8 primaries + Track 4 backup):
  1. target_y_HARD_TECH_v2_null (v10b lgb_proximity)
  2. target_y_NPF_v3unified (v10b lgb_proximity)
  3. binary_v4_HARD_TECH_v3unified (v10c lgb_binary_dense)
  4. C_t5_HARD_TECH_v2_null (v11 lgb_binary_sparse)
  5. B_t1_CSI500_v2_null (v11 lgb_binary_sparse) — Track 8 NEW

For each cell: train + save model.txt + feature_cols.json + lgb_params.json +
              train_spec.json + universe_mask + pred_full_panel.parquet.

Plus: bootstrap CI for 2 NEW cells (7y_A_t3_NPF + 7y_A_t5_NPF, since
matrix_v11_v12_bootstrap_ci.json doesn't include 7y experiments).
"""
from __future__ import annotations
import os, gc, json, time
from pathlib import Path

_HANDOFF_INBOX = os.environ.get("AURUMQ_HANDOFF_INBOX", "data/handoffs/inbox")
import lightgbm as lgb
import numpy as np
import pandas as pd
import datetime as _dt_mod

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kronos_matrix_v10 import (
    UNIVERSES, TRAIN_START, TRAIN_END, COST_ROUND_TRIP,
    _dt, load_universes, filter_universe, compute_realized_and_exits,
    family_importance, load_panel, eval_cell, PANEL_CLOSE,
)
from kronos_matrix_v11 import LGB_BINARY_PARAMS as V11_PARAMS, PHASE1_DIR, load_sparse_label

OUT_BASE = Path("data/kronos/outputs/batch2_artifacts")
OUT_BASE.mkdir(parents=True, exist_ok=True)

LGB_PARAMS_V10 = dict(  # v10 wave_v* regression hyperparams
    n_estimators=200,
    num_leaves=127,
    learning_rate=0.05,
    min_child_samples=200,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
)

# v10c binary dense LGB params (same as v10c script)
LGB_BINARY_DENSE_PARAMS = dict(
    objective="binary",
    metric="average_precision",
    num_leaves=127,
    learning_rate=0.05,
    n_estimators=200,
    min_child_samples=200,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
)


def save_cell_v10b_proximity(exp_id, panel_name, univ, static_sets, pit_dfs, realized, exits):
    """v10b target_y_{univ}_{panel}: lgb_proximity regression on target_y label."""
    print(f"\n=== {exp_id} (v10b target_y proximity regression) ===")
    panel = load_panel(panel_name)
    base_cols = [c for c in panel.columns if c not in ("ts_code", "trade_date")]
    upanel = filter_universe(panel, univ, static_sets, pit_dfs)
    print(f"  upanel: {len(upanel):,} rows × {len(base_cols)} cols")

    label_path = Path("data/p3_4070_long/target_y.parquet")
    label_df = _dt(pd.read_parquet(label_path, columns=["trade_date", "ts_code", "y"]))
    joined = upanel.merge(label_df, on=["ts_code", "trade_date"], how="inner")
    train = joined[(joined["trade_date"] >= TRAIN_START) & (joined["trade_date"] <= TRAIN_END)]
    print(f"  train rows: {len(train):,}, mean y: {train['y'].mean():.4f}")

    t = time.time()
    model = lgb.LGBMRegressor(**LGB_PARAMS_V10)
    model.fit(train[base_cols], train["y"])
    del train, joined; gc.collect()
    print(f"  train {time.time()-t:.0f}s")

    preds = model.predict(upanel[base_cols]).astype(np.float32)
    pred_df = upanel[["ts_code", "trade_date"]].copy()
    pred_df["score"] = preds

    out_dir = OUT_BASE / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out_dir / "model.txt"))
    (out_dir / "feature_cols.json").write_text(json.dumps(base_cols, indent=2))
    (out_dir / "lgb_params.json").write_text(json.dumps(LGB_PARAMS_V10, indent=2))
    (out_dir / "train_spec.json").write_text(json.dumps({
        "cell_id": exp_id, "source_matrix": "v10b target_y proximity regression",
        "panel": panel_name, "universe": univ,
        "label": "data/p3_4070_long/target_y.parquet (paris primary calibrated proximity)",
        "train_window": [str(TRAIN_START), str(TRAIN_END)],
        "n_features": len(base_cols),
    }, indent=2))
    if univ in static_sets:
        mdf = pd.DataFrame({"ts_code": sorted(static_sets[univ])})
        mdf.to_parquet(out_dir / f"universe_mask_{univ}.parquet", compression="zstd")
    pred_df.to_parquet(out_dir / "pred_full_panel.parquet", compression="zstd")

    result = eval_cell(pred_df, realized, exits, adaptive_gating=None)
    r = result["static"]["H2_2025"]["fwd20"]
    ic = r["ic"] * 100
    sn = r["sizing"].get("50", {}).get("sharpe_net", float("nan"))
    q1 = result["static"]["Q1_2026"]["fwd20"]["ic"] * 100
    print(f"  {exp_id}: H2 IC={ic:+.3f}% Sharpe50_NET={sn:+.2f} | Q1 IC={q1:+.3f}%")
    del model, preds, pred_df, panel, upanel; gc.collect()
    return result


def save_cell_v10c_binary_dense(exp_id, panel_name, univ, label_v, static_sets, pit_dfs, realized, exits):
    """v10c binary_v{N}_{univ}_{panel}: LGB binary dense (P75 ~25% pos) on wave_v{N}."""
    print(f"\n=== {exp_id} (v10c binary dense) ===")
    panel = load_panel(panel_name)
    base_cols = [c for c in panel.columns if c not in ("ts_code", "trade_date")]
    upanel = filter_universe(panel, univ, static_sets, pit_dfs)
    print(f"  upanel: {len(upanel):,} rows × {len(base_cols)} cols")

    label_path = Path(f"data/p3_4070_long/target_y_wave_v{label_v}.parquet")
    label_df = _dt(pd.read_parquet(label_path, columns=["trade_date", "ts_code", "y"]))
    # v10c threshold: global P75 of POSITIVE values across full label set
    pos_y = label_df["y"][label_df["y"] > 0]
    threshold = np.percentile(pos_y, 75) if len(pos_y) > 0 else 0
    label_df["y_binary"] = (label_df["y"] > threshold).astype(np.int8)
    print(f"  binary threshold (P75 of positives): {threshold:.5f}, full pos rate: {label_df['y_binary'].mean():.4f}")
    joined = upanel.merge(label_df[["ts_code", "trade_date", "y_binary"]].rename(columns={"y_binary": "y"}),
                          on=["ts_code", "trade_date"], how="inner")
    train = joined[(joined["trade_date"] >= TRAIN_START) & (joined["trade_date"] <= TRAIN_END)]
    train["y_binary"] = train["y"]
    print(f"  train rows: {len(train):,}, P75 binary pos rate: {(train['y_binary']>0).mean():.4f}")

    val_size = max(500, int(len(train) * 0.10))
    val = train.tail(val_size)
    train_fit = train.head(len(train) - val_size)

    t = time.time()
    model = lgb.LGBMClassifier(**LGB_BINARY_DENSE_PARAMS)
    model.fit(
        train_fit[base_cols], train_fit["y_binary"],
        eval_set=[(val[base_cols], val["y_binary"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best_iter = model.best_iteration_ or LGB_BINARY_DENSE_PARAMS["n_estimators"]
    del train, train_fit, val, joined; gc.collect()
    print(f"  train {time.time()-t:.0f}s, best_iter={best_iter}")

    preds = model.predict_proba(upanel[base_cols])[:, 1].astype(np.float32)
    pred_df = upanel[["ts_code", "trade_date"]].copy()
    pred_df["score"] = preds

    out_dir = OUT_BASE / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out_dir / "model.txt"))
    (out_dir / "feature_cols.json").write_text(json.dumps(base_cols, indent=2))
    (out_dir / "lgb_params.json").write_text(json.dumps(LGB_BINARY_DENSE_PARAMS, indent=2))
    (out_dir / "train_spec.json").write_text(json.dumps({
        "cell_id": exp_id, "source_matrix": "v10c LGB binary dense P75 (~25% pos cross-section threshold)",
        "panel": panel_name, "universe": univ, "label": f"wave_v{label_v} P75 cross-section binary",
        "train_window": [str(TRAIN_START), str(TRAIN_END)], "best_iter": int(best_iter),
        "n_features": len(base_cols),
    }, indent=2))
    if univ in static_sets:
        mdf = pd.DataFrame({"ts_code": sorted(static_sets[univ])})
        mdf.to_parquet(out_dir / f"universe_mask_{univ}.parquet", compression="zstd")
    pred_df.to_parquet(out_dir / "pred_full_panel.parquet", compression="zstd")

    result = eval_cell(pred_df, realized, exits, adaptive_gating=None)
    r = result["static"]["H2_2025"]["fwd20"]
    ic = r["ic"] * 100
    sn = r["sizing"].get("50", {}).get("sharpe_net", float("nan"))
    q1 = result["static"]["Q1_2026"]["fwd20"]["ic"] * 100
    print(f"  {exp_id}: H2 IC={ic:+.3f}% Sharpe50_NET={sn:+.2f} | Q1 IC={q1:+.3f}%")
    del model, preds, pred_df, panel, upanel; gc.collect()
    return result


def save_cell_v11_sparse_binary(exp_id, panel_name, univ, method, horizon, static_sets, pit_dfs, realized, exits):
    """v11 {method}_{horizon}_{univ}_{panel}: LGB binary sparse (paris 0.8% pos)."""
    print(f"\n=== {exp_id} (v11 sparse binary {method}_{horizon}) ===")
    panel = load_panel(panel_name)
    base_cols = [c for c in panel.columns if c not in ("ts_code", "trade_date")]
    upanel = filter_universe(panel, univ, static_sets, pit_dfs)
    print(f"  upanel: {len(upanel):,} rows × {len(base_cols)} cols")

    label_df = load_sparse_label(univ, method, horizon)
    joined = upanel.merge(label_df, on=["ts_code", "trade_date"], how="inner")
    train = joined[(joined["trade_date"] >= TRAIN_START) & (joined["trade_date"] <= TRAIN_END)]
    print(f"  train rows: {len(train):,}, pos rate: {(train['y']>0).mean():.4f}")

    val_size = max(1000, int(len(train) * 0.10))
    val = train.tail(val_size)
    train_fit = train.head(len(train) - val_size)

    t = time.time()
    model = lgb.LGBMClassifier(**V11_PARAMS)
    model.fit(
        train_fit[base_cols], train_fit["y"],
        eval_set=[(val[base_cols], val["y"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best_iter = model.best_iteration_ or V11_PARAMS["n_estimators"]
    del train, train_fit, val, joined; gc.collect()
    print(f"  train {time.time()-t:.0f}s, best_iter={best_iter}")

    preds = model.predict_proba(upanel[base_cols])[:, 1].astype(np.float32)
    pred_df = upanel[["ts_code", "trade_date"]].copy()
    pred_df["score"] = preds

    out_dir = OUT_BASE / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out_dir / "model.txt"))
    (out_dir / "feature_cols.json").write_text(json.dumps(base_cols, indent=2))
    (out_dir / "lgb_params.json").write_text(json.dumps(V11_PARAMS, indent=2))
    (out_dir / "train_spec.json").write_text(json.dumps({
        "cell_id": exp_id, "source_matrix": "v11 paris sparse 0.8% binary",
        "panel": panel_name, "universe": univ, "method": method, "horizon": horizon,
        "label": f"labels_{method}_{horizon}_{univ}_year={{YYYY}}.parquet from paris Phase 1",
        "train_window": [str(TRAIN_START), str(TRAIN_END)], "best_iter": int(best_iter),
        "n_features": len(base_cols),
    }, indent=2))
    if univ in static_sets:
        mdf = pd.DataFrame({"ts_code": sorted(static_sets[univ])})
        mdf.to_parquet(out_dir / f"universe_mask_{univ}.parquet", compression="zstd")
    pred_df.to_parquet(out_dir / "pred_full_panel.parquet", compression="zstd")

    result = eval_cell(pred_df, realized, exits, adaptive_gating=None)
    r = result["static"]["H2_2025"]["fwd20"]
    ic = r["ic"] * 100
    sn = r["sizing"].get("50", {}).get("sharpe_net", float("nan"))
    q1 = result["static"]["Q1_2026"]["fwd20"]["ic"] * 100
    print(f"  {exp_id}: H2 IC={ic:+.3f}% Sharpe50_NET={sn:+.2f} | Q1 IC={q1:+.3f}%")
    del model, preds, pred_df, panel, upanel; gc.collect()
    return result


def block_bootstrap_sharpe(daily_returns, k_horizon, n_iter=1000, block_len=5):
    arr = np.asarray(daily_returns)
    arr_net = arr - COST_ROUND_TRIP
    n = len(arr_net)
    if n < block_len * 4: return float("nan"), float("nan"), float("nan")
    n_blocks = (n + block_len - 1) // block_len
    ann = np.sqrt(252.0 / max(k_horizon, 1))
    sharpes = []
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        start_idxs = rng.integers(0, n - block_len + 1, size=n_blocks)
        resampled = np.concatenate([arr_net[s:s+block_len] for s in start_idxs])[:n]
        sd = resampled.std(ddof=1)
        if sd < 1e-9: continue
        sharpes.append(resampled.mean() / sd * ann)
    if not sharpes: return float("nan"), float("nan"), float("nan")
    arr_sh = np.asarray(sharpes)
    return float(arr_sh.mean()), float(np.percentile(arr_sh, 2.5)), float(np.percentile(arr_sh, 97.5))


def compute_cell_ci(pred_path, realized, top_k=50, horizon=20):
    df = pd.read_parquet(pred_path)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    H2_start = _dt_mod.date(2025, 7, 1); H2_end = _dt_mod.date(2025, 12, 31)
    df = df[(df["trade_date"] >= H2_start) & (df["trade_date"] <= H2_end)]
    eval_df = df.merge(realized, on=["trade_date", "ts_code"], how="inner")
    if "score" not in eval_df.columns: return None
    fwd_col = f"ret_fwd{horizon}"
    daily_rets = []
    for d, g in eval_df.dropna(subset=["score", fwd_col]).groupby("trade_date"):
        if len(g) < top_k: continue
        daily_rets.append(float(g.nlargest(top_k, "score")[fwd_col].mean()))
    if len(daily_rets) < 30: return None
    return block_bootstrap_sharpe(daily_rets, k_horizon=horizon)


def bootstrap_ci_for_cells(cell_pred_paths, realized):
    """Bootstrap CI for given list of (cell_id, pred_path) tuples."""
    out = {}
    for i, (cell_id, pred_path) in enumerate(cell_pred_paths):
        print(f"  [{i+1}/{len(cell_pred_paths)}] {cell_id}", flush=True)
        out[cell_id] = {}
        for top_k in [10, 50]:
            for horizon in [5, 20]:
                key = f"K{top_k}_fwd{horizon}"
                ci = compute_cell_ci(pred_path, realized, top_k=top_k, horizon=horizon)
                if ci is None:
                    out[cell_id][key] = None; continue
                out[cell_id][key] = {
                    "sharpe_net_mean": ci[0], "ci95_low": ci[1],
                    "ci95_high": ci[2], "ci_width": ci[2] - ci[1],
                }
    return out


def main():
    t_total = time.time()
    print("[setup] universes + realized ...")
    static_sets, pit_dfs = load_universes(UNIVERSES)
    realized, exits = compute_realized_and_exits()

    print("\n=== Phase 1: Save 5 cell artifacts ===")

    # 1. target_y_HARD_TECH_v2_null (v10b proximity) — resume skip if done
    if not (OUT_BASE / "target_y_HARD_TECH_v2_null" / "model.txt").exists():
        save_cell_v10b_proximity("target_y_HARD_TECH_v2_null", "v2_null", "HARD_TECH",
                                  static_sets, pit_dfs, realized, exits)
    else:
        print("\n[SKIP target_y_HARD_TECH_v2_null] artifact exists")

    # 2. target_y_NPF_v3unified (v10b proximity)
    if not (OUT_BASE / "target_y_NPF_v3unified" / "model.txt").exists():
        save_cell_v10b_proximity("target_y_NPF_v3unified", "v3unified", "NPF",
                                  static_sets, pit_dfs, realized, exits)
    else:
        print("\n[SKIP target_y_NPF_v3unified] artifact exists")

    # 3. binary_v4_HARD_TECH_v3unified (v10c binary dense)
    # Skip if already done (resume from artifact existence)
    if not (OUT_BASE / "binary_v4_HARD_TECH_v3unified" / "model.txt").exists():
        save_cell_v10c_binary_dense("binary_v4_HARD_TECH_v3unified", "v3unified", "HARD_TECH", "4",
                                     static_sets, pit_dfs, realized, exits)
    else:
        print("\n[SKIP binary_v4_HARD_TECH_v3unified] artifact exists, skipping retrain")

    # 4. C_t5_HARD_TECH_v2_null (v11 sparse binary)
    save_cell_v11_sparse_binary("C_t5_HARD_TECH_v2_null", "v2_null", "HARD_TECH", "C", "t5",
                                 static_sets, pit_dfs, realized, exits)

    # 5. B_t1_CSI500_v2_null (v11 sparse binary, NEW Track 8)
    save_cell_v11_sparse_binary("B_t1_CSI500_v2_null", "v2_null", "CSI500", "B", "t1",
                                 static_sets, pit_dfs, realized, exits)

    print("\n=== Phase 2: Bootstrap CI for 2 NEW 7y NPF cells (others already in existing bootstrap JSONs) ===")
    p = pd.read_parquet(PANEL_CLOSE, columns=["ts_code", "trade_date", "close", "adj_factor"])
    p = _dt(p).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    latest = p.groupby("ts_code")["adj_factor"].transform("last")
    p["adj_close"] = (p["close"] * p["adj_factor"] / latest).astype(np.float32)
    for k in (5, 10, 20):
        p[f"ret_fwd{k}"] = (p.groupby("ts_code")["adj_close"].shift(-k) / p["adj_close"] - 1.0).astype(np.float32)
    realized_for_ci = p[["ts_code", "trade_date"] + [f"ret_fwd{k}" for k in (5, 10, 20)]]
    print(f"  realized: {len(realized_for_ci):,} rows")

    # 5 new artifact preds + 2 NEW 7y NPF cells
    cell_paths = []
    for cell in ["target_y_HARD_TECH_v2_null", "target_y_NPF_v3unified",
                 "binary_v4_HARD_TECH_v3unified", "C_t5_HARD_TECH_v2_null",
                 "B_t1_CSI500_v2_null"]:
        cell_paths.append((cell, OUT_BASE / cell / "pred_full_panel.parquet"))
    # 7y NPF cells (paris asks CI for 7y_A_t3_NPF + 7y_A_t5_NPF specifically)
    v11_7y_npf_dir = Path("data/kronos/outputs/matrix_v11_7y_NPF_NPF_FULL")
    for cell in ["7y_A_t3_NPF_v2_no_phase_c", "7y_A_t5_NPF_v2_no_phase_c"]:
        cell_paths.append((cell, v11_7y_npf_dir / cell / "pred_full_panel.parquet"))

    print(f"\n  Running bootstrap CI on {len(cell_paths)} cells ...")
    ci_results = bootstrap_ci_for_cells(cell_paths, realized_for_ci)

    Path("data/kronos/outputs/matrix_batch2_bootstrap_ci.json").write_text(
        json.dumps({"config": {"n_bootstrap": 1000, "block_length": 5,
                               "cost_round_trip": COST_ROUND_TRIP},
                    "cells_bootstrap": list(ci_results.keys()),
                    "results": ci_results}, indent=2, default=str))

    print(f"\n[saved] batch2_bootstrap_ci.json, {len(ci_results)} cells, total {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
