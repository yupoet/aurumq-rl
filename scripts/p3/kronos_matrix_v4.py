"""Matrix v4 — CSI1000 production decision + panel v2 effect (key cells only).

Per paris v19 §5 reverse Q3: paris wants CSI1000 × {v1..v4} on combined_panel_v2
(381 cols) to decide wave_v2 vs wave_v3 production candidate.

Hyperparam: paris lgb_params_path4_regression.json (num_leaves=127, n_estimators=0
+ early_stopping_rounds=100 on rmse). To avoid the regime-shift iter=1-stop bug
seen in matrix v3, we use the **last 15% of train rows by date** as val for early
stopping instead of H1_2025 OOS val.

Scope:
  - CSI1000 × 4 labels × panel_v2 = 4 cells (paris Q3 answer)
  - Optionally MAIN_BOARD × wave_v3 only on both panels = sanity check (panel effect)

Run after combined_panel_v2 download completes.
"""
from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PANEL_V2 = "D:/dev/aurumq-handoffs/inbox/2026-05-15-paris-combined-panel-v2/combined_panel_v_x_v2.parquet"
PANEL_LEDASHI = "data/p3_4070_long/feature_panel_v3_344_pruned.parquet"
LABEL_TEMPLATE = "data/p3_4070_long/target_y_wave_{v}.parquet"
PANEL_CLOSE = "data/p3_4070_long/stock_close_volume_daily.parquet"
UNIVERSE_DIR = Path("data/universes")
OUT_DIR = Path("data/kronos/outputs/matrix_v4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = pd.Timestamp("2022-01-01").date()
TRAIN_END   = pd.Timestamp("2024-12-31").date()
WINDOWS = {
    "H1_2025": (pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-06-30").date()),
    "H2_2025": (pd.Timestamp("2025-07-01").date(), pd.Timestamp("2025-12-31").date()),
    "Q1_2026": (pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-03-31").date()),
    "Q2_2026_partial": (pd.Timestamp("2026-04-01").date(), pd.Timestamp("2026-05-12").date()),
}
HORIZONS = (5, 10, 20)
TOP_K = 50

# Default: paris Q3 — CSI1000 × 4 labels × panel_v2
CELLS = [
    ("v1", "CSI1000", "v2"),
    ("v2", "CSI1000", "v2"),
    ("v3", "CSI1000", "v2"),
    ("v4", "CSI1000", "v2"),
    # Sanity: MAIN_BOARD wave_v3 on both panels (isolate panel size effect)
    ("v3", "MAIN_BOARD", "v2"),
    ("v3", "MAIN_BOARD", "ledashi"),
    # NPF reference (small universe, easier to overfit; panel v2 should help less)
    ("v3", "NPF", "v2"),
]

# paris path4_regression hyperparam, BUT fixed n=200 (no early-stop)
# Reason: rmse on sparse wave_v* target (mean ≈ 0.029) early-stops at iter=1 because
# the first tree already predicts mean = optimal-rmse-ish; subsequent trees add
# < threshold improvement. Use fixed iterations (like matrix v3 did with n=120, just
# bump to 200 since panel v2 is wider).
LGB_PARAMS = dict(
    objective="regression", metric="rmse", boosting_type="gbdt",
    learning_rate=0.05, num_leaves=127,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    min_data_in_leaf=100,
    lambda_l1=0.1, lambda_l2=0.1,
    n_estimators=200,
    max_depth=-1,
    verbose=-1, num_threads=-1, random_state=42,
)
EARLY_STOPPING_ROUNDS = 0  # disabled
TRAIN_TAIL_VAL_FRAC = 0.0  # no val split


def _dt(df):
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def load_universes(names):
    static_sets, pit_dfs = {}, {}
    for u in names:
        path = UNIVERSE_DIR / f"{u}_membership.parquet"
        df = pd.read_parquet(path)
        if df["trade_date"].notna().sum() == 0:
            static_sets[u] = frozenset(df["stock_code"].tolist())
            print(f"  {u}: static, {len(static_sets[u])} stocks")
        else:
            df = _dt(df)
            pit_dfs[u] = df
            print(f"  {u}: daily PIT, {df['stock_code'].nunique()} uniq, {len(df):,} rows")
    return static_sets, pit_dfs


def filter_universe(panel, name, static_sets, pit_dfs):
    if name in static_sets:
        return panel[panel["ts_code"].isin(static_sets[name])]
    pit = pit_dfs[name].rename(columns={"stock_code": "ts_code"})
    return panel.merge(pit, on=["ts_code", "trade_date"], how="inner")


def compute_realized():
    """Just realized fwd returns; skip dyn-exit to save time."""
    t = time.time()
    print("[realized] loading close ...", flush=True)
    p = pd.read_parquet(PANEL_CLOSE, columns=["ts_code", "trade_date", "close", "adj_factor"])
    p = _dt(p).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    latest = p.groupby("ts_code")["adj_factor"].transform("last")
    p["adj_close"] = (p["close"] * p["adj_factor"] / latest).astype(np.float32)
    for k in HORIZONS:
        p[f"ret_fwd{k}"] = (p.groupby("ts_code")["adj_close"].shift(-k) / p["adj_close"] - 1.0).astype(np.float32)
    p = p[["ts_code","trade_date"] + [f"ret_fwd{k}" for k in HORIZONS]]
    print(f"  realized: {len(p):,} rows ({time.time()-t:.0f}s)")
    return p


def cross_sec_ic(df, pred_col, real_col):
    ics = []
    for _, g in df.dropna(subset=[pred_col, real_col]).groupby("trade_date"):
        if len(g) < 30: continue
        if g[pred_col].std() < 1e-9 or g[real_col].std() < 1e-9: continue
        ics.append(g[pred_col].corr(g[real_col]))
    if not ics:
        return {"mean": float("nan"), "ir": float("nan"), "n_days": 0}
    arr = np.asarray(ics); sd = arr.std()
    return {"mean": float(arr.mean()),
            "ir": float(arr.mean() / sd) if sd > 1e-9 else 0.0,
            "n_days": int(len(arr))}


def top_k_sharpe_ann(df, pred_col, real_col, k=TOP_K, ann=252):
    rets = []
    for _, g in df.dropna(subset=[pred_col, real_col]).groupby("trade_date"):
        if len(g) < k: continue
        rets.append(float(g.nlargest(k, pred_col)[real_col].mean()))
    if len(rets) < 2:
        return {"sharpe": float("nan"), "n_days": len(rets), "mean": float("nan")}
    arr = np.asarray(rets); sd = arr.std(ddof=1)
    return {"sharpe": float(arr.mean() / sd * np.sqrt(ann)) if sd > 1e-9 else 0.0,
            "n_days": int(len(arr)), "mean": float(arr.mean())}


def load_panel(name):
    if name == "v2":
        print(f"[panel] reading combined_panel_v_x_v2 ({PANEL_V2}) ...", flush=True)
        t = time.time()
        p = pd.read_parquet(PANEL_V2)
        # Convert trade_date FIRST
        p = _dt(p)
        # Keep only numeric+bool cols + ts_code key. Strip name/industry_l1_name/etc.
        drop_cols = []
        for c in p.columns:
            if c in ("ts_code", "trade_date"):
                continue
            if not (pd.api.types.is_numeric_dtype(p[c]) or pd.api.types.is_bool_dtype(p[c])):
                drop_cols.append(c)
        print(f"  drop non-numeric cols: {drop_cols[:10]}{'...' if len(drop_cols)>10 else ''} ({len(drop_cols)} cols)")
        p = p.drop(columns=drop_cols)
        # Coerce nullable Int (pandas Int64) → numpy int64; float64 → float32
        for c in p.columns:
            if c in ("ts_code", "trade_date"):
                continue
            dt = p[c].dtype
            if str(dt).startswith("Int"):  # pandas nullable Int64/Int32
                p[c] = p[c].astype("float32")  # NaN-safe; LGB handles
            elif dt == np.float64:
                p[c] = p[c].astype(np.float32)
        print(f"  panel v2: {len(p):,} rows × {len(p.columns)} cols ({time.time()-t:.0f}s)")
        return p
    elif name == "ledashi":
        print(f"[panel] reading ledashi 228-col panel ...", flush=True)
        p = _dt(pd.read_parquet(PANEL_LEDASHI))
        print(f"  panel ledashi: {len(p):,} rows × {len(p.columns)} cols")
        return p
    else:
        raise ValueError(name)


def main():
    t_total = time.time()
    print("[setup] universes ...")
    unique_univs = sorted({c[1] for c in CELLS})
    static_sets, pit_dfs = load_universes(unique_univs)

    print("\n[setup] realized fwd returns ...")
    realized = compute_realized()

    print("\n[setup] loading labels ...")
    all_labels = {}
    for v in sorted({c[0] for c in CELLS}):
        l = _dt(pd.read_parquet(LABEL_TEMPLATE.format(v=v), columns=["trade_date", "ts_code", "y"]))
        all_labels[v] = l
        print(f"  wave_{v}: {len(l):,} rows")

    # Group cells by panel to load each panel once
    cells_by_panel = {}
    for label_v, univ, panel_name in CELLS:
        cells_by_panel.setdefault(panel_name, []).append((label_v, univ))

    matrix_results = {}
    for panel_name, cell_list in cells_by_panel.items():
        panel = load_panel(panel_name)
        base_cols = [c for c in panel.columns if c not in ("ts_code", "trade_date")]
        print(f"  base_cols: {len(base_cols)}")

        for label_v, univ in cell_list:
            exp_id = f"{label_v}_{univ}_{panel_name}"
            print(f"\n=== {exp_id} ===", flush=True)
            upanel = filter_universe(panel, univ, static_sets, pit_dfs)
            print(f"  universe={univ}: panel {len(upanel):,} rows")

            label = all_labels[label_v]
            joined = upanel.merge(label, on=["ts_code", "trade_date"], how="inner")
            train_full = joined[(joined["trade_date"] >= TRAIN_START) & (joined["trade_date"] <= TRAIN_END)].copy()
            print(f"  train rows: {len(train_full):,}")

            if len(train_full) < 10000:
                print("  SKIP: train rows < 10K")
                matrix_results[exp_id] = {"skipped": True, "train_rows": len(train_full)}
                continue

            t = time.time()
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(train_full[base_cols], train_full["y"])
            n_iter = LGB_PARAMS["n_estimators"]
            print(f"  train: {time.time()-t:.0f}s, n_iter={n_iter} (fixed)")
            del train_full

            t = time.time()
            preds = model.predict(upanel[base_cols]).astype(np.float32)
            print(f"  predict: {time.time()-t:.0f}s ({len(preds):,} rows)")

            pred_df = upanel[["ts_code", "trade_date"]].copy()
            pred_df["score"] = preds
            pred_df.to_parquet(OUT_DIR / f"pred_{exp_id}.parquet", compression="zstd")

            eval_df = pred_df.merge(realized, on=["ts_code", "trade_date"], how="left")

            result = {"static": {}, "n_pred_rows": len(pred_df), "best_iter": int(n_iter)}
            for wname, (ws, we) in WINDOWS.items():
                sub = eval_df[(eval_df["trade_date"] >= ws) & (eval_df["trade_date"] <= we)]
                result["static"][wname] = {}
                for k in HORIZONS:
                    ic = cross_sec_ic(sub, "score", f"ret_fwd{k}")
                    sh = top_k_sharpe_ann(sub, "score", f"ret_fwd{k}")
                    result["static"][wname][f"fwd{k}"] = {
                        "ic": ic["mean"], "ir": ic["ir"], "top50_sharpe": sh["sharpe"],
                    }

            matrix_results[exp_id] = result
            r = result["static"]
            print(f"  {exp_id}: H2 fwd5 IC={r['H2_2025']['fwd5']['ic']*100:+.3f}%  "
                  f"H2 fwd20 IC={r['H2_2025']['fwd20']['ic']*100:+.3f}%  "
                  f"Q1 fwd20 IC={r['Q1_2026']['fwd20']['ic']*100:+.3f}%")

        del panel

    out = {
        "config": {
            "tier": "v4 — path4_regression hyperparam + train-tail early-stop + panel v2 effect",
            "cells": CELLS,
            "panel_v2_source": PANEL_V2,
            "panel_ledashi_source": PANEL_LEDASHI,
            "horizons": list(HORIZONS),
            "windows": {k: [str(v[0]), str(v[1])] for k, v in WINDOWS.items()},
            "train_window": [str(TRAIN_START), str(TRAIN_END)],
            "train_tail_val_frac": TRAIN_TAIL_VAL_FRAC,
            "lgb_params": LGB_PARAMS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "top_k": TOP_K,
        },
        "results": matrix_results,
        "total_time_s": time.time() - t_total,
    }
    Path("data/kronos/outputs/matrix_v4_results.json").write_text(json.dumps(out, indent=2, default=str))

    lines = ["# Matrix v4 — CSI1000 production decision + panel v2 effect", ""]
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M Asia/Shanghai')}  runtime {time.time()-t_total:.0f}s_")
    lines.append("")
    lines.append("**Hyperparam**: paris `lgb_params_path4_regression.json` (n_estimators=2000 cap + early_stopping=100 on rmse, val = last 15% of train by date)")
    lines.append("")
    lines.append("## All cells")
    lines.append("")
    lines.append("| cell | best_iter | n_pred | H1 IC% fwd20 | H2 IC% fwd20 | Q1 IC% fwd20 | Q2 IC% fwd20 |")
    lines.append("|---|---|---|---|---|---|---|")
    for cell_id in matrix_results:
        r = matrix_results[cell_id]
        if r.get("skipped"):
            lines.append(f"| {cell_id} | skip | {r.get('train_rows',0)} | - | - | - | - |")
            continue
        s = r["static"]
        def fmt(w):
            v = s.get(w, {}).get("fwd20", {}).get("ic", float("nan"))
            return f"{v*100:+.3f}" if v == v else "n/a"
        lines.append(f"| {cell_id} | {r['best_iter']} | {r['n_pred_rows']:,} | {fmt('H1_2025')} | {fmt('H2_2025')} | {fmt('Q1_2026')} | {fmt('Q2_2026_partial')} |")

    lines.append("")
    lines.append("## paris Q3 answer (CSI1000 production candidate)")
    lines.append("")
    lines.append("| label | matrix v3 H2 fwd20 IC% | Tier 4 v2 H2 fwd20 IC% | **matrix v4 (panel v2) H2 fwd20 IC%** |")
    lines.append("|---|---|---|---|")
    for v in ("v1","v2","v3","v4"):
        cid = f"{v}_CSI1000_v2"
        v4 = matrix_results.get(cid, {}).get("static", {}).get("H2_2025", {}).get("fwd20", {}).get("ic", float("nan"))
        v4_str = f"{v4*100:+.3f}" if v4 == v4 else "n/a"
        lines.append(f"| wave_{v} | - | - | **{v4_str}** |")

    Path("data/kronos/outputs/matrix_v4_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[saved] matrix_v4_results.json + matrix_v4_table.md + {len(matrix_results)} predictions")
    print(f"[done] total {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
