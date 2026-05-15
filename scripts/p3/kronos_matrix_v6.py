"""Matrix v6 — same 7 cells, on paris R2-A ablation panel (340 cols).

Per paris REPLY v22 §2: R2-A panel = panel_v2_no_phase_c (366 cols) - 26 non-tech
extras (mfp+mf+hm+senti+mkt-extras+sh_holder+north_hk+inst). Tests ledashi v5 §4
hypothesis that these "non-stationary / reverse-causal" cols hurt OOS.

Decision tree (REPLY v22 §1):
  - if v3_MAIN_BOARD H2 fwd20 IC ≥ +4.14% (ledashi 228 baseline) → ledashi hypothesis CONFIRMED
  - if v3_MAIN_BOARD H2 fwd20 IC < +4.14% → paris alpha-pruning hypothesis takes precedence
                                            → paris ships R2-B (drop 108 alpha+gtja extras)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PANEL_R2A = "D:/dev/aurumq-handoffs/inbox/2026-05-15-paris-panel-v2-ablation-r2/combined_panel_v_x_v2_R2A_minus_non_tech.parquet"
PANEL_LEDASHI = "data/p3_4070_long/feature_panel_v3_344_pruned.parquet"
LABEL_TEMPLATE = "data/p3_4070_long/target_y_wave_{v}.parquet"
PANEL_CLOSE = "data/p3_4070_long/stock_close_volume_daily.parquet"
UNIVERSE_DIR = Path("data/universes")
OUT_DIR = Path("data/kronos/outputs/matrix_v6")
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

CELLS = [
    ("v1", "CSI1000", "r2a"),
    ("v2", "CSI1000", "r2a"),
    ("v3", "CSI1000", "r2a"),
    ("v4", "CSI1000", "r2a"),
    ("v3", "MAIN_BOARD", "r2a"),
    ("v3", "MAIN_BOARD", "ledashi"),
    ("v3", "NPF", "r2a"),
]

LGB_PARAMS = dict(
    objective="regression", metric="rmse", boosting_type="gbdt",
    learning_rate=0.05, num_leaves=127,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    min_data_in_leaf=100,
    lambda_l1=0.1, lambda_l2=0.1,
    n_estimators=200, max_depth=-1,
    verbose=-1, num_threads=-1, random_state=42,
)


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
    if name == "r2a":
        print(f"[panel] reading {PANEL_R2A} ...", flush=True)
        t = time.time()
        p = pd.read_parquet(PANEL_R2A)
        p = _dt(p)
        drop_cols = [
            c for c in p.columns
            if c not in ("ts_code", "trade_date")
            and not (pd.api.types.is_numeric_dtype(p[c]) or pd.api.types.is_bool_dtype(p[c]))
        ]
        print(f"  drop non-numeric: {drop_cols[:10]}{'...' if len(drop_cols)>10 else ''} ({len(drop_cols)} cols)")
        p = p.drop(columns=drop_cols)
        for c in p.columns:
            if c in ("ts_code","trade_date"): continue
            dt = p[c].dtype
            if str(dt).startswith("Int"):
                p[c] = p[c].astype("float32")
            elif dt == np.float64:
                p[c] = p[c].astype(np.float32)
        print(f"  panel R2-A: {len(p):,} rows × {len(p.columns)} cols ({time.time()-t:.0f}s)")
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
    unique_univs = sorted({c[1] for c in CELLS})
    static_sets, pit_dfs = load_universes(unique_univs)

    print("\n[setup] realized ...")
    realized = compute_realized()

    print("\n[setup] labels ...")
    all_labels = {}
    for v in sorted({c[0] for c in CELLS}):
        l = _dt(pd.read_parquet(LABEL_TEMPLATE.format(v=v), columns=["trade_date", "ts_code", "y"]))
        all_labels[v] = l
        print(f"  wave_{v}: {len(l):,} rows")

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
                print("  SKIP")
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
                    result["static"][wname][f"fwd{k}"] = {"ic": ic["mean"], "ir": ic["ir"], "top50_sharpe": sh["sharpe"]}

            matrix_results[exp_id] = result
            r = result["static"]
            print(f"  {exp_id}: H1 fwd20={r['H1_2025']['fwd20']['ic']*100:+.3f}%  "
                  f"H2 fwd5={r['H2_2025']['fwd5']['ic']*100:+.3f}%  "
                  f"H2 fwd20={r['H2_2025']['fwd20']['ic']*100:+.3f}%  "
                  f"Q1 fwd20={r['Q1_2026']['fwd20']['ic']*100:+.3f}%")

        del panel

    # Decision verdict for paris R2-A → R2-B routing
    v3_r2a = matrix_results.get("v3_MAIN_BOARD_r2a", {}).get("static", {}).get("H2_2025", {}).get("fwd20", {}).get("ic", float("nan"))
    v3_ledashi = matrix_results.get("v3_MAIN_BOARD_ledashi", {}).get("static", {}).get("H2_2025", {}).get("fwd20", {}).get("ic", float("nan"))
    verdict = ""
    if v3_r2a == v3_r2a and v3_ledashi == v3_ledashi:
        delta = (v3_r2a - v3_ledashi) * 100
        if v3_r2a >= v3_ledashi:
            verdict = f"R2-A_BEATS_LEDASHI Δ=+{delta:.3f}pp → CONFIRM ledashi non-tech-extras hypothesis"
        else:
            verdict = f"R2-A_LOSES_LEDASHI Δ={delta:.3f}pp → paris R2-B needed (drop 108 alpha+gtja extras)"

    out = {
        "config": {
            "tier": "v6 — paris R2-A panel (340 cols, drop 26 non-tech extras: mfp/mf/hm/senti/mkt/sh_holder/north_hk/inst)",
            "cells": CELLS,
            "panel_r2a": PANEL_R2A,
            "panel_ledashi": PANEL_LEDASHI,
            "lgb_params": LGB_PARAMS,
            "decision_verdict": verdict,
        },
        "results": matrix_results,
        "total_time_s": time.time() - t_total,
    }
    Path("data/kronos/outputs/matrix_v6_results.json").write_text(json.dumps(out, indent=2, default=str))

    lines = ["# Matrix v6 — paris R2-A panel ablation (340 cols)", ""]
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M Asia/Shanghai')}  runtime {time.time()-t_total:.0f}s_")
    lines.append("")
    lines.append("**Panel R2-A**: paris combined_panel_v_x_v2_R2A_minus_non_tech (340 cols = panel_v2_no_phase_c 366 - 26 non-tech extras: mfp+mf+hm+senti+mkt-extras+sh_holder+north_hk+inst)")
    lines.append("**Hyperparam**: `n_estimators=200` fixed")
    lines.append("")
    lines.append(f"## DECISION VERDICT: **{verdict}**")
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
    lines.append("## paris decision tree (REPLY v22 §1)")
    lines.append("")
    lines.append("Compare v3_MAIN_BOARD H2 fwd20 IC % across panels:")
    lines.append("")
    lines.append("| panel | n_cols | H2 fwd5 IC% | H2 fwd20 IC% | Q1 fwd20 IC% |")
    lines.append("|---|---|---|---|---|")
    r2a = matrix_results.get("v3_MAIN_BOARD_r2a", {})
    ledashi = matrix_results.get("v3_MAIN_BOARD_ledashi", {})
    def fmt_w(d, w, k="fwd20"):
        v = d.get("static", {}).get(w, {}).get(k, {}).get("ic", float("nan"))
        return f"{v*100:+.3f}" if v == v else "n/a"
    lines.append(f"| paris R2-A (340 cols) | 340 | {fmt_w(r2a,'H2_2025','fwd5')} | {fmt_w(r2a,'H2_2025')} | {fmt_w(r2a,'Q1_2026')} |")
    lines.append(f"| ledashi 228 pruned (matrix v6 baseline) | 228 | {fmt_w(ledashi,'H2_2025','fwd5')} | {fmt_w(ledashi,'H2_2025')} | {fmt_w(ledashi,'Q1_2026')} |")
    lines.append("")
    lines.append("Reference (matrix v4/v5):")
    lines.append("")
    lines.append("| panel | H2 fwd5 IC% | H2 fwd20 IC% | Q1 fwd20 IC% |")
    lines.append("|---|---|---|---|")
    lines.append("| paris v2 (381 cols, with Phase C NULL, matrix v4) | +3.107 | +3.080 | +0.288 |")
    lines.append("| paris v2_no_phase_c (366 cols, matrix v5) | +2.812 | +3.320 | -0.060 |")
    lines.append("")

    Path("data/kronos/outputs/matrix_v6_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[saved] matrix_v6_results.json + matrix_v6_table.md")
    print(f"[done] total {time.time()-t_total:.0f}s")
    print(f"\n========== VERDICT: {verdict} ==========")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
