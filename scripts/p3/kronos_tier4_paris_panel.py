"""Tier 4 — paris 376-col combined_panel ablation (mirror of matrix v2).

Same 8 universes × 4 labels × 2022+ window × panel_only, but using paris's
376-col combined_panel (vs matrix v2's 228-col pruned panel).

Output: matrix_tier4 cells directly comparable to matrix_v2 cells.

Cell delta = paris panel (149 extra cols incl phase_c) vs my panel.

Per paris v15.2 caveat: phase_c 13 cols fill is low (dc_hot ~2.3%, tdx ~0.1%)
in v1 combined_panel. paris will re-ship v2 after 5/16-5/18 backfill. We run on
v1 now (fillna(0) for these cols + LGB natively handles NaN).
"""
from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PANEL = "D:/dev/aurumq-handoffs/inbox/2026-05-17-paris-combined-panel/combined_panel_v_x.parquet"
LABEL_TEMPLATE = "data/p3_4070_long/target_y_wave_{v}.parquet"
PANEL_CLOSE = "data/p3_4070_long/stock_close_volume_daily.parquet"
TECH_LINES = "D:/dev/aurumq-handoffs/inbox/2026-05-14-paris-macd-kdj-raw/tech_lines_daily.parquet"
UNIVERSE_DIR = Path("data/universes")
OUT_DIR = Path("data/kronos/outputs/tier4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = pd.Timestamp("2022-01-01").date()
TRAIN_END   = pd.Timestamp("2024-12-31").date()
WINDOWS = {
    "H1_2025": (pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-06-30").date()),
    "H2_2025": (pd.Timestamp("2025-07-01").date(), pd.Timestamp("2025-12-31").date()),
    "Q1_2026": (pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-03-31").date()),
    "Q2_2026_partial": (pd.Timestamp("2026-04-01").date(), pd.Timestamp("2026-05-12").date()),
}
HORIZONS = (1, 3, 5, 10, 20, 30)
K_MAX = 30
TOP_K = 50

LABELS = ("v1", "v2", "v3", "v4")
UNIVERSES = ("MAIN_BOARD", "NPF", "NPF_FULL", "NPF_CROSS_BOARD",
             "HARD_TECH", "CSI300", "CSI500", "CSI1000")
TRIGGERS = ("A_stop_5pct", "C_vol_drop", "E_trail_5pct", "F_trend_break",
            "G_K_max", "H_macd_death", "I_kdj_death", "S_ma5_below_ma10",
            "J_take_profit_5", "J_take_profit_10", "Q_OR_FIE")

# Non-feature cols in paris combined_panel
NON_FEATURE_COLS = {"ts_code", "trade_date", "name"}

# Phase C cols (low fill rate per paris v15.2), LGB natively handles NaN
PHASE_C_PREFIXES = ("dc_hot_", "tdx_", "concept_", "stk_shock_")

LGB_PARAMS = dict(
    objective="regression", metric="rmse",
    learning_rate=0.05, num_leaves=63,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    min_data_in_leaf=200, verbose=-1, n_estimators=400, num_threads=-1,
)

CAT_COLS: list[str] = []  # populated after panel load (industry_code etc)


def _dt(df):
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def load_universes() -> tuple[dict, dict]:
    static_sets, pit_dfs = {}, {}
    for u in UNIVERSES:
        path = UNIVERSE_DIR / f"{u}_membership.parquet"
        df = pd.read_parquet(path)
        if df["trade_date"].notna().sum() == 0:
            static_sets[u] = frozenset(df["stock_code"].tolist())
            print(f"  {u}: static, {len(static_sets[u])} stocks")
        else:
            df = _dt(df)
            pit_dfs[u] = df
            print(f"  {u}: PIT, {df['stock_code'].nunique()} uniq, {len(df):,} rows")
    return static_sets, pit_dfs


def filter_universe(panel, name, static_sets, pit_dfs):
    if name in static_sets:
        return panel[panel["ts_code"].isin(static_sets[name])]
    pit = pit_dfs[name].rename(columns={"stock_code": "ts_code"})
    return panel.merge(pit, on=["ts_code", "trade_date"], how="inner")


def compute_realized_and_sim():
    """Realized fwdK + dyn-exit sim (same as matrix v2)."""
    t = time.time()
    print("[realized] loading close+vol ...", flush=True)
    p = pd.read_parquet(PANEL_CLOSE, columns=["ts_code", "trade_date", "close", "adj_factor", "volume"])
    p = _dt(p).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    latest = p.groupby("ts_code")["adj_factor"].transform("last")
    p["adj_close"] = (p["close"] * p["adj_factor"] / latest).astype(np.float32)
    p["vol"] = p["volume"].astype(np.float32)
    p["pct_chg"] = (p.groupby("ts_code")["adj_close"].pct_change()).astype(np.float32)
    p["ma20_vol"] = p.groupby("ts_code")["vol"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()).astype(np.float32)
    p["ma5_close"] = p.groupby("ts_code")["adj_close"].transform(
        lambda x: x.rolling(5, min_periods=2).mean()).astype(np.float32)
    p["ma10_close"] = p.groupby("ts_code")["adj_close"].transform(
        lambda x: x.rolling(10, min_periods=3).mean()).astype(np.float32)

    print("[realized] computing fwd-K ...", flush=True)
    for k in HORIZONS:
        p[f"ret_fwd{k}"] = (p.groupby("ts_code")["adj_close"].shift(-k) / p["adj_close"] - 1.0).astype(np.float32)

    print("[realized] merging tech_lines ...", flush=True)
    tech = _dt(pd.read_parquet(TECH_LINES))
    p = p.merge(tech, on=["ts_code", "trade_date"], how="left")
    del tech

    realized = p[["ts_code", "trade_date"] + [f"ret_fwd{k}" for k in HORIZONS]].copy()
    print(f"  realized: {len(realized):,}  ({time.time()-t:.0f}s)")

    print("[dyn-exit] simulating 11 triggers per-stock ...", flush=True)
    exit_rows = []
    n_syms = 0
    for sym, sub in p.groupby("ts_code"):
        sub = sub.reset_index(drop=True)
        adj_close = sub["adj_close"].to_numpy()
        vol = sub["vol"].to_numpy()
        pct_chg = sub["pct_chg"].to_numpy()
        ma20_vol = sub["ma20_vol"].to_numpy()
        ma5_close = sub["ma5_close"].to_numpy()
        ma10_close = sub["ma10_close"].to_numpy()
        macd_line = sub["macd_line"].to_numpy() if "macd_line" in sub.columns else np.full(len(sub), np.nan)
        macd_signal = sub["macd_signal"].to_numpy() if "macd_signal" in sub.columns else np.full(len(sub), np.nan)
        kdj_k = sub["kdj_k"].to_numpy() if "kdj_k" in sub.columns else np.full(len(sub), np.nan)
        kdj_d = sub["kdj_d"].to_numpy() if "kdj_d" in sub.columns else np.full(len(sub), np.nan)
        n = len(adj_close)
        if n < K_MAX + 5:
            continue

        with np.errstate(invalid="ignore"):
            C_abs = (vol > 2.0 * ma20_vol) & (pct_chg < -0.03)
            F_abs = adj_close < ma5_close
        C_abs = np.nan_to_num(C_abs, nan=False).astype(bool)
        F_abs = np.nan_to_num(F_abs, nan=False).astype(bool)

        S_abs = np.zeros(n, dtype=bool)
        with np.errstate(invalid="ignore"):
            ma_above = ma5_close > ma10_close
        ma_above_safe = np.nan_to_num(ma_above, nan=False)
        S_abs[1:] = ma_above_safe[:-1] & ~ma_above_safe[1:]

        H_abs = np.zeros(n, dtype=bool)
        if not np.all(np.isnan(macd_line)):
            H_abs[1:] = (macd_line[:-1] > macd_signal[:-1]) & (macd_line[1:] <= macd_signal[1:])
            H_abs = H_abs & ~np.isnan(macd_line) & ~np.isnan(macd_signal)

        I_abs = np.zeros(n, dtype=bool)
        if not np.all(np.isnan(kdj_k)):
            I_abs[1:] = (kdj_k[:-1] > kdj_d[:-1]) & (kdj_k[1:] <= kdj_d[1:])
            I_abs = I_abs & ~np.isnan(kdj_k) & ~np.isnan(kdj_d)

        max_entry = n - K_MAX - 1
        if max_entry < 0:
            continue
        entries = np.arange(0, max_entry + 1)
        fwd_idx = entries[:, None] + np.arange(1, K_MAX + 1)[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            cum_ret = adj_close[fwd_idx] / adj_close[entries][:, None] - 1.0
        cum_ret = np.where(np.isfinite(cum_ret), cum_ret, 0.0)
        peak_ret = np.maximum.accumulate(cum_ret, axis=1)

        A_f = cum_ret < -0.05
        C_f = C_abs[fwd_idx]
        E_f = (peak_ret - cum_ret) > 0.05
        F_f = F_abs[fwd_idx]
        G_f = np.zeros_like(A_f)
        H_f = H_abs[fwd_idx]
        I_f = I_abs[fwd_idx]
        S_f = S_abs[fwd_idx]
        J5_f = cum_ret > 0.05
        J10_f = cum_ret > 0.10
        Q_f = F_f | I_f | E_f

        firmap = {
            "A_stop_5pct": A_f, "C_vol_drop": C_f, "E_trail_5pct": E_f,
            "F_trend_break": F_f, "G_K_max": G_f, "H_macd_death": H_f,
            "I_kdj_death": I_f, "S_ma5_below_ma10": S_f,
            "J_take_profit_5": J5_f, "J_take_profit_10": J10_f, "Q_OR_FIE": Q_f,
        }
        dates = sub["trade_date"].values[entries]
        rec = {"ts_code": np.full(len(entries), sym, dtype=object), "trade_date": dates}
        idx = np.arange(len(entries))
        for trig, fired in firmap.items():
            any_fired = fired.any(axis=1)
            first = np.where(any_fired, fired.argmax(axis=1), K_MAX - 1)
            rec[f"{trig}_ret"] = cum_ret[idx, first].astype(np.float32)
            rec[f"{trig}_hold"] = (first + 1).astype(np.int8)
            rec[f"{trig}_fired"] = any_fired
        exit_rows.append(pd.DataFrame(rec))
        n_syms += 1
        if n_syms % 500 == 0:
            print(f"  ... {n_syms} stocks ({time.time()-t:.0f}s)", flush=True)

    exits = pd.concat(exit_rows, ignore_index=True)
    print(f"  exits: {len(exits):,} rows from {n_syms} stocks ({time.time()-t:.0f}s)")
    del p
    return realized, exits


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


def dyn_agg(df, score_col, trigger):
    rc = f"{trigger}_ret"; hc = f"{trigger}_hold"; tc = f"{trigger}_fired"
    sub = df.dropna(subset=[score_col, rc])
    if len(sub) == 0:
        return {"top50_sharpe": float("nan"), "top50_mean": float("nan"),
                "mean_hold": float("nan"), "pct_fired": float("nan"), "n_trades": 0}
    mean_hold = float(sub[hc].mean())
    pct_fired = float(sub[tc].mean())
    daily = []
    for _, g in sub.groupby("trade_date"):
        if len(g) < 50: continue
        daily.append(float(g.nlargest(50, score_col)[rc].mean()))
    if len(daily) >= 2:
        arr = np.asarray(daily); sd = arr.std(ddof=1)
        ann_f = np.sqrt(252.0 / max(mean_hold, 1.0))
        sharpe = float(arr.mean() / sd * ann_f) if sd > 1e-9 else 0.0
        mean = float(arr.mean())
    else:
        sharpe = float("nan"); mean = float("nan")
    return {"top50_sharpe": sharpe, "top50_mean": mean,
            "mean_hold": mean_hold, "pct_fired": pct_fired, "n_trades": int(len(sub))}


def main():
    t_total = time.time()
    print("[load] universes ...")
    static_sets, pit_dfs = load_universes()

    print("\n[load] paris 376-col combined_panel ...", flush=True)
    t = time.time()
    panel = pd.read_parquet(PANEL)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.date

    # paris 376-col panel has 1 string col we want to keep as feature: industry_code (中文行业名, 111 unique)
    # Convert to pandas category dtype → LGB native categorical_feature='auto' picks it up
    if "industry_code" in panel.columns and panel["industry_code"].dtype == "object" or \
       (hasattr(panel["industry_code"].dtype, "name") and panel["industry_code"].dtype.name.startswith("string")):
        panel["industry_code"] = panel["industry_code"].astype("category")
        print(f"  converted industry_code → category ({panel['industry_code'].cat.categories.size} levels)")

    # Auto-drop non-numeric / non-bool / non-category cols (LGB rejects string/object/date raw)
    dropped_non_numeric = []
    feat_cols = []
    cat_cols = []
    for c in panel.columns:
        if c in NON_FEATURE_COLS:
            continue
        dt = panel[c].dtype
        if isinstance(dt, pd.CategoricalDtype):
            feat_cols.append(c)
            cat_cols.append(c)
        elif dt.kind in ("i", "u", "f", "b"):
            feat_cols.append(c)
        else:
            dropped_non_numeric.append((c, str(dt)))
    if dropped_non_numeric:
        print(f"  dropped {len(dropped_non_numeric)} non-numeric cols: {dropped_non_numeric}")
    # Cast uint to int (LGB pandas-check sometimes rejects uint32)
    for c in feat_cols:
        if not isinstance(panel[c].dtype, pd.CategoricalDtype) and panel[c].dtype.kind == "u":
            panel[c] = panel[c].astype(np.int64)
    print(f"  panel: {len(panel):,} rows × {len(feat_cols)} features  ({time.time()-t:.0f}s)")
    print(f"  categorical features ({len(cat_cols)}): {cat_cols}")
    # Stash for LGB.fit
    global CAT_COLS
    CAT_COLS = cat_cols

    print("\n[load] 4 labels ...", flush=True)
    all_labels = {}
    for v in LABELS:
        l = _dt(pd.read_parquet(LABEL_TEMPLATE.format(v=v), columns=["trade_date", "ts_code", "y"]))
        all_labels[v] = l
        print(f"  wave_{v}: {len(l):,} rows")

    print("\n[realized + dyn-exit] ...")
    realized, exits = compute_realized_and_sim()

    matrix_results = {}
    for label_v, univ in product(LABELS, UNIVERSES):
        exp_id = f"{label_v}_{univ}"
        print(f"\n=== {exp_id} ===", flush=True)
        upanel = filter_universe(panel, univ, static_sets, pit_dfs)
        print(f"  universe={univ}: panel {len(upanel):,} rows")

        label = all_labels[label_v]
        train = upanel.merge(label, on=["ts_code", "trade_date"], how="inner")
        train = train[(train["trade_date"] >= TRAIN_START) & (train["trade_date"] <= TRAIN_END)]
        print(f"  train rows: {len(train):,}")

        if len(train) < 10000:
            print("  SKIP: train rows < 10K")
            matrix_results[exp_id] = {"skipped": True, "train_rows": len(train)}
            continue

        t = time.time()
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        cat_kwargs = {"categorical_feature": [c for c in CAT_COLS if c in feat_cols]} if CAT_COLS else {}
        model.fit(train[feat_cols], train["y"], **cat_kwargs)
        print(f"  train: {time.time()-t:.0f}s")
        del train

        t = time.time()
        preds = model.predict(upanel[feat_cols]).astype(np.float32)
        print(f"  predict: {time.time()-t:.0f}s ({len(preds):,} rows)")

        pred_df = upanel[["ts_code", "trade_date"]].copy()
        pred_df["score"] = preds
        pred_df.to_parquet(OUT_DIR / f"pred_{exp_id}.parquet", compression="zstd")

        eval_df = pred_df.merge(realized, on=["ts_code", "trade_date"], how="left")
        exit_eval = pred_df.merge(exits, on=["ts_code", "trade_date"], how="inner")

        result = {"static": {}, "dynamic": {}, "n_pred_rows": len(pred_df)}
        for wname, (ws, we) in WINDOWS.items():
            sub = eval_df[(eval_df["trade_date"] >= ws) & (eval_df["trade_date"] <= we)]
            esub = exit_eval[(exit_eval["trade_date"] >= ws) & (exit_eval["trade_date"] <= we)]
            result["static"][wname] = {}
            for k in HORIZONS:
                ic = cross_sec_ic(sub, "score", f"ret_fwd{k}")
                sh = top_k_sharpe_ann(sub, "score", f"ret_fwd{k}")
                result["static"][wname][f"fwd{k}"] = {"ic": ic["mean"], "ir": ic["ir"], "top50_sharpe": sh["sharpe"]}
            result["dynamic"][wname] = {}
            for trig in TRIGGERS:
                result["dynamic"][wname][trig] = dyn_agg(esub, "score", trig)

        matrix_results[exp_id] = result
        print(f"  {exp_id}: H1 fwd5 IC={result['static']['H1_2025']['fwd5']['ic']:+.4f}  "
              f"H1 fwd20 IC={result['static']['H1_2025']['fwd20']['ic']:+.4f}  "
              f"Q1 fwd20 IC={result['static']['Q1_2026']['fwd20']['ic']:+.4f}")

    out = {
        "config": {
            "tier": 4,
            "panel_source": "paris combined_panel v1 (376 cols, 7.55 GB zstd)",
            "panel_rows": int(len(panel)),
            "n_features": len(feat_cols),
            "labels": list(LABELS), "universes": list(UNIVERSES),
            "horizons": list(HORIZONS),
            "windows": {k: [str(v[0]), str(v[1])] for k, v in WINDOWS.items()},
            "train_window": [str(TRAIN_START), str(TRAIN_END)],
            "lgb_params": LGB_PARAMS,
            "top_k": TOP_K, "triggers": list(TRIGGERS),
            "phase_c_caveat": "phase_c 13 cols fill is low (dc_hot ~2.3%, tdx ~0.1%) in panel v1, paris re-ship v2 after 5/16-5/18 backfill",
        },
        "results": matrix_results,
        "total_time_s": time.time() - t_total,
    }
    Path("data/kronos/outputs/tier4_results.json").write_text(json.dumps(out, indent=2, default=str))

    lines = ["# Tier 4 — paris 376-col combined_panel ablation", ""]
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M Asia/Shanghai')}  runtime {time.time()-t_total:.0f}s_")
    lines.append("")
    lines.append("**Panel**: paris combined_panel v1 (376 features, 7.55 GB)")
    lines.append("**Train window**: 2022-01-01 ~ 2024-12-31")
    lines.append("**Caveat**: phase_c 13 cols fill ~2.3% in panel v1 (paris re-ship v2 after 5/16-5/18 backfill)")
    lines.append("")

    for wname in WINDOWS:
        lines.append(f"\n## {wname} — mean IC × 100")
        for univ in UNIVERSES:
            lines.append(f"\n### {univ}")
            hdr = ["label"] + [f"fwd{k}" for k in HORIZONS]
            lines.append("| " + " | ".join(hdr) + " |")
            lines.append("|" + "---|" * len(hdr))
            for label_v in LABELS:
                r = matrix_results.get(f"{label_v}_{univ}", {})
                row = [f"wave_{label_v}"]
                if r.get("skipped"):
                    row += ["skip"] * len(HORIZONS)
                else:
                    for k in HORIZONS:
                        v = r.get("static", {}).get(wname, {}).get(f"fwd{k}", {}).get("ic", float("nan"))
                        row.append(f"{v*100:+.3f}" if v == v else "n/a")
                lines.append("| " + " | ".join(row) + " |")

    Path("data/kronos/outputs/tier4_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[saved] tier4_results.json + tier4_table.md + {len(matrix_results)} predictions")
    print(f"[done] total {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
