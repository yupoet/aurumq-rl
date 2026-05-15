"""Matrix v8 — same 7 cells on paris panel_v3_unified (244 cols = R2-B + 12 tech_evt).

Per paris REPLY v23: panel v3 unified = R2-B (232 cols) + 12 tech_evt cols (6 _raw
+ 6 _decay10) join from `scripts/build_tech_event_panel.py` Phase 26E/F/G output.

Final verification: if v3_MAIN_BOARD H2 fwd20 IC ≥ +4.143% (ledashi 228 baseline)
→ panel v3 unified is production canonical, paris deprecates v2/R2 series.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PANEL_V3 = "D:/dev/aurumq-handoffs/inbox/2026-05-15-paris-panel-v3-unified/combined_panel_v_x_v3_unified.parquet"
PANEL_LEDASHI = "data/p3_4070_long/feature_panel_v3_344_pruned.parquet"
LABEL_TEMPLATE = "data/p3_4070_long/target_y_wave_{v}.parquet"
PANEL_CLOSE = "data/p3_4070_long/stock_close_volume_daily.parquet"
TECH_LINES = "D:/dev/aurumq-handoffs/inbox/2026-05-14-paris-macd-kdj-raw/tech_lines_daily.parquet"
UNIVERSE_DIR = Path("data/universes")
OUT_DIR = Path("data/kronos/outputs/matrix_v8")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = pd.Timestamp("2022-01-01").date()
TRAIN_END   = pd.Timestamp("2024-12-31").date()
WINDOWS = {
    "H1_2025": (pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-06-30").date()),
    "H2_2025": (pd.Timestamp("2025-07-01").date(), pd.Timestamp("2025-12-31").date()),
    "Q1_2026": (pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-03-31").date()),
    "Q2_2026_partial": (pd.Timestamp("2026-04-01").date(), pd.Timestamp("2026-05-12").date()),
}
HORIZONS = (1, 2, 3, 5, 10, 20, 30)
K_MAX = 30
TOP_K = 50
TRIGGERS = ("A_stop_5pct", "C_vol_drop", "E_trail_5pct", "F_trend_break",
            "G_K_max", "H_macd_death", "I_kdj_death", "S_ma5_below_ma10",
            "J_take_profit_5", "J_take_profit_10", "Q_OR_FIE")

# A-share transaction cost (round-trip):
#   印花税 0.05% (sell) + 券商佣金 ~0.05% bi-dir + 过户费 0.001% + slippage ~5-10 bps
#   conservative: 0.20% round-trip for top-50 MAIN_BOARD
COST_ROUND_TRIP = 0.0020  # 20 bps

CELLS = [
    ("v1", "CSI1000", "v3unified"),
    ("v2", "CSI1000", "v3unified"),
    ("v3", "CSI1000", "v3unified"),
    ("v4", "CSI1000", "v3unified"),
    ("v3", "MAIN_BOARD", "v3unified"),
    ("v3", "MAIN_BOARD", "ledashi"),
    ("v3", "NPF", "v3unified"),
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


def compute_realized_and_exits():
    """Compute fwd returns + dyn-exit simulation for 11 triggers.

    Returns (realized_df, exits_df) — same as matrix v3."""
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

    print("[realized] merging tech_lines for MACD/KDJ ...", flush=True)
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
        if max_entry < 0: continue
        entries = np.arange(0, max_entry + 1)
        fwd_idx = entries[:, None] + np.arange(1, K_MAX + 1)[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            cum_ret = adj_close[fwd_idx] / adj_close[entries][:, None] - 1.0
        cum_ret = np.where(np.isfinite(cum_ret), cum_ret, 0.0)
        peak_ret = np.maximum.accumulate(cum_ret, axis=1)

        firmap = {
            "A_stop_5pct": cum_ret < -0.05,
            "C_vol_drop": C_abs[fwd_idx],
            "E_trail_5pct": (peak_ret - cum_ret) > 0.05,
            "F_trend_break": F_abs[fwd_idx],
            "G_K_max": np.zeros_like(cum_ret, dtype=bool),
            "H_macd_death": H_abs[fwd_idx],
            "I_kdj_death": I_abs[fwd_idx],
            "S_ma5_below_ma10": S_abs[fwd_idx],
            "J_take_profit_5": cum_ret > 0.05,
            "J_take_profit_10": cum_ret > 0.10,
            "Q_OR_FIE": F_abs[fwd_idx] | I_abs[fwd_idx] | ((peak_ret - cum_ret) > 0.05),
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

    exits = pd.concat(exit_rows, ignore_index=True)
    print(f"  exits: {len(exits):,} rows from {n_syms} stocks ({time.time()-t:.0f}s)")
    del p
    return realized, exits


def dyn_agg(df, score_col, trigger):
    rc = f"{trigger}_ret"; hc = f"{trigger}_hold"; tc = f"{trigger}_fired"
    sub = df.dropna(subset=[score_col, rc])
    if len(sub) == 0:
        return {"top50_sharpe": float("nan"), "top50_sharpe_net": float("nan"),
                "top50_mean": float("nan"), "top50_mean_net": float("nan"),
                "mean_hold": float("nan"), "pct_fired": float("nan"), "n_trades": 0}
    mean_hold = float(sub[hc].mean())
    pct_fired = float(sub[tc].mean())
    daily = []
    for _, g in sub.groupby("trade_date"):
        if len(g) < TOP_K: continue
        daily.append(float(g.nlargest(TOP_K, score_col)[rc].mean()))
    if len(daily) >= 2:
        arr = np.asarray(daily); sd = arr.std(ddof=1)
        ann_f = np.sqrt(252.0 / max(mean_hold, 1.0))
        sharpe = float(arr.mean() / sd * ann_f) if sd > 1e-9 else 0.0
        mean = float(arr.mean())
        # net: subtract cost from each trade's realized return
        arr_net = arr - COST_ROUND_TRIP
        sd_net = arr_net.std(ddof=1)
        sharpe_net = float(arr_net.mean() / sd_net * ann_f) if sd_net > 1e-9 else 0.0
        mean_net = float(arr_net.mean())
    else:
        sharpe = float("nan"); mean = float("nan")
        sharpe_net = float("nan"); mean_net = float("nan")
    return {"top50_sharpe": sharpe, "top50_sharpe_net": sharpe_net,
            "top50_mean": mean, "top50_mean_net": mean_net,
            "mean_hold": mean_hold, "pct_fired": pct_fired, "n_trades": int(len(sub))}


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


def top_k_sharpe_ann(df, pred_col, real_col, k=TOP_K, horizon=20):
    """Annualized Sharpe for daily-rebalance, K-day-hold strategy.

    Two flavors:
      - gross: pre-cost (raw IC backtest); ann factor √(252/K)
      - net:   subtract COST_ROUND_TRIP per period (each new portfolio = 1 round-trip)
               then anualize. Daily rebalance + K-day hold ≈ K-day round-trip cycle.
    """
    rets = []
    for _, g in df.dropna(subset=[pred_col, real_col]).groupby("trade_date"):
        if len(g) < k: continue
        rets.append(float(g.nlargest(k, pred_col)[real_col].mean()))
    if len(rets) < 2:
        return {"sharpe": float("nan"), "sharpe_net": float("nan"),
                "n_days": len(rets), "mean": float("nan"), "mean_net": float("nan")}
    arr = np.asarray(rets); sd = arr.std(ddof=1)
    ann_factor = np.sqrt(252.0 / max(horizon, 1))
    arr_net = arr - COST_ROUND_TRIP  # subtract cost from each K-day return
    sd_net = arr_net.std(ddof=1)
    return {
        "sharpe": float(arr.mean() / sd * ann_factor) if sd > 1e-9 else 0.0,
        "sharpe_net": float(arr_net.mean() / sd_net * ann_factor) if sd_net > 1e-9 else 0.0,
        "n_days": int(len(arr)),
        "mean": float(arr.mean()),
        "mean_net": float(arr_net.mean()),
    }


def load_panel(name):
    if name == "v3unified":
        print(f"[panel] reading {PANEL_V3} ...", flush=True)
        t = time.time()
        p = pd.read_parquet(PANEL_V3)
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
        print(f"  panel v3 unified: {len(p):,} rows × {len(p.columns)} cols ({time.time()-t:.0f}s)")
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

    print("\n[setup] realized + dyn-exit ...")
    realized, exits = compute_realized_and_exits()

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
            exit_eval = pred_df.merge(exits, on=["ts_code", "trade_date"], how="inner")

            result = {"static": {}, "dynamic": {}, "n_pred_rows": len(pred_df), "best_iter": int(n_iter)}
            for wname, (ws, we) in WINDOWS.items():
                sub = eval_df[(eval_df["trade_date"] >= ws) & (eval_df["trade_date"] <= we)]
                esub = exit_eval[(exit_eval["trade_date"] >= ws) & (exit_eval["trade_date"] <= we)]
                result["static"][wname] = {}
                for k in HORIZONS:
                    ic = cross_sec_ic(sub, "score", f"ret_fwd{k}")
                    sh = top_k_sharpe_ann(sub, "score", f"ret_fwd{k}", horizon=k)
                    result["static"][wname][f"fwd{k}"] = {
                        "ic": ic["mean"], "ir": ic["ir"],
                        "top50_sharpe_gross": sh["sharpe"], "top50_sharpe_net": sh["sharpe_net"],
                        "top50_mean_gross": sh["mean"], "top50_mean_net": sh["mean_net"],
                    }
                result["dynamic"][wname] = {}
                for trig in TRIGGERS:
                    result["dynamic"][wname][trig] = dyn_agg(esub, "score", trig)

            matrix_results[exp_id] = result
            r = result["static"]
            print(f"  {exp_id}: H1 fwd20={r['H1_2025']['fwd20']['ic']*100:+.3f}%  "
                  f"H2 fwd5={r['H2_2025']['fwd5']['ic']*100:+.3f}%  "
                  f"H2 fwd20={r['H2_2025']['fwd20']['ic']*100:+.3f}%  "
                  f"Q1 fwd20={r['Q1_2026']['fwd20']['ic']*100:+.3f}%")

        del panel

    v3_unified = matrix_results.get("v3_MAIN_BOARD_v3unified", {}).get("static", {}).get("H2_2025", {}).get("fwd20", {}).get("ic", float("nan"))
    v3_ledashi = matrix_results.get("v3_MAIN_BOARD_ledashi", {}).get("static", {}).get("H2_2025", {}).get("fwd20", {}).get("ic", float("nan"))
    verdict = ""
    if v3_unified == v3_unified and v3_ledashi == v3_ledashi:
        delta = (v3_unified - v3_ledashi) * 100
        if v3_unified >= v3_ledashi:
            verdict = f"V3_UNIFIED_BEATS_LEDASHI Δ=+{delta:.3f}pp → CLOSED LOOP → paris promotes v3 unified as production canonical"
        elif v3_unified >= v3_ledashi - 0.001:  # within 0.1pp noise
            verdict = f"V3_UNIFIED_TIES_LEDASHI Δ={delta:.3f}pp (within noise) → accept v3 unified as production"
        else:
            verdict = f"V3_UNIFIED_LOSES_LEDASHI Δ={delta:.3f}pp → investigate further or keep R2-B"

    out = {
        "config": {
            "tier": "v8 — paris panel_v3_unified (244 cols = R2-B 232 + 12 tech_evt)",
            "cells": CELLS,
            "panel_v3": PANEL_V3,
            "panel_ledashi": PANEL_LEDASHI,
            "lgb_params": LGB_PARAMS,
            "decision_verdict": verdict,
        },
        "results": matrix_results,
        "total_time_s": time.time() - t_total,
    }
    Path("data/kronos/outputs/matrix_v8_results.json").write_text(json.dumps(out, indent=2, default=str))

    lines = ["# Matrix v8 — paris panel_v3_unified (244 cols = R2-B + 12 tech_evt)", ""]
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M Asia/Shanghai')}  runtime {time.time()-t_total:.0f}s_")
    lines.append("")
    lines.append(f"## DECISION VERDICT (fwd20 narrow): **{verdict}**")
    lines.append("")
    lines.append("## Multi-horizon H2_2025 IC % (broader picture)")
    lines.append("")
    lines.append("| cell | fwd1 | fwd2 | fwd3 | fwd5 | fwd10 | fwd20 | fwd30 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cell_id in matrix_results:
        r = matrix_results[cell_id]
        if r.get("skipped"):
            lines.append(f"| {cell_id} | skip | - | - | - | - | - | - |")
            continue
        s = r["static"].get("H2_2025", {})
        cells_row = [f"{s.get(f'fwd{k}',{}).get('ic',float('nan'))*100:+.2f}" for k in HORIZONS]
        lines.append(f"| {cell_id} | " + " | ".join(cells_row) + " |")
    lines.append("")
    lines.append("## Multi-horizon H2_2025 top-50 Sharpe GROSS (pre-cost, ann √(252/K))")
    lines.append("")
    lines.append("| cell | fwd1 | fwd2 | fwd3 | fwd5 | fwd10 | fwd20 | fwd30 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cell_id in matrix_results:
        r = matrix_results[cell_id]
        if r.get("skipped"): continue
        s = r["static"].get("H2_2025", {})
        cells_row = [f"{s.get(f'fwd{k}',{}).get('top50_sharpe_gross',float('nan')):+.2f}" for k in HORIZONS]
        lines.append(f"| {cell_id} | " + " | ".join(cells_row) + " |")
    lines.append("")
    lines.append(f"## Multi-horizon H2_2025 top-50 Sharpe NET (after {COST_ROUND_TRIP*100:.2f}% round-trip cost)")
    lines.append("")
    lines.append("| cell | fwd1 | fwd2 | fwd3 | fwd5 | fwd10 | fwd20 | fwd30 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cell_id in matrix_results:
        r = matrix_results[cell_id]
        if r.get("skipped"): continue
        s = r["static"].get("H2_2025", {})
        cells_row = [f"{s.get(f'fwd{k}',{}).get('top50_sharpe_net',float('nan')):+.2f}" for k in HORIZONS]
        lines.append(f"| {cell_id} | " + " | ".join(cells_row) + " |")
    lines.append("")
    lines.append("## Dynamic-exit H2_2025 top-50 Sharpe GROSS (pre-cost, ann √(252/mean_hold))")
    lines.append("")
    short_names = {
        "A_stop_5pct":"A","C_vol_drop":"C","E_trail_5pct":"E","F_trend_break":"F",
        "G_K_max":"G","H_macd_death":"H","I_kdj_death":"I","S_ma5_below_ma10":"S",
        "J_take_profit_5":"J5","J_take_profit_10":"J10","Q_OR_FIE":"Q"
    }
    h = "| cell | " + " | ".join(short_names[t] for t in TRIGGERS) + " |"
    lines.append(h); lines.append("|" + "---|"*(len(TRIGGERS)+1))
    for cell_id in matrix_results:
        r = matrix_results[cell_id]
        if r.get("skipped"): continue
        d = r.get("dynamic", {}).get("H2_2025", {})
        row = [cell_id] + [f"{d.get(t,{}).get('top50_sharpe',float('nan')):+.1f}" for t in TRIGGERS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(f"## Dynamic-exit H2_2025 top-50 Sharpe NET (after {COST_ROUND_TRIP*100:.2f}% round-trip cost)")
    lines.append("")
    lines.append(h); lines.append("|" + "---|"*(len(TRIGGERS)+1))
    for cell_id in matrix_results:
        r = matrix_results[cell_id]
        if r.get("skipped"): continue
        d = r.get("dynamic", {}).get("H2_2025", {})
        row = [cell_id] + [f"{d.get(t,{}).get('top50_sharpe_net',float('nan')):+.1f}" for t in TRIGGERS]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Full 5-step ablation closing (v3_MAIN_BOARD H2 fwd20 IC%)")
    lines.append("")
    lines.append("| step | panel | n_cols | IC% | Δ vs ledashi 228 |")
    lines.append("|---|---|---|---|---|")
    lines.append("| matrix v4 | paris v2 (with Phase C NULL) | 381 | +3.08 | -1.06pp |")
    lines.append("| matrix v5 | panel_v2_no_phase_c | 366 | +3.32 | -0.82pp |")
    lines.append("| matrix v6 | R2-A (drop 26 non-tech) | 340 | +3.69 | -0.46pp |")
    lines.append("| matrix v7 | R2-B (drop 108 alpha+gtja) | 232 | +3.96 | -0.18pp |")
    v3u_ic = matrix_results.get('v3_MAIN_BOARD_v3unified', {}).get('static', {}).get('H2_2025', {}).get('fwd20', {}).get('ic', float('nan'))
    v3u_str = f"{v3u_ic*100:+.3f}" if v3u_ic == v3u_ic else "n/a"
    delta_str = f"{(v3u_ic - v3_ledashi)*100:+.3f}pp" if v3u_ic == v3u_ic else "n/a"
    lines.append(f"| **matrix v8** | **v3 unified (R2-B + 12 tech_evt)** | 244 | **{v3u_str}** | **{delta_str}** |")
    lines.append("| ledashi 228 pruned | baseline | 228 | +4.143 ⭐ | 0 |")

    Path("data/kronos/outputs/matrix_v8_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[saved] matrix_v8_results.json + matrix_v8_table.md")
    print(f"[done] total {time.time()-t_total:.0f}s")
    print(f"\n========== VERDICT: {verdict} ==========")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
