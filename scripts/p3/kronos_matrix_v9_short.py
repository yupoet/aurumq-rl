"""Matrix v9 — SHORT proximity / super-short target ablation.

ledashi has been training on K=20 wave labels but evaluating on fwd1/3/5 — biased.
This script trains models DIRECTLY on ret_fwdK regression target for K ∈ {1, 3, 5}.

Cells (12 = 2 panels × 2 universes × 3 K-targets):
  Panel: ledashi 228 + v3 unified
  Universe: MAIN_BOARD + CSI1000 (production targets)
  Target: regression on ret_fwdK directly (NO proximity weighting)
  K = 1, 3, 5 (super-short)

Hyperparam: same as matrix v8 (n=200 fixed) for apples-to-apples comparison.
Plus 11 dyn-exit triggers + cost-aware net Sharpe.
"""
from __future__ import annotations

import os

# Public consumers: set AURUMQ_HANDOFF_INBOX to your local data dir.
# Default: data/handoffs/inbox/<bundle_dir>/<file>
_HANDOFF_INBOX = os.environ.get("AURUMQ_HANDOFF_INBOX", "data/handoffs/inbox")

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PANEL_V3 = f"{_HANDOFF_INBOX}/2026-05-15-paris-panel-v3-unified/combined_panel_v_x_v3_unified.parquet"
PANEL_LEDASHI = "data/p3_4070_long/feature_panel_v3_344_pruned.parquet"
PANEL_CLOSE = "data/p3_4070_long/stock_close_volume_daily.parquet"
TECH_LINES = f"{_HANDOFF_INBOX}/2026-05-14-paris-macd-kdj-raw/tech_lines_daily.parquet"
UNIVERSE_DIR = Path("data/universes")
OUT_DIR = Path("data/kronos/outputs/matrix_v9")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = pd.Timestamp("2022-01-01").date()
TRAIN_END   = pd.Timestamp("2024-12-31").date()
WINDOWS = {
    "H1_2025": (pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-06-30").date()),
    "H2_2025": (pd.Timestamp("2025-07-01").date(), pd.Timestamp("2025-12-31").date()),
    "Q1_2026": (pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-03-31").date()),
    "Q2_2026_partial": (pd.Timestamp("2026-04-01").date(), pd.Timestamp("2026-05-12").date()),
}
K_TRAIN_TARGETS = (1, 3, 5)   # train target horizons (short)
K_EVAL = (1, 2, 3, 5, 10, 20)  # eval horizons for cross-K transfer check
K_MAX = 30
TOP_K = 50
TRIGGERS = ("A_stop_5pct", "C_vol_drop", "E_trail_5pct", "F_trend_break",
            "G_K_max", "H_macd_death", "I_kdj_death", "S_ma5_below_ma10",
            "J_take_profit_5", "J_take_profit_10", "Q_OR_FIE")
COST_ROUND_TRIP = 0.0020  # 20 bps

CELLS = [
    # (target K, universe, panel)
    (1, "MAIN_BOARD", "ledashi"),
    (3, "MAIN_BOARD", "ledashi"),
    (5, "MAIN_BOARD", "ledashi"),
    (1, "MAIN_BOARD", "v3unified"),
    (3, "MAIN_BOARD", "v3unified"),
    (5, "MAIN_BOARD", "v3unified"),
    (1, "CSI1000", "ledashi"),
    (3, "CSI1000", "ledashi"),
    (5, "CSI1000", "ledashi"),
    (1, "CSI1000", "v3unified"),
    (3, "CSI1000", "v3unified"),
    (5, "CSI1000", "v3unified"),
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
    for k in set(K_TRAIN_TARGETS) | set(K_EVAL):
        p[f"ret_fwd{k}"] = (p.groupby("ts_code")["adj_close"].shift(-k) / p["adj_close"] - 1.0).astype(np.float32)

    print("[realized] merging tech_lines ...", flush=True)
    tech = _dt(pd.read_parquet(TECH_LINES))
    p = p.merge(tech, on=["ts_code", "trade_date"], how="left")
    del tech

    realized = p[["ts_code", "trade_date"] + [f"ret_fwd{k}" for k in K_EVAL]].copy()
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
        if n < K_MAX + 5: continue

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


def top_k_sharpe_ann(df, pred_col, real_col, k=TOP_K, horizon=5):
    rets = []
    for _, g in df.dropna(subset=[pred_col, real_col]).groupby("trade_date"):
        if len(g) < k: continue
        rets.append(float(g.nlargest(k, pred_col)[real_col].mean()))
    if len(rets) < 2:
        return {"sharpe": float("nan"), "sharpe_net": float("nan"),
                "mean": float("nan"), "mean_net": float("nan")}
    arr = np.asarray(rets); sd = arr.std(ddof=1)
    ann = np.sqrt(252.0 / max(horizon, 1))
    arr_net = arr - COST_ROUND_TRIP
    sd_net = arr_net.std(ddof=1)
    return {
        "sharpe": float(arr.mean()/sd*ann) if sd > 1e-9 else 0.0,
        "sharpe_net": float(arr_net.mean()/sd_net*ann) if sd_net > 1e-9 else 0.0,
        "mean": float(arr.mean()), "mean_net": float(arr_net.mean()),
    }


def dyn_agg(df, score_col, trigger):
    rc = f"{trigger}_ret"; hc = f"{trigger}_hold"; tc = f"{trigger}_fired"
    sub = df.dropna(subset=[score_col, rc])
    if len(sub) == 0:
        return {"sharpe": float("nan"), "sharpe_net": float("nan"),
                "mean_hold": float("nan"), "pct_fired": float("nan")}
    mean_hold = float(sub[hc].mean())
    pct_fired = float(sub[tc].mean())
    daily = []
    for _, g in sub.groupby("trade_date"):
        if len(g) < TOP_K: continue
        daily.append(float(g.nlargest(TOP_K, score_col)[rc].mean()))
    if len(daily) >= 2:
        arr = np.asarray(daily); sd = arr.std(ddof=1)
        ann = np.sqrt(252.0 / max(mean_hold, 1.0))
        sharpe = float(arr.mean()/sd*ann) if sd > 1e-9 else 0.0
        arr_net = arr - COST_ROUND_TRIP
        sd_net = arr_net.std(ddof=1)
        sharpe_net = float(arr_net.mean()/sd_net*ann) if sd_net > 1e-9 else 0.0
    else:
        sharpe = float("nan"); sharpe_net = float("nan")
    return {"sharpe": sharpe, "sharpe_net": sharpe_net,
            "mean_hold": mean_hold, "pct_fired": pct_fired}


def load_panel(name):
    if name == "v3unified":
        print(f"[panel] reading paris v3 unified ...", flush=True)
        t = time.time()
        p = pd.read_parquet(PANEL_V3)
        p = _dt(p)
        drop = [c for c in p.columns if c not in ("ts_code","trade_date")
                and not (pd.api.types.is_numeric_dtype(p[c]) or pd.api.types.is_bool_dtype(p[c]))]
        p = p.drop(columns=drop)
        for c in p.columns:
            if c in ("ts_code","trade_date"): continue
            if str(p[c].dtype).startswith("Int"): p[c] = p[c].astype("float32")
            elif p[c].dtype == np.float64: p[c] = p[c].astype(np.float32)
        print(f"  v3 unified: {len(p):,} × {len(p.columns)} cols ({time.time()-t:.0f}s)")
        return p
    elif name == "ledashi":
        print(f"[panel] reading ledashi 228 ...", flush=True)
        p = _dt(pd.read_parquet(PANEL_LEDASHI))
        print(f"  ledashi: {len(p):,} × {len(p.columns)} cols")
        return p
    raise ValueError(name)


def main():
    t_total = time.time()
    unique_univs = sorted({c[1] for c in CELLS})
    static_sets, pit_dfs = load_universes(unique_univs)

    print("\n[setup] realized + dyn-exit ...")
    realized, exits = compute_realized_and_exits()

    cells_by_panel = {}
    for k, univ, panel in CELLS:
        cells_by_panel.setdefault(panel, []).append((k, univ))

    matrix_results = {}
    for panel_name, cell_list in cells_by_panel.items():
        panel = load_panel(panel_name)
        base_cols = [c for c in panel.columns if c not in ("ts_code","trade_date")]
        print(f"  base_cols: {len(base_cols)}")

        for k_target, univ in cell_list:
            exp_id = f"K{k_target}_{univ}_{panel_name}"
            print(f"\n=== {exp_id} ===", flush=True)
            upanel = filter_universe(panel, univ, static_sets, pit_dfs)
            print(f"  panel: {len(upanel):,} rows")

            # Build short target inline: ret_fwdK_target from close panel
            close_realized = realized[["ts_code","trade_date",f"ret_fwd{k_target}"]].rename(
                columns={f"ret_fwd{k_target}":"y"}
            )
            joined = upanel.merge(close_realized, on=["ts_code","trade_date"], how="inner")
            joined = joined.dropna(subset=["y"])
            train = joined[(joined["trade_date"]>=TRAIN_START) & (joined["trade_date"]<=TRAIN_END)]
            print(f"  train rows: {len(train):,}")
            if len(train) < 10000:
                matrix_results[exp_id] = {"skipped": True}; continue

            t = time.time()
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(train[base_cols], train["y"])
            print(f"  train: {time.time()-t:.0f}s")
            del train

            preds = model.predict(upanel[base_cols]).astype(np.float32)
            pred_df = upanel[["ts_code","trade_date"]].copy()
            pred_df["score"] = preds
            pred_df.to_parquet(OUT_DIR / f"pred_{exp_id}.parquet", compression="zstd")

            eval_df = pred_df.merge(realized, on=["ts_code","trade_date"], how="left")
            exit_eval = pred_df.merge(exits, on=["ts_code","trade_date"], how="inner")

            result = {"static": {}, "dynamic": {}, "n_pred": len(pred_df)}
            for wname, (ws, we) in WINDOWS.items():
                sub = eval_df[(eval_df["trade_date"]>=ws)&(eval_df["trade_date"]<=we)]
                esub = exit_eval[(exit_eval["trade_date"]>=ws)&(exit_eval["trade_date"]<=we)]
                result["static"][wname] = {}
                for k_eval in K_EVAL:
                    ic = cross_sec_ic(sub, "score", f"ret_fwd{k_eval}")
                    sh = top_k_sharpe_ann(sub, "score", f"ret_fwd{k_eval}", horizon=k_eval)
                    result["static"][wname][f"fwd{k_eval}"] = {
                        "ic": ic["mean"], "ir": ic["ir"],
                        "sharpe_gross": sh["sharpe"], "sharpe_net": sh["sharpe_net"],
                    }
                result["dynamic"][wname] = {t: dyn_agg(esub, "score", t) for t in TRIGGERS}

            matrix_results[exp_id] = result
            r = result["static"]["H2_2025"]
            print(f"  H2 fwd{k_target} IC={r.get(f'fwd{k_target}',{}).get('ic',0)*100:+.3f}%  "
                  f"H2 fwd5={r['fwd5']['ic']*100:+.3f}%  "
                  f"H2 fwd20={r['fwd20']['ic']*100:+.3f}%")
        del panel

    out = {
        "config": {
            "tier": "v9 — direct ret_fwdK regression (short-target)",
            "cells": CELLS, "k_train": list(K_TRAIN_TARGETS), "k_eval": list(K_EVAL),
            "cost_round_trip": COST_ROUND_TRIP, "lgb_params": LGB_PARAMS,
        },
        "results": matrix_results,
        "total_time_s": time.time()-t_total,
    }
    Path("data/kronos/outputs/matrix_v9_results.json").write_text(json.dumps(out, indent=2, default=str))

    # Build markdown
    lines = ["# Matrix v9 — short-target (ret_fwdK regression)", ""]
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M Asia/Shanghai')}  runtime {time.time()-t_total:.0f}s_")
    lines.append("")
    lines.append("12 cells = 2 panels × 2 universes × 3 K-targets. Target: direct ret_fwdK regression (no proximity).")
    lines.append("")
    lines.append("## IC fwd1/3/5 by training target K (H2_2025)")
    lines.append("")
    lines.append("| cell | fwd1 | fwd2 | fwd3 | fwd5 | fwd10 | fwd20 |")
    lines.append("|---|---|---|---|---|---|---|")
    for cid, r in matrix_results.items():
        if r.get("skipped"): continue
        s = r["static"].get("H2_2025", {})
        row = [cid]
        for k in K_EVAL:
            v = s.get(f"fwd{k}",{}).get("ic", float("nan"))
            row.append(f"{v*100:+.2f}" if v == v else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Sharpe NET (after 0.20% cost) H2_2025")
    lines.append("")
    lines.append("| cell | fwd1 | fwd2 | fwd3 | fwd5 | fwd10 | fwd20 |")
    lines.append("|---|---|---|---|---|---|---|")
    for cid, r in matrix_results.items():
        if r.get("skipped"): continue
        s = r["static"].get("H2_2025", {})
        row = [cid]
        for k in K_EVAL:
            v = s.get(f"fwd{k}",{}).get("sharpe_net", float("nan"))
            row.append(f"{v:+.2f}" if v == v else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Dyn-exit Sharpe NET H2_2025 (key triggers only)")
    lines.append("")
    short_keys = ["F_trend_break","I_kdj_death","J_take_profit_5","Q_OR_FIE"]
    lines.append("| cell | F | I | J5 | Q |")
    lines.append("|---|---|---|---|---|")
    for cid, r in matrix_results.items():
        if r.get("skipped"): continue
        d = r["dynamic"].get("H2_2025", {})
        row = [cid] + [f"{d.get(t,{}).get('sharpe_net',float('nan')):+.1f}" for t in short_keys]
        lines.append("| " + " | ".join(row) + " |")

    Path("data/kronos/outputs/matrix_v9_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[saved] matrix_v9_results.json + matrix_v9_table.md")
    print(f"[done] total {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
