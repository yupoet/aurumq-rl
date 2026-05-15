"""GROWTH phase 3 — train 4 LGB on paris growth labels (4 methods × t20 horizon).

Per paris v17/v18.1 path A: paris ship growth labels (4 methods A/B/C/D × 6 horizons),
ledashi trains LGB on 4070. 4 cells = 4 methods × GROWTH_BOARDS universe × t20 horizon
(WAVE family analog of wave_v1/v2/v3/v4 ablation).

Uses my 228-col pruned panel filtered to GROWTH_BOARDS (2253 stocks).
Labels are binary 0/1 → use objective="binary" for proper handling.
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

PANEL = "data/p3_4070_long/feature_panel_v3_344_pruned.parquet"
GROWTH_LABELS_DIR = Path(f"{_HANDOFF_INBOX}/2026-05-15-paris-growth-labels")
PANEL_CLOSE = "data/p3_4070_long/stock_close_volume_daily.parquet"
TECH_LINES = f"{_HANDOFF_INBOX}/2026-05-14-paris-macd-kdj-raw/tech_lines_daily.parquet"
UNIVERSE_DIR = Path("data/universes")
OUT_DIR = Path("data/kronos/outputs/growth_v4")  # wave_binary hyperparam rerun (paris v19 lgb_params)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = pd.Timestamp("2022-01-01").date()
TRAIN_END   = pd.Timestamp("2024-12-31").date()
WINDOWS = {
    "H1_2025_val":     (pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-06-30").date()),
    "H2_2025":         (pd.Timestamp("2025-07-01").date(), pd.Timestamp("2025-12-31").date()),
    "Q1_2026":         (pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-03-31").date()),
    "Q2_2026_partial": (pd.Timestamp("2026-04-01").date(), pd.Timestamp("2026-05-12").date()),
}
HORIZONS = (1, 3, 5, 10, 20, 30)
K_MAX = 30
TOP_K = 50
TRIGGERS = ("A_stop_5pct", "C_vol_drop", "E_trail_5pct", "F_trend_break",
            "G_K_max", "H_macd_death", "I_kdj_death", "S_ma5_below_ma10",
            "J_take_profit_5", "J_take_profit_10", "Q_OR_FIE")

# 4 methods × t20 → 4 "wave-v1/2/3/4" equivalent cells for GROWTH family
METHOD_TO_NAME = {"A": "v1", "B": "v2", "C": "v3", "D": "v4"}
HORIZON = "t20"
YEARS = [2022, 2023, 2024, 2025, 2026]

# paris production wave_binary hyperparam (v19 lgb_params_wave_binary.json)
LGB_PARAMS = dict(
    objective="binary", metric="average_precision",
    boosting_type="gbdt",
    learning_rate=0.05, num_leaves=63,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    min_data_in_leaf=200,
    n_estimators=500,
    verbose=-1, num_threads=-1, random_state=42,
)
EARLY_STOPPING_ROUNDS = 50
VAL_FRAC = 0.15  # last 15% of train rows by date for early stopping


def _dt(df):
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def load_growth_universe() -> frozenset[str]:
    df = pd.read_parquet(UNIVERSE_DIR / "GROWTH_BOARDS_membership.parquet")
    return frozenset(df["stock_code"].tolist())


def load_growth_labels(method: str) -> pd.DataFrame:
    """Concat 5-year growth labels for one method × t20 horizon."""
    dfs = []
    for year in YEARS:
        path = GROWTH_LABELS_DIR / f"labels_{method}_{HORIZON}_growth_year={year}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    full = pd.concat(dfs, ignore_index=True)
    full = _dt(full)
    return full


def compute_realized_and_sim_growth(growth_set: frozenset[str]):
    """Realized fwd-K from paris combined_panel (stock_close is MAIN_BOARD-only).
    Skip dyn-exit (no GROWTH MACD/KDJ available)."""
    t = time.time()
    print("[realized] loading paris combined_panel close cols for GROWTH ...", flush=True)
    p = pd.read_parquet(
        f"{_HANDOFF_INBOX}/2026-05-17-paris-combined-panel/combined_panel_v_x.parquet",
        columns=["ts_code", "trade_date", "close", "pct_chg"],
    )
    p["trade_date"] = pd.to_datetime(p["trade_date"]).dt.date
    p = p[p["ts_code"].isin(growth_set)].sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"  filtered GROWTH from paris panel: {len(p):,} rows × {p['ts_code'].nunique()} stocks")
    p["adj_close"] = p["close"].astype(np.float32)  # paris combined_panel close already adj
    p["vol"] = 0  # not available
    p["pct_chg"] = p["pct_chg"].astype(np.float32) if p["pct_chg"].dtype != np.float32 else p["pct_chg"]
    # No vol → skip MA-based triggers, set dummies
    p["ma20_vol"] = 0.0
    p["ma5_close"] = p.groupby("ts_code")["adj_close"].transform(
        lambda x: x.rolling(5, min_periods=2).mean()).astype(np.float32)
    p["ma10_close"] = p.groupby("ts_code")["adj_close"].transform(
        lambda x: x.rolling(10, min_periods=3).mean()).astype(np.float32)

    for k in HORIZONS:
        p[f"ret_fwd{k}"] = (p.groupby("ts_code")["adj_close"].shift(-k) / p["adj_close"] - 1.0).astype(np.float32)

    # No paris tech_lines for GROWTH; skip merge, fill NaN to disable H/I
    for c in ("macd_line", "macd_signal", "macd_hist", "kdj_k", "kdj_d", "kdj_j"):
        p[c] = np.nan

    realized = p[["ts_code", "trade_date"] + [f"ret_fwd{k}" for k in HORIZONS]].copy()

    print("[dyn-exit] simulating ...", flush=True)
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
    growth_set = load_growth_universe()
    print(f"[load] GROWTH universe: {len(growth_set)} stocks")

    print("\n[load] panel (filtering GROWTH) ...", flush=True)
    panel = _dt(pd.read_parquet(PANEL))
    panel = panel[panel["ts_code"].isin(growth_set)]
    base_cols = [c for c in panel.columns if c not in ("ts_code", "trade_date")]
    print(f"  panel GROWTH: {len(panel):,} rows × {len(base_cols)} feats")

    print("\n[realized + dyn-exit] (GROWTH only) ...")
    realized, exits = compute_realized_and_sim_growth(growth_set)

    results = {}
    for method, name in METHOD_TO_NAME.items():
        exp_id = f"{name}_method_{method}_growth"
        print(f"\n=== {exp_id} (label {method}_{HORIZON}) ===", flush=True)
        label = load_growth_labels(method)
        print(f"  label: {len(label):,} rows, pos rate {label['y'].mean()*100:.1f}%")

        joined = panel.merge(label, on=["ts_code", "trade_date"], how="inner")
        train = joined[(joined["trade_date"] >= TRAIN_START) & (joined["trade_date"] <= TRAIN_END)]
        print(f"  train rows: {len(train):,}, pos rate {train['y'].mean()*100:.1f}%")

        if len(train) < 10000:
            print("  SKIP")
            results[exp_id] = {"skipped": True}
            continue

        t = time.time()
        train = train.sort_values("trade_date").reset_index(drop=True)
        n_val = int(len(train) * VAL_FRAC)
        tr = train.iloc[:-n_val]
        va = train.iloc[-n_val:]
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            tr[base_cols], tr["y"],
            eval_set=[(va[base_cols], va["y"])],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        print(f"  train: {time.time()-t:.0f}s, best_iter={model.best_iteration_}, tr_rows={len(tr):,} va_rows={len(va):,}")
        del train, tr, va

        t = time.time()
        preds = model.predict_proba(panel[base_cols])[:, 1].astype(np.float32)
        print(f"  predict: {time.time()-t:.0f}s ({len(preds):,} rows)")

        pred_df = panel[["ts_code", "trade_date"]].copy()
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

        results[exp_id] = result
        print(f"  {exp_id}: H2 fwd5 IC={result['static']['H2_2025']['fwd5']['ic']:+.4f}  "
              f"H2 fwd20 IC={result['static']['H2_2025']['fwd20']['ic']:+.4f}  "
              f"Q1 fwd20 IC={result['static']['Q1_2026']['fwd20']['ic']:+.4f}")

    out = {
        "config": {
            "tier": "GROWTH phase 3 — paris labels × ledashi LGB + paris hyperparam",
            "panel_source": PANEL,
            "growth_labels": "paris 2026-05-15-paris-growth-labels (binary 0/1, 4 methods × t20)",
            "method_to_name": METHOD_TO_NAME,
            "horizon": HORIZON,
            "labels": list(METHOD_TO_NAME.values()),
            "universes": ["GROWTH_BOARDS"],
            "horizons": list(HORIZONS),
            "windows": {k: [str(v[0]), str(v[1])] for k, v in WINDOWS.items()},
            "lgb_params": LGB_PARAMS,
            "top_k": TOP_K, "triggers": list(TRIGGERS),
        },
        "results": results,
        "total_time_s": time.time() - t_total,
    }
    Path("data/kronos/outputs/growth_v4_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] growth_v4_results.json + {len(results)} predictions")
    print(f"[done] total {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
