"""Matrix v10 — FULL coverage: 7 panels × 6 universes × 4 wave labels × 7 sizings.

Per approved plan 2026-05-16 (overnight run):

Training cells: 7 panels × 6 universes × 4 wave_v* labels = 168 cells
Eval-only cells: 6 paris/ledashi proximity model predictions = 6 cells (MAIN_BOARD only)
Total: 174 cells

Per-cell eval:
- 7 horizons (fwd1/2/3/5/10/20/30)
- 7 sizings: 6 fixed top_k {5,10,15,20,30,50} + 1 adaptive (top-pct + gating)
- 11 dyn-exit triggers (A/C/E/F/G/H/I/S/J5/J10/Q)
- 4 windows (H1_2025 / H2_2025 / Q1_2026 / Q2_2026_partial)
- Cost-aware gross + net Sharpe (0.20% round-trip)
- Factor importance (gain) per cell + family aggregation

Hyperparam: matrix v8 default (n=200 fixed, num_leaves=127, regression+rmse).
"""
from __future__ import annotations

import os

_HANDOFF_INBOX = os.environ.get("AURUMQ_HANDOFF_INBOX", "data/handoffs/inbox")

import gc
import json
import time
from pathlib import Path
from functools import reduce

import lightgbm as lgb
import numpy as np
import pandas as pd

# ===== Panels (7) =====
PANEL_PATHS = {
    "ledashi":      "data/p3_4070_long/feature_panel_v3_344_pruned.parquet",
    "tier4_v2_old": f"{_HANDOFF_INBOX}/2026-05-17-paris-combined-panel/combined_panel_v_x.parquet",
    "v2_null":      f"{_HANDOFF_INBOX}/2026-05-15-paris-combined-panel-v2/combined_panel_v_x_v2.parquet",
    "v2_no_phase_c":f"{_HANDOFF_INBOX}/2026-05-15-paris-panel-v2-no-phase-c/combined_panel_v_x_v2_no_phase_c.parquet",
    "r2a":          f"{_HANDOFF_INBOX}/2026-05-15-paris-panel-v2-ablation-r2/combined_panel_v_x_v2_R2A_minus_non_tech.parquet",
    "r2b":          f"{_HANDOFF_INBOX}/2026-05-15-paris-panel-v2-ablation-r2b/combined_panel_v_x_v2_R2B_minus_alpha_gtja_extras.parquet",
    "v3unified":    f"{_HANDOFF_INBOX}/2026-05-15-paris-panel-v3-unified/combined_panel_v_x_v3_unified.parquet",
}
PANELS = ("ledashi", "tier4_v2_old", "v2_null", "v2_no_phase_c", "r2a", "r2b", "v3unified")

LABEL_TEMPLATE = "data/p3_4070_long/target_y_wave_{v}.parquet"
PANEL_CLOSE = "data/p3_4070_long/stock_close_volume_daily.parquet"
TECH_LINES = f"{_HANDOFF_INBOX}/2026-05-14-paris-macd-kdj-raw/tech_lines_daily.parquet"
UNIVERSE_DIR = Path("data/universes")
OUT_DIR = Path("data/kronos/outputs/matrix_v10")
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
TRIGGERS = ("A_stop_5pct", "C_vol_drop", "E_trail_5pct", "F_trend_break",
            "G_K_max", "H_macd_death", "I_kdj_death", "S_ma5_below_ma10",
            "J_take_profit_5", "J_take_profit_10", "Q_OR_FIE")
COST_ROUND_TRIP = 0.0020

# Sizing — 6 fixed tiers + 1 adaptive
TOP_K_LIST = (5, 10, 15, 20, 30, 50)
ADAPTIVE_PCT = 0.05
ADAPTIVE_K_CAP = 50
ADAPTIVE_K_FLOOR = 10
ADAPTIVE_Q_LOW = 0.25
ADAPTIVE_Q_MID = 0.50

LABELS = ("v1", "v2", "v3", "v4")
UNIVERSES = ("MAIN_BOARD", "CSI500", "CSI1000", "NPF", "NPF_FULL", "HARD_TECH")

# Factor family prefixes (for importance aggregation)
FAMILY_PREFIXES = {
    "alpha":    "alpha_",
    "gtja":     "gtja_",
    "mfp":      "mfp_",
    "mf":       "mf_",
    "hm":       "hm_",
    "hk":       "hk_",
    "inst":     "inst_",
    "mg":       "mg_",
    "cyq":      "cyq_",
    "senti":    "senti_",
    "sh":       "sh_",
    "fund":     "fund_",
    "ind":      "ind_",
    "mkt":      "mkt_",
    "tech_evt": "tech_evt_",
    "tech":     "tech_",
    "concept":  "concept_",
    "dc_hot":   "dc_hot_",
    "stk_shock":"stk_shock_",
    "tdx":      "tdx_",
    "kronos":   "kronos_",
    "industry": "industry_",
}

LGB_PARAMS = dict(
    objective="regression", metric="rmse", boosting_type="gbdt",
    learning_rate=0.05, num_leaves=127,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    min_data_in_leaf=100,
    lambda_l1=0.1, lambda_l2=0.1,
    n_estimators=200, max_depth=-1,
    verbose=-1, num_threads=-1, random_state=42,
)

# Eval-only score sources (6 paris/ledashi proximity models, MAIN_BOARD only)
ES_CELLS = [
    ("ES_path4",          f"{_HANDOFF_INBOX}/2026-05-15-paris-9seed-ensemble-preds/path4_ensemble_preds.parquet", "pred_ensemble_calibrated"),
    ("ES_path5_long_v2",  "runs/sl_regime_stack_long_v2/predictions.parquet", "score_calibrated"),
    ("ES_path5_short_v2", "runs/sl_regime_stack_v2/predictions.parquet",       "score_calibrated"),
    ("ES_path1_long",     "runs/sl_path1_long",  None),
    ("ES_path4_long",     "runs/sl_path4_long",  None),
    ("ES_path2_long",     "runs/sl_path2_long",  None),
]


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

    for k in HORIZONS:
        p[f"ret_fwd{k}"] = (p.groupby("ts_code")["adj_close"].shift(-k) / p["adj_close"] - 1.0).astype(np.float32)

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
    if not ics: return {"mean": float("nan"), "ir": float("nan"), "n_days": 0}
    arr = np.asarray(ics); sd = arr.std()
    return {"mean": float(arr.mean()),
            "ir": float(arr.mean()/sd) if sd > 1e-9 else 0.0,
            "n_days": int(len(arr))}


def top_k_sharpe_ann_multi(df, pred_col, real_col, k_list=TOP_K_LIST, horizon=20, adaptive_gating=None):
    """Compute Sharpe gross+net for multiple top_k + optional adaptive scheme."""
    results = {}
    grouped = list(df.dropna(subset=[pred_col, real_col]).groupby("trade_date"))

    for K in k_list:
        rets = [float(g.nlargest(K, pred_col)[real_col].mean()) for _, g in grouped if len(g) >= K]
        if len(rets) < 2:
            results[str(K)] = {"sharpe": float("nan"), "sharpe_net": float("nan")}
            continue
        arr = np.asarray(rets); sd = arr.std(ddof=1)
        ann = np.sqrt(252.0 / max(horizon, 1))
        arr_net = arr - COST_ROUND_TRIP; sd_net = arr_net.std(ddof=1)
        results[str(K)] = {
            "sharpe": float(arr.mean()/sd*ann) if sd > 1e-9 else 0.0,
            "sharpe_net": float(arr_net.mean()/sd_net*ann) if sd_net > 1e-9 else 0.0,
            "mean": float(arr.mean()), "mean_net": float(arr_net.mean()),
        }

    if adaptive_gating is not None:
        q25 = adaptive_gating["q25"]; q50 = adaptive_gating["q50"]
        target_k_fn = adaptive_gating["target_k_fn"]
        adaptive_rets = []
        days_full = days_half = days_zero = 0
        ks_used = []
        for d, g in grouped:
            n = len(g)
            target_k = target_k_fn(n)
            if n < ADAPTIVE_K_FLOOR: continue
            top_g = g.nlargest(min(target_k, n), pred_col)
            mean_top = float(top_g[pred_col].mean())
            if mean_top < q25:
                days_zero += 1
                adaptive_rets.append(0.0); ks_used.append(0)
            elif mean_top < q50:
                k_actual = max(1, target_k // 2)
                days_half += 1
                adaptive_rets.append(float(g.nlargest(k_actual, pred_col)[real_col].mean()))
                ks_used.append(k_actual)
            else:
                days_full += 1
                adaptive_rets.append(float(top_g[real_col].mean()))
                ks_used.append(target_k)
        if len(adaptive_rets) >= 2:
            arr = np.asarray(adaptive_rets); sd = arr.std(ddof=1)
            ann = np.sqrt(252.0 / max(horizon, 1))
            non_zero_mask = np.array(ks_used) > 0
            arr_net = arr.copy()
            arr_net[non_zero_mask] = arr_net[non_zero_mask] - COST_ROUND_TRIP
            sd_net = arr_net.std(ddof=1)
            n_total = len(adaptive_rets)
            results["adaptive"] = {
                "sharpe": float(arr.mean()/sd*ann) if sd > 1e-9 else 0.0,
                "sharpe_net": float(arr_net.mean()/sd_net*ann) if sd_net > 1e-9 else 0.0,
                "pct_days_full": days_full/n_total, "pct_days_half": days_half/n_total,
                "pct_days_zero": days_zero/n_total, "avg_k_used": float(np.mean(ks_used)) if ks_used else 0.0,
            }
        else:
            results["adaptive"] = {"sharpe": float("nan"), "sharpe_net": float("nan")}
    return results


def dyn_agg_multi(df, score_col, trigger, k_list=TOP_K_LIST, adaptive_gating=None):
    rc = f"{trigger}_ret"; hc = f"{trigger}_hold"; tc = f"{trigger}_fired"
    sub = df.dropna(subset=[score_col, rc])
    if len(sub) == 0:
        return {"mean_hold": float("nan"), "pct_fired": float("nan")}
    mean_hold = float(sub[hc].mean())
    pct_fired = float(sub[tc].mean())
    grouped = list(sub.groupby("trade_date"))
    results = {"mean_hold": mean_hold, "pct_fired": pct_fired}
    ann = np.sqrt(252.0 / max(mean_hold, 1.0))

    for K in k_list:
        daily = [float(g.nlargest(K, score_col)[rc].mean()) for _, g in grouped if len(g) >= K]
        if len(daily) >= 2:
            arr = np.asarray(daily); sd = arr.std(ddof=1)
            arr_net = arr - COST_ROUND_TRIP; sd_net = arr_net.std(ddof=1)
            results[str(K)] = {
                "sharpe": float(arr.mean()/sd*ann) if sd > 1e-9 else 0.0,
                "sharpe_net": float(arr_net.mean()/sd_net*ann) if sd_net > 1e-9 else 0.0,
            }
        else:
            results[str(K)] = {"sharpe": float("nan"), "sharpe_net": float("nan")}

    if adaptive_gating is not None:
        q25 = adaptive_gating["q25"]; q50 = adaptive_gating["q50"]
        target_k_fn = adaptive_gating["target_k_fn"]
        rets = []
        ks_used = []
        for d, g in grouped:
            n = len(g)
            target_k = target_k_fn(n)
            if n < ADAPTIVE_K_FLOOR: continue
            top_g = g.nlargest(min(target_k, n), score_col)
            mean_top = float(top_g[score_col].mean())
            if mean_top < q25:
                rets.append(0.0); ks_used.append(0)
            elif mean_top < q50:
                k_actual = max(1, target_k // 2)
                rets.append(float(g.nlargest(k_actual, score_col)[rc].mean())); ks_used.append(k_actual)
            else:
                rets.append(float(top_g[rc].mean())); ks_used.append(target_k)
        if len(rets) >= 2:
            arr = np.asarray(rets); sd = arr.std(ddof=1)
            non_zero_mask = np.array(ks_used) > 0
            arr_net = arr.copy(); arr_net[non_zero_mask] = arr_net[non_zero_mask] - COST_ROUND_TRIP
            sd_net = arr_net.std(ddof=1)
            results["adaptive"] = {
                "sharpe": float(arr.mean()/sd*ann) if sd > 1e-9 else 0.0,
                "sharpe_net": float(arr_net.mean()/sd_net*ann) if sd_net > 1e-9 else 0.0,
            }
        else:
            results["adaptive"] = {"sharpe": float("nan"), "sharpe_net": float("nan")}
    return results


def compute_adaptive_thresholds(pred_df, train_start=TRAIN_START, train_end=TRAIN_END):
    train = pred_df[(pred_df["trade_date"] >= train_start) & (pred_df["trade_date"] <= train_end)]
    daily_top_means = []
    for d, g in train.groupby("trade_date"):
        n = len(g)
        target_k = max(ADAPTIVE_K_FLOOR, min(ADAPTIVE_K_CAP, round(n * ADAPTIVE_PCT)))
        if n < ADAPTIVE_K_FLOOR: continue
        daily_top_means.append(float(g.nlargest(target_k, "score")["score"].mean()))
    if len(daily_top_means) < 30: return None
    return {
        "q25": float(np.quantile(daily_top_means, ADAPTIVE_Q_LOW)),
        "q50": float(np.quantile(daily_top_means, ADAPTIVE_Q_MID)),
        "target_k_fn": lambda n: max(ADAPTIVE_K_FLOOR, min(ADAPTIVE_K_CAP, round(n * ADAPTIVE_PCT))),
    }


def family_importance(model, feature_names):
    """Aggregate LGB gain importance by family prefix."""
    gains = model.feature_importance(importance_type="gain")
    total_gain = float(gains.sum())
    families = {fam: 0.0 for fam in FAMILY_PREFIXES}
    families["other"] = 0.0
    top_features = []
    for name, gain in zip(feature_names, gains):
        matched = False
        for fam, prefix in FAMILY_PREFIXES.items():
            if name.startswith(prefix):
                families[fam] += float(gain)
                matched = True
                break
        if not matched:
            families["other"] += float(gain)
        top_features.append((name, float(gain)))
    family_pct = {fam: round(g / total_gain * 100, 2) if total_gain > 0 else 0.0
                  for fam, g in families.items()}
    top_features.sort(key=lambda x: -x[1])
    top20 = [(n, round(g / total_gain * 100, 2) if total_gain > 0 else 0.0) for n, g in top_features[:20]]
    return {"family_pct": family_pct, "top20": top20, "total_gain": total_gain}


def load_panel(name):
    if name == "ledashi":
        return _dt(pd.read_parquet(PANEL_PATHS[name]))
    p = pd.read_parquet(PANEL_PATHS[name])
    p = _dt(p)
    drop = [c for c in p.columns if c not in ("ts_code","trade_date")
            and not (pd.api.types.is_numeric_dtype(p[c]) or pd.api.types.is_bool_dtype(p[c]))]
    p = p.drop(columns=drop)
    for c in p.columns:
        if c in ("ts_code","trade_date"): continue
        if str(p[c].dtype).startswith("Int"): p[c] = p[c].astype("float32")
        elif p[c].dtype == np.float64: p[c] = p[c].astype(np.float32)
    return p


def load_es_predictions(name, path, score_col):
    p = Path(path)
    if p.is_dir():
        seed_dfs = []
        for sub in p.iterdir():
            if sub.is_dir():
                pf = sub / "predictions.parquet"
                if pf.exists():
                    seed_dfs.append(pd.read_parquet(pf))
        if not seed_dfs:
            print(f"  [{name}] no predictions.parquet in subdirs, skip")
            return None
        merged = seed_dfs[0][["trade_date", "ts_code"]].copy()
        score_lists = []
        for i, sdf in enumerate(seed_dfs):
            sdf = sdf[["trade_date","ts_code","score"]].rename(columns={"score": f"score_{i}"})
            merged = merged.merge(sdf, on=["trade_date","ts_code"], how="outer")
            score_lists.append(f"score_{i}")
        merged["score"] = merged[score_lists].mean(axis=1)
        return _dt(merged[["trade_date","ts_code","score"]])
    else:
        df = pd.read_parquet(path)
        if score_col not in df.columns:
            print(f"  [{name}] missing column {score_col}, available: {list(df.columns)[:10]}")
            return None
        df = df.rename(columns={score_col: "score"})
        return _dt(df[["trade_date","ts_code","score"]])


def eval_cell(pred_df, realized, exits, adaptive_gating):
    """Run full eval (static + dynamic) on prediction dataframe."""
    eval_df = pred_df.merge(realized, on=["ts_code","trade_date"], how="left")
    exit_eval = pred_df.merge(exits, on=["ts_code","trade_date"], how="inner")

    result = {"static": {}, "dynamic": {}, "n_pred_rows": len(pred_df)}
    for wname, (ws, we) in WINDOWS.items():
        sub = eval_df[(eval_df["trade_date"]>=ws)&(eval_df["trade_date"]<=we)]
        esub = exit_eval[(exit_eval["trade_date"]>=ws)&(exit_eval["trade_date"]<=we)]
        result["static"][wname] = {}
        for k in HORIZONS:
            ic = cross_sec_ic(sub, "score", f"ret_fwd{k}")
            sharpe_dict = top_k_sharpe_ann_multi(sub, "score", f"ret_fwd{k}",
                                                  horizon=k, adaptive_gating=adaptive_gating)
            result["static"][wname][f"fwd{k}"] = {"ic": ic["mean"], "ir": ic["ir"], "sizing": sharpe_dict}
        result["dynamic"][wname] = {}
        for trig in TRIGGERS:
            result["dynamic"][wname][trig] = dyn_agg_multi(esub, "score", trig,
                                                            adaptive_gating=adaptive_gating)
    return result


def main():
    t_total = time.time()
    print("[setup] universes ...")
    static_sets, pit_dfs = load_universes(UNIVERSES)

    print("\n[setup] realized + dyn-exit ...")
    realized, exits = compute_realized_and_exits()

    print("\n[setup] labels ...")
    all_labels = {}
    for v in LABELS:
        l = _dt(pd.read_parquet(LABEL_TEMPLATE.format(v=v), columns=["trade_date","ts_code","y"]))
        all_labels[v] = l
        print(f"  wave_{v}: {len(l):,} rows")

    matrix_results = {}
    family_importance_results = {}

    # Load existing checkpoint (resume support)
    CHECKPOINT = Path("data/kronos/outputs/matrix_v10_checkpoint.json")
    if CHECKPOINT.exists():
        prev = json.loads(CHECKPOINT.read_text())
        matrix_results = prev.get("results", {})
        family_importance_results = prev.get("family_importance", {})
        print(f"[resume] loaded checkpoint with {len(matrix_results)} cells", flush=True)

    # ===== Training cells (7 panels × 6 universes × 4 labels = 168) =====
    for panel_name in PANELS:
        # Skip panel entirely if all its cells are in checkpoint
        panel_cells_expected = [f"{label_v}_{univ}_{panel_name}"
                                 for univ in UNIVERSES for label_v in LABELS]
        if all(c in matrix_results for c in panel_cells_expected):
            print(f"\n[skip panel {panel_name}] all {len(panel_cells_expected)} cells already in checkpoint", flush=True)
            continue

        try:
            panel = load_panel(panel_name)
        except Exception as e:
            print(f"  [{panel_name}] LOAD FAILED: {e}")
            continue
        base_cols = [c for c in panel.columns if c not in ("ts_code","trade_date")]
        print(f"\n[panel {panel_name}] loaded, {len(panel):,} rows × {len(base_cols)} feats")

        for univ in UNIVERSES:
            upanel = filter_universe(panel, univ, static_sets, pit_dfs)
            print(f"\n--- {univ} on {panel_name}: {len(upanel):,} rows ---", flush=True)

            for label_v in LABELS:
                exp_id = f"{label_v}_{univ}_{panel_name}"
                if exp_id in matrix_results:
                    print(f"  [resume skip] {exp_id} already in checkpoint", flush=True)
                    continue
                label = all_labels[label_v]
                joined = upanel.merge(label, on=["ts_code","trade_date"], how="inner")
                train = joined[(joined["trade_date"]>=TRAIN_START) & (joined["trade_date"]<=TRAIN_END)]
                if len(train) < 10000:
                    print(f"  [SKIP] {exp_id}: train {len(train)} < 10K")
                    matrix_results[exp_id] = {"skipped": True, "train_rows": len(train)}
                    continue

                t = time.time()
                model = lgb.LGBMRegressor(**LGB_PARAMS)
                model.fit(train[base_cols], train["y"])
                del train, joined; gc.collect()

                # Factor importance
                fam_imp = family_importance(model.booster_, base_cols)
                family_importance_results[exp_id] = fam_imp

                preds = model.predict(upanel[base_cols]).astype(np.float32)
                pred_df = upanel[["ts_code","trade_date"]].copy()
                pred_df["score"] = preds

                # save pred parquet only for production candidate panels (ledashi + v3unified)
                if panel_name in ("ledashi", "v3unified"):
                    pred_df.to_parquet(OUT_DIR / f"pred_{exp_id}.parquet", compression="zstd")

                # adaptive thresholds (per cell, on train window predictions)
                adaptive_gating = compute_adaptive_thresholds(pred_df)

                result = eval_cell(pred_df, realized, exits, adaptive_gating)
                if adaptive_gating:
                    result["adaptive_meta"] = {
                        "q25": adaptive_gating["q25"], "q50": adaptive_gating["q50"],
                    }
                matrix_results[exp_id] = result

                r = result["static"]["H2_2025"]
                ic20 = r["fwd20"]["ic"] * 100
                sn50 = r["fwd20"]["sizing"].get("50", {}).get("sharpe_net", float("nan"))
                snadp = r["fwd20"]["sizing"].get("adaptive", {}).get("sharpe_net", float("nan"))
                q1 = result["static"]["Q1_2026"]["fwd20"]["ic"] * 100
                print(f"  {exp_id}: train {time.time()-t:.0f}s | H2 fwd20 IC={ic20:+.3f}% Sharpe50_NET={sn50:+.2f} ADAPT_NET={snadp:+.2f} | Q1 IC={q1:+.3f}%", flush=True)

                del model, preds, pred_df, result; gc.collect()
            del upanel; gc.collect()
            # Per-universe checkpoint (finer-grained than per-panel)
            CHECKPOINT.write_text(json.dumps({
                "results": matrix_results, "family_importance": family_importance_results,
            }, indent=2, default=str))
            print(f"[checkpoint] saved after universe {univ} on {panel_name}: {len(matrix_results)} cells", flush=True)

        del panel; gc.collect()

    # ===== ES eval-only cells (6 paris/ledashi proximity models, MAIN_BOARD) =====
    print("\n[ES] eval pre-trained proximity models on MAIN_BOARD ...")
    main_board_set = static_sets["MAIN_BOARD"]
    for es_name, es_path, score_col in ES_CELLS:
        print(f"\n--- {es_name} from {es_path} ---", flush=True)
        try:
            pred_df = load_es_predictions(es_name, es_path, score_col)
        except Exception as e:
            print(f"  load error: {e}")
            continue
        if pred_df is None: continue
        # Filter to MAIN_BOARD
        pred_df = pred_df[pred_df["ts_code"].isin(main_board_set)].copy()
        if len(pred_df) < 1000:
            print(f"  too few rows after MAIN_BOARD filter: {len(pred_df)}")
            continue

        adaptive_gating = compute_adaptive_thresholds(pred_df)
        result = eval_cell(pred_df, realized, exits, adaptive_gating)
        if adaptive_gating:
            result["adaptive_meta"] = {"q25": adaptive_gating["q25"], "q50": adaptive_gating["q50"]}
        cell_id = f"{es_name}_MAIN_BOARD"
        matrix_results[cell_id] = result
        r = result["static"]["H2_2025"]
        ic20 = r["fwd20"]["ic"] * 100
        sn50 = r["fwd20"]["sizing"].get("50", {}).get("sharpe_net", float("nan"))
        print(f"  {cell_id}: rows={len(pred_df):,} | H2 fwd20 IC={ic20:+.3f}% Sharpe50_NET={sn50:+.2f}", flush=True)

    # ===== Output =====
    out = {
        "config": {
            "tier": "v10 — 7 panel × 6 universe × 4 label × 7 sizing + 6 ES eval-only",
            "panels": list(PANELS), "universes": list(UNIVERSES), "labels": list(LABELS),
            "horizons": list(HORIZONS), "top_k_list": list(TOP_K_LIST),
            "triggers": list(TRIGGERS), "cost_round_trip": COST_ROUND_TRIP,
            "adaptive": {"pct": ADAPTIVE_PCT, "k_cap": ADAPTIVE_K_CAP,
                         "k_floor": ADAPTIVE_K_FLOOR, "q_low": ADAPTIVE_Q_LOW, "q_mid": ADAPTIVE_Q_MID},
            "lgb_params": LGB_PARAMS,
        },
        "results": matrix_results,
        "family_importance": family_importance_results,
        "total_time_s": time.time()-t_total,
    }
    out_path = Path("data/kronos/outputs/matrix_v10_results.json")
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

    # Markdown table
    lines = ["# Matrix v10 — FULL coverage (174 cells)", ""]
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M Asia/Shanghai')}  runtime {time.time()-t_total:.0f}s_")
    lines.append("")
    lines.append("**Scope**: 7 panel × 6 universe × 4 wave label = 168 trained cells + 6 ES eval-only = 174 cells")
    lines.append("")

    # Table 1: H2 fwd20 IC by panel × universe per label
    for label_v in LABELS:
        lines.append(f"## H2_2025 fwd20 IC % — wave_{label_v}")
        lines.append("")
        lines.append("| universe | " + " | ".join(PANELS) + " |")
        lines.append("|---|" + "---|"*len(PANELS))
        for univ in UNIVERSES:
            row = [univ]
            for panel in PANELS:
                cid = f"{label_v}_{univ}_{panel}"
                r = matrix_results.get(cid, {})
                if r.get("skipped"): row.append("skip")
                else:
                    v = r.get("static", {}).get("H2_2025", {}).get("fwd20", {}).get("ic", float("nan"))
                    row.append(f"{v*100:+.3f}" if v == v else "n/a")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Table 2: Best Sharpe NET per (universe, label) — which panel wins
    lines.append("## Best Sharpe NET (top_k=50) per (universe, label) — winning panel")
    lines.append("")
    lines.append("| universe | wave_v1 | wave_v2 | wave_v3 | wave_v4 |")
    lines.append("|---|---|---|---|---|")
    for univ in UNIVERSES:
        row = [univ]
        for label_v in LABELS:
            best_sh, best_pan = float("-inf"), "?"
            for panel in PANELS:
                cid = f"{label_v}_{univ}_{panel}"
                r = matrix_results.get(cid, {})
                if r.get("skipped"): continue
                s = r.get("static", {}).get("H2_2025", {}).get("fwd20", {}).get("sizing", {}).get("50", {}).get("sharpe_net", float("-inf"))
                if s > best_sh: best_sh, best_pan = s, panel
            row.append(f"{best_sh:+.2f} ({best_pan})" if best_sh > float("-inf") else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Table 3: Sharpe NET vs top_k sweep (best panel × best label per universe, on H2 fwd20)
    lines.append("## Sharpe NET sweep by sizing (H2 fwd20, best panel/label per universe)")
    lines.append("")
    lines.append("| universe | best (panel, label) | K=5 | K=10 | K=15 | K=20 | K=30 | K=50 | adaptive |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for univ in UNIVERSES:
        best_sh, best_key = float("-inf"), None
        for label_v in LABELS:
            for panel in PANELS:
                cid = f"{label_v}_{univ}_{panel}"
                r = matrix_results.get(cid, {})
                if r.get("skipped"): continue
                s = r.get("static", {}).get("H2_2025", {}).get("fwd20", {}).get("sizing", {}).get("50", {}).get("sharpe_net", float("-inf"))
                if s > best_sh: best_sh, best_key = s, (panel, label_v)
        if best_key is None:
            lines.append(f"| {univ} | n/a | - | - | - | - | - | - | - |")
            continue
        panel, label_v = best_key
        cid = f"{label_v}_{univ}_{panel}"
        sizing = matrix_results[cid]["static"]["H2_2025"]["fwd20"]["sizing"]
        row = [univ, f"({panel}, wave_{label_v})"]
        for K in TOP_K_LIST:
            v = sizing.get(str(K), {}).get("sharpe_net", float("nan"))
            row.append(f"{v:+.2f}" if v == v else "n/a")
        v_a = sizing.get("adaptive", {}).get("sharpe_net", float("nan"))
        row.append(f"{v_a:+.2f}" if v_a == v_a else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Table 4: Best dyn-exit Sharpe NET (top_k=50) per cell
    lines.append("## Best dyn-exit Sharpe NET per universe (top_k=50, H2_2025)")
    lines.append("")
    lines.append("| universe | label | panel | trigger | Sharpe NET |")
    lines.append("|---|---|---|---|---|")
    for univ in UNIVERSES:
        rows = []
        for label_v in LABELS:
            for panel in PANELS:
                cid = f"{label_v}_{univ}_{panel}"
                r = matrix_results.get(cid, {})
                if r.get("skipped"): continue
                d = r.get("dynamic", {}).get("H2_2025", {})
                for trig in TRIGGERS:
                    s = d.get(trig, {}).get("50", {}).get("sharpe_net", float("-inf"))
                    if s > float("-inf"):
                        rows.append((s, label_v, panel, trig))
        rows.sort(reverse=True)
        if rows:
            s, lv, p, tr = rows[0]
            lines.append(f"| {univ} | wave_{lv} | {p} | {tr} | {s:+.2f} |")
    lines.append("")

    # Table 5: ES (eval-only) MAIN_BOARD cells
    lines.append("## ES (paris/ledashi pre-trained proximity models) — MAIN_BOARD")
    lines.append("")
    lines.append("| ES model | H2 fwd20 IC% | Sharpe50_NET | Q1 fwd20 IC% | best dyn-exit | Sharpe NET |")
    lines.append("|---|---|---|---|---|---|")
    for es_name, _, _ in ES_CELLS:
        cid = f"{es_name}_MAIN_BOARD"
        r = matrix_results.get(cid, {})
        if not r:
            lines.append(f"| {es_name} | n/a | n/a | n/a | n/a | n/a |")
            continue
        s = r["static"]
        ic_h2 = s["H2_2025"]["fwd20"]["ic"] * 100
        sh_h2 = s["H2_2025"]["fwd20"]["sizing"].get("50", {}).get("sharpe_net", float("nan"))
        ic_q1 = s["Q1_2026"]["fwd20"]["ic"] * 100
        d = r.get("dynamic", {}).get("H2_2025", {})
        best_trig, best_sh = "n/a", float("-inf")
        for trig in TRIGGERS:
            t_sh = d.get(trig, {}).get("50", {}).get("sharpe_net", float("-inf"))
            if t_sh > best_sh: best_sh, best_trig = t_sh, trig
        lines.append(f"| {es_name} | {ic_h2:+.3f} | {sh_h2:+.2f} | {ic_q1:+.3f} | {best_trig} | {best_sh:+.2f} |")
    lines.append("")

    # Table 6: Family importance — top 5 families per panel (averaged over universes × labels)
    lines.append("## Factor family importance — avg gain % per panel (top 5 families)")
    lines.append("")
    for panel in PANELS:
        lines.append(f"### Panel: {panel}")
        family_avg = {}
        family_count = 0
        for exp_id, fi in family_importance_results.items():
            if not exp_id.endswith(f"_{panel}"): continue
            family_count += 1
            for fam, pct in fi["family_pct"].items():
                family_avg[fam] = family_avg.get(fam, 0.0) + pct
        if family_count > 0:
            family_avg = {f: v / family_count for f, v in family_avg.items()}
            ranked = sorted(family_avg.items(), key=lambda x: -x[1])[:8]
            lines.append("| family | avg gain % |")
            lines.append("|---|---|")
            for fam, pct in ranked:
                if pct > 0.5:
                    lines.append(f"| {fam} | {pct:.2f} |")
        lines.append("")

    Path("data/kronos/outputs/matrix_v10_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[saved] matrix_v10_table.md")
    print(f"[done] total {time.time()-t_total:.0f}s ({(time.time()-t_total)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
