"""Matrix ablation — 4 labels × 3 feature sets × 4 universes × NEW train window.

48 LGB trains. For each cell:
  - Train on (panel ∩ label ∩ universe ∩ NEW_WINDOW 2024-04 ~ 2024-12)
  - Predict on full (panel ∩ universe)
  - Eval IC + top-50 ann Sharpe on H1/H2/Q1/Q2 2025-2026 × fwd1/3/5/10/20/30
  - Eval dynamic-exit triggers (A-G + H/I MACD/KDJ death cross) — shared per-stock sim

Output:
  data/kronos/outputs/matrix_results.json
  data/kronos/outputs/matrix_predictions_<exp_id>.parquet (48 files)
  data/kronos/outputs/matrix_table.md
"""
from __future__ import annotations

import os

# Public consumers: set AURUMQ_HANDOFF_INBOX to your local data dir.
# Default: data/handoffs/inbox/<bundle_dir>/<file>
_HANDOFF_INBOX = os.environ.get("AURUMQ_HANDOFF_INBOX", "data/handoffs/inbox")


import json
import time
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# --- inputs ---
PANEL = "data/p3_4070_long/feature_panel_v3_344_pruned.parquet"
LABEL_TEMPLATE = "data/p3_4070_long/target_y_wave_{v}.parquet"
KRONOS = "data/kronos/outputs/kronos_predictions_daily.parquet"
PANEL_CLOSE = "data/p3_4070_long/stock_close_volume_daily.parquet"
TECH_LINES = f"{_HANDOFF_INBOX}/2026-05-14-paris-macd-kdj-raw/tech_lines_daily.parquet"
UNIVERSE_DIR = Path("data/universes")

OUT_DIR = Path("data/kronos/outputs/matrix")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = pd.Timestamp("2024-04-01").date()
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
FEATURE_SETS = ("panel_only", "panel_kronos", "panel_kronos_tech")
UNIVERSES = ("MAIN_BOARD", "NPF", "NPF_FULL", "GROWTH_BOARDS")
KRONOS_COLS = ["kron_pred_return_fwd5", "kron_pred_return_fwd20"]
TECH_COLS = ["macd_line", "macd_signal", "macd_hist", "kdj_k", "kdj_d", "kdj_j"]

TRIGGERS = ("A_stop_5pct", "C_vol_drop", "E_trail_5pct", "F_trend_break",
            "G_K_max", "H_macd_death", "I_kdj_death")

LGB_PARAMS = dict(
    objective="regression", metric="rmse",
    learning_rate=0.05, num_leaves=63,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    min_data_in_leaf=200, verbose=-1, n_estimators=400, num_threads=-1,
)


def _dt(df):
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def load_universes() -> dict[str, frozenset[str]]:
    """Load 4 static universe membership sets."""
    out = {}
    for u in UNIVERSES:
        p = UNIVERSE_DIR / f"{u}_membership.parquet"
        df = pd.read_parquet(p)
        out[u] = frozenset(df["stock_code"].tolist())
        print(f"  {u}: {len(out[u]):,} stocks")
    return out


def load_panel_with_features() -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Load full panel + Kronos cols (renamed kron_*) + tech_lines (macd/kdj)."""
    t = time.time()
    print("[load] panel ...", flush=True)
    panel = _dt(pd.read_parquet(PANEL))
    base_cols = [c for c in panel.columns if c not in ("ts_code", "trade_date")]
    print(f"  panel: {len(panel):,} rows × {len(base_cols)} feats  ({time.time()-t:.0f}s)")

    print("[load] kronos ...", flush=True)
    kron = _dt(pd.read_parquet(
        KRONOS, columns=["trade_date", "ts_code", "pred_return_fwd5", "pred_return_fwd20"]
    ))
    kron = kron.rename(columns={"pred_return_fwd5": "kron_pred_return_fwd5",
                                 "pred_return_fwd20": "kron_pred_return_fwd20"})
    panel = panel.merge(kron, on=["ts_code", "trade_date"], how="left")
    del kron
    for c in KRONOS_COLS:
        panel[c] = panel[c].fillna(0.0).astype(np.float32)

    print("[load] tech_lines ...", flush=True)
    tech = _dt(pd.read_parquet(TECH_LINES))
    panel = panel.merge(tech, on=["ts_code", "trade_date"], how="left")
    del tech
    for c in TECH_COLS:
        panel[c] = panel[c].fillna(0.0).astype(np.float32)

    print(f"[load] DONE: {len(panel):,} rows × {panel.shape[1]} cols  ({time.time()-t:.0f}s)")
    return panel, base_cols, KRONOS_COLS, TECH_COLS


def compute_realized_and_sim() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Realized fwdK + per (entry_date, ts_code) dynamic-exit trigger outcomes.

    Returns
    -------
    realized: (ts_code, trade_date, ret_fwd1, ret_fwd3, ret_fwd5, ret_fwd10, ret_fwd20, ret_fwd30)
    exits: (ts_code, trade_date, {trigger}_realized_ret, {trigger}_holding_days, {trigger}_triggered) × 7 triggers
    """
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

    print("[realized] computing fwd-K ...", flush=True)
    for k in HORIZONS:
        p[f"ret_fwd{k}"] = (p.groupby("ts_code")["adj_close"].shift(-k) / p["adj_close"] - 1.0).astype(np.float32)

    print("[realized] loading tech_lines + merging ...", flush=True)
    tech = _dt(pd.read_parquet(TECH_LINES))
    p = p.merge(tech, on=["ts_code", "trade_date"], how="left")
    del tech

    realized = p[["ts_code", "trade_date"] + [f"ret_fwd{k}" for k in HORIZONS]].copy()
    print(f"  realized: {len(realized):,}  ({time.time()-t:.0f}s)")

    print("[dyn-exit] simulating per-stock ...", flush=True)
    exit_rows = []
    n_syms = 0
    for sym, sub in p.groupby("ts_code"):
        sub = sub.reset_index(drop=True)
        adj_close = sub["adj_close"].to_numpy()
        vol = sub["vol"].to_numpy()
        pct_chg = sub["pct_chg"].to_numpy()
        ma20_vol = sub["ma20_vol"].to_numpy()
        ma5_close = sub["ma5_close"].to_numpy()
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

        firmap = {
            "A_stop_5pct": A_f, "C_vol_drop": C_f, "E_trail_5pct": E_f,
            "F_trend_break": F_f, "G_K_max": G_f, "H_macd_death": H_f, "I_kdj_death": I_f,
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
    print(f"  exits: {len(exits):,} rows from {n_syms} stocks  ({time.time()-t:.0f}s)")
    del p
    return realized, exits


def cross_sec_ic(df, pred_col, real_col) -> dict:
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


def top_k_sharpe_ann(df, pred_col, real_col, k=TOP_K, ann=252) -> dict:
    rets = []
    for _, g in df.dropna(subset=[pred_col, real_col]).groupby("trade_date"):
        if len(g) < k: continue
        rets.append(float(g.nlargest(k, pred_col)[real_col].mean()))
    if len(rets) < 2:
        return {"sharpe": float("nan"), "n_days": len(rets), "mean": float("nan")}
    arr = np.asarray(rets); sd = arr.std(ddof=1)
    return {"sharpe": float(arr.mean() / sd * np.sqrt(ann)) if sd > 1e-9 else 0.0,
            "n_days": int(len(arr)), "mean": float(arr.mean())}


def dyn_agg(df, score_col, trigger) -> dict:
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


def run_experiment(panel, base_cols, label_v, feature_set, universe, universes_dict, all_labels):
    exp_id = f"{label_v}_{feature_set}_{universe}"
    print(f"\n=== {exp_id} ===", flush=True)
    feat_cols = list(base_cols)
    if feature_set in ("panel_kronos", "panel_kronos_tech"):
        feat_cols += KRONOS_COLS
    if feature_set == "panel_kronos_tech":
        feat_cols += TECH_COLS

    # Universe filter
    codes = universes_dict[universe]
    upanel = panel[panel["ts_code"].isin(codes)]
    print(f"  universe={universe} (n={len(codes)}): panel {len(upanel):,} rows")

    # Train subset
    label = all_labels[label_v]
    train = upanel.merge(label, on=["ts_code", "trade_date"], how="inner")
    train = train[(train["trade_date"] >= TRAIN_START) & (train["trade_date"] <= TRAIN_END)]
    print(f"  train rows: {len(train):,}  ({len(feat_cols)} feats)")

    if len(train) < 10000:
        print(f"  SKIP: train rows < 10K")
        return None

    t = time.time()
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(train[feat_cols], train["y"])
    print(f"  train: {time.time()-t:.0f}s")
    del train

    t = time.time()
    preds = model.predict(upanel[feat_cols]).astype(np.float32)
    print(f"  predict: {time.time()-t:.0f}s ({len(preds):,} rows)")

    out_pred_df = upanel[["ts_code", "trade_date"]].copy()
    out_pred_df["score"] = preds
    out_pred_df.to_parquet(OUT_DIR / f"pred_{exp_id}.parquet", compression="zstd")
    return out_pred_df


def main() -> int:
    t_total = time.time()
    universes_dict = load_universes()
    panel, base_cols, _, _ = load_panel_with_features()

    print("[load] 4 labels ...", flush=True)
    all_labels = {}
    for v in LABELS:
        l = _dt(pd.read_parquet(LABEL_TEMPLATE.format(v=v), columns=["trade_date", "ts_code", "y"]))
        all_labels[v] = l
        print(f"  wave_{v}: {len(l):,} rows")

    print("\n[realized + dyn-exit] ...")
    realized, exits = compute_realized_and_sim()

    # Run 48 experiments
    matrix_results = {}
    for label_v, fset, univ in product(LABELS, FEATURE_SETS, UNIVERSES):
        exp_id = f"{label_v}_{fset}_{univ}"
        pred_df = run_experiment(panel, base_cols, label_v, fset, univ, universes_dict, all_labels)
        if pred_df is None:
            matrix_results[exp_id] = {"skipped": True}
            continue

        # Eval
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
              f"H1 fwd20 IC={result['static']['H1_2025']['fwd20']['ic']:+.4f}")

    out = {
        "config": {
            "labels": list(LABELS), "feature_sets": list(FEATURE_SETS),
            "universes": list(UNIVERSES), "horizons": list(HORIZONS),
            "windows": {k: [str(v[0]), str(v[1])] for k, v in WINDOWS.items()},
            "train_window": [str(TRAIN_START), str(TRAIN_END)],
            "lgb_params": LGB_PARAMS, "top_k": TOP_K, "triggers": list(TRIGGERS),
        },
        "results": matrix_results,
        "total_time_s": time.time() - t_total,
    }
    Path("data/kronos/outputs/matrix_results.json").write_text(json.dumps(out, indent=2, default=str))

    # Build markdown table: per universe, label × feature_set × fwd5/20 H1 IC
    lines = ["# Matrix ablation results", ""]
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M Asia/Shanghai')}  runtime {time.time()-t_total:.0f}s_")
    lines.append("")
    lines.append(f"**Train window**: {TRAIN_START} ~ {TRAIN_END} (paris-locked NEW window)")
    lines.append("")

    short_trig = {"A_stop_5pct":"A", "C_vol_drop":"C", "E_trail_5pct":"E", "F_trend_break":"F",
                  "G_K_max":"G", "H_macd_death":"H", "I_kdj_death":"I"}

    for wname in WINDOWS:
        lines.append(f"\n## {wname} — mean IC × 100")
        for univ in UNIVERSES:
            lines.append(f"\n### {univ}")
            hdr = ["label"] + [f"{fs} fwd{k}" for fs in FEATURE_SETS for k in (5, 20)]
            lines.append("| " + " | ".join(hdr) + " |")
            lines.append("|" + "---|" * len(hdr))
            for label_v in LABELS:
                row = [f"wave_{label_v}"]
                for fset in FEATURE_SETS:
                    for k in (5, 20):
                        eid = f"{label_v}_{fset}_{univ}"
                        r = matrix_results.get(eid, {})
                        if r.get("skipped"):
                            row.append("skip")
                        else:
                            v = r.get("static", {}).get(wname, {}).get(f"fwd{k}", {}).get("ic", float("nan"))
                            row.append(f"{v*100:+.3f}" if v == v else "n/a")
                lines.append("| " + " | ".join(row) + " |")

        lines.append(f"\n### {wname} top-50 ann Sharpe @ fwd20")
        hdr = ["label / feature_set"] + list(UNIVERSES)
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "---|" * len(hdr))
        for label_v in LABELS:
            for fset in FEATURE_SETS:
                row = [f"wave_{label_v} / {fset}"]
                for univ in UNIVERSES:
                    eid = f"{label_v}_{fset}_{univ}"
                    r = matrix_results.get(eid, {})
                    if r.get("skipped"):
                        row.append("skip")
                    else:
                        v = r.get("static", {}).get(wname, {}).get("fwd20", {}).get("top50_sharpe", float("nan"))
                        row.append(f"{v:+.2f}" if v == v else "n/a")
                lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Dynamic-exit top-50 ann Sharpe (H1 2025, F_trend_break + I_kdj_death only)")
    lines.append("")
    for trig in ("F_trend_break", "I_kdj_death"):
        lines.append(f"\n### H1_2025 — {trig}")
        hdr = ["label / feature_set"] + list(UNIVERSES)
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "---|" * len(hdr))
        for label_v in LABELS:
            for fset in FEATURE_SETS:
                row = [f"wave_{label_v} / {fset}"]
                for univ in UNIVERSES:
                    eid = f"{label_v}_{fset}_{univ}"
                    r = matrix_results.get(eid, {})
                    if r.get("skipped"):
                        row.append("skip")
                    else:
                        v = r.get("dynamic", {}).get("H1_2025", {}).get(trig, {}).get("top50_sharpe", float("nan"))
                        row.append(f"{v:+.2f}" if v == v else "n/a")
                lines.append("| " + " | ".join(row) + " |")

    Path("data/kronos/outputs/matrix_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[saved] matrix_results.json + matrix_table.md + 48 predictions parquets")
    print(f"[done] total {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
