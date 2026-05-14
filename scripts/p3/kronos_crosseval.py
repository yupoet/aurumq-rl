"""Kronos cross-eval — static fwdK + dynamic-exit triggers × all available score sources.

Score sources (6):
  paris_t3_baseline       (data/p3_4070_long/baseline_predictions.parquet)
  ledashi_sl_final        (runs/sl_final/predictions.parquet)
  ledashi_lgb_baseline    (LGB on 226 features, no Kronos)
  ledashi_lgb_kronos      (LGB on 228 features = 226 + 2 Kronos cols)
  kronos_raw_fwd5         (kronos_predictions_daily.parquet:pred_return_fwd5)
  kronos_raw_fwd20        (kronos_predictions_daily.parquet:pred_return_fwd20)

Static fwdK horizons: 1, 2, 3, 5, 10, 15, 20, 30.

Dynamic-exit triggers (7 — H MACD, I KDJ deferred until paris ships raw MACD/KDJ
time series; current panel only has decayed event flags, not raw line crossings):
  A. stop_loss_5pct      : cum_ret < -5%
  B. stop_loss_3pct      : cum_ret < -3%
  C. vol_drop            : vol > 2*MA20(vol) AND day_pct_chg < -3%
  D. vol_drop_or_stop    : C OR A
  E. trailing_stop_5pct  : peak_cum_ret - cur_cum_ret > 5%
  F. trend_break         : adj_close < MA5(adj_close)
  G. K_max_only          : 30d forced (baseline; never triggers)

K_max = 30 (forced exit). Entry assumed at close[D]; exit on close[D+k] when fires.

Eval windows (3): H1_2025 / H2_2025 / Q1_2026.

NOTE on the Q1 2026 NaN bug (from initial ablation): training uses panel∩label inner
join (label ends 2025-11), but PREDICTION + EVAL operate on FULL panel (covers Q1 2026
via realized returns from stock_close_volume_daily up to 2026-05-12). Fixed here by
separating train_subset from full_panel.

Output:
  data/kronos/outputs/crosseval_predictions.parquet  (all 6 score cols, full panel)
  data/kronos/outputs/crosseval_results.json
  data/kronos/outputs/crosseval_table.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# --- inputs ---
PANEL = "data/p3_4070_long/feature_panel_v3_344_pruned.parquet"
LABEL = "data/p3_4070_long/target_y_wave_v2.parquet"
KRONOS = "data/kronos/outputs/kronos_predictions_daily.parquet"
PANEL_CLOSE = "data/p3_4070_long/stock_close_volume_daily.parquet"
PARIS_T3 = "data/p3_4070_long/baseline_predictions.parquet"
SL_FINAL = "runs/sl_final/predictions.parquet"

TRAIN_START = pd.Timestamp("2018-01-02").date()
TRAIN_END   = pd.Timestamp("2024-12-31").date()

WINDOWS = {
    "H1_2025": (pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-06-30").date()),
    "H2_2025": (pd.Timestamp("2025-07-01").date(), pd.Timestamp("2025-12-31").date()),
    "Q1_2026": (pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-03-31").date()),
}
HORIZONS = (1, 2, 3, 5, 10, 15, 20, 30)
K_MAX = 30

LGB_PARAMS = dict(
    objective="regression", metric="rmse",
    learning_rate=0.05, num_leaves=63,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    min_data_in_leaf=200, verbose=-1, n_estimators=400, num_threads=-1,
)

TRIGGERS = ("A_stop_5pct", "B_stop_3pct", "C_vol_drop", "D_vol_or_stop",
            "E_trail_5pct", "F_trend_break", "G_K_max_only")

SCORES = ("paris_t3_baseline", "ledashi_sl_final",
          "ledashi_lgb_baseline", "ledashi_lgb_kronos",
          "kronos_raw_fwd5", "kronos_raw_fwd20")


def _dt(df):
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def load_features() -> tuple[pd.DataFrame, list[str]]:
    """Load FULL panel + Kronos preds. Returns panel and feature_cols.
    The panel is the prediction surface (not yet inner-joined with label)."""
    t = time.time()
    print("[panel] loading feature_panel_v3_344_pruned ...", flush=True)
    panel = _dt(pd.read_parquet(PANEL))
    print(f"  rows: {len(panel):,}  cols: {panel.shape[1]}  ({time.time()-t:.0f}s)")

    print("[kronos] left-joining kronos preds ...", flush=True)
    kron = _dt(pd.read_parquet(
        KRONOS, columns=["trade_date", "ts_code", "pred_return_fwd5", "pred_return_fwd20"]
    ))
    panel = panel.merge(kron, on=["trade_date", "ts_code"], how="left")
    del kron
    panel["pred_return_fwd5"]  = panel["pred_return_fwd5"].fillna(0.0).astype(np.float32)
    panel["pred_return_fwd20"] = panel["pred_return_fwd20"].fillna(0.0).astype(np.float32)
    print(f"  panel after kronos: {len(panel):,} rows  ({time.time()-t:.0f}s)")

    feature_cols = [c for c in panel.columns if c not in ("ts_code", "trade_date")]
    print(f"[feat] {len(feature_cols)} feature cols (incl. 2 Kronos)")
    return panel, feature_cols


def train_predict_lgb(panel: pd.DataFrame, feature_cols: list[str], tag: str) -> np.ndarray:
    """Train on panel ∩ label ∩ train_range, predict on FULL panel."""
    print(f"\n[{tag}] training ({len(feature_cols)} features) ...", flush=True)
    label = _dt(pd.read_parquet(LABEL, columns=["trade_date", "ts_code", "y"]))
    train = panel.merge(label, on=["trade_date", "ts_code"], how="inner")
    del label
    train = train[(train["trade_date"] >= TRAIN_START) & (train["trade_date"] <= TRAIN_END)]
    print(f"  train rows: {len(train):,}  range {train['trade_date'].min()} ~ {train['trade_date'].max()}")
    t = time.time()
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(train[feature_cols], train["y"])
    print(f"  train time: {time.time()-t:.0f}s")
    del train
    t = time.time()
    preds = model.predict(panel[feature_cols]).astype(np.float32)
    print(f"  predict on full panel: {len(panel):,} rows  ({time.time()-t:.0f}s)")
    return preds, model.booster_


def compute_realized_panel() -> tuple[pd.DataFrame, dict]:
    """Realized fwd-K returns from stock_close_volume_daily, plus dynamic-exit per-stock arrays.

    Returns
    -------
    realized : DataFrame with (ts_code, trade_date, ret_fwd1, ret_fwd2, ..., ret_fwd30)
    per_stock : dict[ts_code] -> {
        "dates": np.ndarray of date,
        "adj_close": np.ndarray float32,
        "vol": np.ndarray float32,
        "pct_chg": np.ndarray float32,
        "ma20_vol": np.ndarray float32,
        "ma5_close": np.ndarray float32,
    }
    """
    t = time.time()
    print("[realized] loading close+vol ...", flush=True)
    p = pd.read_parquet(PANEL_CLOSE, columns=["ts_code", "trade_date", "close",
                                              "adj_factor", "volume"])
    p = _dt(p).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    latest = p.groupby("ts_code")["adj_factor"].transform("last")
    p["adj_close"] = (p["close"] * p["adj_factor"] / latest).astype(np.float32)
    p["vol"] = p["volume"].astype(np.float32)
    p["pct_chg"] = (p.groupby("ts_code")["adj_close"].pct_change()).astype(np.float32)
    p["ma20_vol"] = (p.groupby("ts_code")["vol"]
                     .transform(lambda x: x.rolling(20, min_periods=5).mean())).astype(np.float32)
    p["ma5_close"] = (p.groupby("ts_code")["adj_close"]
                      .transform(lambda x: x.rolling(5, min_periods=2).mean())).astype(np.float32)

    print(f"  basic stats done ({time.time()-t:.0f}s)")

    # Realized fwd-K returns (for static eval)
    realized_cols = ["ts_code", "trade_date"]
    for k in HORIZONS:
        col = f"ret_fwd{k}"
        p[col] = (p.groupby("ts_code")["adj_close"].shift(-k) / p["adj_close"] - 1.0).astype(np.float32)
        realized_cols.append(col)
    realized = p[realized_cols].copy()
    print(f"  realized fwd-K done ({time.time()-t:.0f}s)")

    # Per-stock arrays for dynamic exit simulation
    print("[realized] building per-stock arrays ...", flush=True)
    per_stock: dict = {}
    for sym, g in p.groupby("ts_code"):
        n = len(g)
        if n < K_MAX + 5:
            continue
        per_stock[sym] = {
            "dates": g["trade_date"].to_numpy(),
            "adj_close": g["adj_close"].to_numpy(),
            "vol": g["vol"].to_numpy(),
            "pct_chg": g["pct_chg"].to_numpy(),
            "ma20_vol": g["ma20_vol"].to_numpy(),
            "ma5_close": g["ma5_close"].to_numpy(),
        }
    print(f"  {len(per_stock)} stocks indexed ({time.time()-t:.0f}s)")
    return realized, per_stock


def simulate_dynamic_exit(per_stock: dict, sym_to_dates: dict) -> pd.DataFrame:
    """For each (entry_date, ts_code), simulate 30-day forward exit under 7 triggers.

    Returns DataFrame: ts_code, trade_date, plus per-trigger columns:
        {TRIG}_realized_ret  (float32)
        {TRIG}_holding_days  (int8, 1..30)
        {TRIG}_triggered     (bool)
    """
    t = time.time()
    print("[dyn-exit] simulating ...", flush=True)

    rows = []
    for sym, arrs in per_stock.items():
        dates = arrs["dates"]
        adj_close = arrs["adj_close"]
        vol = arrs["vol"]
        pct_chg = arrs["pct_chg"]
        ma20_vol = arrs["ma20_vol"]
        ma5_close = arrs["ma5_close"]
        n = len(dates)

        # Absolute trigger arrays (do not depend on entry_t)
        # C: vol > 2*MA20(vol) AND pct_chg < -3%
        C_abs = (vol > 2.0 * ma20_vol) & (pct_chg < -0.03)
        # F: close < MA5(close)
        F_abs = adj_close < ma5_close

        # Valid entry indices: must have K_MAX days of future data
        max_entry = n - K_MAX - 1
        if max_entry < 0:
            continue
        entries = np.arange(0, max_entry + 1)  # 0..max_entry
        # Build (n_entries, K_MAX) forward index matrix
        forward_idx = entries[:, None] + np.arange(1, K_MAX + 1)[None, :]
        forward_close = adj_close[forward_idx]
        entry_close = adj_close[entries][:, None]
        cum_ret = forward_close / entry_close - 1.0       # shape (n_entries, K_MAX)
        peak_ret = np.maximum.accumulate(cum_ret, axis=1)  # path-max

        # Trigger fired matrices (shape (n_entries, K_MAX))
        A_fired = cum_ret < -0.05
        B_fired = cum_ret < -0.03
        C_fired = C_abs[forward_idx]
        D_fired = A_fired | C_fired
        E_fired = (peak_ret - cum_ret) > 0.05
        F_fired = F_abs[forward_idx]
        G_fired = np.zeros_like(A_fired)

        # For each trigger, find first-fire idx (0..K_MAX-1) or K_MAX-1 if never (force exit)
        firmap = {
            "A_stop_5pct": A_fired,
            "B_stop_3pct": B_fired,
            "C_vol_drop": C_fired,
            "D_vol_or_stop": D_fired,
            "E_trail_5pct": E_fired,
            "F_trend_break": F_fired,
            "G_K_max_only": G_fired,
        }

        rec_dict = {"ts_code": np.full(len(entries), sym, dtype=object),
                    "trade_date": dates[entries]}
        for trig, fired in firmap.items():
            any_fired = fired.any(axis=1)
            first_idx = np.where(any_fired, fired.argmax(axis=1), K_MAX - 1)
            realized = cum_ret[np.arange(len(entries)), first_idx].astype(np.float32)
            holding = (first_idx + 1).astype(np.int8)
            rec_dict[f"{trig}_realized_ret"] = realized
            rec_dict[f"{trig}_holding_days"] = holding
            rec_dict[f"{trig}_triggered"] = any_fired
        rows.append(pd.DataFrame(rec_dict))

    out = pd.concat(rows, ignore_index=True)
    print(f"  {len(out):,} (entry_date, ts_code) rows ({time.time()-t:.0f}s)")
    return out


def cross_sec_ic(df, pred_col, real_col) -> dict:
    ics = []
    for _, g in df.dropna(subset=[pred_col, real_col]).groupby("trade_date"):
        if len(g) < 30: continue
        if g[pred_col].std() < 1e-9 or g[real_col].std() < 1e-9: continue
        ics.append(g[pred_col].corr(g[real_col]))
    if not ics:
        return {"mean": float("nan"), "ir": float("nan"), "pct_pos": float("nan"), "n_days": 0}
    arr = np.asarray(ics); sd = arr.std()
    return {"mean": float(arr.mean()),
            "ir": float(arr.mean() / sd) if sd > 1e-9 else 0.0,
            "pct_pos": float((arr > 0).mean()),
            "n_days": int(len(arr))}


def top_k_sharpe(df, pred_col, real_col, k=50, ann_factor=252) -> dict:
    rets = []
    for _, g in df.dropna(subset=[pred_col, real_col]).groupby("trade_date"):
        if len(g) < k: continue
        rets.append(float(g.nlargest(k, pred_col)[real_col].mean()))
    if len(rets) < 2:
        return {"sharpe": float("nan"), "cum_ret": float("nan"),
                "n_days": len(rets), "mean": float("nan")}
    arr = np.asarray(rets); sd = arr.std(ddof=1)
    return {"sharpe": float(arr.mean() / sd * np.sqrt(ann_factor)) if sd > 1e-9 else 0.0,
            "cum_ret": float(np.prod(1.0 + arr) - 1.0),
            "n_days": int(len(arr)),
            "mean": float(arr.mean())}


def dyn_agg(df, score_col, trigger) -> dict:
    """Aggregate dynamic-exit metrics: mean_realized_ret, mean_holding_days,
    pct_triggered, top50 of mean ret (non-annualized) and Sharpe per-day series."""
    rc = f"{trigger}_realized_ret"
    hc = f"{trigger}_holding_days"
    tc = f"{trigger}_triggered"
    if score_col not in df.columns:
        return {}
    sub = df.dropna(subset=[score_col, rc])
    if len(sub) == 0:
        return {"mean_realized_ret": float("nan"), "mean_holding_days": float("nan"),
                "pct_triggered": float("nan"), "n_trades": 0,
                "top50_mean_ret": float("nan"), "top50_sharpe": float("nan")}
    # All-trades aggregates
    mean_ret = float(sub[rc].mean())
    mean_hold = float(sub[hc].mean())
    pct_trig = float(sub[tc].mean())
    n_trades = int(len(sub))

    # Per-day top-50 mean return + Sharpe of the daily series
    daily = []
    for _, g in sub.groupby("trade_date"):
        if len(g) < 50: continue
        daily.append(float(g.nlargest(50, score_col)[rc].mean()))
    if len(daily) >= 2:
        arr = np.asarray(daily)
        sd = arr.std(ddof=1)
        top50_mean = float(arr.mean())
        # Sharpe annualized assuming N_trades_per_year = 252/mean_hold
        ann_factor = 252.0 / max(mean_hold, 1.0)
        top50_sharpe = float(arr.mean() / sd * np.sqrt(ann_factor)) if sd > 1e-9 else 0.0
    else:
        top50_mean = float("nan"); top50_sharpe = float("nan")

    return {
        "mean_realized_ret": mean_ret,
        "mean_holding_days": mean_hold,
        "pct_triggered": pct_trig,
        "n_trades": n_trades,
        "top50_mean_ret": top50_mean,
        "top50_sharpe": top50_sharpe,
    }


def main() -> int:
    t_total = time.time()
    pred_path = Path("data/kronos/outputs/crosseval_predictions.parquet")
    kron_cols = ["pred_return_fwd5", "pred_return_fwd20"]

    if pred_path.exists():
        print(f"[resume] loading saved predictions {pred_path}", flush=True)
        scores = pd.read_parquet(pred_path)
        scores["trade_date"] = pd.to_datetime(scores["trade_date"]).dt.date
        print(f"  rows: {len(scores):,}  cols: {list(scores.columns)}")
        tm = None  # importance fallback below
    else:
        panel, all_feats = load_features()
        baseline_feats = [c for c in all_feats if c not in kron_cols]
        pred_baseline, _ = train_predict_lgb(panel, baseline_feats, "ledashi_lgb_baseline")
        pred_kronos, tm  = train_predict_lgb(panel, all_feats,      "ledashi_lgb_kronos")
        scores = panel[["ts_code", "trade_date"]].copy()
        scores["ledashi_lgb_baseline"] = pred_baseline
        scores["ledashi_lgb_kronos"]   = pred_kronos
        scores["kronos_raw_fwd5"]      = panel["pred_return_fwd5"].values
        scores["kronos_raw_fwd20"]     = panel["pred_return_fwd20"].values
        del panel, pred_baseline, pred_kronos

        # External scores (only on first run; saved parquet already has them)
        paris = _dt(pd.read_parquet(PARIS_T3))
        scores = scores.merge(
            paris.rename(columns={"p_t3_baseline": "paris_t3_baseline"})[
                ["ts_code", "trade_date", "paris_t3_baseline"]
            ],
            on=["ts_code", "trade_date"], how="left",
        )
        del paris
        sl = _dt(pd.read_parquet(SL_FINAL))
        scores = scores.merge(
            sl.rename(columns={"score_calibrated": "ledashi_sl_final"})[
                ["ts_code", "trade_date", "ledashi_sl_final"]
            ],
            on=["ts_code", "trade_date"], how="left",
        )
        del sl

    print("\n[scores] coverage:")
    for s in SCORES:
        nn = scores[s].notna().sum()
        print(f"  {s:<26s} {nn:>10,} non-null ({nn/len(scores)*100:.1f}%)")

    out_pred_path = Path("data/kronos/outputs/crosseval_predictions.parquet")
    out_pred_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out_pred_path, compression="zstd", compression_level=5)
    print(f"[saved] predictions -> {out_pred_path}")

    realized, per_stock = compute_realized_panel()
    sym_to_dates = {s: arrs["dates"] for s, arrs in per_stock.items()}

    print("\n[static] merging scores + realized fwd-K ...")
    static_df = scores.merge(realized, on=["ts_code", "trade_date"], how="left")
    print(f"  rows: {len(static_df):,}")

    print("\n[static] computing cross-sec IC + top-50 ann Sharpe ...")
    static_results = {}
    for s in SCORES:
        static_results[s] = {}
        for wname, (ws, we) in WINDOWS.items():
            sub = static_df[(static_df["trade_date"] >= ws) & (static_df["trade_date"] <= we)]
            static_results[s][wname] = {}
            for k in HORIZONS:
                rcol = f"ret_fwd{k}"
                ic = cross_sec_ic(sub, s, rcol)
                tk = top_k_sharpe(sub, s, rcol, k=50, ann_factor=252)
                static_results[s][wname][f"fwd{k}"] = {**ic, "top50_sharpe": tk["sharpe"],
                                                        "top50_mean": tk["mean"]}
    del realized, static_df

    print("\n[dynamic] simulating exit paths ...")
    exits = simulate_dynamic_exit(per_stock, sym_to_dates)
    del per_stock

    print("\n[dynamic] merging scores + exit returns ...")
    exits = scores.merge(exits, on=["ts_code", "trade_date"], how="inner")
    print(f"  rows after merge: {len(exits):,}")
    del scores

    print("\n[dynamic] aggregating per (score, window, trigger) ...")
    dyn_results = {}
    for s in SCORES:
        dyn_results[s] = {}
        for wname, (ws, we) in WINDOWS.items():
            sub = exits[(exits["trade_date"] >= ws) & (exits["trade_date"] <= we)]
            dyn_results[s][wname] = {}
            for trig in TRIGGERS:
                dyn_results[s][wname][trig] = dyn_agg(sub, s, trig)
    del exits

    # Importance (only when LGB was retrained)
    if tm is not None:
        imp = pd.DataFrame({
            "feature": tm.feature_name(),
            "gain": tm.feature_importance(importance_type="gain"),
            "split": tm.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False).reset_index(drop=True)
        imp["rank_gain"] = imp.index + 1
        imp.to_csv("data/kronos/outputs/crosseval_importance.csv", index=False)
        kron_imp = imp[imp["feature"].isin(kron_cols)]
        print("\n[importance] Kronos cols in +Kronos LGB:")
        print(kron_imp.to_string(index=False))
    else:
        kron_imp = pd.DataFrame(columns=["feature","rank_gain","gain","split"])
        print("\n[importance] skipped (resumed from saved predictions, LGB not retrained)")

    # ---- Save big JSON ----
    results = {
        "config": {
            "windows": {k: [str(v[0]), str(v[1])] for k, v in WINDOWS.items()},
            "static_horizons": list(HORIZONS),
            "dynamic_triggers": list(TRIGGERS),
            "dynamic_K_max": K_MAX,
            "top_k": 50,
            "lgb_params": LGB_PARAMS,
            "train_range": [str(TRAIN_START), str(TRAIN_END)],
        },
        "static": static_results,
        "dynamic": dyn_results,
        "total_time_s": time.time() - t_total,
        "notes": [
            "H_macd_death_cross and I_kdj_death_cross deferred: panel only has decayed",
            "event flags (tech_evt_*), not raw MACD/KDJ time series. Paris will be asked",
            "to ship raw daily MACD line/signal/hist and KDJ K/D/J for 2025-01 ~ 2026-05.",
        ],
    }
    Path("data/kronos/outputs/crosseval_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )

    # ---- Build markdown table ----
    lines = ["# Kronos cross-eval — static fwdK + dynamic-exit triggers", ""]
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M Asia/Shanghai')}, "
                 f"runtime {time.time()-t_total:.0f}s_")
    lines.append("")
    lines.append("## Score sources (6)")
    lines.append("")
    lines.append("| score | description |")
    lines.append("|---|---|")
    lines.append("| paris_t3_baseline | paris t3 binary label baseline `p_t3_baseline` |")
    lines.append("| ledashi_sl_final | my SL ensemble `score_calibrated` |")
    lines.append("| ledashi_lgb_baseline | my LGB on 226 features (no Kronos) |")
    lines.append("| ledashi_lgb_kronos | my LGB on 228 features (= 226 + 2 Kronos cols) |")
    lines.append("| kronos_raw_fwd5 | raw Kronos pred_return_fwd5 |")
    lines.append("| kronos_raw_fwd20 | raw Kronos pred_return_fwd20 |")
    lines.append("")

    # Static IC × 100
    lines.append("## Static fwdK — Mean cross-sec IC (× 100)")
    lines.append("")
    for wname in WINDOWS:
        lines.append(f"### {wname}")
        h = "| score | " + " | ".join(f"fwd{k}" for k in HORIZONS) + " |"
        sep = "|" + "---|" * (len(HORIZONS) + 1)
        lines.append(h); lines.append(sep)
        for s in SCORES:
            row = [s]
            for k in HORIZONS:
                v = static_results[s][wname][f"fwd{k}"]["mean"]
                row.append(f"{v*100:+.3f}" if v == v else "n/a")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # IR
    lines.append("## Static fwdK — IR")
    lines.append("")
    for wname in WINDOWS:
        lines.append(f"### {wname}")
        h = "| score | " + " | ".join(f"fwd{k}" for k in HORIZONS) + " |"
        sep = "|" + "---|" * (len(HORIZONS) + 1)
        lines.append(h); lines.append(sep)
        for s in SCORES:
            row = [s]
            for k in HORIZONS:
                v = static_results[s][wname][f"fwd{k}"]["ir"]
                row.append(f"{v:+.2f}" if v == v else "n/a")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Top-50 ann Sharpe (252 days)
    lines.append("## Static fwdK — top-50 annualized Sharpe (252)")
    lines.append("")
    for wname in WINDOWS:
        lines.append(f"### {wname}")
        h = "| score | " + " | ".join(f"fwd{k}" for k in HORIZONS) + " |"
        sep = "|" + "---|" * (len(HORIZONS) + 1)
        lines.append(h); lines.append(sep)
        for s in SCORES:
            row = [s]
            for k in HORIZONS:
                v = static_results[s][wname][f"fwd{k}"]["top50_sharpe"]
                row.append(f"{v:+.2f}" if v == v else "n/a")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Dynamic exit
    lines.append("## Dynamic-exit triggers (K_max=30)")
    lines.append("")
    lines.append("**top-50 mean realized return per trade** (%, NOT annualized — context: see mean_holding_days)")
    lines.append("")
    for wname in WINDOWS:
        lines.append(f"### {wname}")
        h = "| score | " + " | ".join(t.split("_",1)[0] for t in TRIGGERS) + " |"
        sep = "|" + "---|" * (len(TRIGGERS) + 1)
        lines.append(h); lines.append(sep)
        for s in SCORES:
            row = [s]
            for trig in TRIGGERS:
                v = dyn_results[s][wname][trig].get("top50_mean", float("nan"))
                row.append(f"{v*100:+.2f}" if v == v else "n/a")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("**top-50 annualized Sharpe (Sharpe × sqrt(252/mean_holding_days))**")
    lines.append("")
    for wname in WINDOWS:
        lines.append(f"### {wname}")
        h = "| score | " + " | ".join(t.split("_",1)[0] for t in TRIGGERS) + " |"
        sep = "|" + "---|" * (len(TRIGGERS) + 1)
        lines.append(h); lines.append(sep)
        for s in SCORES:
            row = [s]
            for trig in TRIGGERS:
                v = dyn_results[s][wname][trig].get("top50_sharpe", float("nan"))
                row.append(f"{v:+.2f}" if v == v else "n/a")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("**mean holding days per trigger** (universe avg, all trades not just top-50)")
    lines.append("")
    for wname in WINDOWS:
        lines.append(f"### {wname}")
        h = "| score | " + " | ".join(t.split("_",1)[0] for t in TRIGGERS) + " |"
        sep = "|" + "---|" * (len(TRIGGERS) + 1)
        lines.append(h); lines.append(sep)
        for s in SCORES:
            row = [s]
            for trig in TRIGGERS:
                v = dyn_results[s][wname][trig].get("mean_holding_days", float("nan"))
                row.append(f"{v:.1f}" if v == v else "n/a")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("**% of trades that fired trigger (vs K_max forced exit)**")
    lines.append("")
    for wname in WINDOWS:
        lines.append(f"### {wname}")
        h = "| score | " + " | ".join(t.split("_",1)[0] for t in TRIGGERS) + " |"
        sep = "|" + "---|" * (len(TRIGGERS) + 1)
        lines.append(h); lines.append(sep)
        for s in SCORES:
            row = [s]
            for trig in TRIGGERS:
                v = dyn_results[s][wname][trig].get("pct_triggered", float("nan"))
                row.append(f"{v*100:.1f}%" if v == v else "n/a")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("---")
    lines.append("## Trigger definitions")
    lines.append("")
    lines.append("- **A_stop_5pct**: exit when cumulative return since entry < -5%")
    lines.append("- **B_stop_3pct**: exit when cumulative return since entry < -3% (tighter)")
    lines.append("- **C_vol_drop**: exit when volume > 2×MA20(volume) AND day pct_chg < -3%")
    lines.append("- **D_vol_or_stop**: C OR A (放量跌 or -5% stop loss, whichever first)")
    lines.append("- **E_trail_5pct**: trailing stop — exit when peak_cum_ret - cur_cum_ret > 5%")
    lines.append("- **F_trend_break**: exit when adj_close < MA5(adj_close)")
    lines.append("- **G_K_max_only**: no early exit, hold to 30 days (baseline)")
    lines.append("")
    lines.append("**H_macd_death_cross** / **I_kdj_death_cross** deferred — panel only has")
    lines.append("decayed event flags (`tech_evt_macd_zero_cross_decay10`,")
    lines.append("`tech_evt_kdj_below_30_cross_decay10`), not raw MACD line / KDJ K-D")
    lines.append("series. Sent ask to paris for raw time series; will rerun this script")
    lines.append("with H, I once delivered.")
    lines.append("")
    lines.append("## Kronos importance in +Kronos LGB")
    lines.append("")
    lines.append("| feature | rank_gain | gain | split |")
    lines.append("|---|---:|---:|---:|")
    for _, r in kron_imp.iterrows():
        lines.append(f"| {r['feature']} | {int(r['rank_gain'])} | {int(r['gain']):,} | {int(r['split']):,} |")

    md_path = Path("data/kronos/outputs/crosseval_table.md")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {md_path}")
    print(f"\n[done] total {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
