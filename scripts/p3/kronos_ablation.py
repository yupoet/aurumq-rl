"""Kronos ablation — baseline vs +Kronos-2cols on wave_v2 panel.

Trains 2 LightGBM models on identical panel/label/split, except treatment adds
`pred_return_fwd5` and `pred_return_fwd20` from Kronos. Evaluates on paris's
eval windows (H1 2025 / H2 2025 / Q1 2026) and reports mean IC + top-50 Sharpe.

Output:
  data/kronos/outputs/ablation_results.json
  data/kronos/outputs/ablation_importance.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PANEL = "data/p3_4070_long/feature_panel_v3_344_pruned.parquet"
LABEL = "data/p3_4070_long/target_y_wave_v2.parquet"
KRONOS = "data/kronos/outputs/kronos_predictions_daily.parquet"
PANEL_CLOSE = "data/p3_4070_long/stock_close_volume_daily.parquet"

TRAIN_START = pd.Timestamp("2018-01-02").date()
TRAIN_END   = pd.Timestamp("2024-12-31").date()

WINDOWS = {
    "H1_2025": (pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-06-30").date()),
    "H2_2025": (pd.Timestamp("2025-07-01").date(), pd.Timestamp("2025-12-31").date()),
    "Q1_2026": (pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-03-31").date()),
}

LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    num_leaves=63,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    min_data_in_leaf=200,
    verbose=-1,
    n_estimators=400,
    num_threads=-1,
)


def _date_col(df: pd.DataFrame) -> pd.DataFrame:
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def load_data() -> tuple[pd.DataFrame, list[str]]:
    t = time.time()
    print("[load] panel ...", flush=True)
    panel = pd.read_parquet(PANEL)
    panel = _date_col(panel)
    print(f"  {len(panel):,} rows  {panel.shape[1]} cols  ({time.time()-t:.0f}s)")

    print("[load] label ...", flush=True)
    label = pd.read_parquet(LABEL, columns=["trade_date", "ts_code", "y"])
    label = _date_col(label)
    panel = panel.merge(label, on=["trade_date", "ts_code"], how="inner")
    del label
    print(f"  after label join: {len(panel):,} rows  ({time.time()-t:.0f}s)")

    print("[load] kronos preds ...", flush=True)
    kron = pd.read_parquet(KRONOS, columns=["trade_date", "ts_code",
                                            "pred_return_fwd5", "pred_return_fwd20"])
    kron = _date_col(kron)
    panel = panel.merge(kron, on=["trade_date", "ts_code"], how="left")
    del kron
    # fillna(0) for pre-Kronos training rows
    panel["pred_return_fwd5"] = panel["pred_return_fwd5"].fillna(0.0).astype(np.float32)
    panel["pred_return_fwd20"] = panel["pred_return_fwd20"].fillna(0.0).astype(np.float32)
    print(f"  after kronos join: {len(panel):,} rows  ({time.time()-t:.0f}s)")

    feature_cols = [c for c in panel.columns
                    if c not in ("ts_code", "trade_date", "y")]
    print(f"[feat] {len(feature_cols)} feature cols (incl. 2 Kronos)")
    return panel, feature_cols


def realized_returns_for_eval() -> pd.DataFrame:
    """Compute fwd-5 and fwd-20 adjusted returns from stock_close_volume_daily."""
    p = pd.read_parquet(PANEL_CLOSE, columns=["ts_code", "trade_date", "close", "adj_factor"])
    p = _date_col(p).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    latest = p.groupby("ts_code")["adj_factor"].transform("last")
    p["adj_close"] = p["close"] * p["adj_factor"] / latest
    p["ret_fwd5"]  = p.groupby("ts_code")["adj_close"].shift(-5) / p["adj_close"] - 1.0
    p["ret_fwd20"] = p.groupby("ts_code")["adj_close"].shift(-20) / p["adj_close"] - 1.0
    return p[["ts_code", "trade_date", "ret_fwd5", "ret_fwd20"]]


def cross_sec_ic(df: pd.DataFrame, pred_col: str, real_col: str) -> dict:
    ics = []
    for _, g in df.dropna(subset=[pred_col, real_col]).groupby("trade_date"):
        if len(g) < 30:
            continue
        if g[pred_col].std() < 1e-9 or g[real_col].std() < 1e-9:
            continue
        ics.append(g[pred_col].corr(g[real_col]))
    if not ics:
        return {"mean": np.nan, "std": np.nan, "ir": np.nan, "pct_pos": np.nan, "n_days": 0}
    arr = np.asarray(ics)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ir": float(arr.mean() / arr.std()) if arr.std() > 1e-9 else 0.0,
        "pct_pos": float((arr > 0).mean()),
        "n_days": int(len(arr)),
    }


def top_k_sharpe(df: pd.DataFrame, pred_col: str, real_col: str, k: int = 50) -> dict:
    """Annualized Sharpe of daily equal-weight top-K portfolio."""
    daily_rets = []
    for _, g in df.dropna(subset=[pred_col, real_col]).groupby("trade_date"):
        if len(g) < k:
            continue
        topk = g.nlargest(k, pred_col)
        daily_rets.append(float(topk[real_col].mean()))
    if len(daily_rets) < 2:
        return {"sharpe": np.nan, "cum_ret": np.nan, "n_days": len(daily_rets)}
    arr = np.asarray(daily_rets)
    sd = arr.std(ddof=1)
    sharpe = float(arr.mean() / sd * np.sqrt(252)) if sd > 1e-9 else 0.0
    cum = float(np.prod(1.0 + arr) - 1.0)
    return {"sharpe": sharpe, "cum_ret": cum, "n_days": int(len(arr))}


def train_eval(panel: pd.DataFrame, feature_cols: list[str], tag: str,
               realized: pd.DataFrame) -> tuple[dict, lgb.Booster]:
    print(f"\n[{tag}] feature cols: {len(feature_cols)}")
    train_mask = (panel["trade_date"] >= TRAIN_START) & (panel["trade_date"] <= TRAIN_END)
    train = panel[train_mask]
    print(f"  train rows: {len(train):,}  ({train['trade_date'].min()} ~ {train['trade_date'].max()})")

    t = time.time()
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(train[feature_cols], train["y"])
    print(f"  train time: {time.time()-t:.0f}s")
    del train

    # Predict on full panel (eval windows are within)
    print(f"  predicting on {len(panel):,} rows ...", flush=True)
    panel = panel.copy()
    panel["pred"] = model.predict(panel[feature_cols])

    # Merge realized returns
    eval_df = panel[["ts_code", "trade_date", "pred"]].merge(
        realized, on=["ts_code", "trade_date"], how="left"
    )

    results = {}
    for wname, (s, e) in WINDOWS.items():
        sub = eval_df[(eval_df["trade_date"] >= s) & (eval_df["trade_date"] <= e)]
        results[wname] = {
            "ic_fwd5":      cross_sec_ic(sub, "pred", "ret_fwd5"),
            "ic_fwd20":     cross_sec_ic(sub, "pred", "ret_fwd20"),
            "top50_sharpe_fwd5":  top_k_sharpe(sub, "pred", "ret_fwd5", k=50),
            "top50_sharpe_fwd20": top_k_sharpe(sub, "pred", "ret_fwd20", k=50),
        }
        print(f"  {wname} IC fwd5  = {results[wname]['ic_fwd5']['mean']:+.4f}  "
              f"IR={results[wname]['ic_fwd5']['ir']:+.2f}  "
              f"top50_S fwd5={results[wname]['top50_sharpe_fwd5']['sharpe']:+.2f}")
        print(f"  {wname} IC fwd20 = {results[wname]['ic_fwd20']['mean']:+.4f}  "
              f"IR={results[wname]['ic_fwd20']['ir']:+.2f}  "
              f"top50_S fwd20={results[wname]['top50_sharpe_fwd20']['sharpe']:+.2f}")

    return results, model.booster_


def main() -> int:
    t_total = time.time()
    panel, all_features = load_data()
    realized = realized_returns_for_eval()

    kronos_cols = ["pred_return_fwd5", "pred_return_fwd20"]
    baseline_cols = [c for c in all_features if c not in kronos_cols]

    baseline_results, b_model = train_eval(panel, baseline_cols, "baseline", realized)
    treatment_results, t_model = train_eval(panel, all_features, "+kronos", realized)

    # Importance of Kronos cols in treatment
    importance_df = pd.DataFrame({
        "feature": t_model.feature_name(),
        "importance_gain": t_model.feature_importance(importance_type="gain"),
        "importance_split": t_model.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    importance_df["rank_gain"] = importance_df.index + 1

    kron_imp = importance_df[importance_df["feature"].isin(kronos_cols)]
    print(f"\n[importance] Kronos cols ranking (of {len(importance_df)} features):")
    print(kron_imp.to_string(index=False))

    # Diff calc
    diffs = {}
    for wname in WINDOWS:
        diffs[wname] = {}
        for hor in ("fwd5", "fwd20"):
            bm = baseline_results[wname][f"ic_{hor}"]["mean"]
            tm = treatment_results[wname][f"ic_{hor}"]["mean"]
            bs = baseline_results[wname][f"top50_sharpe_{hor}"]["sharpe"]
            ts = treatment_results[wname][f"top50_sharpe_{hor}"]["sharpe"]
            diffs[wname][f"ic_{hor}_delta_bps"] = (tm - bm) * 1e4
            diffs[wname][f"top50_sharpe_{hor}_delta"] = ts - bs

    out = {
        "config": {
            "panel": PANEL,
            "label": LABEL,
            "kronos": KRONOS,
            "train_range": [str(TRAIN_START), str(TRAIN_END)],
            "windows": {k: [str(v[0]), str(v[1])] for k, v in WINDOWS.items()},
            "lgb_params": LGB_PARAMS,
            "n_baseline_features": len(baseline_cols),
            "n_treatment_features": len(all_features),
        },
        "baseline": baseline_results,
        "treatment_kronos": treatment_results,
        "deltas": diffs,
        "kronos_importance": kron_imp.to_dict(orient="records"),
        "total_time_s": time.time() - t_total,
    }

    out_path = Path("data/kronos/outputs/ablation_results.json")
    out_path.write_text(json.dumps(out, indent=2, default=str))
    importance_df.to_csv("data/kronos/outputs/ablation_importance.csv", index=False)

    print(f"\n[done] total {time.time()-t_total:.0f}s. Results -> {out_path}")
    print(f"\n=== DELTAS (treatment - baseline) ===")
    for wname in WINDOWS:
        d = diffs[wname]
        print(f"  {wname}: ic_fwd5={d['ic_fwd5_delta_bps']:+.1f} bps  "
              f"ic_fwd20={d['ic_fwd20_delta_bps']:+.1f} bps  "
              f"top50_S_fwd5={d['top50_sharpe_fwd5_delta']:+.3f}  "
              f"top50_S_fwd20={d['top50_sharpe_fwd20_delta']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
