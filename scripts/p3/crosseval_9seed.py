"""Cross-eval single-seed vs 3-seed-1-config vs 9-model ensemble.

Answers paris reverse Q2 from v19: how much does the 9-model ensemble add over
single seed=42 or 3-seed × 1 config?

Score columns (from paris path4_ensemble_preds.parquet):
  single_seed42_C  = pred_nl63_lr030_mdl50_seed42   (single seed, best-config-by-name)
  3seed_configC    = mean(pred_nl63_lr030_mdl50_seed{42,43,44})
  3seed_configB    = mean(pred_nl63_lr030_mdl100_seed{42,43,44})
  3seed_configA    = mean(pred_nl31_lr050_mdl50_seed{42,43,44})
  ensemble_raw     = pred_ensemble_raw
  ensemble_cal     = pred_ensemble_calibrated   (production)

Forward returns from MAIN_BOARD close panel (paris preds are MAIN_BOARD-only).
Windows: H2_2025 (val), Q1_2026 (test), Q2_2026_partial.
Horizons: fwd5, fwd10, fwd20.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PRED = "D:/dev/aurumq-handoffs/inbox/2026-05-15-paris-9seed-ensemble-preds/path4_ensemble_preds.parquet"
CLOSE = "D:/dev/aurumq-rl/data/p3_4070_long/stock_close_volume_daily.parquet"

WINDOWS = {
    "H2_2025":         (pd.Timestamp("2025-07-01").date(), pd.Timestamp("2025-12-31").date()),
    "Q1_2026":         (pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-03-31").date()),
    "Q2_2026_partial": (pd.Timestamp("2026-04-01").date(), pd.Timestamp("2026-04-30").date()),
}
HORIZONS = (5, 10, 20)
TOP_K = 50


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


def main() -> int:
    t0 = time.time()

    print(f"[load] preds {PRED}", flush=True)
    pred = pd.read_parquet(PRED)
    pred["trade_date"] = pd.to_datetime(pred["trade_date"]).dt.date
    print(f"  pred: {len(pred):,} rows × {pred.shape[1]} cols, dates {pred.trade_date.min()} ~ {pred.trade_date.max()}")

    # Build score columns
    pred["single_seed42_C"] = pred["pred_nl63_lr030_mdl50_seed42"].astype(np.float32)
    pred["3seed_configC"] = pred[["pred_nl63_lr030_mdl50_seed42","pred_nl63_lr030_mdl50_seed43","pred_nl63_lr030_mdl50_seed44"]].mean(axis=1).astype(np.float32)
    pred["3seed_configB"] = pred[["pred_nl63_lr030_mdl100_seed42","pred_nl63_lr030_mdl100_seed43","pred_nl63_lr030_mdl100_seed44"]].mean(axis=1).astype(np.float32)
    pred["3seed_configA"] = pred[["pred_nl31_lr050_mdl50_seed42","pred_nl31_lr050_mdl50_seed43","pred_nl31_lr050_mdl50_seed44"]].mean(axis=1).astype(np.float32)
    pred["ensemble_raw"] = pred["pred_ensemble_raw"].astype(np.float32)
    pred["ensemble_cal"] = pred["pred_ensemble_calibrated"].astype(np.float32)

    SCORES = ["single_seed42_C", "3seed_configA", "3seed_configB", "3seed_configC", "ensemble_raw", "ensemble_cal"]
    keep_cols = ["ts_code","trade_date"] + SCORES
    pred = pred[keep_cols]

    print(f"[load] close panel {CLOSE}", flush=True)
    p = pd.read_parquet(CLOSE, columns=["ts_code","trade_date","close","adj_factor"])
    p["trade_date"] = pd.to_datetime(p["trade_date"]).dt.date
    p = p.sort_values(["ts_code","trade_date"]).reset_index(drop=True)
    latest = p.groupby("ts_code")["adj_factor"].transform("last")
    p["adj_close"] = (p["close"] * p["adj_factor"] / latest).astype(np.float32)

    for k in HORIZONS:
        p[f"ret_fwd{k}"] = (p.groupby("ts_code")["adj_close"].shift(-k) / p["adj_close"] - 1.0).astype(np.float32)
    p = p[["ts_code","trade_date"] + [f"ret_fwd{k}" for k in HORIZONS]]
    print(f"  close panel + fwd: {len(p):,} rows")

    # Restrict to test windows for memory
    test_start = min(s for s,e in WINDOWS.values())
    test_end   = max(e for s,e in WINDOWS.values())
    pred_t = pred[(pred["trade_date"]>=test_start) & (pred["trade_date"]<=test_end)]
    p_t = p[(p["trade_date"]>=test_start) & (p["trade_date"]<=test_end)]
    print(f"  pred test rows: {len(pred_t):,}, fwd test rows: {len(p_t):,}")

    eval_df = pred_t.merge(p_t, on=["ts_code","trade_date"], how="inner")
    print(f"  merged eval rows: {len(eval_df):,}")

    results = {}
    for s in SCORES:
        results[s] = {}
        for wname, (ws, we) in WINDOWS.items():
            sub = eval_df[(eval_df["trade_date"]>=ws) & (eval_df["trade_date"]<=we)]
            results[s][wname] = {}
            for k in HORIZONS:
                ic = cross_sec_ic(sub, s, f"ret_fwd{k}")
                sh = top_k_sharpe_ann(sub, s, f"ret_fwd{k}")
                results[s][wname][f"fwd{k}"] = {
                    "ic": ic["mean"], "ir": ic["ir"], "n_days_ic": ic["n_days"],
                    "top50_sharpe": sh["sharpe"], "top50_mean": sh["mean"], "n_days_sh": sh["n_days"],
                }
            print(f"  {s:<22} {wname:<18} fwd20 IC={results[s][wname]['fwd20']['ic']:+.4f} Sharpe={results[s][wname]['fwd20']['top50_sharpe']:+.3f}", flush=True)

    # Compute deltas (vs ensemble_cal)
    deltas = {}
    base = "ensemble_cal"
    for s in SCORES:
        if s == base: continue
        deltas[f"{s}_vs_{base}"] = {}
        for wname in WINDOWS:
            deltas[f"{s}_vs_{base}"][wname] = {}
            for k in HORIZONS:
                d_ic = results[s][wname][f"fwd{k}"]["ic"] - results[base][wname][f"fwd{k}"]["ic"]
                d_sh = results[s][wname][f"fwd{k}"]["top50_sharpe"] - results[base][wname][f"fwd{k}"]["top50_sharpe"]
                deltas[f"{s}_vs_{base}"][wname][f"fwd{k}"] = {"delta_ic": d_ic, "delta_sharpe": d_sh}

    out = {
        "config": {
            "source": PRED,
            "fwd_returns": CLOSE,
            "scores": SCORES,
            "windows": {k: [str(v[0]), str(v[1])] for k, v in WINDOWS.items()},
            "horizons": list(HORIZONS), "top_k": TOP_K,
        },
        "results": results,
        "deltas_vs_ensemble_cal": deltas,
        "total_time_s": time.time() - t0,
    }
    out_path = Path("D:/dev/aurumq-rl/data/kronos/outputs/crosseval_9seed.json")
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {out_path}")
    print(f"[done] {time.time()-t0:.0f}s")

    # Print summary table
    print("\n=== IC fwd20 ===")
    print(f"{'score':<22} " + " ".join(f"{w:>20}" for w in WINDOWS))
    for s in SCORES:
        line = f"{s:<22} "
        for wname in WINDOWS:
            line += f"{results[s][wname]['fwd20']['ic']:>+20.4f}"
        print(line)

    print("\n=== top-50 Sharpe (ann) fwd20 ===")
    print(f"{'score':<22} " + " ".join(f"{w:>20}" for w in WINDOWS))
    for s in SCORES:
        line = f"{s:<22} "
        for wname in WINDOWS:
            line += f"{results[s][wname]['fwd20']['top50_sharpe']:>+20.3f}"
        print(line)

    print("\n=== delta vs ensemble_cal (Sharpe fwd20) ===")
    print(f"{'score':<22} " + " ".join(f"{w:>20}" for w in WINDOWS))
    for s in SCORES:
        if s == base:
            line = f"{s:<22} (baseline)"
        else:
            line = f"{s:<22} "
            for wname in WINDOWS:
                d = deltas[f"{s}_vs_{base}"][wname]["fwd20"]["delta_sharpe"]
                line += f"{d:>+20.3f}"
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
