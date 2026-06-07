"""Bootstrap CI on v11 + v12 top production cells.

Fills the gap from v10h (which only covered v10/v10b/v10c/v10d/v10e cells, not v11/v12).
Uses same block-bootstrap methodology: 1000 iter × 5-day block × Sharpe NET top-K.

Focuses on top 20 cells from v11 (paris sparse binary) + top 10 from v12 (anchor).
"""
from __future__ import annotations
import os, gc, json, time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kronos_matrix_v10 import (
    UNIVERSES, PANEL_CLOSE, HORIZONS, COST_ROUND_TRIP,
    _dt, load_universes,
)

N_BOOTSTRAP = 1000
BLOCK_LENGTH = 5

V11_PRED_DIR = Path("data/kronos/outputs/matrix_v11")
V12_PRED_DIR = Path("data/kronos/outputs/matrix_v12")


def block_bootstrap_sharpe(daily_returns, k_horizon, n_iter=N_BOOTSTRAP, block_len=BLOCK_LENGTH):
    arr = np.asarray(daily_returns)
    arr_net = arr - COST_ROUND_TRIP
    n = len(arr_net)
    if n < block_len * 4:
        return float("nan"), float("nan"), float("nan")
    n_blocks = (n + block_len - 1) // block_len
    ann = np.sqrt(252.0 / max(k_horizon, 1))
    sharpes = []
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        start_idxs = rng.integers(0, n - block_len + 1, size=n_blocks)
        resampled = np.concatenate([arr_net[s:s+block_len] for s in start_idxs])[:n]
        sd = resampled.std(ddof=1)
        if sd < 1e-9: continue
        sharpes.append(resampled.mean() / sd * ann)
    if not sharpes:
        return float("nan"), float("nan"), float("nan")
    arr_sh = np.asarray(sharpes)
    return float(arr_sh.mean()), float(np.percentile(arr_sh, 2.5)), float(np.percentile(arr_sh, 97.5))


def compute_cell_ci(pred_path, realized, top_k=50, horizon=20):
    df = pd.read_parquet(pred_path)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    H2_start = pd.Timestamp("2025-07-01").date()
    H2_end = pd.Timestamp("2025-12-31").date()
    df = df[(df["trade_date"] >= H2_start) & (df["trade_date"] <= H2_end)]
    eval_df = df.merge(realized, on=["trade_date", "ts_code"], how="inner")
    if "score" not in eval_df.columns:
        return None
    fwd_col = f"ret_fwd{horizon}"
    daily_rets = []
    for d, g in eval_df.dropna(subset=["score", fwd_col]).groupby("trade_date"):
        if len(g) < top_k: continue
        daily_rets.append(float(g.nlargest(top_k, "score")[fwd_col].mean()))
    if len(daily_rets) < 30:
        return None
    return block_bootstrap_sharpe(daily_rets, k_horizon=horizon)


def main():
    t_total = time.time()
    print("[setup] realized + universes ...")
    _, _ = load_universes(UNIVERSES[:1])  # warm
    p = pd.read_parquet(PANEL_CLOSE, columns=["ts_code", "trade_date", "close", "adj_factor"])
    p = _dt(p).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    latest = p.groupby("ts_code")["adj_factor"].transform("last")
    p["adj_close"] = (p["close"] * p["adj_factor"] / latest).astype(np.float32)
    for k in (5, 10, 20):
        p[f"ret_fwd{k}"] = (p.groupby("ts_code")["adj_close"].shift(-k) / p["adj_close"] - 1.0).astype(np.float32)
    realized = p[["ts_code", "trade_date"] + [f"ret_fwd{k}" for k in (5, 10, 20)]]
    print(f"[realized] {len(realized):,} rows")

    pred_files = []
    if V11_PRED_DIR.exists():
        pred_files += sorted(V11_PRED_DIR.glob("pred_*.parquet"))
    if V12_PRED_DIR.exists():
        pred_files += sorted(V12_PRED_DIR.glob("pred_*.parquet"))
    print(f"[scan] {len(pred_files)} pred files across v11+v12")

    ci_results = {}
    for i, pf in enumerate(pred_files):
        cell_id = pf.stem.replace("pred_", "")
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(pred_files)}] {cell_id}", flush=True)
        ci_results[cell_id] = {}
        for top_k in [10, 50]:
            for horizon in [5, 20]:
                key = f"K{top_k}_fwd{horizon}"
                ci = compute_cell_ci(pf, realized, top_k=top_k, horizon=horizon)
                if ci is None:
                    ci_results[cell_id][key] = None; continue
                ci_results[cell_id][key] = {
                    "sharpe_net_mean": ci[0],
                    "ci95_low": ci[1], "ci95_high": ci[2],
                    "ci_width": ci[2] - ci[1],
                }

    Path("data/kronos/outputs/matrix_v11_v12_bootstrap_ci.json").write_text(
        json.dumps({"config": {"n_bootstrap": N_BOOTSTRAP, "block_length": BLOCK_LENGTH,
                               "cost_round_trip": COST_ROUND_TRIP},
                    "results": ci_results}, indent=2, default=str))
    print(f"\n[saved] matrix_v11_v12_bootstrap_ci.json, {len(ci_results)} cells, {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
