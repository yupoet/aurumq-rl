"""Matrix v10h — Bootstrap 95% CI on Sharpe NET for all v10-g cells.

Post-processing of all per-cell prediction parquets. For each cell:
  1. Load score parquet + forward returns
  2. For each (sizing, dyn-exit, window) combo:
     - Compute daily portfolio returns
     - Block-bootstrap 1000 iterations on date axis
     - Output Sharpe NET 95% CI [low, high]

This addresses "弱结论" — NPF/HARD_TECH small universe & CatBoost/XGBoost cells
need CI to determine whether differences are statistically significant.

Output: matrix_v10h_bootstrap.json with CI per cell.
"""
from __future__ import annotations

import os
import gc
import json
import time
from pathlib import Path

_HANDOFF_INBOX = os.environ.get("AURUMQ_HANDOFF_INBOX", "data/handoffs/inbox")

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kronos_matrix_v10 import (
    UNIVERSES, TRIGGERS, COST_ROUND_TRIP, TOP_K_LIST,
    PANEL_CLOSE, HORIZONS,
    _dt, compute_realized_and_exits, load_universes,
)

PRED_DIRS = [
    Path("data/kronos/outputs/matrix_v10"),
    Path("data/kronos/outputs/matrix_v10b"),
    Path("data/kronos/outputs/matrix_v10c"),
    Path("data/kronos/outputs/matrix_v10d"),
    Path("data/kronos/outputs/matrix_v10e"),
    Path("data/kronos/outputs/matrix_v10f"),
    Path("data/kronos/outputs/matrix_v10g"),
]
N_BOOTSTRAP = 1000
BLOCK_LENGTH = 5  # 5-day block (handles autocorrelation)


def block_bootstrap_sharpe(daily_returns, k_horizon, n_iter=N_BOOTSTRAP, block_len=BLOCK_LENGTH):
    """Block bootstrap on daily returns. Returns (sharpe_mean, sharpe_ci_low, sharpe_ci_high) NET."""
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
    """For one cell + given top_k/horizon, compute Sharpe NET CI on H2_2025."""
    df = pd.read_parquet(pred_path)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    H2_start = pd.Timestamp("2025-07-01").date()
    H2_end = pd.Timestamp("2025-12-31").date()
    df = df[(df["trade_date"]>=H2_start) & (df["trade_date"]<=H2_end)]
    eval_df = df.merge(realized, on=["trade_date","ts_code"], how="inner")
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
    print("[setup] realized fwd returns ..."); _, _ = load_universes(UNIVERSES[:1])  # warm cache
    p = pd.read_parquet(PANEL_CLOSE, columns=["ts_code", "trade_date", "close", "adj_factor"])
    p = _dt(p).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    latest = p.groupby("ts_code")["adj_factor"].transform("last")
    p["adj_close"] = (p["close"] * p["adj_factor"] / latest).astype(np.float32)
    for k in (5, 10, 20):
        p[f"ret_fwd{k}"] = (p.groupby("ts_code")["adj_close"].shift(-k) / p["adj_close"] - 1.0).astype(np.float32)
    realized = p[["ts_code", "trade_date"] + [f"ret_fwd{k}" for k in (5, 10, 20)]]
    print(f"[realized] {len(realized):,} rows")

    # Iterate all pred parquets
    all_pred_files = []
    for d in PRED_DIRS:
        if d.exists():
            all_pred_files.extend(sorted(d.glob("pred_*.parquet")) + sorted(d.glob("meta_*.parquet")) + sorted(d.glob("hybrid_*.parquet")))
    print(f"[scan] {len(all_pred_files)} pred files across {len(PRED_DIRS)} dirs")

    ci_results = {}
    for i, pf in enumerate(all_pred_files):
        cell_id = pf.stem.replace("pred_", "").replace("meta_", "meta_").replace("hybrid_", "hybrid_")
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(all_pred_files)}] {cell_id}", flush=True)
        t_cell = time.time()
        ci_results[cell_id] = {}
        for top_k in [10, 50]:  # 2 key K values
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

    Path("data/kronos/outputs/matrix_v10h_bootstrap_ci.json").write_text(
        json.dumps({"config": {"n_bootstrap": N_BOOTSTRAP, "block_length": BLOCK_LENGTH,
                               "cost_round_trip": COST_ROUND_TRIP},
                    "results": ci_results}, indent=2, default=str))
    print(f"\n[saved] matrix_v10h_bootstrap_ci.json, {len(ci_results)} cells, {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
