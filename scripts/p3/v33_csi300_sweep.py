"""v33 task 2 — CSI300 paradigm 1 sweep (5 panels × lgb_proximity).

paris 5/28+ deferred priority, promoted P2 post-v13.

Train paradigm 1 lgb_proximity on CSI300 PIT × 5 panels × target_y label.
Expected: large-cap alpha is squeezed so signal weak, but worth one definitive sweep.
"""
from __future__ import annotations
import os, gc, json, time
from pathlib import Path

_HANDOFF_INBOX = os.environ.get("AURUMQ_HANDOFF_INBOX", "data/handoffs/inbox")
import lightgbm as lgb
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kronos_matrix_v10 import (
    PANELS, TRAIN_START, TRAIN_END, _dt,
    load_universes, filter_universe, compute_realized_and_exits,
    load_panel, eval_cell,
)
from save_batch2_artifacts import LGB_PARAMS_V10

OUT_DIR = Path("data/kronos/outputs/matrix_v33_csi300_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANELS_SWEEP = ("ledashi", "v2_null", "v2_no_phase_c", "v3unified", "r2b")
UNIVERSE = "CSI300"


def train_cell(panel_name, static_sets, pit_dfs, realized, exits):
    cid = f"target_y_CSI300_{panel_name}"
    print(f"\n=== {cid} (paradigm 1 lgb_proximity, target_y label) ===")
    panel = load_panel(panel_name)
    base_cols = [c for c in panel.columns if c not in ("ts_code", "trade_date")]
    upanel = filter_universe(panel, UNIVERSE, static_sets, pit_dfs)
    print(f"  upanel: {len(upanel):,} rows × {len(base_cols)} cols")

    label_path = Path("data/p3_4070_long/target_y.parquet")
    label_df = _dt(pd.read_parquet(label_path, columns=["trade_date", "ts_code", "y"]))
    joined = upanel.merge(label_df, on=["ts_code", "trade_date"], how="inner")
    train = joined[(joined["trade_date"] >= TRAIN_START) & (joined["trade_date"] <= TRAIN_END)]
    print(f"  train rows: {len(train):,}")

    t = time.time()
    model = lgb.LGBMRegressor(**LGB_PARAMS_V10)
    model.fit(train[base_cols], train["y"])
    del train, joined; gc.collect()
    print(f"  train {time.time()-t:.0f}s")

    preds = model.predict(upanel[base_cols]).astype(np.float32)
    pred_df = upanel[["ts_code", "trade_date"]].copy()
    pred_df["score"] = preds
    pred_path = OUT_DIR / f"pred_{cid}.parquet"
    pred_df.to_parquet(pred_path, compression="zstd")

    result = eval_cell(pred_df, realized, exits, adaptive_gating=None)
    r = result["static"]["H2_2025"]["fwd20"]
    q1 = result["static"]["Q1_2026"]["fwd20"]
    ic_h2 = r["ic"] * 100
    ic_q1 = q1["ic"] * 100
    sn_h2 = r["sizing"].get("50", {}).get("sharpe_net", float("nan"))
    sn_q1 = q1["sizing"].get("50", {}).get("sharpe_net", float("nan"))
    print(f"  {cid}: H2 IC={ic_h2:+.3f}% S50_NET={sn_h2:+.2f} | Q1 IC={ic_q1:+.3f}% S50_NET={sn_q1:+.2f}")
    del model, preds, pred_df, panel, upanel; gc.collect()
    return result


def main():
    t_total = time.time()
    print("[setup] universes ...")
    static_sets, pit_dfs = load_universes(("MAIN_BOARD", "CSI300", "CSI500", "CSI1000",
                                            "NPF", "NPF_FULL", "HARD_TECH"))
    print("\n[setup] realized + dyn-exit ...")
    realized, exits = compute_realized_and_exits()

    results = {}
    for panel_name in PANELS_SWEEP:
        cid = f"target_y_CSI300_{panel_name}"
        try:
            results[cid] = train_cell(panel_name, static_sets, pit_dfs, realized, exits)
        except Exception as e:
            print(f"  [FAIL] {cid}: {e}")
            results[cid] = {"error": str(e)}

    Path("data/kronos/outputs/matrix_v33_csi300_sweep_results.json").write_text(json.dumps({
        "task": "v33 task 2 — CSI300 paradigm 1 sweep",
        "panels": list(PANELS_SWEEP),
        "results": results,
        "total_time_s": time.time() - t_total,
    }, indent=2, default=str))
    print(f"\n[done] CSI300 sweep {time.time()-t_total:.0f}s, {len(results)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
