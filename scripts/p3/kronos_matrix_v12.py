"""Matrix v12 — paris Phase 2 anchor paradigm 2 (β PELT-hybrid + α ledashi-draft).

paradigm 2 = event-anchored pattern recognition:
  1. Backward scan history for "main rising wave" events
  2. T-1/T-3/T-5 pre-event days = positive class
  3. NEG_RATIO=5:1 sampled non-event days = negative class
  4. Binary classifier learns "is pre-wave?" feature signature

Compare vs paradigm 1 (v11 paris sparse proximity) on same (panel, universe, horizon):
  paris v24 §4 prediction:
    - HARD_TECH anchor T-3 binary IC > proximity t3 IC by +1.5pp
    - MAIN_BOARD ≈ tied (event sample too dispersed)
    - β vs α: +0.3-0.6pp from PELT changepoint + benchmark excess + dedupe

v12 fires: 7 panel × 6 universe × 3 anchor (T1/T3/T5) × 2 spec (α, β) = 252 cells.

Labels: paris ship at
  inbox/2026-05-16-paris-reply-v24/
    anchor_T{1,3,5}_{univ}_year={YYYY}.parquet  (β main, 90 files)
    anchor_labels_alpha/anchor_T{1,3,5}_*.parquet  (α baseline, 90 files)

Label schema: (trade_date, ts_code, y_height, y_strength, wave_start_date, y_binary, anchor_horizon)
  - y_binary = 1 if pre-event sample, 0 if neg-sampled (NEG_RATIO=5:1, ~20% pos rate)
  - y_height = wave height (continuous, for regression future)
  - y_strength = paris P_strength composite (continuous, for regression future)
  - wave_start_date = event-anchor date (null for negatives)

Note: anchor labels are SAMPLED (~20% pos), not full universe cross-section.
  - Train on sampled (trade_date, ts_code) → ~20% pos LGB binary
  - Eval: SCORE all rows of universe panel → cross-section IC + Sharpe top-K
"""
from __future__ import annotations

import os
import gc
import json
import time
from pathlib import Path

_HANDOFF_INBOX = os.environ.get("AURUMQ_HANDOFF_INBOX", "data/handoffs/inbox")

import lightgbm as lgb
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kronos_matrix_v10 import (
    PANEL_PATHS, PANELS, UNIVERSES, PANEL_CLOSE, TECH_LINES, UNIVERSE_DIR,
    TRAIN_START, TRAIN_END, WINDOWS, HORIZONS, K_MAX, TRIGGERS, COST_ROUND_TRIP,
    TOP_K_LIST, ADAPTIVE_PCT, ADAPTIVE_K_CAP, ADAPTIVE_K_FLOOR, ADAPTIVE_Q_LOW, ADAPTIVE_Q_MID,
    FAMILY_PREFIXES,
    _dt, load_universes, filter_universe, compute_realized_and_exits,
    cross_sec_ic, top_k_sharpe_ann_multi, dyn_agg_multi, compute_adaptive_thresholds,
    family_importance, load_panel, eval_cell,
)

# paris-aligned LGB binary (same as v11)
LGB_BINARY_PARAMS = dict(
    objective="binary",
    metric="average_precision",
    num_leaves=63,
    learning_rate=0.05,
    n_estimators=500,
    min_child_samples=200,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
)

FIX_INBOX = Path(f"{_HANDOFF_INBOX}/2026-05-18-paris-anchor-alpha-fix")
ANCHOR_DIR_BETA = FIX_INBOX / "beta_pelt_hybrid_main"   # β PELT-hybrid + dedupe (~15% pos, ~912 rows/MAIN_BOARD-year)
ANCHOR_DIR_ALPHA = FIX_INBOX / "alpha_5cond_baseline"   # α 5-condition forward argmax (~20% pos, ~7421 rows/MAIN_BOARD-year)
ANCHORS = ("T1", "T3", "T5")
SPECS = ("alpha", "beta")
YEARS = ("2022", "2023", "2024", "2025", "2026")

OUT_DIR = Path("data/kronos/outputs/matrix_v12")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_anchor_label(univ: str, anchor_T: str, spec: str) -> pd.DataFrame | None:
    """Concat all years of paris Phase 2 anchor labels for (univ, anchor_T, spec)."""
    base = ANCHOR_DIR_BETA if spec == "beta" else ANCHOR_DIR_ALPHA
    dfs = []
    for year in YEARS:
        f = base / f"anchor_{anchor_T}_{univ}_year={year}.parquet"
        if f.exists():
            df = pd.read_parquet(f, columns=["trade_date", "ts_code", "y_binary"])
            df = df.rename(columns={"y_binary": "y"})
            dfs.append(df)
    if not dfs: return None
    out = pd.concat(dfs, ignore_index=True)
    out = _dt(out)
    return out


def main():
    t_total = time.time()
    print("[setup] universes ...")
    static_sets, pit_dfs = load_universes(UNIVERSES)

    print("\n[setup] realized + dyn-exit ...")
    realized, exits = compute_realized_and_exits()

    print(f"\n[setup] verify Phase 2 anchor inbox: {FIX_INBOX}")
    sample_b = ANCHOR_DIR_BETA / "anchor_T3_MAIN_BOARD_year=2024.parquet"
    sample_a = ANCHOR_DIR_ALPHA / "anchor_T3_MAIN_BOARD_year=2024.parquet"
    if not sample_b.exists() or not sample_a.exists():
        print(f"  ERROR: Phase 2 anchor labels missing")
        print(f"    β sample exists: {sample_b.exists()}")
        print(f"    α sample exists: {sample_a.exists()}")
        return 1
    print(f"  Phase 2 anchor β + α OK")

    matrix_results = {}
    family_importance_results = {}
    CHECKPOINT = Path("data/kronos/outputs/matrix_v12_checkpoint.json")
    if CHECKPOINT.exists():
        prev = json.loads(CHECKPOINT.read_text())
        matrix_results = prev.get("results", {})
        family_importance_results = prev.get("family_importance", {})
        print(f"[resume] {len(matrix_results)} cells from checkpoint")

    total_cells = len(PANELS) * len(UNIVERSES) * len(ANCHORS) * len(SPECS)
    print(f"\n[matrix] {len(PANELS)} panels × {len(UNIVERSES)} univ × {len(ANCHORS)} anchor × {len(SPECS)} spec = {total_cells} cells")

    for panel_name in PANELS:
        panel_cells = [f"{spec}_{anchor}_{u}_{panel_name}"
                       for u in UNIVERSES for anchor in ANCHORS for spec in SPECS]
        if all(c in matrix_results for c in panel_cells):
            print(f"\n[skip panel {panel_name}] all in checkpoint"); continue

        try: panel = load_panel(panel_name)
        except Exception as e:
            print(f"  [{panel_name}] LOAD FAILED: {e}"); continue
        base_cols = [c for c in panel.columns if c not in ("ts_code","trade_date")]
        print(f"\n[panel {panel_name}] {len(panel):,} rows × {len(base_cols)} feats", flush=True)

        for univ in UNIVERSES:
            upanel = filter_universe(panel, univ, static_sets, pit_dfs)

            for anchor in ANCHORS:
                for spec in SPECS:
                    exp_id = f"{spec}_{anchor}_{univ}_{panel_name}"
                    if exp_id in matrix_results:
                        continue

                    label_df = load_anchor_label(univ, anchor, spec)
                    if label_df is None or len(label_df) == 0:
                        print(f"  [SKIP] no labels for {exp_id}"); continue

                    # Inner join: only rows that ARE in anchor sample (events + neg-sampled)
                    joined = upanel.merge(label_df, on=["ts_code", "trade_date"], how="inner")
                    train = joined[(joined["trade_date"] >= TRAIN_START) & (joined["trade_date"] <= TRAIN_END)]
                    if len(train) < 1000:
                        print(f"  [SKIP] {exp_id} train {len(train)} < 1K (anchor labels sparse)")
                        matrix_results[exp_id] = {"skipped": True, "train_rows": len(train)}
                        continue

                    pos_rate_train = (train["y"] > 0).mean()
                    if pos_rate_train < 0.05 or pos_rate_train > 0.50:
                        print(f"  [WARN] {exp_id} train pos_rate={pos_rate_train:.4f} unusual (expect ~0.17-0.20)")

                    t = time.time()
                    val_size = max(500, int(len(train) * 0.10))
                    val = train.tail(val_size)
                    train_fit = train.head(len(train) - val_size)

                    model = lgb.LGBMClassifier(**LGB_BINARY_PARAMS)
                    model.fit(
                        train_fit[base_cols], train_fit["y"],
                        eval_set=[(val[base_cols], val["y"])],
                        callbacks=[lgb.early_stopping(50, verbose=False)],
                    )
                    best_iter = model.best_iteration_ or LGB_BINARY_PARAMS["n_estimators"]
                    del train, train_fit, val, joined; gc.collect()

                    fam_imp = family_importance(model.booster_, base_cols)
                    family_importance_results[exp_id] = fam_imp

                    # SCORE all rows of universe panel (not just sampled labels) for eval
                    preds = model.predict_proba(upanel[base_cols])[:, 1].astype(np.float32)
                    pred_df = upanel[["ts_code", "trade_date"]].copy()
                    pred_df["score"] = preds

                    if panel_name in ("ledashi", "v3unified"):
                        pred_df.to_parquet(OUT_DIR / f"pred_{exp_id}.parquet", compression="zstd")

                    result = eval_cell(pred_df, realized, exits, adaptive_gating=None)
                    matrix_results[exp_id] = result

                    r = result["static"]["H2_2025"]
                    ic20 = r["fwd20"]["ic"] * 100
                    sn50 = r["fwd20"]["sizing"].get("50", {}).get("sharpe_net", float("nan"))
                    q1 = result["static"]["Q1_2026"]["fwd20"]["ic"] * 100
                    print(f"  {exp_id}: train {time.time()-t:.0f}s (best_iter={best_iter}, pos_rate={pos_rate_train:.3f}) | H2 fwd20 IC={ic20:+.3f}% Sharpe50_NET={sn50:+.2f} | Q1 IC={q1:+.3f}%", flush=True)

                    del model, preds, pred_df, result; gc.collect()
                    CHECKPOINT.write_text(json.dumps({
                        "results": matrix_results, "family_importance": family_importance_results,
                    }, indent=2, default=str))
            del upanel; gc.collect()
        del panel; gc.collect()

    out = {
        "config": {"tier": "v12 — paris Phase 2 anchor paradigm 2 (β PELT-hybrid + α ledashi-draft)",
                   "panels": list(PANELS), "universes": list(UNIVERSES),
                   "anchors": list(ANCHORS), "specs": list(SPECS),
                   "lgb_params": LGB_BINARY_PARAMS,
                   "label_pos_rate_note": "anchor labels SAMPLED ~20% pos (NEG_RATIO=5:1), not full cross-section"},
        "results": matrix_results, "family_importance": family_importance_results,
        "total_time_s": time.time() - t_total,
    }
    Path("data/kronos/outputs/matrix_v12_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] matrix_v12_results.json, {len(matrix_results)} cells, {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
