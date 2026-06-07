"""Matrix v12 7y anchor T3 — paris Batch 3B + 3C trainings.

Per paris ACK v11_7y §3.3B + §3.3C:
- 7y_anchor_T3_HARD_TECH_v3unified: paradigm 2 on 7y data (paris 2018-2021 anchor events + ledashi 2022-2026)
- anchor_T3_CSI1000_v3unified: paradigm 2 on NEW universe (CSI1000 2022-2026 only)

paris anchor events 2018-2021 shipped: `oss://ledashi-oss/aurumq-rl/handoffs/2026-05-22-paris-anchor-events-2018-2021/`

Spec: v3unified panel (paris production candidate, 244 cols) + anchor α 5-condition labels.
"""
from __future__ import annotations
import os, gc, json, time
from pathlib import Path

_HANDOFF_INBOX = os.environ.get("AURUMQ_HANDOFF_INBOX", "data/handoffs/inbox")
import datetime as _dt_mod
import lightgbm as lgb
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kronos_matrix_v10 import (
    UNIVERSES, _dt, load_universes, filter_universe, compute_realized_and_exits,
    family_importance, load_panel, eval_cell,
)  # noqa: filter_universe used below
from kronos_matrix_v12 import LGB_BINARY_PARAMS, ANCHOR_DIR_ALPHA as V12_ALPHA_DIR

PARIS_7Y_ANCHOR_DIR = Path(f"{_HANDOFF_INBOX}/2026-05-22-paris-anchor-events-2018-2021")

OUT_DIR = Path("data/kronos/outputs/matrix_v12_anchor_7y")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START_HARD_TECH = _dt_mod.date(2018, 1, 1)
TRAIN_START_CSI1000 = _dt_mod.date(2022, 1, 1)
TRAIN_END = _dt_mod.date(2024, 12, 31)

PARIS_7Y_PANEL_DIR = Path(f"{_HANDOFF_INBOX}/2026-05-19-paris-panel-v2-no-phase-c-7y")
LEDASHI_PANEL_PATH_TEMPLATE = "D:/dev/aurumq-handoffs/inbox/2026-05-15-paris-panel-v3-unified/combined_panel_v_x_v3_unified.parquet"


def load_anchor_labels(universe, anchor):
    """Concat paris bundle 2018-2021 + 2022-2026 anchor labels (paris ship all years)."""
    dfs = []
    for year in (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026):
        fp = PARIS_7Y_ANCHOR_DIR / f"anchor_{anchor}_{universe}_year={year}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp, columns=["trade_date", "ts_code", "y_binary"])
            df = df.rename(columns={"y_binary": "y"})
            dfs.append(df)
    if not dfs: return None
    out = pd.concat(dfs, ignore_index=True)
    out = _dt(out)
    return out


def load_v3unified_panel_for_universe(universe, static_sets, pit_dfs, use_7y_for_hard_tech=True):
    """For HARD_TECH: concat paris 7y v2_no_phase_c panel + ledashi v3unified...
    Actually use ledashi v3unified panel directly for both (paris 7y panel is v2_no_phase_c which differs from v3unified).

    For HARD_TECH 7y: we need 2018-2021 panel that has v3unified-compatible cols.
    paris ship only 2018-2021 v2_no_phase_c (357 cols), NOT v3unified.
    Workaround: use intersection of paris v2_no_phase_c + ledashi v3unified for 7y HARD_TECH,
                use ledashi v3unified only for CSI1000.
    """
    if universe == "HARD_TECH" and use_7y_for_hard_tech:
        # 7y path: paris 2018-2021 v2_no_phase_c + ledashi v3unified 2022+ intersection
        paris_sample = pd.read_parquet(PARIS_7Y_PANEL_DIR / "combined_panel_v2_no_phase_c_year=2018.parquet").head(1)
        ledashi_panel = pd.read_parquet(LEDASHI_PANEL_PATH_TEMPLATE).head(1)
        common_cols = sorted(set(paris_sample.columns) & set(ledashi_panel.columns))
        del paris_sample, ledashi_panel; gc.collect()
        print(f"  HARD_TECH 7y col intersection: {len(common_cols)}")

        univ_codes = set(static_sets[universe])
        parts = []
        for year in (2018, 2019, 2020, 2021):
            fp = PARIS_7Y_PANEL_DIR / f"combined_panel_v2_no_phase_c_year={year}.parquet"
            df = pd.read_parquet(fp, columns=common_cols)
            df = df[df['ts_code'].isin(univ_codes)]
            parts.append(df)
            gc.collect()
        ledashi_panel = pd.read_parquet(LEDASHI_PANEL_PATH_TEMPLATE, columns=common_cols)
        ledashi_panel = ledashi_panel[ledashi_panel['ts_code'].isin(univ_codes)]
        _train_2022 = _dt_mod.date(2022, 1, 1)
        ledashi_panel = ledashi_panel[ledashi_panel['trade_date'] >= _train_2022]
        parts.append(ledashi_panel)
        combined = pd.concat(parts, ignore_index=True)
        del parts; gc.collect()
    else:
        # CSI1000: ledashi v3unified panel only (2022-2026), use v10 filter_universe util
        panel = pd.read_parquet(LEDASHI_PANEL_PATH_TEMPLATE)
        panel = _dt(panel)
        combined = filter_universe(panel, universe, static_sets, pit_dfs)
        del panel; gc.collect()

    combined = _dt(combined)
    drop_cols = [c for c in combined.columns if c not in ("ts_code", "trade_date")
                 and not (pd.api.types.is_numeric_dtype(combined[c]) or pd.api.types.is_bool_dtype(combined[c]))]
    if drop_cols:
        print(f"  drop non-numeric: {drop_cols}")
        combined = combined.drop(columns=drop_cols)
    for c in combined.columns:
        if c in ("ts_code", "trade_date"): continue
        if str(combined[c].dtype).startswith("Int"):
            combined[c] = combined[c].astype("float32")
        elif combined[c].dtype == np.float64:
            combined[c] = combined[c].astype(np.float32)
    combined = combined.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return combined


def train_eval_cell(exp_id, universe, anchor, panel_name, train_start, static_sets, pit_dfs, realized, exits, use_7y_panel):
    print(f"\n=== {exp_id} ===")
    upanel = load_v3unified_panel_for_universe(universe, static_sets, pit_dfs, use_7y_for_hard_tech=use_7y_panel)
    base_cols = [c for c in upanel.columns if c not in ("ts_code", "trade_date")]
    print(f"  upanel: {len(upanel):,} rows × {len(base_cols)} cols ({upanel['trade_date'].min()} ~ {upanel['trade_date'].max()})")

    label_df = load_anchor_labels(universe, anchor)
    if label_df is None or len(label_df) == 0:
        print(f"  [SKIP] no labels"); return None

    joined = upanel.merge(label_df, on=["ts_code", "trade_date"], how="inner")
    train = joined[(joined["trade_date"] >= train_start) & (joined["trade_date"] <= TRAIN_END)]
    print(f"  train rows: {len(train):,}, pos rate: {(train['y']>0).mean():.4f}")
    print(f"  train window: {train['trade_date'].min()} ~ {train['trade_date'].max()}")
    if len(train) < 1000:
        print(f"  [SKIP] {exp_id} train {len(train)} < 1K"); return None

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

    preds = model.predict_proba(upanel[base_cols])[:, 1].astype(np.float32)
    pred_df = upanel[["ts_code", "trade_date"]].copy()
    pred_df["score"] = preds

    artifact_dir = OUT_DIR / exp_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(artifact_dir / "model.txt"))
    (artifact_dir / "feature_cols.json").write_text(json.dumps(base_cols, indent=2))
    (artifact_dir / "lgb_params.json").write_text(json.dumps(LGB_BINARY_PARAMS, indent=2))
    (artifact_dir / "train_spec.json").write_text(json.dumps({
        "train_window": [str(train_start), str(TRAIN_END)],
        "best_iter": int(best_iter), "n_features": len(base_cols),
        "panel_basis": "7y intersection HARD_TECH or v3unified-only CSI1000",
        "label": f"anchor_{anchor}_{universe} α 5-condition",
    }, indent=2))
    pred_df.to_parquet(artifact_dir / "pred_full_panel.parquet", compression="zstd")

    result = eval_cell(pred_df, realized, exits, adaptive_gating=None)
    r = result["static"]["H2_2025"]
    ic20 = r["fwd20"]["ic"] * 100
    sn50 = r["fwd20"]["sizing"].get("50", {}).get("sharpe_net", float("nan"))
    q1 = result["static"]["Q1_2026"]["fwd20"]["ic"] * 100
    print(f"  {exp_id}: train {time.time()-t:.0f}s (best_iter={best_iter}) | H2 IC={ic20:+.3f}% Sharpe50={sn50:+.2f} | Q1 IC={q1:+.3f}%")
    del model, preds, pred_df; gc.collect()
    return result


def main():
    t_total = time.time()
    print("[setup] universes + realized ...")
    static_sets, pit_dfs = load_universes(UNIVERSES)
    realized, exits = compute_realized_and_exits()

    matrix_results = {}

    # 7y HARD_TECH anchor T3
    res = train_eval_cell(
        exp_id="7y_alpha_T3_HARD_TECH_v3unified",
        universe="HARD_TECH",
        anchor="T3",
        panel_name="v3unified",
        train_start=TRAIN_START_HARD_TECH,
        static_sets=static_sets, pit_dfs=pit_dfs, realized=realized, exits=exits,
        use_7y_panel=True,
    )
    if res: matrix_results["7y_alpha_T3_HARD_TECH_v3unified"] = res

    # CSI1000 anchor T3 (3y, paris ship CSI1000 2022-2026 anchor events)
    res = train_eval_cell(
        exp_id="alpha_T3_CSI1000_v3unified",
        universe="CSI1000",
        anchor="T3",
        panel_name="v3unified",
        train_start=TRAIN_START_CSI1000,
        static_sets=static_sets, pit_dfs=pit_dfs, realized=realized, exits=exits,
        use_7y_panel=False,
    )
    if res: matrix_results["alpha_T3_CSI1000_v3unified"] = res

    out = {
        "config": {"tier": "v12 anchor 7y (paris Batch 3B + 3C)",
                   "cells": ["7y_alpha_T3_HARD_TECH_v3unified", "alpha_T3_CSI1000_v3unified"],
                   "anchor": "T3 alpha 5-condition baseline",
                   "lgb_params": LGB_BINARY_PARAMS},
        "results": matrix_results,
        "total_time_s": time.time() - t_total,
    }
    Path("data/kronos/outputs/matrix_v12_anchor_7y_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] matrix_v12_anchor_7y_results.json, {len(matrix_results)} cells, {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
