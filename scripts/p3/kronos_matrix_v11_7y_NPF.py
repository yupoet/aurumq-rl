"""Matrix v11 7y self-train for NPF + NPF_FULL universes.

Reuses kronos_matrix_v11_7y_HARD_TECH.py logic, swaps universe + adds NPF + NPF_FULL.
Train cells: 2 universe × 3 horizon = 6 cells (T1/T3/T5 × NPF/NPF_FULL).

Compare against v11 3y NPF/NPF_FULL cells to test if 7y dilutes Q1 alpha similarly
to HARD_TECH T5 finding.
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
import datetime as _dt_mod

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kronos_matrix_v10 import (
    UNIVERSES, _dt, load_universes, filter_universe, compute_realized_and_exits,
    family_importance, eval_cell,
)
from kronos_matrix_v11 import LGB_BINARY_PARAMS

PARIS_7Y_DIR = Path(f"{_HANDOFF_INBOX}/2026-05-19-paris-panel-v2-no-phase-c-7y")
LEDASHI_3Y_PANEL_PATH = Path(f"{_HANDOFF_INBOX}/2026-05-15-paris-panel-v2-no-phase-c/combined_panel_v_x_v2_no_phase_c.parquet")
LEDASHI_3Y_LABELS_DIR = Path(f"{_HANDOFF_INBOX}/2026-05-16-paris-reply-v24")

OUT_DIR = Path("data/kronos/outputs/matrix_v11_7y_NPF_NPF_FULL")
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSES_TEST = ("NPF", "NPF_FULL")
METHOD = "A"
HORIZONS_TEST = ("t1", "t3", "t5")

TRAIN_START_7Y = _dt_mod.date(2018, 1, 1)
TRAIN_END = _dt_mod.date(2024, 12, 31)


def load_concat_panel(universe, static_sets, pit_dfs):
    """paris 2018-2021 + ledashi 2022-2026, universe-filtered."""
    print(f"\n[panel] {universe}: col intersection ...")
    paris_sample = pd.read_parquet(PARIS_7Y_DIR / "combined_panel_v2_no_phase_c_year=2018.parquet").head(1)
    ledashi_sample = pd.read_parquet(LEDASHI_3Y_PANEL_PATH).head(1)
    common_cols = sorted(set(paris_sample.columns) & set(ledashi_sample.columns))
    del paris_sample, ledashi_sample; gc.collect()

    univ_codes = set(static_sets[universe]) if universe in static_sets else None

    print(f"[panel] loading paris 2018-2021 + ledashi 2022-2026 ({universe} filter) ...")
    parts = []
    for year in (2018, 2019, 2020, 2021):
        fp = PARIS_7Y_DIR / f"combined_panel_v2_no_phase_c_year={year}.parquet"
        df = pd.read_parquet(fp, columns=common_cols)
        if univ_codes is not None:
            df = df[df['ts_code'].isin(univ_codes)]
        parts.append(df)
        print(f"  paris {year}: {len(df):,} rows")
        gc.collect()
    ledashi_panel = pd.read_parquet(LEDASHI_3Y_PANEL_PATH, columns=common_cols)
    if univ_codes is not None:
        ledashi_panel = ledashi_panel[ledashi_panel['ts_code'].isin(univ_codes)]
    _train_2022 = _dt_mod.date(2022, 1, 1)
    ledashi_panel = ledashi_panel[ledashi_panel['trade_date'] >= _train_2022]
    print(f"  ledashi 2022-2026: {len(ledashi_panel):,} rows")
    parts.append(ledashi_panel)

    combined = pd.concat(parts, ignore_index=True)
    del parts, ledashi_panel; gc.collect()
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
    print(f"  combined {universe} 7y: {len(combined):,} rows × {combined.shape[1]} cols")
    return combined, combined.columns.tolist()


def load_concat_label(universe, horizon):
    """Concat paris 2018-2021 + ledashi 2022-2026 labels for given universe."""
    dfs = []
    # paris 7y bundle only has HARD_TECH labels. For NPF/NPF_FULL, we need to use ledashi 2022-2026 only.
    # FOR NPF/NPF_FULL: use only ledashi 2022-2026 labels (no paris 2018-2021 NPF labels).
    # Train_start effectively 2022-01 for NPF cells (paris didn't ship NPF 7y labels).
    for year in (2022, 2023, 2024, 2025, 2026):
        fp = LEDASHI_3Y_LABELS_DIR / f"labels_A_{horizon}_{universe}_year={year}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp, columns=["trade_date", "ts_code", "y"])
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

    matrix_results = {}
    family_importance_results = {}

    for universe in UNIVERSES_TEST:
        upanel, cols_full = load_concat_panel(universe, static_sets, pit_dfs)
        base_cols = [c for c in cols_full if c not in ("ts_code", "trade_date")]
        print(f"\n[upanel] {universe} 7y: {len(upanel):,} rows × {len(base_cols)} feats ({upanel['trade_date'].min()} ~ {upanel['trade_date'].max()})")

        for horizon in HORIZONS_TEST:
            exp_id = f"7y_A_{horizon}_{universe}_v2_no_phase_c"
            print(f"\n=== {exp_id} ===")

            label_df = load_concat_label(universe, horizon)
            if label_df is None:
                print(f"  [SKIP] no labels found for {universe}_{horizon}"); continue

            joined = upanel.merge(label_df, on=["ts_code", "trade_date"], how="inner")
            train = joined[(joined["trade_date"] >= TRAIN_START_7Y) & (joined["trade_date"] <= TRAIN_END)]
            print(f"  train rows: {len(train):,}, pos rate: {(train['y']>0).mean():.4f}")
            print(f"  train window: {train['trade_date'].min()} ~ {train['trade_date'].max()}")

            if len(train) < 5000:
                print(f"  [SKIP] {exp_id} train {len(train)} < 5K"); matrix_results[exp_id] = {"skipped": True}; continue

            t = time.time()
            val_size = max(1000, int(len(train) * 0.10))
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

            preds = model.predict_proba(upanel[base_cols])[:, 1].astype(np.float32)
            pred_df = upanel[["ts_code", "trade_date"]].copy()
            pred_df["score"] = preds

            artifact_dir = OUT_DIR / exp_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            model.booster_.save_model(str(artifact_dir / "model.txt"))
            (artifact_dir / "feature_cols.json").write_text(json.dumps(base_cols, indent=2))
            (artifact_dir / "lgb_params.json").write_text(json.dumps(LGB_BINARY_PARAMS, indent=2))
            (artifact_dir / "train_spec.json").write_text(json.dumps({
                "train_window": [str(TRAIN_START_7Y), str(TRAIN_END)],
                "best_iter": int(best_iter), "n_features": len(base_cols),
                "comment": f"7y train (paris panel 2018-2021 + ledashi panel 2022-2024, intersection {len(base_cols)} cols); labels 2022-2024 only (no paris {universe} 2018-2021 labels)",
            }, indent=2))
            pred_df.to_parquet(artifact_dir / "pred_full_panel.parquet", compression="zstd")

            result = eval_cell(pred_df, realized, exits, adaptive_gating=None)
            matrix_results[exp_id] = result
            r = result["static"]["H2_2025"]
            ic20 = r["fwd20"]["ic"] * 100
            sn50 = r["fwd20"]["sizing"].get("50", {}).get("sharpe_net", float("nan"))
            q1 = result["static"]["Q1_2026"]["fwd20"]["ic"] * 100
            print(f"  {exp_id}: train {time.time()-t:.0f}s (best_iter={best_iter}) | H2 IC={ic20:+.3f}% Sharpe50_NET={sn50:+.2f} | Q1 IC={q1:+.3f}%", flush=True)
            del model, preds, pred_df, result; gc.collect()
        del upanel; gc.collect()

    out = {
        "config": {"tier": "v11 7y NPF + NPF_FULL self-train",
                   "universes": list(UNIVERSES_TEST), "method": METHOD, "horizons": list(HORIZONS_TEST),
                   "train_window": [str(TRAIN_START_7Y), str(TRAIN_END)],
                   "label_window": "ledashi 2022-2024 only (no paris 2018-2021 NPF labels)",
                   "lgb_params": LGB_BINARY_PARAMS},
        "results": matrix_results, "family_importance": family_importance_results,
        "total_time_s": time.time() - t_total,
    }
    Path("data/kronos/outputs/matrix_v11_7y_NPF_NPF_FULL_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] matrix_v11_7y_NPF_NPF_FULL_results.json, {len(matrix_results)} cells, {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
