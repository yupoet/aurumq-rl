"""Matrix v11 7y self-train — paris experiment: long-window 2018-2024 vs short 3y.

paris v26 raised: "v11 train 2022-2024 vs paris production 2018-2024 (4-year gap).
This affects model quality." paris shipped 7-year HARD_TECH panel + labels (5/19)
to settle the comparison.

Spec:
- Universe: HARD_TECH (193 stocks)
- Method: A (paris production-aligned)
- Horizons: t1, t3, t5 (3 cells)
- Panel: v2_no_phase_c (intersection 356 cols, paris 357 ∩ ledashi 366 = 356, dropping 10 ledashi-only + 1 paris-only)
- Train: 2018-01-01 ~ 2024-12-31 (7 years)
- Eval: H2_2025 + Q1_2026 (same as v11)
- LGB binary classifier (same hyperparams as v11)

Decision gate (paris 5 conditions for 7y → production):
  1. Sharpe NET ≥ 3y + 0.3
  2. H2 fwd20 IC ≥ 3y - 0.5pp
  3. Q1 fwd20 IC ≥ 3y - 1pp
  4. regime spread H2-Q1 reduced
  5. signal density ≥ 0.8× 3y

Output: data/kronos/outputs/matrix_v11_7y_HARD_TECH/
  + matrix_v11_7y_HARD_TECH_results.json (per-horizon results)
  + 3 pred_full_panel parquets
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
    UNIVERSES, PANEL_CLOSE, WINDOWS, HORIZONS, K_MAX, TRIGGERS, COST_ROUND_TRIP,
    TOP_K_LIST, ADAPTIVE_PCT, ADAPTIVE_K_CAP, ADAPTIVE_K_FLOOR, ADAPTIVE_Q_LOW, ADAPTIVE_Q_MID,
    FAMILY_PREFIXES,
    _dt, load_universes, filter_universe, compute_realized_and_exits,
    cross_sec_ic, top_k_sharpe_ann_multi, dyn_agg_multi, compute_adaptive_thresholds,
    family_importance, load_panel, eval_cell,
)
from kronos_matrix_v11 import LGB_BINARY_PARAMS, PHASE1_DIR, load_sparse_label

PARIS_7Y_DIR = Path(f"{_HANDOFF_INBOX}/2026-05-19-paris-panel-v2-no-phase-c-7y")
LEDASHI_3Y_PANEL_PATH = Path(f"{_HANDOFF_INBOX}/2026-05-15-paris-panel-v2-no-phase-c/combined_panel_v_x_v2_no_phase_c.parquet")
LEDASHI_3Y_LABELS_DIR = Path(f"{_HANDOFF_INBOX}/2026-05-16-paris-reply-v24")

OUT_DIR = Path("data/kronos/outputs/matrix_v11_7y_HARD_TECH")
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = "HARD_TECH"
METHOD = "A"
HORIZONS_TEST = ("t1", "t3", "t5")

TRAIN_START_7Y = pd.Timestamp("2018-01-01").date()
TRAIN_END = pd.Timestamp("2024-12-31").date()


def load_concat_panel(static_sets, pit_dfs):
    """Concat paris 2018-2021 + ledashi 2022-2024, FILTER HARD_TECH per-year to save memory."""
    # First peek to compute common_cols
    print(f"\n[panel] computing col intersection (paris 2018 sample vs ledashi 2022+) ...")
    paris_sample = pd.read_parquet(PARIS_7Y_DIR / "combined_panel_v2_no_phase_c_year=2018.parquet", columns=None).head(1)
    ledashi_sample = pd.read_parquet(LEDASHI_3Y_PANEL_PATH, columns=None).head(1)
    common_cols = sorted(set(paris_sample.columns) & set(ledashi_sample.columns))
    print(f"  common cols (intersection): {len(common_cols)}")
    del paris_sample, ledashi_sample; gc.collect()

    # HARD_TECH membership
    hard_tech_codes = set(static_sets[UNIVERSE])
    print(f"  HARD_TECH 193 stocks pre-filter")

    # Year-by-year load + filter
    print(f"\n[panel] loading paris 2018-2021 (4 years) filtered to HARD_TECH ...")
    parts = []
    for year in (2018, 2019, 2020, 2021):
        fp = PARIS_7Y_DIR / f"combined_panel_v2_no_phase_c_year={year}.parquet"
        df = pd.read_parquet(fp, columns=common_cols)
        df = df[df['ts_code'].isin(hard_tech_codes)]
        parts.append(df)
        print(f"  paris {year}: {len(df):,} rows × {df.shape[1]} cols (filtered HARD_TECH)")
        gc.collect()

    print(f"\n[panel] loading ledashi 2022-2026 filtered to HARD_TECH ...")
    ledashi_panel = pd.read_parquet(LEDASHI_3Y_PANEL_PATH, columns=common_cols)
    ledashi_panel = ledashi_panel[ledashi_panel['ts_code'].isin(hard_tech_codes)]
    import datetime as _dt_mod
    _train_2022 = _dt_mod.date(2022, 1, 1)
    ledashi_panel = ledashi_panel[ledashi_panel['trade_date'] >= _train_2022]
    print(f"  ledashi 2022-2026: {len(ledashi_panel):,} rows × {ledashi_panel.shape[1]} cols (filtered HARD_TECH)")
    parts.append(ledashi_panel)

    combined = pd.concat(parts, ignore_index=True)
    del parts, ledashi_panel; gc.collect()
    combined = _dt(combined)

    # Drop non-numeric / non-bool cols (LGB requires numeric features)
    drop_cols = [c for c in combined.columns if c not in ("ts_code", "trade_date")
                 and not (pd.api.types.is_numeric_dtype(combined[c]) or pd.api.types.is_bool_dtype(combined[c]))]
    if drop_cols:
        print(f"  drop non-numeric cols ({len(drop_cols)}): {drop_cols[:8]}{'...' if len(drop_cols) > 8 else ''}")
        combined = combined.drop(columns=drop_cols)

    # Cast Int/float64 → float32 for LGB memory efficiency
    for c in combined.columns:
        if c in ("ts_code", "trade_date"): continue
        if str(combined[c].dtype).startswith("Int"):
            combined[c] = combined[c].astype("float32")
        elif combined[c].dtype == np.float64:
            combined[c] = combined[c].astype(np.float32)

    combined = combined.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    final_cols = combined.columns.tolist()
    print(f"\n  combined 7y HARD_TECH panel: {len(combined):,} rows × {combined.shape[1]} cols (after drop non-numeric)")
    return combined, final_cols


def load_concat_label(horizon):
    """Concat paris 2018-2021 + ledashi 2022-2026 labels_A_{horizon}_HARD_TECH."""
    dfs = []
    for year in (2018, 2019, 2020, 2021):
        fp = PARIS_7Y_DIR / f"labels_A_{horizon}_HARD_TECH_year={year}.parquet"
        df = pd.read_parquet(fp, columns=["trade_date", "ts_code", "y"])
        dfs.append(df)
    # ledashi 2022-2026 (from v24 ship)
    for year in (2022, 2023, 2024, 2025, 2026):
        fp = LEDASHI_3Y_LABELS_DIR / f"labels_A_{horizon}_HARD_TECH_year={year}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp, columns=["trade_date", "ts_code", "y"])
            dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out = _dt(out)
    return out


def main():
    t_total = time.time()
    print("[setup] universes ...")
    static_sets, pit_dfs = load_universes(UNIVERSES)

    print("\n[setup] realized + dyn-exit ...")
    realized, exits = compute_realized_and_exits()

    upanel, base_cols_full = load_concat_panel(static_sets, pit_dfs)
    base_cols = [c for c in base_cols_full if c not in ("ts_code", "trade_date")]
    print(f"\n[feature cols]: {len(base_cols)}")
    print(f"[upanel] HARD_TECH 7y: {len(upanel):,} rows ({upanel['trade_date'].min()} ~ {upanel['trade_date'].max()})")

    matrix_results = {}
    family_importance_results = {}

    for horizon in HORIZONS_TEST:
        exp_id = f"7y_A_{horizon}_HARD_TECH_v2_no_phase_c"
        print(f"\n=== {exp_id} ===")

        label_df = load_concat_label(horizon)
        print(f"  label rows: {len(label_df):,}, full pos rate: {(label_df['y']>0).mean():.4f}")

        joined = upanel.merge(label_df, on=["ts_code", "trade_date"], how="inner")
        train = joined[(joined["trade_date"] >= TRAIN_START_7Y) & (joined["trade_date"] <= TRAIN_END)]
        print(f"  train rows: {len(train):,}, pos rate train: {(train['y']>0).mean():.4f}")
        print(f"  train window: {train['trade_date'].min()} ~ {train['trade_date'].max()}")

        if len(train) < 5000:
            print(f"  [SKIP] {exp_id} train {len(train)} < 5K")
            matrix_results[exp_id] = {"skipped": True}; continue

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

        # Save artifacts for paris production swap
        artifact_dir = OUT_DIR / exp_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(artifact_dir / "model.txt"))
        (artifact_dir / "feature_cols.json").write_text(json.dumps(base_cols, indent=2))
        (artifact_dir / "lgb_params.json").write_text(json.dumps(LGB_BINARY_PARAMS, indent=2))
        (artifact_dir / "train_spec.json").write_text(json.dumps({
            "train_window": [str(TRAIN_START_7Y), str(TRAIN_END)],
            "best_iter": int(best_iter),
            "n_features": len(base_cols),
            "comment": "7-year train (paris 2018-2021 + ledashi 2022-2024 intersection 356 cols)",
        }, indent=2))
        pred_df.to_parquet(artifact_dir / "pred_full_panel.parquet", compression="zstd")

        result = eval_cell(pred_df, realized, exits, adaptive_gating=None)
        matrix_results[exp_id] = result

        r = result["static"]["H2_2025"]
        ic20 = r["fwd20"]["ic"] * 100
        sn50 = r["fwd20"]["sizing"].get("50", {}).get("sharpe_net", float("nan"))
        sn10 = r["fwd20"]["sizing"].get("10", {}).get("sharpe_net", float("nan"))
        q1 = result["static"]["Q1_2026"]["fwd20"]["ic"] * 100
        print(f"  {exp_id}: train {time.time()-t:.0f}s (best_iter={best_iter}) | H2 fwd20 IC={ic20:+.3f}% Sharpe50_NET={sn50:+.2f} Sharpe10_NET={sn10:+.2f} | Q1 IC={q1:+.3f}%", flush=True)

        del model, preds, pred_df; gc.collect()

    out = {
        "config": {"tier": "v11 7y self-train HARD_TECH (paris paradigm 1 long-window comparison)",
                   "universe": UNIVERSE,
                   "method": METHOD,
                   "horizons": list(HORIZONS_TEST),
                   "train_window": [str(TRAIN_START_7Y), str(TRAIN_END)],
                   "panel": "intersection of paris-7y + ledashi-3y v2_no_phase_c (356 cols)",
                   "lgb_params": LGB_BINARY_PARAMS},
        "results": matrix_results,
        "family_importance": family_importance_results,
        "total_time_s": time.time() - t_total,
    }
    Path("data/kronos/outputs/matrix_v11_7y_HARD_TECH_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] matrix_v11_7y_HARD_TECH_results.json, {len(matrix_results)} cells, {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
