"""v33 task 3 — 7y HARD_TECH paradigm 1 Bayesian hyperparam sweep (paris ACK_v19 §2.3 P1).

30 Optuna trials sampling LGB hyperparams. Success criterion: any trial dual-positive
H2 ≥ +2.0 AND Q1 ≥ +6.0 AND Sharpe ≥ +3.5.

Cell base: A_t5_HARD_TECH_v2_no_phase_c (v11 7y best so far)
Sweep params:
  - min_child_samples: 50 ~ 500
  - feature_fraction: 0.5 ~ 1.0
  - num_leaves: 31 ~ 255
  - learning_rate: 0.01 ~ 0.10

Panel + labels loaded ONCE, shared across trials.
"""
from __future__ import annotations
import os, gc, json, time
from pathlib import Path
import datetime as _dt_mod

_HANDOFF_INBOX = os.environ.get("AURUMQ_HANDOFF_INBOX", "data/handoffs/inbox")
import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kronos_matrix_v10 import _dt, load_universes, compute_realized_and_exits, eval_cell, WINDOWS
from kronos_matrix_v11_7y_HARD_TECH import (
    PARIS_7Y_DIR, LEDASHI_3Y_PANEL_PATH, LEDASHI_3Y_LABELS_DIR,
    UNIVERSE, TRAIN_START_7Y, TRAIN_END,
)

OUT_DIR = Path("data/kronos/outputs/matrix_v33_hard_tech_7y_bayesian")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = "t5"  # paris ACK_v19 best 3y HARD_TECH cell was T5
N_TRIALS = 30
SEED = 42


def load_panel_filtered():
    """Reuse pattern from kronos_matrix_v11_7y_HARD_TECH.load_concat_panel."""
    print("\n[panel] loading 7y HARD_TECH panel + labels ...")
    paris_sample = pd.read_parquet(PARIS_7Y_DIR / "combined_panel_v2_no_phase_c_year=2018.parquet").head(1)
    ledashi_sample = pd.read_parquet(LEDASHI_3Y_PANEL_PATH).head(1)
    common_cols = sorted(set(paris_sample.columns) & set(ledashi_sample.columns))
    print(f"  common cols: {len(common_cols)}")
    del paris_sample, ledashi_sample; gc.collect()

    static_sets, _ = load_universes((UNIVERSE,))
    hard_tech_codes = set(static_sets[UNIVERSE])

    parts = []
    for year in (2018, 2019, 2020, 2021):
        fp = PARIS_7Y_DIR / f"combined_panel_v2_no_phase_c_year={year}.parquet"
        df = pd.read_parquet(fp, columns=common_cols)
        df = df[df['ts_code'].isin(hard_tech_codes)]
        parts.append(df)
        print(f"  paris {year}: {len(df):,} rows")
        gc.collect()
    ledashi_panel = pd.read_parquet(LEDASHI_3Y_PANEL_PATH, columns=common_cols)
    ledashi_panel = ledashi_panel[ledashi_panel['ts_code'].isin(hard_tech_codes)]
    _train_2022 = _dt_mod.date(2022, 1, 1)
    ledashi_panel = ledashi_panel[ledashi_panel['trade_date'] >= _train_2022]
    print(f"  ledashi 2022+: {len(ledashi_panel):,} rows")
    parts.append(ledashi_panel)
    combined = pd.concat(parts, ignore_index=True); del parts, ledashi_panel; gc.collect()
    combined = _dt(combined)

    drop_cols = [c for c in combined.columns if c not in ("ts_code", "trade_date")
                 and not (pd.api.types.is_numeric_dtype(combined[c]) or pd.api.types.is_bool_dtype(combined[c]))]
    if drop_cols:
        combined = combined.drop(columns=drop_cols)
    for c in combined.columns:
        if c in ("ts_code", "trade_date"): continue
        if str(combined[c].dtype).startswith("Int"): combined[c] = combined[c].astype("float32")
        elif combined[c].dtype == np.float64: combined[c] = combined[c].astype(np.float32)
    return combined


def load_labels_7y():
    """Concat paris 2018-2021 + ledashi 2022+ sparse t5 HARD_TECH labels."""
    dfs = []
    # paris 2018-2021
    for year in (2018, 2019, 2020, 2021):
        fp = PARIS_7Y_DIR / f"labels_A_t5_HARD_TECH_year={year}.parquet"
        if fp.exists():
            dfs.append(pd.read_parquet(fp))
    # ledashi 2022-2026 from v24 reply
    for year in (2022, 2023, 2024, 2025, 2026):
        fp = LEDASHI_3Y_LABELS_DIR / f"labels_A_t5_HARD_TECH_year={year}.parquet"
        if fp.exists():
            dfs.append(pd.read_parquet(fp))
    label_df = pd.concat(dfs, ignore_index=True)
    label_df = _dt(label_df)
    print(f"  labels (sparse t5 HARD_TECH 7y): {len(label_df):,} rows, pos rate {(label_df['y']>0).mean():.4f}")
    return label_df


def objective_factory(panel, label_df, realized, exits, base_cols):
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "n_estimators": 500,
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 500),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "bagging_fraction": 0.85, "bagging_freq": 1,
            "n_jobs": -1, "verbose": -1, "random_state": SEED,
        }
        joined = panel.merge(label_df, on=["ts_code", "trade_date"], how="inner")
        train = joined[(joined["trade_date"] >= TRAIN_START_7Y) & (joined["trade_date"] <= TRAIN_END)]
        if len(train) < 1000:
            return -1.0
        val_size = max(500, int(len(train) * 0.10))
        val = train.tail(val_size)
        train_fit = train.head(len(train) - val_size)
        del train, joined; gc.collect()

        t = time.time()
        model = lgb.LGBMClassifier(**params)
        model.fit(train_fit[base_cols], train_fit["y"],
                  eval_set=[(val[base_cols], val["y"])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        train_time = time.time() - t

        preds = model.predict_proba(panel[base_cols])[:, 1].astype(np.float32)
        pred_df = panel[["ts_code", "trade_date"]].copy()
        pred_df["score"] = preds

        result = eval_cell(pred_df, realized, exits, adaptive_gating=None)
        r = result["static"]["H2_2025"]["fwd20"]
        q1 = result["static"]["Q1_2026"]["fwd20"]
        ic_h2 = r["ic"] * 100
        ic_q1 = q1["ic"] * 100
        sn_h2 = r["sizing"].get("50", {}).get("sharpe_net", float("nan"))
        sn_q1 = q1["sizing"].get("50", {}).get("sharpe_net", float("nan"))

        # Custom objective: composite Sharpe_NET prioritizing dual-regime
        composite = sn_h2 + max(sn_q1, 0) + 0.1 * (ic_h2 + ic_q1)

        trial.set_user_attr("h2_ic", ic_h2)
        trial.set_user_attr("q1_ic", ic_q1)
        trial.set_user_attr("sharpe_h2", sn_h2)
        trial.set_user_attr("sharpe_q1", sn_q1)
        trial.set_user_attr("train_time_s", train_time)
        trial.set_user_attr("best_iter", model.best_iteration_)
        meets_gate = (ic_h2 >= 2.0 and ic_q1 >= 6.0 and sn_h2 >= 3.5)
        trial.set_user_attr("meets_gate", meets_gate)
        print(f"  trial {trial.number}: H2 IC={ic_h2:+.2f} Q1 IC={ic_q1:+.2f} S_H2={sn_h2:+.2f} S_Q1={sn_q1:+.2f} | composite={composite:+.2f} gate={meets_gate} ({train_time:.0f}s)",
              flush=True)
        del model, preds, pred_df, train_fit, val, result; gc.collect()
        return composite

    return objective


def main():
    t_total = time.time()
    print(f"=== v33 task 3: 7y HARD_TECH paradigm 1 Bayesian sweep ({N_TRIALS} trials) ===\n")

    panel = load_panel_filtered()
    print(f"\n[panel] final: {len(panel):,} rows × {panel.shape[1]} cols")
    base_cols = [c for c in panel.columns if c not in ("ts_code", "trade_date")]
    label_df = load_labels_7y()

    print("\n[setup] realized + dyn-exit ...")
    realized, exits = compute_realized_and_exits()

    print(f"\n[optuna] starting Bayesian sweep ({N_TRIALS} trials) ...")
    study = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=SEED),
                                  study_name="hard_tech_7y_t5")
    objective_fn = objective_factory(panel, label_df, realized, exits, base_cols)
    study.optimize(objective_fn, n_trials=N_TRIALS, gc_after_trial=True)

    best = study.best_trial
    print(f"\n=== best trial: #{best.number} composite={best.value:+.2f} ===")
    print(f"  H2 IC={best.user_attrs['h2_ic']:+.2f}% Q1 IC={best.user_attrs['q1_ic']:+.2f}%")
    print(f"  Sharpe H2={best.user_attrs['sharpe_h2']:+.2f} Q1={best.user_attrs['sharpe_q1']:+.2f}")
    print(f"  Meets gate: {best.user_attrs['meets_gate']}")
    print(f"  Params: {best.params}")

    # Save full study
    all_trials = []
    for t in study.trials:
        all_trials.append({
            "trial": t.number, "value": t.value, "params": t.params,
            **{k: v for k, v in t.user_attrs.items()},
        })
    n_gate_pass = sum(1 for t in all_trials if t.get("meets_gate"))

    Path("data/kronos/outputs/matrix_v33_hard_tech_7y_bayesian_results.json").write_text(json.dumps({
        "task": "v33 task 3 — 7y HARD_TECH paradigm 1 Bayesian sweep",
        "config": {"n_trials": N_TRIALS, "horizon": HORIZON, "universe": UNIVERSE,
                   "train_window": [str(TRAIN_START_7Y), str(TRAIN_END)],
                   "sweep_params": ["min_child_samples", "feature_fraction", "num_leaves", "learning_rate"],
                   "gate": "H2 IC ≥ 2.0 AND Q1 IC ≥ 6.0 AND Sharpe_H2 ≥ 3.5"},
        "best_trial": {"trial": best.number, "value": best.value, "params": best.params,
                       **{k: v for k, v in best.user_attrs.items()}},
        "n_gate_pass": n_gate_pass,
        "all_trials": all_trials,
        "total_time_s": time.time() - t_total,
    }, indent=2, default=str))
    print(f"\n[done] {N_TRIALS} trials in {time.time()-t_total:.0f}s, {n_gate_pass} gate-pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
