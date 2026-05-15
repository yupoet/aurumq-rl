"""Build growth_method_c inference bundle for paris promote-after-shadow.

Per paris REPLY v21 §3:
  oss://ledashi-oss-sgp/fromsz/handoffs/2026-05-15-growth-method-c-bundle/
  ├── meta_lgb_model.txt
  ├── feature_cols.json
  ├── isotonic.pkl
  ├── manifest.json
  └── INFER.md

Method C = paris growth labels triple_barrier t20 binary, GROWTH_BOARDS universe (2253).
Hyperparam: paris production wave_binary (lgb_params_wave_binary.json).

Pipeline:
  1. Load 226-col panel filtered to GROWTH_BOARDS
  2. Load paris method-C t20 labels 2022-2026 concat
  3. Train LGBMClassifier with paris wave_binary params + early stopping on val tail
  4. Save model + features + isotonic on H1_2025 + manifest + INFER
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

PANEL = "data/p3_4070_long/feature_panel_v3_344_pruned.parquet"
LABEL_DIR = Path("D:/dev/aurumq-handoffs/inbox/2026-05-15-paris-growth-labels")
UNIVERSE = Path("data/universes/GROWTH_BOARDS_membership.parquet")
OUT = Path("D:/dev/aurumq-rl/runs/growth_method_c_bundle")
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_START = pd.Timestamp("2022-01-01").date()
TRAIN_END   = pd.Timestamp("2024-12-31").date()
H1_START    = pd.Timestamp("2025-01-01").date()
H1_END      = pd.Timestamp("2025-06-30").date()
YEARS = [2022, 2023, 2024, 2025, 2026]

LGB_PARAMS = dict(
    objective="binary", metric="average_precision",
    boosting_type="gbdt",
    learning_rate=0.05, num_leaves=63,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    min_data_in_leaf=200,
    n_estimators=500,
    verbose=-1, num_threads=-1, random_state=42,
)
EARLY_STOPPING = 50
VAL_FRAC = 0.15


def _dt(df):
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def main() -> int:
    t0 = time.time()
    print("[load] GROWTH universe ...")
    growth_set = frozenset(pd.read_parquet(UNIVERSE)["stock_code"].tolist())
    print(f"  {len(growth_set)} stocks")

    print("[load] panel + filter GROWTH ...")
    panel = _dt(pd.read_parquet(PANEL))
    panel = panel[panel["ts_code"].isin(growth_set)].reset_index(drop=True)
    base_cols = [c for c in panel.columns if c not in ("ts_code", "trade_date")]
    print(f"  panel GROWTH: {len(panel):,} rows × {len(base_cols)} feats")

    print("[load] paris method-C t20 labels ...")
    dfs = []
    for year in YEARS:
        p = LABEL_DIR / f"labels_C_t20_growth_year={year}.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p))
    labels = _dt(pd.concat(dfs, ignore_index=True))
    print(f"  labels: {len(labels):,} rows, pos rate {labels['y'].mean()*100:.1f}%")

    print("[merge] panel + labels ...")
    joined = panel.merge(labels, on=["ts_code", "trade_date"], how="inner")
    train_full = joined[(joined["trade_date"] >= TRAIN_START) & (joined["trade_date"] <= TRAIN_END)].copy()
    train_full = train_full.sort_values("trade_date").reset_index(drop=True)
    n_val = int(len(train_full) * VAL_FRAC)
    tr = train_full.iloc[:-n_val]
    va = train_full.iloc[-n_val:]
    print(f"  train rows: {len(tr):,}, val rows: {len(va):,}, pos rate {tr['y'].mean()*100:.1f}%/{va['y'].mean()*100:.1f}%")

    print("[train] LGBMClassifier ...")
    t = time.time()
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        tr[base_cols], tr["y"],
        eval_set=[(va[base_cols], va["y"])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    n_iter = model.best_iteration_
    print(f"  train: {time.time()-t:.0f}s, best_iter={n_iter}")
    del tr, va, train_full

    print("[predict] on full panel ...")
    t = time.time()
    preds = model.predict_proba(panel[base_cols])[:, 1].astype(np.float32)
    print(f"  predict: {time.time()-t:.0f}s ({len(preds):,} rows)")
    pred_df = panel[["ts_code", "trade_date"]].copy()
    pred_df["score_raw"] = preds

    print("[isotonic] fit on H1_2025 with raw label y ...")
    h1_panel = pred_df.merge(labels, on=["ts_code", "trade_date"], how="inner")
    h1_panel = h1_panel[(h1_panel["trade_date"] >= H1_START) & (h1_panel["trade_date"] <= H1_END)]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(h1_panel["score_raw"].to_numpy(), h1_panel["y"].to_numpy())
    pred_df["score_calibrated"] = iso.transform(preds).astype(np.float32)
    print(f"  iso fit on {len(h1_panel):,} H1 rows")

    print(f"[save] writing bundle to {OUT} ...")
    model.booster_.save_model(str(OUT / "meta_lgb_model.txt"))
    with (OUT / "isotonic.pkl").open("wb") as f:
        pickle.dump(iso, f)
    (OUT / "feature_cols.json").write_text(json.dumps(base_cols, indent=2))

    # Schema hash
    schema_hash = hashlib.sha256(",".join(base_cols).encode()).hexdigest()[:16]

    # Sample predictions parquet (last 30 trade_dates for sanity)
    sample = pred_df.tail(50000)
    sample.to_parquet(OUT / "sample_predictions_tail50k.parquet", compression="zstd")

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "path": "growth_method_c",
        "model_type": "Single LightGBM classifier (binary) on paris growth method-C triple_barrier t20 label, GROWTH_BOARDS universe",
        "label_source": "paris 2026-05-15-paris-growth-labels (method C / triple_barrier / t20 horizon)",
        "panel_source": "ledashi 226-col pruned panel (feature_panel_v3_344_pruned.parquet)",
        "universe": "GROWTH_BOARDS (2253 stocks, 创业 300xxx + 科创 688xxx + 北交 8xxxxx)",
        "n_features": len(base_cols),
        "feature_cols_first5": base_cols[:5],
        "feature_cols_last5": base_cols[-5:],
        "schema_hash": schema_hash,
        "train_window": [str(TRAIN_START), str(TRAIN_END)],
        "val_window_for_isotonic": [str(H1_START), str(H1_END)],
        "lgb_params": LGB_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING,
        "best_iteration": int(n_iter),
        "calibration": {
            "method": "IsotonicRegression (sklearn)",
            "fit_on": "H1_2025 raw score → realized binary label y",
            "fit_rows": int(len(h1_panel)),
        },
        "ledashi_validation_metrics": {
            "source": "growth_v4_results.json v3_method_C_growth (binary classifier with wave_binary params)",
            "H2_2025_fwd5_IC": 0.0116,
            "H2_2025_fwd20_IC": 0.0263,
            "Q1_2026_fwd20_IC": 0.0534,
            "Q1_2026_fwd20_IC_pct": "+5.34%",
            "delta_vs_old_hybrid_hyperparam": "+1.06 bps (paris estimated +0.3~+1 bps, this is upper-bound)",
        },
        "inference_recipe": {
            "step1": "load model: lgb.Booster(model_file='meta_lgb_model.txt')",
            "step2": "load isotonic: pickle.load('isotonic.pkl')",
            "step3": "get panel[GROWTH_universe][base_cols] for today's date (226 cols, exact order in feature_cols.json)",
            "step4": "score_raw = model.predict(X)[:, 1] if shape[1]==2 else model.predict(X)",
            "step5": "score_calibrated = isotonic.transform(score_raw)",
        },
        "shadow_plan_per_paris_v21": {
            "shadow_window": "5/16 ~ 5/20 (3-5 working days)",
            "shadow_table": "wave_scores_daily (no email exposure)",
            "promote_threshold": "rows > 0 + score distribution stable + IC fwd20 > +0.5%",
            "promote_target": "17-model bench (paris _PATH_DISPLAY_INFO adds growth_method_c row)",
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    infer_md = f"""# growth_method_c — Inference Recipe

**Bundle**: `growth_method_c/`
**Date**: 2026-05-15 (paris promote-after-shadow workflow)
**Variant**: Single LightGBM classifier on paris growth method-C triple_barrier t20 label

## Headline numbers (ledashi validation, growth_v4)

| Window | IC fwd5 | IC fwd20 |
|---|---|---|
| H2_2025 | +0.0116 | +0.0263 |
| Q1_2026 | - | **+0.0534** (best Q1 across 4 methods) |

vs OLD GROWTH (matrix v2/v3 era with hybrid regression hyperparam): +1.06 bps gain on Q1 fwd20.

## Architecture

```
ledashi 226-col pruned panel × GROWTH_BOARDS (2253 stocks)
  ↓
LightGBM binary classifier (paris wave_binary params: num_leaves=63, n=500+early_stop,
                            objective=binary, metric=average_precision)
  ↓
raw probability score (0-1)
  ↓
isotonic calibration (fit on H1_2025 → realized binary label y)
  ↓
score_calibrated (production)
```

## Required input panel

GROWTH_BOARDS universe (2253 stocks: 创业 300xxx + 科创 688xxx + 北交 8xxxxx).
226 features from ledashi `feature_panel_v3_344_pruned.parquet`, EXACT order in `feature_cols.json`.

## Step-by-step inference

```python
import json, pickle
import polars as pl
import lightgbm as lgb
import numpy as np
from pathlib import Path

BUNDLE = Path("growth_method_c_bundle")
feature_cols = json.loads((BUNDLE / "feature_cols.json").read_text())
manifest = json.loads((BUNDLE / "manifest.json").read_text())
model = lgb.Booster(model_file=str(BUNDLE / "meta_lgb_model.txt"))
with (BUNDLE / "isotonic.pkl").open("rb") as f:
    iso = pickle.load(f)

# 1. Get GROWTH universe for today
growth_set = pl.read_parquet("data/universes/GROWTH_BOARDS_membership.parquet")["stock_code"].to_list()

# 2. Get today's panel restricted to GROWTH stocks
panel = pl.read_parquet("data/p3_4070_long/feature_panel_v3_344_pruned.parquet")
today_panel = panel.filter(
    (pl.col("trade_date") == today) & (pl.col("ts_code").is_in(growth_set))
)

# 3. Predict
X = today_panel.select(feature_cols).to_numpy().astype(np.float32)
score_raw = model.predict(X).astype(np.float32)   # if shape[1]==2: take [:,1]
score_calibrated = iso.transform(score_raw).astype(np.float32)

# 4. Top-50 picker (production)
df = today_panel.select(["trade_date", "ts_code"]).with_columns(
    pl.Series("score_calibrated", score_calibrated)
)
top50 = df.sort("score_calibrated", descending=True).head(50)
```

## Feature schema ({len(base_cols)} cols, exact order)

See `feature_cols.json`. Same columns as `feature_panel_v3_344_pruned.parquet` (ledashi production pruning).

## Notes

- `meta_lgb_model.txt` is single LGB (not ensemble); paris production wave_binary params
- isotonic fit on H1_2025; not refit on H2 (avoids leak)
- shadow mode: write `wave_scores_daily` table, no email; promote after 3-5 working days
- paris cron: 19:03 (off-peak vs 18:56/57 path5/path5_long)
"""
    (OUT / "INFER.md").write_text(infer_md, encoding="utf-8")

    print(f"\n[done] bundle at {OUT}")
    print(f"  schema_hash={schema_hash}, best_iter={n_iter}, total {time.time()-t0:.0f}s")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}: {f.stat().st_size/1e3:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
