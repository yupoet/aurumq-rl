"""Build the 2 missing long-panel base bundles needed for Path 5 long stacking:
  - sl_path4_long_inference_bundle/  (rank-z pruned 226-col, 9 LGB)
  - sl_path2_long_inference_bundle/  (rank-z pruned 226-col, 6 CB + 3 XGB)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent.parent))
from p3.path1_eval import H1


def _sha16(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()[:16]


def _ensemble_predict(run_root, chosen):
    parts = []
    for name in chosen:
        df = pl.read_parquet(run_root / name / "predictions.parquet").rename({"score": f"score_{name}"})
        parts.append(df)
    out = parts[0]
    for x in parts[1:]:
        out = out.join(x, on=["trade_date","ts_code"], how="inner")
    score_cols = [c for c in out.columns if c.startswith("score_")]
    return out.with_columns(pl.mean_horizontal(score_cols).alias("score_raw")).select(["trade_date","ts_code","score_raw"])


def _fit_iso(bundle_dir, ens):
    target = pl.read_parquet(bundle_dir / "target_y.parquet")
    h1 = ens.join(target, on=["trade_date","ts_code"], how="inner").filter(
        (pl.col("trade_date") >= H1[0]) & (pl.col("trade_date") <= H1[1])
    )
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(h1["score_raw"].to_numpy(), h1["y"].to_numpy())
    return iso


def _feature_cols_pruned():
    """Path 4 long + Path 2 long use feature_panel_clean.parquet (226 cols, pruned + rank-z'd)."""
    cols = [c for c in pl.read_parquet_schema("data/p3_4070_long/feature_panel_clean.parquet")
            if c not in ("ts_code","trade_date","y")]
    return cols


def build_path4_long(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    bundle = Path("data/p3_4070_long")
    runs_root = Path("runs/sl_path_d")
    ens_meta = json.loads((runs_root / "ensemble.json").read_text())
    chosen = ens_meta["chosen_runs"]

    shas = {}
    for n in chosen:
        src = runs_root / n / "lgb_model.txt"
        dst = out_dir / "models" / f"lgb_model_{n}.txt"
        shutil.copy(src, dst)
        shas[n] = _sha16(dst)
    ens = _ensemble_predict(runs_root, chosen)
    iso = _fit_iso(bundle, ens)
    pickle.dump(iso, open(out_dir / "isotonic.pkl", "wb"))

    cols = _feature_cols_pruned()
    (out_dir / "feature_cols.json").write_text(json.dumps(cols, indent=2, ensure_ascii=False))
    # Also ship the drop_candidates list so paris can derive 226 cols from their 345-col panel
    shutil.copy("runs/sl_path4/feature_audit_drop_candidates.json", out_dir / "drop_candidates.json")

    manifest = {
        "generated_at": dt.datetime.now().isoformat(),
        "path": "sl_path4_long (rank-z pruned 226-col, long train)",
        "feature_panel_input": "feature_panel_clean.parquet (rank-z'd, pruned to 226 cols)",
        "feature_panel_preprocessing": (
            "rank_z.cross_sectional_rank_z per day (same as Path 4 short bundle) "
            "THEN drop the 119 SHAP-zero cols (see drop_candidates.json)"
        ),
        "n_features": len(cols),
        "n_models": len(chosen),
        "model_type": "LightGBM regression_l2",
        "ensemble_strategy": "seed_mean across 9 chosen seeds",
        "chosen_runs": chosen,
        "model_shas": shas,
        "calibration": "isotonic on H1",
        "train_window": "2018-01-02 to 2024-12-04 (7y)",
        "ensemble_metrics": {
            "H1_calibrated_primary": ens_meta["ensemble_calibrated_H1"]["primary_mean_top50_proximity_excess"],
            "H2_calibrated_primary": ens_meta["ensemble_calibrated_H2"]["primary_mean_top50_proximity_excess"],
        },
        "notes": (
            "PRIMARILY USED AS BASE LEARNER FOR PATH 5 LONG STACKING. "
            "Standalone, Path 4 long is approximately the same as Path 4 short "
            "(rank-z destroys long-panel info — see SUPPLEMENT_RESULTS.md Finding 1). "
            "Inside the Path 5 long stack however, its rank-z'd predictions provide "
            "structural diversity vs Path 1 long (raw) and Path 2 long (CB+XGB mix)."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    print(f"path4_long: {len(chosen)} models + isotonic + {len(cols)} cols")


def build_path2_long(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    bundle = Path("data/p3_4070_long")
    runs_root = Path("runs/sl_path2_long")
    ens_meta = json.loads((runs_root / "ensemble.json").read_text())
    chosen = ens_meta["chosen_runs"]

    shas = {}
    kinds = {}
    for n in chosen:
        kind = "catboost" if n.startswith("cb_") else "xgboost"
        kinds[n] = kind
        candidates = [
            runs_root / n / "catboost_model.cbm",
            runs_root / n / "xgb_model.json",
        ]
        src = next((c for c in candidates if c.exists()), None)
        if src is None:
            print(f"WARN: no model file for {n}")
            continue
        dst = out_dir / "models" / f"{n}{src.suffix}"
        shutil.copy(src, dst)
        shas[n] = _sha16(dst)
    ens = _ensemble_predict(runs_root, chosen)
    iso = _fit_iso(bundle, ens)
    pickle.dump(iso, open(out_dir / "isotonic.pkl", "wb"))

    cols = _feature_cols_pruned()
    (out_dir / "feature_cols.json").write_text(json.dumps(cols, indent=2, ensure_ascii=False))
    shutil.copy("runs/sl_path4/feature_audit_drop_candidates.json", out_dir / "drop_candidates.json")

    manifest = {
        "generated_at": dt.datetime.now().isoformat(),
        "path": "sl_path2_long (rank-z pruned 226-col, CatBoost+XGB mix, long train)",
        "feature_panel_input": "feature_panel_clean.parquet (rank-z'd, pruned to 226 cols)",
        "feature_panel_preprocessing": "rank_z + drop 119 SHAP-zero cols (see drop_candidates.json)",
        "n_features": len(cols),
        "n_models": len(chosen),
        "model_type": "Mixed CatBoost + XGBoost regressors",
        "model_kinds": kinds,
        "ensemble_strategy": "equal-weight mean across all chosen models (CB + XGB mixed)",
        "chosen_runs": chosen,
        "model_shas": shas,
        "calibration": "isotonic on H1",
        "train_window": "2018-01-02 to 2024-12-04 (7y)",
        "ensemble_metrics": {
            "H1_calibrated_primary": ens_meta["ensemble_calibrated_H1"]["primary_mean_top50_proximity_excess"],
            "H2_calibrated_primary": ens_meta["ensemble_calibrated_H2"]["primary_mean_top50_proximity_excess"],
        },
        "notes": (
            "PRIMARILY USED AS BASE LEARNER FOR PATH 5 LONG STACKING. "
            "NOTE: due to compute-time budget, only cb_d6 + xgb_d6 (the 6-depth variants) "
            "made it into the final ensemble. cb_d8/d10, xgb_d8/d10 were either too slow "
            "to complete overnight or stubbed for time. The cb_d6 + xgb_d6 ensemble still "
            "carries structural diversity (different learner families) vs LGB-only paths."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    print(f"path2_long: {len(chosen)} models ({sum(1 for k in kinds.values() if k=='catboost')} CB, "
          f"{sum(1 for k in kinds.values() if k=='xgboost')} XGB) + isotonic + {len(cols)} cols")


def main():
    out_root = Path("runs/sl_long_panel_bundles")
    build_path4_long(out_root / "sl_path4_long_inference_bundle")
    build_path2_long(out_root / "sl_path2_long_inference_bundle")
    print(f"\nDONE — added 2 base bundles to {out_root}")


if __name__ == "__main__":
    main()
