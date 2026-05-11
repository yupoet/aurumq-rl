"""Build long-panel inference bundles for OSS handoff:
  - sl_path1_long_inference_bundle/   (9 LGB on raw long panel)
  - sl_path5_long_inference_bundle/   (meta + regime, long bases)
  - sl_hybrid_inference_bundle/       (Path 1 long + Path 4 short, no retrain)
  - sl_path4_raw_long_inference_bundle/ (proves rank-z hypothesis; reference)
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
        p = run_root / name / "predictions.parquet"
        df = pl.read_parquet(p).rename({"score": f"score_{name}"})
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


def build_path1_long(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    bundle = Path("data/p3_4070_long")
    runs_root = Path("runs/sl_path1_long")
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
    # feature_cols from raw panel
    cols = [c for c in pl.read_parquet_schema(bundle / "feature_panel_v3_344.parquet") if c not in ("ts_code","trade_date","y")]
    (out_dir / "feature_cols.json").write_text(json.dumps(cols, indent=2, ensure_ascii=False))
    manifest = {
        "generated_at": dt.datetime.now().isoformat(),
        "path": "sl_path1_long",
        "feature_panel_input": "feature_panel_v3_344.parquet (RAW, NO rank-z)",
        "feature_panel_preprocessing": "NONE — raw paris panel directly",
        "n_features": len(cols),
        "n_models": len(chosen),
        "model_type": "LightGBM regression_l2",
        "ensemble_strategy": "seed_mean of 9 chosen seeds across top-3 configs",
        "chosen_runs": chosen,
        "model_shas": shas,
        "calibration": "isotonic on H1",
        "train_window": "2018-01-02 to 2024-12-04 (7y) — see ablation: 5y plateau",
        "ensemble_metrics": {
            "H1_calibrated_primary": ens_meta["ensemble_calibrated_H1"]["primary_mean_top50_proximity_excess"],
            "H2_calibrated_primary": ens_meta["ensemble_calibrated_H2"]["primary_mean_top50_proximity_excess"],
        },
        "notes": (
            "Path 1 LONG (raw input) — the SCIENTIFIC winner of overnight sweep. "
            "Raw input + 7y train window outperforms Path 1 short (raw) by +5.97 bps on H1, "
            "+5.92 bps on H2. Ablation shows 5y window (2020-2024) gives same accuracy "
            "as 7y at 30% less compute. For production: consider 2020-2024 train window."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    print(f"path1_long: wrote {len(chosen)} models, isotonic, feature_cols ({len(cols)} cols)")


def build_path5_long(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root = Path("runs/sl_regime_stack_long")
    ens_meta = json.loads((runs_root / "ensemble.json").read_text())
    shutil.copy(runs_root / "meta_lgb_model.txt", out_dir / "meta_lgb_model.txt")
    shutil.copy(runs_root / "meta_isotonic.pkl", out_dir / "meta_isotonic.pkl")
    shutil.copy("scripts/p3/path5_regime_features.py", out_dir / "regime_features.py")
    manifest = {
        "generated_at": dt.datetime.now().isoformat(),
        "path": "sl_path5_long (regime stacking on LONG-panel bases)",
        "model_type": "LightGBM meta-learner stacking 3 long-panel bases + 11 regime features",
        "stacking_inputs": {
            "base_paths": ens_meta.get("paths_used"),
            "regime_features_source": "compute via regime_features.py",
            "base_panels": {
                "sl_path1_long": "raw feature_panel_v3_344 (NO rank-z), 7y train",
                "sl_path4_long": "rank-z + pruned 226-col panel, 7y train",
                "sl_path2_long": "rank-z 226-col + CatBoost+XGB mix, 7y train",
            },
        },
        "feature_cols": ens_meta["feature_cols"],
        "n_features": ens_meta["n_features"],
        "feature_breakdown": {
            "base_path_predictions": [c for c in ens_meta["feature_cols"] if c.startswith("path_") and "__x__" not in c],
            "regime_features": [
                "market_pct", "market_pct_t_minus_1", "market_pct_t_minus_5",
                "univ_vol_20d", "univ_autocorr_20d", "univ_skew_20d",
                "cs_dispersion", "cs_skew_daily", "top_bottom_spread",
                "sector_dispersion", "n_stocks_daily",
            ],
            "interaction_features": [c for c in ens_meta["feature_cols"] if "__x__" in c],
        },
        "chosen_per_path": ens_meta["chosen_per_path"],
        "meta_params": ens_meta["meta_params"],
        "ensemble_metrics": {
            "stacking_calibrated_H1": ens_meta["stacking_calibrated_H1"]["primary_mean_top50_proximity_excess"],
            "stacking_calibrated_H2": ens_meta["stacking_calibrated_H2"]["primary_mean_top50_proximity_excess"],
        },
        "notes": (
            "PATH 5 LONG = the OVERNIGHT WINNER. H1 +0.02882 H2 +0.03113 (+5-9 bps over Path 4 short prod). "
            "Requires running 3 base path inferences (path1_long, path4_long=Path D, path2_long), then "
            "feeding their scores + regime features into the meta. See INFER.md for end-to-end recipe."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    print("path5_long: wrote meta + regime_features.py + manifest")


def build_hybrid(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root = Path("runs/sl_hybrid_p1long_p4short")
    shutil.copy(runs_root / "predictions.parquet", out_dir / "predictions_sample.parquet")
    ens = json.loads((runs_root / "ensemble.json").read_text())
    manifest = {
        "generated_at": dt.datetime.now().isoformat(),
        "path": "sl_hybrid_p1long_p4short",
        "model_type": "Simple 50/50 average of Path 1 long + Path 4 short ensembles — NO RETRAIN",
        "method": "score_hybrid = (score_path1_long_calibrated + score_path4_short_calibrated) / 2",
        "H1_primary": ens["H1_primary"],
        "H2_primary": ens["H2_primary"],
        "requires": [
            "sl_path1_long_inference_bundle (one of overnight bundle)",
            "sl_path4_inference_bundle (already in production from earlier handoff)",
        ],
        "notes": (
            "Simplest production upgrade: run both bundles' inference, average their calibrated scores, "
            "then apply Strategy D top-50 sizing. No new models to train or maintain. "
            "Performance ~equal to Path 5 long stacking but vastly simpler ops."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    print("hybrid: wrote manifest + sample preds")


def build_path4_raw_long(out_dir):
    """Reference bundle — proves the rank-z hypothesis. Same numbers as Path 1 long."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    bundle = Path("data/p3_4070_long")
    runs_root = Path("runs/sl_path4_raw_long")
    ens_meta = json.loads((runs_root / "ensemble.json").read_text())
    chosen = ens_meta["chosen_runs"]
    shas = {}
    for n in chosen:
        src = runs_root / n / "lgb_model.txt"
        if not src.exists():
            continue
        dst = out_dir / "models" / f"lgb_model_{n}.txt"
        shutil.copy(src, dst)
        shas[n] = _sha16(dst)
    ens = _ensemble_predict(runs_root, chosen)
    iso = _fit_iso(bundle, ens)
    pickle.dump(iso, open(out_dir / "isotonic.pkl", "wb"))
    cols = [c for c in pl.read_parquet_schema(bundle / "feature_panel_v3_344.parquet") if c not in ("ts_code","trade_date","y")]
    (out_dir / "feature_cols.json").write_text(json.dumps(cols, indent=2, ensure_ascii=False))
    manifest = {
        "generated_at": dt.datetime.now().isoformat(),
        "path": "sl_path4_raw_long (rank-z hypothesis test)",
        "feature_panel_input": "feature_panel_v3_344_raw (NO rank-z)",
        "n_features": len(cols),
        "n_models": len(chosen),
        "ensemble_metrics": {
            "H1_calibrated_primary": ens_meta["ensemble_calibrated_H1"]["primary_mean_top50_proximity_excess"],
            "H2_calibrated_primary": ens_meta["ensemble_calibrated_H2"]["primary_mean_top50_proximity_excess"],
        },
        "chosen_runs": chosen,
        "model_shas": shas,
        "notes": (
            "REFERENCE only — proves the rank-z hypothesis. Runs Path 4's hyperparam grid (nl{31,63}) "
            "on RAW long panel. Result is IDENTICAL to Path 1 long, confirming that the +5-6 bps "
            "long-panel gain is from the RAW input pipeline, NOT the hyperparam configuration. "
            "For PRODUCTION, use sl_path1_long_inference_bundle (same models, same results)."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    print(f"path4_raw_long: wrote {len(chosen)} models + manifest")


def main():
    out_root = Path("runs/sl_long_panel_bundles")
    out_root.mkdir(parents=True, exist_ok=True)
    build_path1_long(out_root / "sl_path1_long_inference_bundle")
    build_path5_long(out_root / "sl_path5_long_inference_bundle")
    build_hybrid(out_root / "sl_hybrid_inference_bundle")
    build_path4_raw_long(out_root / "sl_path4_raw_long_inference_bundle")
    print(f"\nALL bundles written to {out_root}")


if __name__ == "__main__":
    main()
