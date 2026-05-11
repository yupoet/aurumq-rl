"""Build inference bundles for Path 1 / 2 / 5 / 6 to ship to paris.

For each path:
  - rebuild ensemble preds from chosen_runs
  - fit isotonic on H1 actual_y -> save isotonic.pkl
  - copy model files into bundle/models/
  - write manifest.json + feature_cols.json + INFER.md
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent.parent))
from p3.path1_eval import H1


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _sha16(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _ensemble_predict_lgb(run_root: Path, chosen: list[str]) -> pl.DataFrame:
    parts = []
    for name in chosen:
        p = run_root / name / "predictions.parquet"
        df = pl.read_parquet(p).rename({"score": f"score_{name}"})
        parts.append(df)
    out = parts[0]
    for x in parts[1:]:
        out = out.join(x, on=["trade_date", "ts_code"], how="inner")
    score_cols = [c for c in out.columns if c.startswith("score_")]
    return out.with_columns(
        pl.mean_horizontal(score_cols).alias("score_raw")
    ).select(["trade_date", "ts_code", "score_raw"])


def _fit_isotonic_h1(bundle_dir: Path, ens_preds: pl.DataFrame) -> IsotonicRegression:
    target_y = pl.read_parquet(bundle_dir / "target_y.parquet")
    h1 = ens_preds.join(target_y, on=["trade_date", "ts_code"], how="inner").filter(
        (pl.col("trade_date") >= H1[0]) & (pl.col("trade_date") <= H1[1])
    )
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(h1["score_raw"].to_numpy(), h1["y"].to_numpy())
    logger.info("isotonic fit on %d H1 rows", len(h1))
    return iso


def build_bundle_lgb(
    path_name: str,
    run_root: Path,
    bundle_dir: Path,
    feature_panel_input: str,
    feature_panel_preprocessing: str,
    n_features_expected: int,
    notes: str,
    out_dir: Path,
) -> None:
    """Build bundle for path 1/4/6 (pure LGB ensemble)."""
    ens_meta = json.loads((run_root / "ensemble.json").read_text())
    chosen = ens_meta["chosen_runs"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)

    model_shas: dict[str, str] = {}
    for name in chosen:
        src = run_root / name / "lgb_model.txt"
        dst = out_dir / "models" / f"lgb_model_{name}.txt"
        shutil.copy(src, dst)
        model_shas[name] = _sha16(dst)

    ens_preds = _ensemble_predict_lgb(run_root, chosen)
    iso = _fit_isotonic_h1(bundle_dir, ens_preds)
    with (out_dir / "isotonic.pkl").open("wb") as f:
        pickle.dump(iso, f)

    feature_cols = list(pl.read_parquet_schema(bundle_dir / feature_panel_input).keys())
    feature_cols = [c for c in feature_cols if c not in ("ts_code", "trade_date", "y")]
    assert len(feature_cols) == n_features_expected, (
        f"{path_name}: expected {n_features_expected} features, got {len(feature_cols)}"
    )
    (out_dir / "feature_cols.json").write_text(
        json.dumps(feature_cols, indent=2, ensure_ascii=False)
    )

    manifest = {
        "generated_at": dt.datetime.now().isoformat(),
        "path": path_name,
        "feature_panel_input": feature_panel_input,
        "feature_panel_preprocessing": feature_panel_preprocessing,
        "n_features": n_features_expected,
        "n_models": len(chosen),
        "model_type": "LightGBM regression (regression_l2)",
        "ensemble_strategy": "seed_mean (mean of LightGBM predict outputs)",
        "calibration": "sklearn.isotonic.IsotonicRegression fit on H1 actual_y",
        "chosen_runs": chosen,
        "model_shas": model_shas,
        "windows": {"H1_calibration": [str(H1[0]), str(H1[1])]},
        "ensemble_metrics": {
            "H1_calibrated_primary": ens_meta.get("ensemble_calibrated_H1", {}).get(
                "primary_mean_top50_proximity_excess"
            ),
            "H2_calibrated_primary": ens_meta.get("ensemble_calibrated_H2", {}).get(
                "primary_mean_top50_proximity_excess"
            ),
        },
        "notes": notes,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
    )
    logger.info("wrote %s manifest, %d models, isotonic.pkl, feature_cols.json (%d cols)",
                path_name, len(chosen), len(feature_cols))


def build_path2(run_root: Path, bundle_dir: Path, out_dir: Path) -> None:
    """Path 2: 12 mixed CatBoost+XGB models."""
    ens_meta = json.loads((run_root / "ensemble.json").read_text())
    chosen = ens_meta["chosen_runs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)

    model_shas: dict[str, str] = {}
    model_kinds: dict[str, str] = {}
    for name in chosen:
        kind = "catboost" if name.startswith("cb_") else "xgboost"
        # CatBoost saves *.cbm; XGB saves *.json
        candidates = [
            run_root / name / "catboost_model.cbm",
            run_root / name / "xgb_model.json",
            run_root / name / "model.cbm",
            run_root / name / "model.json",
            run_root / name / "model.txt",
        ]
        src = next((c for c in candidates if c.exists()), None)
        if src is None:
            available = list((run_root / name).iterdir())
            raise FileNotFoundError(f"no model file in {run_root / name}: {available}")
        dst = out_dir / "models" / f"{name}{src.suffix}"
        shutil.copy(src, dst)
        model_shas[name] = _sha16(dst)
        model_kinds[name] = kind

    # Use predictions.parquet at run-level for ensemble
    parts = []
    for name in chosen:
        p = run_root / name / "predictions.parquet"
        df = pl.read_parquet(p).rename({"score": f"score_{name}"})
        parts.append(df)
    out = parts[0]
    for x in parts[1:]:
        out = out.join(x, on=["trade_date", "ts_code"], how="inner")
    score_cols = [c for c in out.columns if c.startswith("score_")]
    ens_preds = out.with_columns(
        pl.mean_horizontal(score_cols).alias("score_raw")
    ).select(["trade_date", "ts_code", "score_raw"])

    iso = _fit_isotonic_h1(bundle_dir, ens_preds)
    with (out_dir / "isotonic.pkl").open("wb") as f:
        pickle.dump(iso, f)

    fc_panel = "feature_panel_clean.parquet"  # rank-z'd
    feature_cols = list(pl.read_parquet_schema(bundle_dir / fc_panel).keys())
    feature_cols = [c for c in feature_cols if c not in ("ts_code", "trade_date", "y")]
    (out_dir / "feature_cols.json").write_text(
        json.dumps(feature_cols, indent=2, ensure_ascii=False)
    )

    manifest = {
        "generated_at": dt.datetime.now().isoformat(),
        "path": "sl_path2",
        "feature_panel_input": "feature_panel_clean.parquet",
        "feature_panel_preprocessing": "src/aurumq_rl/p3/rank_z.py:cross_sectional_rank_z (same as path4)",
        "n_features": len(feature_cols),
        "n_models": len(chosen),
        "model_type": "Mixed CatBoost + XGBoost regressors (model multiplicity ensemble)",
        "model_kinds": model_kinds,
        "ensemble_strategy": "seed_mean across all 12 models (catboost + xgb mixed equal-weight)",
        "calibration": "sklearn.isotonic.IsotonicRegression fit on H1 actual_y",
        "chosen_runs": chosen,
        "model_shas": model_shas,
        "windows": {"H1_calibration": [str(H1[0]), str(H1[1])]},
        "ensemble_metrics": {
            "H1_calibrated_primary": ens_meta.get("ensemble_calibrated_H1", {}).get(
                "primary_mean_top50_proximity_excess"
            ),
            "H2_calibrated_primary": ens_meta.get("ensemble_calibrated_H2", {}).get(
                "primary_mean_top50_proximity_excess"
            ),
        },
        "notes": (
            "Path 2 = 'model multiplicity' experiment. Same input as Path 4 "
            "(rank-z'd 345-col panel) but mixes CatBoost (catboost.CatBoostRegressor) "
            "with XGBoost (xgboost.XGBRegressor) for ensemble diversity. Equal-weight "
            "mean of all 12 models' scores."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
    )
    logger.info("wrote path2 manifest, %d models", len(chosen))


def build_path5(run_root: Path, bundle_dir: Path, out_dir: Path) -> None:
    """Path 5: regime stacking — meta-LGB over base path preds + regime features."""
    ens_meta = json.loads((run_root / "ensemble.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy meta artifacts (already saved by training script)
    shutil.copy(run_root / "meta_lgb_model.txt", out_dir / "meta_lgb_model.txt")
    shutil.copy(run_root / "meta_isotonic.pkl", out_dir / "meta_isotonic.pkl")

    # Copy regime feature compute script
    shutil.copy(
        Path("scripts/p3/path5_regime_features.py"),
        out_dir / "regime_features.py",
    )

    manifest = {
        "generated_at": dt.datetime.now().isoformat(),
        "path": "sl_regime_stack",
        "model_type": "LightGBM meta-learner stacking 3 base paths + 11 regime features",
        "stacking_inputs": {
            "base_paths": ens_meta["paths_used"],
            "regime_features_source": "data/p3_4070/regime_features.parquet",
            "regime_features_compute_script": "regime_features.py (use as: python regime_features.py --bundle <DATA_DIR>)",
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
        "meta_best_iteration": ens_meta["meta_best_iteration"],
        "calibration": "sklearn.isotonic.IsotonicRegression on H1 (saved as meta_isotonic.pkl)",
        "ensemble_metrics": {
            "stacking_calibrated_H1": ens_meta["stacking_calibrated_H1"]["primary_mean_top50_proximity_excess"],
            "stacking_calibrated_H2": ens_meta["stacking_calibrated_H2"]["primary_mean_top50_proximity_excess"],
            "stacking_T1_hit_H1": ens_meta["stacking_calibrated_H1"]["top50_T1_hit_rate"],
            "stacking_T1_hit_H2": ens_meta["stacking_calibrated_H2"]["top50_T1_hit_rate"],
        },
        "notes": (
            "Path 5 = regime-aware stacking. Run inference in 4 steps: "
            "(1) get base path predictions for path1 + path4 + path2 (each their own bundle); "
            "(2) compute regime_features for today via regime_features.py; "
            "(3) build the 23-col feature matrix in EXACTLY this column order; "
            "(4) meta_lgb predict -> isotonic transform. Score interpretation identical to path4."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
    )
    logger.info("wrote path5 (regime_stack) bundle")


def main() -> None:
    bundle_dir = Path("data/p3_4070")
    out_root = Path("runs/sl_paths_inference_bundles")
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- Path 1 ----
    build_bundle_lgb(
        path_name="sl_path1",
        run_root=Path("runs/sl_path1"),
        bundle_dir=bundle_dir,
        feature_panel_input="feature_panel_v3_344.parquet",
        feature_panel_preprocessing="NONE — raw features (paris's panel directly, no rank-z)",
        n_features_expected=345,
        notes=(
            "Path 1 = vanilla LightGBM regression on RAW features. "
            "Input panel is paris's feature_panel_v3_344.parquet directly (NO rank-z transform). "
            "feature_cols.json column order matches that panel. "
            "Used the same TRAIN window 2023-01-03 ~ 2024-12-04 as path4."
        ),
        out_dir=out_root / "sl_path1_inference_bundle",
    )

    # ---- Path 6 ----
    build_bundle_lgb(
        path_name="sl_path6",
        run_root=Path("runs/sl_path6"),
        bundle_dir=bundle_dir,
        feature_panel_input="feature_panel_clean_pruned.parquet",
        feature_panel_preprocessing=(
            "rank_z.cross_sectional_rank_z (same as path4) THEN drop 119 SHAP-zero features "
            "(see runs/sl_path4/feature_audit_drop_candidates.json for which 119)"
        ),
        n_features_expected=226,
        notes=(
            "Path 6 = Bayesian opt over Path 4 hyperparam space, on the PRUNED 226-col panel. "
            "Standalone — does NOT chain on Path 4 output. "
            "feature_cols.json gives the exact 226 columns to feed in (rank-z'd). "
            "If you have the 345-col rank-z panel from path4, drop the 119 cols listed in "
            "runs/sl_path4/feature_audit_drop_candidates.json (shipped here as drop_candidates.json) "
            "to get the 226-col input."
        ),
        out_dir=out_root / "sl_path6_inference_bundle",
    )
    # Also copy the drop list into path6 bundle for paris's convenience
    shutil.copy(
        Path("runs/sl_path4/feature_audit_drop_candidates.json"),
        out_root / "sl_path6_inference_bundle" / "drop_candidates.json",
    )

    # ---- Path 2 ----
    build_path2(
        run_root=Path("runs/sl_path2"),
        bundle_dir=bundle_dir,
        out_dir=out_root / "sl_path2_inference_bundle",
    )

    # ---- Path 5 ----
    build_path5(
        run_root=Path("runs/sl_regime_stack"),
        bundle_dir=bundle_dir,
        out_dir=out_root / "sl_path5_inference_bundle",
    )

    logger.info("ALL bundles written to %s", out_root)


if __name__ == "__main__":
    main()
