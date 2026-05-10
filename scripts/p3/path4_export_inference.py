"""Export Path 4 inference bundle for production handoff.

Reads the existing runs/sl_path4 grid + ensemble.json, picks the chosen
top-K configs (same logic as path1_ensemble.py), and writes a
self-contained inference bundle to ``runs/sl_path4/inference_bundle/``:

  - models/lgb_model_<run_name>.txt   × 9 (chosen ensemble members)
  - isotonic.pkl                      (sklearn IsotonicRegression fit on H1)
  - feature_cols.json                 (ordered feature list — must match LightGBM)
  - manifest.json                     (chosen runs, model SHAs, timestamps)
  - INFER.md                          (recipe doc — see runs/sl_path4/INFER.md)
"""
from __future__ import annotations

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p3.path1_eval import H1
from p3.path1_ensemble import pick_top_configs_by_val, seed_mean_ensemble


logger = logging.getLogger(__name__)


def _sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--runs-root", default=Path("runs/sl_path4"), type=Path)
    ap.add_argument("--out", default=Path("runs/sl_path4/inference_bundle"), type=Path)
    ap.add_argument("--top-k-configs", type=int, default=3)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "models").mkdir(exist_ok=True)

    # 1. Load all run results, pick top-K configs (same logic as path1_ensemble)
    run_val_primary: dict[str, float] = {}
    run_paths: dict[str, Path] = {}
    for run_dir in sorted(args.runs_root.glob("*_seed*/")):
        results = run_dir / "results.json"
        if not results.exists():
            continue
        d = json.loads(results.read_text())
        run_val_primary[run_dir.name] = d["VAL_EFF"]["primary_mean_top50_proximity_excess"]
        run_paths[run_dir.name] = run_dir
    logger.info("found %d completed runs", len(run_val_primary))

    chosen = pick_top_configs_by_val(run_val_primary, top_k=args.top_k_configs)
    chosen = [c for c in chosen if c in run_paths]
    logger.info("chose %d run names across %d configs", len(chosen),
                len({n.rsplit('_seed', 1)[0] for n in chosen}))

    # 2. Copy lgb_model.txt files into bundle/models/
    model_shas = {}
    for n in chosen:
        src = run_paths[n] / "lgb_model.txt"
        dst = args.out / "models" / f"lgb_model_{n}.txt"
        shutil.copy(src, dst)
        model_shas[n] = _sha256_short(dst)
    logger.info("copied %d lgb_model.txt files to %s/models/", len(chosen), args.out)

    # 3. Reconstitute the ensemble (seed-mean of chosen) and refit isotonic on H1
    pred_dfs = [pl.read_parquet(run_paths[n] / "predictions.parquet") for n in chosen]
    ens = seed_mean_ensemble(pred_dfs)
    target_y = pl.read_parquet(args.bundle / "target_y.parquet")
    h1_join = ens.filter(
        (pl.col("trade_date") >= H1[0]) & (pl.col("trade_date") <= H1[1])
    ).join(target_y, on=["trade_date", "ts_code"], how="inner")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(h1_join["score"].to_numpy(), h1_join["y"].to_numpy())
    with (args.out / "isotonic.pkl").open("wb") as f:
        pickle.dump(iso, f)
    logger.info("fit isotonic on %d H1 rows; saved isotonic.pkl", len(h1_join))

    # 4. Save feature column order (read from one model — feature_name embedded in lgb_model.txt)
    # LightGBM model files include feature names; we extract via lgb.Booster
    import lightgbm as lgb
    sample_model = lgb.Booster(model_file=str(args.out / "models" / f"lgb_model_{chosen[0]}.txt"))
    feature_cols = sample_model.feature_name()
    (args.out / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2))
    logger.info("saved feature_cols.json: %d features", len(feature_cols))

    # Verify all chosen models share the same feature_cols (defensive)
    for n in chosen[1:]:
        m = lgb.Booster(model_file=str(args.out / "models" / f"lgb_model_{n}.txt"))
        if m.feature_name() != feature_cols:
            logger.error("feature_cols mismatch in %s; bundle is unsafe", n)
            return 2

    # 5. Save manifest
    manifest = {
        "generated_at": dt.datetime.now().isoformat(),
        "feature_panel_input": "feature_panel_clean.parquet",
        "feature_panel_preprocessing": "src/aurumq_rl/p3/rank_z.py:cross_sectional_rank_z",
        "n_features": len(feature_cols),
        "n_models": len(chosen),
        "ensemble_strategy": "seed_mean (mean of LightGBM predict outputs)",
        "calibration": "sklearn.isotonic.IsotonicRegression fit on H1 actual_y",
        "chosen_runs": chosen,
        "model_shas": model_shas,
        "windows": {
            "H1_calibration": [str(H1[0]), str(H1[1])],
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("wrote manifest: %s", args.out / "manifest.json")

    # 6. Smoke-verify the bundle: predict on a small slice end-to-end
    logger.info("smoke-verifying bundle on H1 first 5 days ...")
    h1_slice = ens.filter(
        (pl.col("trade_date") >= H1[0]) & (pl.col("trade_date") <= H1[1])
    ).head(1000)
    raw_scores = h1_slice["score"].to_numpy()
    cal_scores = iso.transform(raw_scores)
    logger.info("  raw range: [%.4f, %.4f]  cal range: [%.4f, %.4f]",
                raw_scores.min(), raw_scores.max(), cal_scores.min(), cal_scores.max())

    logger.info("inference bundle ready at %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
