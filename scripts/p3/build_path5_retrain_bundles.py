"""Build path5 + path5_long inference bundles after leakage-fix retrain.

Reads outputs of:
  - runs/sl_regime_stack_v2/         (path5 short)
  - runs/sl_regime_stack_long_v2/    (path5_long)

Writes:
  - runs/sl_path5_retrain_v2_bundle/
  - runs/sl_path5_long_retrain_v2_bundle/

Each bundle contains: meta_lgb_model.txt, meta_isotonic.pkl, regime_features.py,
manifest.json (new schema_hash + new IC/Sharpe/T1_hit), INFER.md (unchanged).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

RUN_PATH5 = Path("D:/dev/aurumq-rl/runs/sl_regime_stack_v2")
RUN_PATH5_LONG = Path("D:/dev/aurumq-rl/runs/sl_regime_stack_long_v2")
SCRIPTS_DIR = Path("D:/dev/aurumq-rl/scripts/p3")
OUT_PATH5 = Path("D:/dev/aurumq-rl/runs/sl_path5_retrain_v2_bundle")
OUT_PATH5_LONG = Path("D:/dev/aurumq-rl/runs/sl_path5_long_retrain_v2_bundle")
OLD_BUNDLE_LONG = Path("D:/dev/aurumq-rl/runs/sl_long_panel_bundles/sl_path5_long_inference_bundle")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build(run_dir: Path, out_dir: Path, variant: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy model artifacts
    for fname in ("meta_lgb_model.txt", "meta_isotonic.pkl"):
        shutil.copy(run_dir / fname, out_dir / fname)

    # Copy regime_features.py (algorithm reference)
    shutil.copy(SCRIPTS_DIR / "path5_regime_features.py", out_dir / "regime_features.py")

    # Load ensemble.json
    ens = json.loads((run_dir / "ensemble.json").read_text())

    # Schema hash from feature_cols (deterministic)
    feature_cols = ens["feature_cols"]
    schema_hash = hashlib.sha256(",".join(feature_cols).encode()).hexdigest()[:16]

    # Build manifest
    is_long = variant == "path5_long"
    base_paths = ens["paths_used"]

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "path": f"sl_{variant} (regime stacking, leakage-fix v2)",
        "model_type": "LightGBM meta-learner stacking 3 bases + 11 regime features (regime_features rekeyed to T+1 close, NO forward leakage)",
        "leakage_fix": {
            "from_handoff": "oss://ledashi-oss/aurumq-rl/handoffs/2026-05-15-path5-regime-leakage-fix/",
            "regime_features_source": "oss://ledashi-oss/aurumq-rl/handoffs/p3_4070_daily/regime_features.parquet (paris-built, rekeyed to T+1 close)",
            "regime_features_sha256_first16": sha256_file(Path("D:/dev/aurumq-rl/data/p3_4070/regime_features.parquet"))[:16],
            "regime_features_max_date": "2026-05-14",
            "regime_features_rows": 810,
        },
        "stacking_inputs": {
            "base_paths": base_paths,
        },
        "feature_cols": feature_cols,
        "n_features": ens["n_features"],
        "chosen_per_path": ens["chosen_per_path"],
        "meta_params": ens["meta_params"],
        "meta_best_iteration": ens["meta_best_iteration"],
        "schema_hash": schema_hash,
        "ensemble_metrics_new_v2": {
            "stacking_raw_H1": ens["stacking_raw_H1"]["primary_mean_top50_proximity_excess"],
            "stacking_raw_H2": ens["stacking_raw_H2"]["primary_mean_top50_proximity_excess"],
            "stacking_calibrated_H1": ens["stacking_calibrated_H1"]["primary_mean_top50_proximity_excess"],
            "stacking_calibrated_H2": ens["stacking_calibrated_H2"]["primary_mean_top50_proximity_excess"],
            "T1_hit_raw_H1": ens["stacking_raw_H1"]["top50_T1_hit_rate"],
            "T1_hit_raw_H2": ens["stacking_raw_H2"]["top50_T1_hit_rate"],
            "T1_hit_cal_H1": ens["stacking_calibrated_H1"]["top50_T1_hit_rate"],
            "T1_hit_cal_H2": ens["stacking_calibrated_H2"]["top50_T1_hit_rate"],
            "spearman_cal_H1": ens["stacking_calibrated_H1"]["spearman"],
            "spearman_cal_H2": ens["stacking_calibrated_H2"]["spearman"],
        },
        "per_base_path_eval": {
            p: {
                "H1_primary": e["H1"]["primary_mean_top50_proximity_excess"],
                "H2_primary": e["H2"]["primary_mean_top50_proximity_excess"],
                "H1_T1_hit": e["H1"]["top50_T1_hit_rate"],
                "H2_T1_hit": e["H2"]["top50_T1_hit_rate"],
            }
            for p, e in ens["per_path_ensemble"].items()
        },
        "paris_baseline_eval": {
            "H1_primary": ens["paris_baseline_H1"]["primary_mean_top50_proximity_excess"],
            "H2_primary": ens["paris_baseline_H2"]["primary_mean_top50_proximity_excess"],
            "H1_T1_hit": ens["paris_baseline_H1"]["top50_T1_hit_rate"],
            "H2_T1_hit": ens["paris_baseline_H2"]["top50_T1_hit_rate"],
        },
    }

    # Old leakage numbers (for reference)
    if variant == "path5":
        # No direct old bundle on disk; just record claim
        manifest["old_leakage_metrics"] = {
            "T1_hit_best": 0.558,
            "source": "paris email 2026-05-15 'NEW BEST T1_hit 55.8%'",
        }
    else:
        old_path = OLD_BUNDLE_LONG / "manifest.json"
        if old_path.exists():
            old = json.loads(old_path.read_text())
            manifest["old_leakage_metrics"] = {
                "stacking_calibrated_H1": old["ensemble_metrics"]["stacking_calibrated_H1"],
                "stacking_calibrated_H2": old["ensemble_metrics"]["stacking_calibrated_H2"],
                "source": "runs/sl_long_panel_bundles/sl_path5_long_inference_bundle/manifest.json",
            }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # Copy INFER.md from old long bundle if available; otherwise note
    if is_long and (OLD_BUNDLE_LONG / "INFER.md").exists():
        shutil.copy(OLD_BUNDLE_LONG / "INFER.md", out_dir / "INFER.md")
    else:
        # path5 short — write a minimal note
        (out_dir / "INFER.md").write_text(
            f"# {variant} retrain v2 — inference recipe\n\n"
            f"**Identical** to original `sl_{variant}_inference_bundle` (see `sl_path5_long_inference_bundle/INFER.md` for full recipe; same architecture).\n\n"
            f"**Only change**: `regime_features.parquet` MUST be the leakage-fixed version (rekeyed to T+1 close,\n"
            f"`oss://ledashi-oss/aurumq-rl/handoffs/p3_4070_daily/regime_features.parquet`, max_date = today).\n\n"
            f"## Feature cols (exact order)\n\n"
            + "\n".join(f"- `{c}`" for c in feature_cols)
            + "\n",
            encoding="utf-8",
        )

    print(f"  [{variant}] bundle built at {out_dir}")
    print(f"    schema_hash={schema_hash}, regime_sha16={manifest['leakage_fix']['regime_features_sha256_first16']}")


def main() -> int:
    build(RUN_PATH5, OUT_PATH5, "path5")
    build(RUN_PATH5_LONG, OUT_PATH5_LONG, "path5_long")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
