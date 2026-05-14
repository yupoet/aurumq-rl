"""Build inference bundle for Wave v1 (Path 1 long architecture retrained on wave label)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import polars as pl
import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent.parent))
from p3.path1_eval import H1


def _sha16(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_wave_v1_bundle():
    bundle = Path("data/p3_4070_long")
    runs_root = Path("runs/sl_path1_long_wave_v1")
    out_dir = Path("runs/sl_long_panel_bundles/sl_wave_v1_inference_bundle")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)

    ens_meta = json.loads((runs_root / "ensemble.json").read_text())
    chosen = ens_meta["chosen_runs"]

    # Copy models
    shas = {}
    for n in chosen:
        src = runs_root / n / "lgb_model.txt"
        dst = out_dir / "models" / f"lgb_model_{n}.txt"
        shutil.copy(src, dst)
        shas[n] = _sha16(dst)

    # Re-fit isotonic on wave label H1 (already computed in ensemble step but bundle it cleanly)
    target_y_wave = pl.read_parquet(bundle / "target_y_wave_v1.parquet")
    preds = pl.read_parquet(runs_root / "predictions.parquet")
    h1_join = preds.join(
        target_y_wave.select(["trade_date", "ts_code", "y"]),
        on=["trade_date", "ts_code"], how="inner"
    ).filter((pl.col("trade_date") >= H1[0]) & (pl.col("trade_date") <= H1[1]))
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(h1_join["score_raw"].to_numpy(), h1_join["y"].to_numpy())
    with (out_dir / "isotonic.pkl").open("wb") as f:
        pickle.dump(iso, f)

    # Feature cols (raw 345)
    cols = [c for c in pl.read_parquet_schema(bundle / "feature_panel_v3_344.parquet")
            if c not in ("ts_code", "trade_date", "y")]
    (out_dir / "feature_cols.json").write_text(json.dumps(cols, indent=2, ensure_ascii=False))

    manifest = {
        "generated_at": dt.datetime.now().isoformat(),
        "path": "sl_wave_v1 (Path 1 long architecture, WAVE label)",
        "family": "WAVE (long-horizon, fwd-20d/40d signal)",
        "feature_panel_input": "feature_panel_v3_344.parquet (RAW, NO rank-z)",
        "feature_panel_preprocessing": "NONE — raw paris panel directly",
        "n_features": len(cols),
        "n_models": len(chosen),
        "model_type": "LightGBM regression_l2 trained on wave label",
        "ensemble_strategy": "seed_mean of 9 chosen seeds across top-3 configs",
        "chosen_runs": chosen,
        "model_shas": shas,
        "calibration": "isotonic on H1 wave-label actual y",
        "train_window": "2018-01-02 ~ 2024-12-04 (7y); label cutoff 2025-11-06",
        "label_spec": {
            "wave_quality": "0.5*fwd_20d_excess + 0.25*fwd_40d_excess + Sharpe_20d_term + 0.10*max_dd_20d",
            "entry_timing_decay": "exp(-(day_to_peak_20d - 1) / 5)",
            "final": "max(0, wave_quality * entry_timing_decay)",
            "version": "v1 — see SUPPLEMENT for known bugs to address in v2",
        },
        "design_horizon_metrics": {
            "fwd_20d_excess_VAL": ens_meta.get("ensemble_calibrated_H1",{}).get("primary_mean_top50_proximity_excess"),
            "fwd_20d_excess_H1":  None,
            "note": "see runs/eval_matrix_v2_with_wave/long_horizon.md for full table",
        },
        "wave_vs_proximity_tradeoff_summary": {
            "VAL_fwd_1d_excess": "-46 bps vs Path 1 long (proximity)",
            "VAL_fwd_20d_excess": "+237 bps vs Path 1 long",
            "H1_fwd_20d_excess": "+214 bps vs Path 1 long",
            "T40_hit": "+1-8 pp across all slices",
        },
        "notes": (
            "WAVE FAMILY model — designed for swing-trade entry (hold 20-40 trade days). "
            "DO NOT use as drop-in replacement for proximity models in next-day strategies. "
            "Pair with an exit model that holds until trend break (not next-day mean reversion)."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    print(f"wave_v1: wrote {len(chosen)} models + isotonic + feature_cols.json ({len(cols)} cols)")


if __name__ == "__main__":
    build_wave_v1_bundle()
