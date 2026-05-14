"""Wave v2 post-grid: ensemble + extend predictions + long-horizon eval."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


LONG = Path("data/p3_4070_long")
RUN_ROOT = Path("runs/sl_path1_long_wave_v2")


def step1_ensemble():
    cmd = [
        sys.executable, "scripts/p3/path1_ensemble.py",
        "--bundle", str(LONG),
        "--runs-root", str(RUN_ROOT),
        "--out-root", str(RUN_ROOT),
        "--top-k-configs", "3",
    ]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"ensemble rc={rc}")
    print("[step1] ensemble done")


def step2_extend_predictions():
    """Predict on dates not in wave_v2 panel (forward 40d window cuts at 2025-11-06)."""
    ens_meta = json.loads((RUN_ROOT / "ensemble.json").read_text())
    chosen = ens_meta["chosen_runs"]
    print(f"  chosen: {len(chosen)} models")

    df = pl.read_parquet(LONG / "feature_panel_v3_344.parquet").filter(
        pl.col("trade_date") >= dt.date(2025, 11, 7)
    )
    uni_parts = [pl.read_parquet(p).select(["trade_date","ts_code","in_universe"])
                 for p in sorted(LONG.glob("universe_mask/year=*.parquet"))]
    uni = pl.concat(uni_parts)
    df = df.join(uni, on=["trade_date","ts_code"], how="left").filter(
        pl.col("in_universe") == True  # noqa: E712
    ).drop("in_universe")
    print(f"  in-uni rows for extend: {len(df):,}")

    feature_cols = [c for c in df.columns if c not in ("ts_code","trade_date")]
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    preds = [lgb.Booster(model_file=str(RUN_ROOT / n / "lgb_model.txt")).predict(X).astype(np.float32)
             for n in chosen]
    score_raw = np.mean(preds, axis=0)

    # Refit isotonic from existing predictions + wave v2 label
    existing = pl.read_parquet(RUN_ROOT / "predictions.parquet")
    target_y = pl.read_parquet(LONG / "target_y_wave_v2.parquet")
    from p3.path1_eval import H1
    from sklearn.isotonic import IsotonicRegression
    h1_join = existing.join(
        target_y.select(["trade_date","ts_code","y"]), on=["trade_date","ts_code"], how="inner"
    ).filter((pl.col("trade_date") >= H1[0]) & (pl.col("trade_date") <= H1[1]))
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(h1_join["score_raw"].to_numpy(), h1_join["y"].to_numpy())
    score_cal = iso.transform(score_raw).astype(np.float32)

    new_preds = df.select(["trade_date","ts_code"]).with_columns(
        pl.Series("score_raw", score_raw),
        pl.Series("score_calibrated", score_cal),
    )
    existing_dates = set(existing["trade_date"].unique().to_list())
    new_filtered = new_preds.filter(~pl.col("trade_date").is_in(list(existing_dates)))
    if len(new_filtered) == 0:
        print("  no new dates"); return
    common = [c for c in new_filtered.columns if c in existing.columns]
    ex_sub = existing.select(common); new_sub = new_filtered.select(common)
    casts = {c: ex_sub[c].dtype for c in common if new_sub[c].dtype != ex_sub[c].dtype}
    if casts:
        new_sub = new_sub.with_columns([pl.col(c).cast(t) for c, t in casts.items()])
    merged = pl.concat([ex_sub, new_sub]).sort(["trade_date","ts_code"])
    merged.write_parquet(RUN_ROOT / "predictions.parquet", compression="zstd", compression_level=9)
    print(f"  saved (ex {len(existing):,} + new {len(new_filtered):,} = {len(merged):,})")


def step3_eval():
    """Run eval_matrix_v2 + long_horizon with wave-v2 added."""
    # eval_matrix_v2
    spec = importlib.util.spec_from_file_location("ev2", "scripts/p3/eval_matrix_v2.py")
    ev2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev2)
    extras = [
        ("Wave v1", "runs/sl_path1_long_wave_v1/predictions.parquet", "score_calibrated"),
        ("Wave v2 (4-fix)", "runs/sl_path1_long_wave_v2/predictions.parquet", "score_calibrated"),
    ]
    for e in extras:
        if e not in ev2.PATHS: ev2.PATHS.append(e)
    ev2.OUT_ROOT = Path("runs/eval_matrix_v2_with_wave_v2")
    ev2.main()

    # long_horizon eval
    spec2 = importlib.util.spec_from_file_location("lh", "scripts/p3/wave_v1_long_horizon_eval.py")
    lh = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(lh)
    extras2 = [
        ("Wave v1",         "runs/sl_path1_long_wave_v1/predictions.parquet", "score_calibrated"),
        ("Wave v2 (4-fix)", "runs/sl_path1_long_wave_v2/predictions.parquet", "score_calibrated"),
    ]
    for e in extras2:
        if e not in lh.PATHS: lh.PATHS.append(e)
    lh.OUT = Path("runs/eval_matrix_v2_with_wave_v2/long_horizon.md")
    lh.main()


def main():
    print("=== STEP 1: ensemble ===")
    step1_ensemble()
    print("=== STEP 2: extend predictions to 2025-11-07+ ===")
    step2_extend_predictions()
    print("=== STEP 3: eval matrix + long-horizon ===")
    step3_eval()


if __name__ == "__main__":
    main()
