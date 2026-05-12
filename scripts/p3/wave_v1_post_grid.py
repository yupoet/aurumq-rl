"""After wave-v1 grid finishes, do:
  1. Ensemble (path1_ensemble.py) → runs/sl_path1_long_wave_v1/ensemble.json + predictions.parquet
  2. Extend predictions to dates not covered by training panel (2025-11-07 onward,
     since wave label requires forward 40d window and panel cuts off at 2025-11-06).
  3. Add as a row to eval_matrix_v2 (use modified script with extra path).
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


LONG_BUNDLE = Path("data/p3_4070_long")
RUN_ROOT = Path("runs/sl_path1_long_wave_v1")


def step1_ensemble():
    """Run path1_ensemble.py on the wave-v1 grid."""
    # baseline_predictions needed by ensemble script
    bp_src = Path("data/p3_4070_long/baseline_predictions.parquet")
    if not bp_src.exists():
        print("[step1] copying baseline_predictions from short bundle")
        import shutil
        shutil.copy("data/p3_4070/baseline_predictions.parquet", bp_src)
    cmd = [
        sys.executable, "scripts/p3/path1_ensemble.py",
        "--bundle", str(LONG_BUNDLE),
        "--runs-root", str(RUN_ROOT),
        "--out-root", str(RUN_ROOT),
        "--top-k-configs", "3",
    ]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"ensemble failed rc={rc}")
    print("[step1] ensemble done")


def step2_extend_predictions():
    """Extend predictions on 2025-11-07 onward + 2026 dates using long raw panel.

    The wave label training cut off at 2025-11-06 (need fwd-40d). To eval against
    proximity-based realized fwd-K excess on H2 + 2026 windows, we need predictions
    on those dates too.
    """
    print("[step2] extending predictions to 2025-11-07+ ...")
    ens_meta = json.loads((RUN_ROOT / "ensemble.json").read_text())
    chosen = ens_meta["chosen_runs"]

    # Load raw long panel + universe
    df = pl.read_parquet(LONG_BUNDLE / "feature_panel_v3_344.parquet").filter(
        pl.col("trade_date") >= dt.date(2025, 11, 7)
    )
    uni_parts = []
    for p in sorted(LONG_BUNDLE.glob("universe_mask/year=*.parquet")):
        uni_parts.append(pl.read_parquet(p).select(["trade_date", "ts_code", "in_universe"]))
    uni = pl.concat(uni_parts)
    df = df.join(uni, on=["trade_date", "ts_code"], how="left").filter(
        pl.col("in_universe") == True  # noqa: E712
    ).drop("in_universe")
    print(f"  in-uni rows for extend: {len(df):,}  dates: {df['trade_date'].n_unique()}")

    feature_cols = [c for c in df.columns if c not in ("ts_code", "trade_date")]
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    preds = []
    for name in chosen:
        m = lgb.Booster(model_file=str(RUN_ROOT / name / "lgb_model.txt"))
        preds.append(m.predict(X).astype(np.float32))
    score_raw = np.mean(preds, axis=0)
    # Use the ensemble's already-fit isotonic
    iso_path = RUN_ROOT / "predictions.parquet"  # the iso is embedded in predictions via ensemble script
    # Read the existing isotonic from ensemble pickling — fall back: refit on H1 here
    # Actually path1_ensemble writes the calibrated predictions directly to predictions.parquet.
    # We need to re-apply isotonic to our new score_raw. Let's load isotonic by refitting
    # from the predictions if needed.
    # Simpler: refit isotonic locally using existing predictions where they overlap H1.
    existing = pl.read_parquet(RUN_ROOT / "predictions.parquet")
    target_y = pl.read_parquet(LONG_BUNDLE / "target_y_wave_v1.parquet")
    from p3.path1_eval import H1
    h1_join = existing.join(
        target_y, on=["trade_date", "ts_code"], how="inner"
    ).filter((pl.col("trade_date") >= H1[0]) & (pl.col("trade_date") <= H1[1]))
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(h1_join["score_raw"].to_numpy(), h1_join["y"].to_numpy())
    score_cal = iso.transform(score_raw).astype(np.float32)
    new_preds = df.select(["trade_date", "ts_code"]).with_columns(
        pl.Series("score_raw", score_raw),
        pl.Series("score_calibrated", score_cal),
    )

    # Concat with existing predictions.parquet — drop overlapping dates
    existing_dates = set(existing["trade_date"].unique().to_list())
    new_preds_filtered = new_preds.filter(~pl.col("trade_date").is_in(list(existing_dates)))
    if len(new_preds_filtered) == 0:
        print("  no new dates to add")
        return
    # align dtypes
    common_cols = [c for c in new_preds_filtered.columns if c in existing.columns]
    existing_sub = existing.select(common_cols)
    new_sub = new_preds_filtered.select(common_cols)
    casts = {c: existing_sub[c].dtype for c in common_cols if new_sub[c].dtype != existing_sub[c].dtype}
    if casts:
        new_sub = new_sub.with_columns([pl.col(c).cast(t) for c, t in casts.items()])
    merged = pl.concat([existing_sub, new_sub]).sort(["trade_date", "ts_code"])
    merged.write_parquet(RUN_ROOT / "predictions.parquet", compression="zstd", compression_level=9)
    print(f"  saved (existing {len(existing):,} + new {len(new_preds_filtered):,} = {len(merged):,})")


def step3_eval():
    """Re-run eval_matrix_v2 with wave-v1 path included."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ev2", "scripts/p3/eval_matrix_v2.py")
    ev2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev2)
    extra = ("Wave v1 (P1L retrain)", "runs/sl_path1_long_wave_v1/predictions.parquet", "score_calibrated")
    if extra not in ev2.PATHS:
        ev2.PATHS.append(extra)
    ev2.OUT_ROOT = Path("runs/eval_matrix_v2_with_wave")
    ev2.main()


def main():
    print("=== STEP 1: ensemble ===")
    step1_ensemble()
    print()
    print("=== STEP 2: extend predictions to 2025-11-07+ ===")
    step2_extend_predictions()
    print()
    print("=== STEP 3: eval matrix v2 with wave-v1 added ===")
    step3_eval()


if __name__ == "__main__":
    main()
