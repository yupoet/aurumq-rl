"""Extend predictions to 2026-Q1 + 2026-Q2-partial for the production paths.

For each path:
  1. Load chosen-run models from runs/<path>/ + isotonic from inference bundle
  2. Build the path's input matrix on long bundle's 2026 rows (already has
     raw 345-col + pruned rank-z 226-col panels through 2026-04-30)
  3. Predict ensemble mean → isotonic → predictions_2026.parquet
  4. Concat with existing predictions.parquet → predictions.parquet (in-place rewrite)

Paths handled (in priority order):
  - sl_path1_long       (raw 345-col, 9 LGB models)
  - sl_path_d (Path 4 long)  (rank-z'd pruned 226-col, 9 LGB)
  - sl_path2_long       (rank-z'd pruned 226-col, 6 CB + 3 XGB)
  - sl_path4 (Path 4 short)  (rank-z'd 345-col)   ← needs fresh rank-z compute on 2026
  - sl_regime_stack_long (P5 meta over 1L+4L+2L bases + regime features + interactions)
  - sl_hybrid (P1L + P4S, equal-weight)            ← chained from path1_long + path4 outputs
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from aurumq_rl.p3.rank_z import cross_sectional_rank_z


SHORT_BUNDLE = Path("data/p3_4070")
LONG_BUNDLE = Path("data/p3_4070_long")
INFER_BUNDLE_ROOT = Path("runs/sl_paths_inference_bundles")
INFER_LONG_ROOT = Path("runs/sl_long_panel_bundles")


def load_long_universe() -> pl.DataFrame:
    """Concat universe_mask shards from long bundle."""
    parts = []
    for p in sorted(LONG_BUNDLE.glob("universe_mask/year=*.parquet")):
        parts.append(pl.read_parquet(p).select(["trade_date", "ts_code", "in_universe"]))
    return pl.concat(parts)


def load_long_raw_2026() -> pl.DataFrame:
    """Long raw panel filtered to 2026 with in-universe rows only."""
    df = pl.read_parquet(LONG_BUNDLE / "feature_panel_v3_344.parquet").filter(
        pl.col("trade_date") >= dt.date(2026, 1, 1)
    )
    uni = load_long_universe()
    df = df.join(uni, on=["trade_date", "ts_code"], how="left").filter(
        pl.col("in_universe") == True  # noqa: E712
    ).drop("in_universe")
    print(f"  raw 2026 in-uni rows: {len(df):,}  dates: {df['trade_date'].n_unique()}")
    return df


def load_long_pruned_2026() -> pl.DataFrame:
    """Long rank-z'd PRUNED 226-col panel filtered to 2026 with in-uni only."""
    df = pl.read_parquet(LONG_BUNDLE / "feature_panel_clean.parquet").filter(
        pl.col("trade_date") >= dt.date(2026, 1, 1)
    )
    uni = load_long_universe()
    df = df.join(uni, on=["trade_date", "ts_code"], how="left").filter(
        pl.col("in_universe") == True  # noqa: E712
    ).drop("in_universe")
    print(f"  pruned 2026 in-uni rows: {len(df):,}  dates: {df['trade_date'].n_unique()}")
    return df


def predict_lgb_ensemble(
    panel: pl.DataFrame,
    feature_cols: list[str],
    run_root: Path,
    chosen_runs: list[str],
    iso_path: Path,
) -> pl.DataFrame:
    """LGB ensemble predict + isotonic. Returns (trade_date, ts_code, score_raw, score_calibrated)."""
    X = panel.select(feature_cols).to_numpy().astype(np.float32)
    preds = []
    for name in chosen_runs:
        model_file = run_root / name / "lgb_model.txt"
        m = lgb.Booster(model_file=str(model_file))
        preds.append(m.predict(X).astype(np.float32))
    score_raw = np.mean(preds, axis=0)
    with iso_path.open("rb") as f:
        iso = pickle.load(f)
    score_cal = iso.transform(score_raw).astype(np.float32)
    return panel.select(["trade_date", "ts_code"]).with_columns(
        pl.Series("score_raw", score_raw),
        pl.Series("score_calibrated", score_cal),
    )


def predict_path2_long_ensemble(
    panel: pl.DataFrame,
    feature_cols: list[str],
    run_root: Path,
    chosen_runs: list[str],
    iso_path: Path,
) -> pl.DataFrame:
    """CatBoost + XGB mixed ensemble."""
    import catboost as cb
    import xgboost as xgb_lib

    X = panel.select(feature_cols).to_numpy().astype(np.float32)
    preds = []
    for name in chosen_runs:
        if name.startswith("cb_"):
            m = cb.CatBoostRegressor()
            m.load_model(str(run_root / name / "catboost_model.cbm"))
            preds.append(m.predict(X).astype(np.float32))
        elif name.startswith("xgb_"):
            booster = xgb_lib.Booster()
            booster.load_model(str(run_root / name / "xgb_model.json"))
            preds.append(booster.predict(xgb_lib.DMatrix(X)).astype(np.float32))
    score_raw = np.mean(preds, axis=0)
    with iso_path.open("rb") as f:
        iso = pickle.load(f)
    score_cal = iso.transform(score_raw).astype(np.float32)
    return panel.select(["trade_date", "ts_code"]).with_columns(
        pl.Series("score_raw", score_raw),
        pl.Series("score_calibrated", score_cal),
    )


def predict_path5_long_2026(
    p1l_2026: pl.DataFrame,
    p4l_2026: pl.DataFrame,
    p2l_2026: pl.DataFrame,
) -> pl.DataFrame:
    """Path 5 long meta predict on 2026 — chain 3 base path scores + regime features + interactions."""
    META_BUNDLE = INFER_LONG_ROOT / "sl_path5_long_inference_bundle"
    manifest = json.loads((META_BUNDLE / "manifest.json").read_text())
    feature_cols = manifest["feature_cols"]  # 23 cols

    # Build base scores. NOTE: training used `_long` suffix on the base columns.
    base = (
        p1l_2026.select(["trade_date", "ts_code", pl.col("score_raw").alias("path_sl_path1_long")])
        .join(p4l_2026.select(["trade_date", "ts_code", pl.col("score_raw").alias("path_sl_path4_long")]),
              on=["trade_date", "ts_code"], how="inner")
        .join(p2l_2026.select(["trade_date", "ts_code", pl.col("score_raw").alias("path_sl_path2_long")]),
              on=["trade_date", "ts_code"], how="inner")
    )
    # Regime features for 2026
    regime = pl.read_parquet(SHORT_BUNDLE / "regime_features.parquet").filter(
        pl.col("trade_date") >= dt.date(2026, 1, 1)
    )
    df = base.join(regime, on="trade_date", how="left")
    # Build interaction columns (training names also use _long suffix)
    for pc in ("path_sl_path1_long", "path_sl_path4_long", "path_sl_path2_long"):
        for rc in ("univ_vol_20d", "cs_dispersion", "sector_dispersion"):
            df = df.with_columns((pl.col(pc) * pl.col(rc)).alias(f"{pc}__x__{rc}"))

    X = df.select(feature_cols).to_numpy().astype(np.float32)
    meta = lgb.Booster(model_file=str(META_BUNDLE / "meta_lgb_model.txt"))
    score_raw = meta.predict(X).astype(np.float32)
    with (META_BUNDLE / "meta_isotonic.pkl").open("rb") as f:
        iso = pickle.load(f)
    score_cal = iso.transform(score_raw).astype(np.float32)
    return df.select(["trade_date", "ts_code"]).with_columns(
        pl.Series("score_raw", score_raw),
        pl.Series("score_calibrated", score_cal),
    )


def append_and_save(existing_path: Path, new: pl.DataFrame):
    """Concat existing predictions.parquet with new rows; rewrite in place after backup."""
    if existing_path.exists():
        existing = pl.read_parquet(existing_path)
        # backup once
        backup = existing_path.parent / (existing_path.name + ".pre_2026")
        if not backup.exists():
            import shutil
            shutil.copy(existing_path, backup)
        # filter new to ensure no overlap (drop rows already present)
        existing_dates = set(existing["trade_date"].unique().to_list())
        new_filtered = new.filter(~pl.col("trade_date").is_in(list(existing_dates)))
        # Align columns (existing may not have score_raw if older format) and cast dtypes
        common_cols = [c for c in new_filtered.columns if c in existing.columns]
        existing_sub = existing.select(common_cols)
        new_sub = new_filtered.select(common_cols)
        # Cast new columns to existing dtypes to avoid Float32/Float64 mismatch
        casts = {c: existing_sub[c].dtype for c in common_cols if new_sub[c].dtype != existing_sub[c].dtype}
        if casts:
            new_sub = new_sub.with_columns([pl.col(c).cast(t) for c, t in casts.items()])
        merged = pl.concat([existing_sub, new_sub]).sort(["trade_date", "ts_code"])
        merged.write_parquet(existing_path, compression="zstd", compression_level=9)
        print(f"  saved {existing_path} (existing {len(existing):,} + new {len(new_filtered):,} = {len(merged):,})")
    else:
        new.write_parquet(existing_path, compression="zstd", compression_level=9)
        print(f"  saved {existing_path} ({len(new):,} rows)")


def main():
    # ---- Path 1 long: raw 345-col ----
    print("\n[Path 1 long] raw 345-col on 2026")
    raw_2026 = load_long_raw_2026()
    feature_cols = [c for c in raw_2026.columns if c not in ("ts_code", "trade_date")]
    print(f"  feature cols: {len(feature_cols)}")
    p1l_chosen = json.loads(Path("runs/sl_path1_long/ensemble.json").read_text())["chosen_runs"]
    p1l = predict_lgb_ensemble(
        panel=raw_2026,
        feature_cols=feature_cols,
        run_root=Path("runs/sl_path1_long"),
        chosen_runs=p1l_chosen,
        iso_path=INFER_LONG_ROOT / "sl_path1_long_inference_bundle" / "isotonic.pkl",
    )
    append_and_save(Path("runs/sl_path1_long/predictions.parquet"), p1l)

    # ---- Path 4 long (Path D): rank-z'd PRUNED 226-col (already shipped in long bundle) ----
    print("\n[Path 4 long / Path D] pruned 226-col on 2026")
    pruned_2026 = load_long_pruned_2026()
    feature_cols_226 = [c for c in pruned_2026.columns if c not in ("ts_code", "trade_date", "y")]
    print(f"  feature cols: {len(feature_cols_226)}")
    p4l_chosen = json.loads(Path("runs/sl_path_d/ensemble.json").read_text())["chosen_runs"]
    p4l = predict_lgb_ensemble(
        panel=pruned_2026,
        feature_cols=feature_cols_226,
        run_root=Path("runs/sl_path_d"),
        chosen_runs=p4l_chosen,
        iso_path=INFER_LONG_ROOT / "sl_path4_long_inference_bundle" / "isotonic.pkl",
    )
    append_and_save(Path("runs/sl_path_d/predictions.parquet"), p4l)

    # ---- Path 2 long: same 226-col panel, CB+XGB mix ----
    print("\n[Path 2 long] pruned 226-col CB+XGB on 2026")
    p2l_chosen = json.loads(Path("runs/sl_path2_long/ensemble.json").read_text())["chosen_runs"]
    p2l = predict_path2_long_ensemble(
        panel=pruned_2026,
        feature_cols=feature_cols_226,
        run_root=Path("runs/sl_path2_long"),
        chosen_runs=p2l_chosen,
        iso_path=INFER_LONG_ROOT / "sl_path2_long_inference_bundle" / "isotonic.pkl",
    )
    append_and_save(Path("runs/sl_path2_long/predictions.parquet"), p2l)

    # ---- Path 4 short: rank-z 345-col, needs fresh rank-z on 2026 raw rows ----
    print("\n[Path 4 short / prod] computing fresh rank-z on 2026 raw 345-col")
    raw_2026_full = pl.read_parquet(LONG_BUNDLE / "feature_panel_v3_344.parquet").filter(
        pl.col("trade_date") >= dt.date(2026, 1, 1)
    )
    uni = load_long_universe()
    fcols_345 = [c for c in raw_2026_full.columns if c not in ("ts_code", "trade_date")]
    print(f"  rank-z over {len(fcols_345)} cols")
    rz_2026 = cross_sectional_rank_z(raw_2026_full, uni, fcols_345)
    # Filter to in-universe AFTER rank-z (rank-z already zeroes out-of-uni rows)
    rz_2026 = rz_2026.join(uni, on=["trade_date", "ts_code"], how="left").filter(
        pl.col("in_universe") == True  # noqa: E712
    ).drop("in_universe")
    print(f"  rank-z'd 2026 in-uni rows: {len(rz_2026):,}")
    p4s_chosen = json.loads(Path("runs/sl_path4/ensemble.json").read_text())["chosen_runs"]
    p4s = predict_lgb_ensemble(
        panel=rz_2026,
        feature_cols=fcols_345,
        run_root=Path("runs/sl_path4"),
        chosen_runs=p4s_chosen,
        iso_path=INFER_BUNDLE_ROOT / "sl_path4_inference_bundle" / "isotonic.pkl"
            if (INFER_BUNDLE_ROOT / "sl_path4_inference_bundle" / "isotonic.pkl").exists()
            else Path("runs/sl_path4/inference_bundle/isotonic.pkl"),
    )
    append_and_save(Path("runs/sl_path4/predictions.parquet"), p4s)

    # ---- Path 5 long meta: chain 3 long base preds + regime ----
    print("\n[Path 5 long] meta chain over 1L/4L/2L 2026 + regime features")
    p5l_2026 = predict_path5_long_2026(p1l, p4l, p2l)
    print(f"  P5 long 2026 rows: {len(p5l_2026):,}")
    append_and_save(Path("runs/sl_regime_stack_long/predictions.parquet"), p5l_2026)

    # ---- Hybrid 2026: equal-weight (P1L + P4S) / 2 ----
    print("\n[Hybrid] (P1L + P4S) / 2 on 2026")
    hyb_2026 = (
        p1l.select(["trade_date", "ts_code", pl.col("score_calibrated").alias("p1l_score")])
        .join(p4s.select(["trade_date", "ts_code", pl.col("score_calibrated").alias("p4s_score")]),
              on=["trade_date", "ts_code"], how="inner")
        .with_columns(((pl.col("p1l_score") + pl.col("p4s_score")) / 2.0).alias("score"))
    )
    print(f"  Hybrid 2026 rows: {len(hyb_2026):,}")
    append_and_save(Path("runs/sl_hybrid_p1long_p4short/predictions.parquet"), hyb_2026)

    print("\n[done] all paths extended to 2026")


if __name__ == "__main__":
    main()
