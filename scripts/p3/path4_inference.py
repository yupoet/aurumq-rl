"""SL Path 4 production inference — load bundle, transform features, predict, calibrate.

Usage::

    .venv/Scripts/python.exe scripts/p3/path4_inference.py \\
        --bundle-dir runs/sl_path4/inference_bundle \\
        --raw-features data/p3_4070/feature_panel_v3_344.parquet \\
        --universe-mask "data/p3_4070/universe_mask/year=*.parquet" \\
        --start-date 2025-07-01 --end-date 2025-09-30 \\
        --top-k 50 \\
        --out predictions_today.parquet

The output parquet has columns (trade_date, ts_code, score_raw, score_calibrated)
across the full universe. Caller filters to in-universe + top-K per day.

For 'today only' inference: pass --start-date today --end-date today.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p3.rank_z import cross_sectional_rank_z


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", required=True, type=Path,
                    help="Path to inference_bundle/ produced by path4_export_inference.py")
    ap.add_argument("--raw-features", required=True, type=Path,
                    help="Raw feature panel parquet (pre rank-z) e.g. feature_panel_v3_344.parquet")
    ap.add_argument("--universe-mask", required=True, type=str,
                    help="Glob to per-year universe_mask parquets")
    ap.add_argument("--start-date", required=True, help="ISO date, inclusive")
    ap.add_argument("--end-date", required=True, help="ISO date, inclusive")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    bundle = args.bundle_dir
    if not bundle.exists():
        logger.error("bundle dir %s not found", bundle)
        return 2

    # 1. Load bundle metadata + isotonic + feature_cols
    manifest = json.loads((bundle / "manifest.json").read_text())
    expected_feature_cols = json.loads((bundle / "feature_cols.json").read_text())
    with (bundle / "isotonic.pkl").open("rb") as f:
        iso = pickle.load(f)
    logger.info("loaded bundle: %d models, %d features, %d chosen runs",
                manifest["n_models"], manifest["n_features"], len(manifest["chosen_runs"]))

    # 2. Load raw features for the requested window
    t0 = time.time()
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    raw = pl.read_parquet(args.raw_features).filter(
        (pl.col("trade_date") >= start) & (pl.col("trade_date") <= end)
    )
    logger.info("loaded raw features: %d rows in [%s, %s] (%.1fs)",
                len(raw), start, end, time.time() - t0)

    # Verify column set matches training-time
    raw_cols = [c for c in raw.columns if c not in ("ts_code", "trade_date")]
    missing = set(expected_feature_cols) - set(raw_cols)
    if missing:
        logger.error("raw panel missing %d expected feature cols: %s",
                     len(missing), sorted(missing)[:10])
        return 3
    extra = set(raw_cols) - set(expected_feature_cols)
    if extra:
        logger.warning("raw panel has %d extra cols (will be dropped): %s",
                       len(extra), sorted(extra)[:5])

    # 3. Load universe mask
    uni_parts = []
    for p in sorted(Path().glob(args.universe_mask)):
        uni_parts.append(pl.read_parquet(p).select(["trade_date", "ts_code", "in_universe"]))
    universe = pl.concat(uni_parts).filter(
        (pl.col("trade_date") >= start) & (pl.col("trade_date") <= end)
    )
    logger.info("loaded universe mask: %d rows", len(universe))

    # 4. Apply rank-z (per-day per-feature rank within in-universe stocks)
    t1 = time.time()
    panel_cols = ["trade_date", "ts_code"] + expected_feature_cols
    raw_subset = raw.select(panel_cols)
    clean = cross_sectional_rank_z(raw_subset, universe, expected_feature_cols)
    logger.info("rank-z done in %.1fs (%d rows)", time.time() - t1, len(clean))

    # 5. Filter to in-universe rows for prediction (out-of-universe scores meaningless)
    clean = clean.join(universe, on=["trade_date", "ts_code"], how="left").filter(
        pl.col("in_universe") == True  # noqa: E712
    ).drop("in_universe")
    logger.info("in-universe rows: %d", len(clean))

    # 6. Stack predictions across the chosen ensemble models
    X = clean.select(expected_feature_cols).to_numpy().astype(np.float32)
    chosen = manifest["chosen_runs"]
    preds = []
    for n in chosen:
        model_path = bundle / "models" / f"lgb_model_{n}.txt"
        m = lgb.Booster(model_file=str(model_path))
        preds.append(m.predict(X).astype(np.float32))
    score_raw = np.mean(preds, axis=0)  # seed-mean ensemble
    score_calibrated = iso.transform(score_raw).astype(np.float32)
    logger.info("predicted %d × %d ensemble: raw[%.4f, %.4f] cal[%.4f, %.4f]",
                len(preds), len(score_raw),
                score_raw.min(), score_raw.max(),
                score_calibrated.min(), score_calibrated.max())

    # 7. Output (trade_date, ts_code, score_raw, score_calibrated)
    out = clean.select(["trade_date", "ts_code"]).with_columns(
        pl.Series("score_raw", score_raw),
        pl.Series("score_calibrated", score_calibrated),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out, compression="zstd", compression_level=9)
    logger.info("wrote %s (%d rows)", args.out, len(out))

    # 8. Demo top-K printout for the most recent date in the window
    last_date = out["trade_date"].max()
    top_k_today = (
        out.filter(pl.col("trade_date") == last_date)
        .sort("score_calibrated", descending=True)
        .head(args.top_k)
    )
    logger.info("top-%d picks on %s:", args.top_k, last_date)
    for r in top_k_today.head(10).rows(named=True):
        logger.info("  %s  %s  raw=%.5f  cal=%.5f",
                    r["trade_date"], r["ts_code"], r["score_raw"], r["score_calibrated"])
    if args.top_k > 10:
        logger.info("  ... (%d more)", args.top_k - 10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
