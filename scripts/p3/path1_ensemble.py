"""Path 1 — ensemble (top-3 configs × 3 seeds, mean) + isotonic calibration on H1.

Outputs:
- runs/sl_path1/predictions.parquet — (trade_date, ts_code, score_raw, score_calibrated)
- runs/sl_path1/ensemble.json — H1 + H2 metric blocks for both raw and calibrated.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

# Self-contained import path for subprocess invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p3.path1_eval import H1, H2, evaluate


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Pure functions
# ------------------------------------------------------------------ #

def seed_mean_ensemble(prediction_dfs: list[pl.DataFrame]) -> pl.DataFrame:
    """Average score across multiple seed predictions, joined on (trade_date, ts_code)."""
    if not prediction_dfs:
        raise ValueError("no prediction DataFrames provided")
    base = prediction_dfs[0].select(["trade_date", "ts_code", "score"]).rename({"score": "score_0"})
    for i, df in enumerate(prediction_dfs[1:], start=1):
        base = base.join(
            df.select(["trade_date", "ts_code", "score"]).rename({"score": f"score_{i}"}),
            on=["trade_date", "ts_code"],
            how="inner",
        )
    score_cols = [f"score_{i}" for i in range(len(prediction_dfs))]
    base = base.with_columns(pl.mean_horizontal([pl.col(c) for c in score_cols]).alias("score"))
    return base.select(["trade_date", "ts_code", "score"])


def calibrate_isotonic(pred_calibration: np.ndarray, actual_calibration: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Fit isotonic regression on (pred, actual) and return a callable transformer.

    The returned callable applies the fitted isotonic to any new pred array.
    Rank order of the input is strictly preserved by adding a negligible
    rank-proportional epsilon to break isotonic flat-region ties.
    """
    pred_calibration = np.asarray(pred_calibration, dtype=np.float64)
    actual_calibration = np.asarray(actual_calibration, dtype=np.float64)
    mask = np.isfinite(pred_calibration) & np.isfinite(actual_calibration)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(pred_calibration[mask], actual_calibration[mask])

    def _transform(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = iso.transform(x)
        # Break isotonic plateau ties with a negligible rank-proportional offset
        # so argsort(x) == argsort(out) always holds.
        eps = 1e-12
        ranks = np.argsort(np.argsort(x))
        return out + eps * ranks

    return _transform


def pick_top_configs_by_val(runs: dict[str, float], top_k: int = 3) -> list[str]:
    """Select the top_k DISTINCT configs (collapsing seeds) by their seed-mean VAL primary.

    runs keys: full run names like "nl63_lr050_mdl100_seed42".
    Returns ALL run names (across seeds) belonging to the top_k configs.
    """
    config_to_seedscores: dict[str, list[float]] = {}
    for name, score in runs.items():
        config_base = name.rsplit("_seed", 1)[0]
        config_to_seedscores.setdefault(config_base, []).append(score)
    config_means = {k: float(np.mean(v)) for k, v in config_to_seedscores.items()}
    top = sorted(config_means.items(), key=lambda kv: -kv[1])[:top_k]
    top_configs = {kv[0] for kv in top}
    return [n for n in runs if n.rsplit("_seed", 1)[0] in top_configs]


# ------------------------------------------------------------------ #
# Driver
# ------------------------------------------------------------------ #

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--runs-root", default=Path("runs/sl_path1"), type=Path)
    ap.add_argument("--out-root", default=Path("runs/sl_path1"), type=Path)
    ap.add_argument("--top-k-configs", type=int, default=3)
    args = ap.parse_args(argv)

    # 1. Load all run results
    run_val_primary: dict[str, float] = {}
    pred_paths: dict[str, Path] = {}
    for run_dir in sorted(args.runs_root.glob("*_seed*/")):
        results = run_dir / "results.json"
        if not results.exists():
            continue
        d = json.loads(results.read_text())
        run_val_primary[run_dir.name] = d["VAL_EFF"]["primary_mean_top50_proximity_excess"]
        pred_paths[run_dir.name] = run_dir / "predictions.parquet"
    logger.info("found %d completed runs", len(run_val_primary))
    if len(run_val_primary) < args.top_k_configs * 2:
        logger.error("not enough runs (%d) to form top-%d ensemble", len(run_val_primary), args.top_k_configs)
        return 2

    # 2. Pick top-K configs
    chosen = pick_top_configs_by_val(run_val_primary, top_k=args.top_k_configs)
    logger.info("chose %d run names across %d distinct configs", len(chosen),
                len({n.rsplit('_seed', 1)[0] for n in chosen}))

    # 3. Seed-mean ensemble
    pred_dfs = [pl.read_parquet(pred_paths[n]) for n in chosen]
    ens = seed_mean_ensemble(pred_dfs)
    logger.info("ensemble: %d rows", len(ens))

    # 4. Calibrate on H1
    target_y = pl.read_parquet(args.bundle / "target_y.parquet")
    realized = pl.read_parquet(args.bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    )
    market = pl.read_parquet(args.bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    )
    h1_join = ens.filter(
        (pl.col("trade_date") >= H1[0]) & (pl.col("trade_date") <= H1[1])
    ).join(target_y, on=["trade_date", "ts_code"], how="inner")
    calibrator = calibrate_isotonic(
        h1_join["score"].to_numpy(),
        h1_join["y"].to_numpy(),
    )
    score_calibrated = calibrator(ens["score"].to_numpy())
    ens_cal = ens.with_columns(pl.Series("score_calibrated", score_calibrated))
    logger.info("isotonic fit on %d H1 rows; calibrated %d total rows", len(h1_join), len(ens_cal))

    # 5. Eval raw + calibrated on H1, H2
    eval_raw_h1 = evaluate(ens, target_y, realized, market, H1)
    eval_raw_h2 = evaluate(ens, target_y, realized, market, H2)
    eval_cal_h1 = evaluate(
        ens_cal.select(["trade_date", "ts_code", pl.col("score_calibrated").alias("score")]),
        target_y, realized, market, H1,
    )
    eval_cal_h2 = evaluate(
        ens_cal.select(["trade_date", "ts_code", pl.col("score_calibrated").alias("score")]),
        target_y, realized, market, H2,
    )

    # 6. Compare to paris baseline
    bp = pl.read_parquet(args.bundle / "baseline_predictions.parquet")
    bp_pred = bp.select(["trade_date", "ts_code", pl.col("p_t3_baseline").alias("score")])
    bp_h1 = evaluate(bp_pred, target_y, realized, market, H1)
    bp_h2 = evaluate(bp_pred, target_y, realized, market, H2)

    summary = {
        "chosen_runs": chosen,
        "n_calibration_rows_H1": len(h1_join),
        "ensemble_raw_H1": eval_raw_h1,
        "ensemble_raw_H2": eval_raw_h2,
        "ensemble_calibrated_H1": eval_cal_h1,
        "ensemble_calibrated_H2": eval_cal_h2,
        "paris_baseline_H1": bp_h1,
        "paris_baseline_H2": bp_h2,
    }

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "ensemble.json").write_text(json.dumps(summary, indent=2, default=str))
    ens_cal.write_parquet(args.out_root / "predictions.parquet", compression="zstd", compression_level=10)

    logger.info("== Path 1 ensemble vs paris baseline ==")
    logger.info("Window | Metric                       | Path1 raw | Path1 cal | Paris    | Δ vs Paris")
    for window, p1r, p1c, par in (
        ("H1", eval_raw_h1, eval_cal_h1, bp_h1),
        ("H2", eval_raw_h2, eval_cal_h2, bp_h2),
    ):
        for k, label in (
            ("primary_mean_top50_proximity_excess", "primary"),
            ("spearman", "spearman"),
            ("top50_T1_hit_rate", "T1_hit"),
            ("ece_10bin", "ECE"),
        ):
            logger.info(
                "%-6s | %-28s | %+.6f | %+.6f | %+.6f | %+.6f",
                window, label, p1r[k], p1c[k], par[k], p1c[k] - par[k],
            )

    logger.info("wrote %s and %s",
                args.out_root / "ensemble.json", args.out_root / "predictions.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
