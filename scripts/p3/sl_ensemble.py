"""Build SL ensemble from sl_grid runs and compare to P2 v2 baseline.

Two-stage averaging:
  1. seed-average per label: mean of 3 seeds' raw probabilities
  2. label-ensemble: rank-rescale per label then mean across labels

Evaluates against same H1/H2 windows.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score


logger = logging.getLogger(__name__)


TRAIN_EFF = (date(2023, 1, 3),  date(2024, 12, 4))
H1        = (date(2025, 7, 1),  date(2025, 9, 30))
H2        = (date(2025, 10, 1), date(2025, 12, 31))


def load_grid(grid_dir: Path) -> dict:
    """Returns {label: [(seed, p_h1, p_h2)]}."""
    out: dict = {}
    for run_dir in sorted(grid_dir.glob("*_seed*/")):
        # name like labels_A_t3_seed42
        parts = run_dir.name.rsplit("_seed", 1)
        label, seed = parts[0], int(parts[1])
        npz = np.load(run_dir / "predictions.npz")
        out.setdefault(label, []).append((seed, npz["p_h1"], npz["p_h2"]))
    return out


def evaluate(y_true, y_pred, top_k_pcts=(0.01, 0.05, 0.10)):
    pr = average_precision_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred)
    base = y_true.mean()
    out = {"pr_auc": float(pr), "roc_auc": float(roc), "base_rate": float(base)}
    for pct in top_k_pcts:
        k = max(int(len(y_true) * pct), 1)
        idx = np.argsort(-y_pred)[:k]
        prec = y_true[idx].mean()
        out[f"top{int(pct*100)}pct_precision"] = float(prec)
        out[f"top{int(pct*100)}pct_lift"] = float(prec / base) if base > 0 else 0.0
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    BUNDLE = Path("data/p3_4070")
    GRID = Path("runs/sl_grid")
    OUT = Path("runs/sl_ensemble")
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Need true labels for H1/H2. Use labels_A_t3 (paris's prod label) for
    #    eval — same convention as their reported PR_AUC 0.122.
    eval_label = "labels_A_t3"
    label_parts = []
    for year in (2023, 2024, 2025, 2026):
        label_parts.append(pl.read_parquet(BUNDLE / "labels" / f"{eval_label}_year={year}.parquet"))
    labels_df = pl.concat(label_parts)
    feat_df = pl.read_parquet(BUNDLE / "feature_panel_v3_344.parquet").select(["trade_date", "ts_code"])
    uni_parts = []
    for year in (2023, 2024, 2025, 2026):
        uni_parts.append(pl.read_parquet(BUNDLE / "universe_mask" / f"year={year}.parquet").select(["trade_date", "ts_code", "in_universe"]))
    uni_df = pl.concat(uni_parts)
    df = feat_df.join(labels_df, on=["trade_date", "ts_code"], how="inner")
    df = df.join(uni_df, on=["trade_date", "ts_code"], how="left")
    df = df.filter(pl.col("in_universe") == True)  # noqa: E712

    h1_df = df.filter((pl.col("trade_date") >= H1[0]) & (pl.col("trade_date") <= H1[1]))
    h2_df = df.filter((pl.col("trade_date") >= H2[0]) & (pl.col("trade_date") <= H2[1]))
    y_h1 = h1_df["y"].to_numpy().astype(np.int8)
    y_h2 = h2_df["y"].to_numpy().astype(np.int8)
    logger.info("eval rows: H1=%d (pos=%d) H2=%d (pos=%d)",
                len(y_h1), int(y_h1.sum()), len(y_h2), int(y_h2.sum()))

    # 2. Load grid + per-label seed-mean
    grid = load_grid(GRID)
    logger.info("loaded labels: %s", list(grid.keys()))

    summary = {"per_label_seed_mean": {}, "label_ensemble_rank": {}}

    label_seedmean_h1 = {}
    label_seedmean_h2 = {}
    for lbl, runs in grid.items():
        ps_h1 = np.stack([r[1] for r in runs])  # (n_seeds, N_h1)
        ps_h2 = np.stack([r[2] for r in runs])  # (n_seeds, N_h2)
        avg_h1 = ps_h1.mean(axis=0)
        avg_h2 = ps_h2.mean(axis=0)
        label_seedmean_h1[lbl] = avg_h1
        label_seedmean_h2[lbl] = avg_h2
        eh1 = evaluate(y_h1, avg_h1)
        eh2 = evaluate(y_h2, avg_h2)
        summary["per_label_seed_mean"][lbl] = {"h1": eh1, "h2": eh2}
        logger.info("[seed-mean] %s  H1 PR=%.4f top1=%.2fx | H2 PR=%.4f top1=%.2fx",
                    lbl, eh1["pr_auc"], eh1["top1pct_lift"],
                    eh2["pr_auc"], eh2["top1pct_lift"])

    # 3. Label ensemble: rank-rescale each label's seed-mean, then average.
    # Rank rescale: per-label scores → rank in [0, 1]. Lets us average across
    # labels with different score distributions (e.g. A_t3 vs C_t3).
    def rank01(x):
        order = np.argsort(np.argsort(x))
        return order / max(len(x) - 1, 1)

    ranked_h1 = np.stack([rank01(label_seedmean_h1[l]) for l in grid])
    ranked_h2 = np.stack([rank01(label_seedmean_h2[l]) for l in grid])
    ens_h1 = ranked_h1.mean(axis=0)
    ens_h2 = ranked_h2.mean(axis=0)
    eh1 = evaluate(y_h1, ens_h1)
    eh2 = evaluate(y_h2, ens_h2)
    summary["label_ensemble_rank"] = {"h1": eh1, "h2": eh2, "labels": list(grid.keys())}
    logger.info("[label-ens]  ALL  H1 PR=%.4f top1=%.2fx | H2 PR=%.4f top1=%.2fx",
                eh1["pr_auc"], eh1["top1pct_lift"], eh2["pr_auc"], eh2["top1pct_lift"])

    # Try only A+C ensemble (best on each window)
    use_labels = ["labels_A_t3", "labels_C_t3"]
    ranked_h1 = np.stack([rank01(label_seedmean_h1[l]) for l in use_labels])
    ranked_h2 = np.stack([rank01(label_seedmean_h2[l]) for l in use_labels])
    ens_h1 = ranked_h1.mean(axis=0)
    ens_h2 = ranked_h2.mean(axis=0)
    eh1 = evaluate(y_h1, ens_h1)
    eh2 = evaluate(y_h2, ens_h2)
    summary["label_ensemble_AC"] = {"h1": eh1, "h2": eh2, "labels": use_labels}
    logger.info("[label-ens]  A+C  H1 PR=%.4f top1=%.2fx | H2 PR=%.4f top1=%.2fx",
                eh1["pr_auc"], eh1["top1pct_lift"], eh2["pr_auc"], eh2["top1pct_lift"])

    # 4. Compare against paris P2 v2 baseline_predictions on H1
    bp = pl.read_parquet(BUNDLE / "baseline_predictions.parquet")
    bp_h1 = bp.filter((pl.col("trade_date") >= H1[0]) & (pl.col("trade_date") <= H1[1]))
    bp_h1 = bp_h1.join(h1_df.select(["trade_date", "ts_code", "y"]), on=["trade_date", "ts_code"], how="inner")
    if len(bp_h1) > 0:
        baseline_eval = evaluate(
            bp_h1["y"].to_numpy().astype(np.int8),
            bp_h1["p_t3_baseline"].to_numpy(),
        )
        summary["paris_p2v2_h1_eval"] = baseline_eval
        logger.info("[paris  ]  P2v2 H1 PR=%.4f top1=%.2fx (n=%d)",
                    baseline_eval["pr_auc"], baseline_eval["top1pct_lift"], len(bp_h1))

    bp_h2 = bp.filter((pl.col("trade_date") >= H2[0]) & (pl.col("trade_date") <= H2[1]))
    bp_h2 = bp_h2.join(h2_df.select(["trade_date", "ts_code", "y"]), on=["trade_date", "ts_code"], how="inner")
    if len(bp_h2) > 0:
        baseline_eval_h2 = evaluate(
            bp_h2["y"].to_numpy().astype(np.int8),
            bp_h2["p_t3_baseline"].to_numpy(),
        )
        summary["paris_p2v2_h2_eval"] = baseline_eval_h2
        logger.info("[paris  ]  P2v2 H2 PR=%.4f top1=%.2fx (n=%d)",
                    baseline_eval_h2["pr_auc"], baseline_eval_h2["top1pct_lift"], len(bp_h2))

    (OUT / "ensemble_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote %s", OUT / "ensemble_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
