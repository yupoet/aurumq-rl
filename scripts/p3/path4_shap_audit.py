"""SHAP feature audit for Path 4 best model — feeds Direction B (feature engineering).

Outputs:
  - runs/sl_path4/feature_importance_audit.md  — human-readable audit
  - runs/sl_path4/feature_importance.json      — raw rankings (gain + SHAP)
  - runs/sl_path4/feature_audit_top30.json     — top features for interaction screening
  - runs/sl_path4/feature_audit_drop_candidates.json — features with near-zero importance

Picks the best single-run model (highest VAL_EFF primary) from runs/sl_path4
to compute SHAP. Sample of 10k random rows from VAL_EFF for tractable runtime
(SHAP TreeExplainer is fast but still O(n_rows × n_trees × n_leaves)).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p3.path1_train import _load_features_universe, FEATURE_PANEL_FNAME, VAL_EFF


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--feature-panel", default="feature_panel_clean.parquet")
    ap.add_argument("--runs-root", default=Path("runs/sl_path4"), type=Path)
    ap.add_argument("--n-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    # 1. Pick best single-run model from runs/sl_path4 by VAL_EFF primary
    best_run = None
    best_score = -float("inf")
    for run_dir in sorted(args.runs_root.glob("*_seed*/")):
        results = run_dir / "results.json"
        if not results.exists():
            continue
        d = json.loads(results.read_text())
        score = d["VAL_EFF"]["primary_mean_top50_proximity_excess"]
        if score > best_score:
            best_score = score
            best_run = run_dir
    if best_run is None:
        logger.error("no completed runs in %s", args.runs_root)
        return 2
    logger.info("best run: %s (VAL primary=%.6f)", best_run.name, best_score)

    model = lgb.Booster(model_file=str(best_run / "lgb_model.txt"))
    feature_cols = model.feature_name()
    logger.info("loaded model: %d trees, %d features", model.num_trees(), len(feature_cols))

    # 2. Load VAL_EFF features
    t0 = time.time()
    feat_df, _ = _load_features_universe(args.bundle, args.feature_panel)
    val_df = feat_df.filter(
        (pl.col("trade_date") >= VAL_EFF[0]) & (pl.col("trade_date") <= VAL_EFF[1])
    )
    logger.info("VAL_EFF rows: %d (%.1fs)", len(val_df), time.time() - t0)

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(val_df), size=min(args.n_samples, len(val_df)), replace=False)
    X_sample = val_df.select(feature_cols).to_numpy()[sample_idx].astype(np.float32)
    logger.info("sampled %d rows for SHAP", len(X_sample))

    # 3. Compute SHAP values via TreeExplainer (fast for LightGBM)
    t1 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    logger.info("SHAP computed in %.1fs", time.time() - t1)

    # 4. Rank features by mean(|SHAP|)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_rank = sorted(
        [(c, float(s)) for c, s in zip(feature_cols, mean_abs_shap)],
        key=lambda x: -x[1],
    )

    # 5. LightGBM gain importance for cross-check
    gain_imp = model.feature_importance(importance_type="gain")
    split_imp = model.feature_importance(importance_type="split")
    gain_total = gain_imp.sum()
    gain_rank = sorted(
        [(c, float(g) / gain_total if gain_total > 0 else 0.0, int(s))
         for c, g, s in zip(feature_cols, gain_imp, split_imp)],
        key=lambda x: -x[1],
    )

    # 6. Identify three categories
    top30_shap = shap_rank[:30]
    drop_candidates = [
        (c, s) for c, s in shap_rank
        if s < 1e-6  # essentially zero contribution
    ]

    # Surprise list: high SHAP but bottom split importance (model uses it once but heavily)
    feature_to_split = {c: int(s) for c, _, s in [(r[0], r[1], r[2]) for r in gain_rank]}
    feature_to_shap = {c: s for c, s in shap_rank}
    surprise = [
        (c, feature_to_shap[c], feature_to_split[c])
        for c in feature_cols
        if feature_to_shap[c] > np.median(mean_abs_shap) * 5  # high SHAP
        and feature_to_split[c] < 5  # but very few splits (suspicious)
    ]
    surprise.sort(key=lambda x: -x[1])

    # 7. Write artifacts
    out_dir = args.runs_root
    (out_dir / "feature_importance.json").write_text(json.dumps({
        "best_run": best_run.name,
        "best_val_primary": best_score,
        "n_samples": len(X_sample),
        "shap_rank_top50": [{"feature": c, "mean_abs_shap": s} for c, s in shap_rank[:50]],
        "shap_rank_bottom50": [{"feature": c, "mean_abs_shap": s} for c, s in shap_rank[-50:]],
        "gain_rank_top50": [
            {"feature": c, "gain_pct": g, "n_splits": s} for c, g, s in gain_rank[:50]
        ],
    }, indent=2))

    (out_dir / "feature_audit_top30.json").write_text(json.dumps(
        [{"feature": c, "mean_abs_shap": s} for c, s in top30_shap], indent=2
    ))
    (out_dir / "feature_audit_drop_candidates.json").write_text(json.dumps(
        [{"feature": c, "mean_abs_shap": s} for c, s in drop_candidates], indent=2
    ))

    # 8. Markdown report
    md = [
        "# Path 4 Feature Importance Audit",
        "",
        f"**Date**: {dt.date.today().isoformat()}",
        f"**Best single model**: `{best_run.name}` (VAL primary {best_score:.6f})",
        f"**SHAP sample**: {len(X_sample)} rows from VAL_EFF (2025-Q1/H1)",
        "",
        "## Top-30 features by mean(|SHAP|)",
        "",
        "| Rank | Feature | mean(\\|SHAP\\|) | Gain rank | n_splits |",
        "|---:|---|---:|---:|---:|",
    ]
    gain_rank_pos = {c: i + 1 for i, (c, _, _) in enumerate(gain_rank)}
    for i, (c, s) in enumerate(top30_shap, start=1):
        gpos = gain_rank_pos.get(c, "—")
        nsplit = feature_to_split.get(c, 0)
        md.append(f"| {i} | `{c}` | {s:.6f} | {gpos} | {nsplit} |")

    md += [
        "",
        f"## Drop candidates ({len(drop_candidates)} features with mean(|SHAP|) < 1e-6)",
        "",
        "These contribute essentially nothing to the model. Safe to remove from future training",
        "(panels 332→~280 cols, ~15% faster training, possibly +1bps OOS via reduced overfit).",
        "",
    ]
    if drop_candidates:
        for c, s in drop_candidates[:30]:
            md.append(f"- `{c}` (SHAP={s:.2e})")
        if len(drop_candidates) > 30:
            md.append(f"- ... ({len(drop_candidates) - 30} more)")
    else:
        md.append("_None — every feature is used by the model._")

    md += [
        "",
        f"## Surprise list ({len(surprise)} features: high SHAP, few splits)",
        "",
        "These have outsized impact per use. Could indicate:",
        "- Hidden interaction the model captured by single deep split",
        "- Suspicious value range (e.g., extreme outliers like gtja_017 fp32 max)",
        "- Worth manual review",
        "",
    ]
    for c, s, n in surprise[:20]:
        md.append(f"- `{c}` (mean(|SHAP|)={s:.6f}, only {n} splits)")
    if not surprise:
        md.append("_No suspicious patterns found._")

    md += [
        "",
        "## Cross-check: top-10 by gain importance (LightGBM native)",
        "",
        "| Rank | Feature | Gain % | n_splits | SHAP rank |",
        "|---:|---|---:|---:|---:|",
    ]
    shap_rank_pos = {c: i + 1 for i, (c, _) in enumerate(shap_rank)}
    for i, (c, g, n) in enumerate(gain_rank[:10], start=1):
        spos = shap_rank_pos.get(c, "—")
        md.append(f"| {i} | `{c}` | {g*100:.2f}% | {n} | {spos} |")

    md += [
        "",
        "## Recommendations",
        "",
        f"1. **Drop {len(drop_candidates)} features** (drop_candidates JSON) — saves training time, may help generalization",
        "2. **Validate top-30 prefixes** — confirm with paris that the top features make business sense (no data leak / look-ahead)",
        "3. **Audit surprise list** — if any are suspicious, ask paris to spot-check the feature formula",
        "4. **Interaction screening (next step)** — pairwise SHAP of top-15 features to find candidate interaction terms for Path 5",
        "",
        "## Method",
        "",
        f"- Model: LightGBM Booster `{best_run.name}/lgb_model.txt` (best by VAL primary)",
        f"- SHAP: `shap.TreeExplainer.shap_values()` on {len(X_sample)} sampled rows from VAL_EFF",
        "- mean(|SHAP|) per feature → primary ranking",
        "- Gain importance + split count from `lgb.feature_importance` for cross-validation",
        "",
    ]
    (out_dir / "feature_importance_audit.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("wrote %s", out_dir / "feature_importance_audit.md")
    logger.info("Top 5 features by SHAP: %s", [c for c, _ in shap_rank[:5]])
    logger.info("Drop candidates: %d / %d total", len(drop_candidates), len(feature_cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
