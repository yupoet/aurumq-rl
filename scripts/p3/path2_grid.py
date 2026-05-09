"""Path 2 — run CatBoost + XGBoost grids on the clean Path 4 panel.

6 hyperparam configs × 3 seeds × 2 model classes = 36 runs total.
Sequential subprocess invocation (each model uses all 36 cores via n_jobs=-1).
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path


logger = logging.getLogger(__name__)


# CatBoost: 3 depths × 2 lr = 6 configs
CATBOOST_CONFIGS = []
for depth in (6, 8, 10):
    for lr in (0.03, 0.05):
        CATBOOST_CONFIGS.append({"depth": depth, "learning_rate": lr})

# XGBoost: 3 max_depth × 2 lr = 6 configs
XGBOOST_CONFIGS = []
for md in (6, 8, 10):
    for lr in (0.03, 0.05):
        XGBOOST_CONFIGS.append({"max_depth": md, "learning_rate": lr})

SEEDS = (42, 43, 44)


def _name(model_class: str, c: dict, seed: int) -> str:
    if model_class == "catboost":
        return f"cb_d{c['depth']}_lr{int(c['learning_rate']*1000):03d}_seed{seed}"
    elif model_class == "xgboost":
        return f"xgb_d{c['max_depth']}_lr{int(c['learning_rate']*1000):03d}_seed{seed}"
    raise ValueError(f"unknown model_class {model_class}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--feature-panel", default="feature_panel_clean.parquet")
    ap.add_argument("--out-root", default=Path("runs/sl_path2"), type=Path)
    ap.add_argument("--num-iterations", type=int, default=2000)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model-classes", nargs="+", default=("catboost", "xgboost"),
                    choices=("catboost", "xgboost"))
    args = ap.parse_args(argv)

    py = sys.executable
    here = Path(__file__).resolve().parent

    combos: list[tuple[str, dict, int]] = []
    if "catboost" in args.model_classes:
        for c in CATBOOST_CONFIGS:
            for s in SEEDS:
                combos.append(("catboost", c, s))
    if "xgboost" in args.model_classes:
        for c in XGBOOST_CONFIGS:
            for s in SEEDS:
                combos.append(("xgboost", c, s))
    if args.limit > 0:
        combos = combos[: args.limit]
    logger.info("path2 grid: %d combos (catboost=%d, xgboost=%d, seeds=%d)",
                len(combos), len(CATBOOST_CONFIGS), len(XGBOOST_CONFIGS), len(SEEDS))

    t_start = time.time()
    n_ok = n_skip = n_fail = 0
    for i, (mc, c, s) in enumerate(combos):
        name = _name(mc, c, s)
        out = args.out_root / name
        if (out / "results.json").exists():
            logger.info("[%d/%d] skip %s", i + 1, len(combos), name)
            n_skip += 1
            continue
        logger.info("[%d/%d] BEGIN %s", i + 1, len(combos), name)

        train_script = here / f"path2_train_{mc}.py"
        cmd = [
            py, str(train_script),
            "--bundle", str(args.bundle),
            "--feature-panel", args.feature_panel,
            "--out", str(out),
            "--seed", str(s),
            "--learning-rate", str(c["learning_rate"]),
            "--num-iterations", str(args.num_iterations),
            "--early-stopping-rounds", str(args.early_stopping_rounds),
        ]
        if mc == "catboost":
            cmd += ["--depth", str(c["depth"])]
        else:
            cmd += ["--max-depth", str(c["max_depth"])]

        rc = subprocess.run(cmd, cwd=Path.cwd()).returncode
        if rc == 0:
            n_ok += 1
        else:
            n_fail += 1
            logger.error("[%d/%d] FAIL %s (rc=%d)", i + 1, len(combos), name, rc)

    elapsed = time.time() - t_start
    logger.info("path2 grid done in %.0fs: ok=%d skip=%d fail=%d", elapsed, n_ok, n_skip, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
