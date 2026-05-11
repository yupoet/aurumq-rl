"""Path D safe grid: drop nl127 configs (crash on long panel), keep nl31+nl63.

Long-panel LightGBM at nl127 + n_jobs=8 hits Windows access violation due to
per-thread buffer × tree size = too much memory. Restrict grid to:
  num_leaves ∈ {31, 63}    (2)
  learning_rate ∈ {0.03, 0.05}  (2)
  min_data_in_leaf ∈ {50, 100}  (2)
  seeds ∈ {42, 43, 44}  (3)
→ 24 runs total. ~6 min/run on long panel × 24 = ~2.4h wall.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path


logger = logging.getLogger(__name__)


CONFIGS = []
for nl in (31, 63):
    for lr in (0.03, 0.05):
        for mdl in (50, 100):
            CONFIGS.append({"num_leaves": nl, "learning_rate": lr, "min_data_in_leaf": mdl})

SEEDS = (42, 43, 44)


def _config_name(c: dict, seed: int) -> str:
    return (
        f"nl{c['num_leaves']}_lr{int(c['learning_rate']*1000):03d}_"
        f"mdl{c['min_data_in_leaf']}_seed{seed}"
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070_long", type=Path)
    ap.add_argument("--feature-panel", default="feature_target_long.parquet")
    ap.add_argument("--out-root", default=Path("runs/sl_path_d"), type=Path)
    ap.add_argument("--train-start", default="2018-01-02")
    ap.add_argument("--train-end", default="2024-12-04")
    ap.add_argument("--num-iterations", type=int, default=2000)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args(argv)

    py = sys.executable
    train_script = Path(__file__).resolve().parent / "path1_train.py"

    combos = [(c, s) for c in CONFIGS for s in SEEDS]
    logger.info("path D safe grid: %d configs × %d seeds = %d combos",
                len(CONFIGS), len(SEEDS), len(combos))

    t_start = time.time()
    n_ok = n_skip = n_fail = 0
    for i, (c, s) in enumerate(combos):
        name = _config_name(c, s)
        out = args.out_root / name
        if (out / "results.json").exists():
            logger.info("[%d/%d] skip %s (already done)", i + 1, len(combos), name)
            n_skip += 1
            continue
        logger.info("[%d/%d] BEGIN %s", i + 1, len(combos), name)
        cmd = [
            py, str(train_script),
            "--bundle", str(args.bundle),
            "--feature-panel", args.feature_panel,
            "--out", str(out),
            "--seed", str(s),
            "--num-leaves", str(c["num_leaves"]),
            "--learning-rate", str(c["learning_rate"]),
            "--min-data-in-leaf", str(c["min_data_in_leaf"]),
            "--num-iterations", str(args.num_iterations),
            "--early-stopping-rounds", str(args.early_stopping_rounds),
            "--train-start", args.train_start,
            "--train-end", args.train_end,
            "--n-jobs", str(args.n_jobs),
        ]
        rc = subprocess.run(cmd, cwd=Path.cwd()).returncode
        if rc == 0:
            n_ok += 1
        else:
            n_fail += 1
            logger.error("[%d/%d] FAIL %s (rc=%d)", i + 1, len(combos), name, rc)

    elapsed = time.time() - t_start
    logger.info("path D grid done in %.0fs: ok=%d skip=%d fail=%d", elapsed, n_ok, n_skip, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
