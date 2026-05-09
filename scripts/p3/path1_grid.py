"""Path 1 — run the 12-config × 3-seed grid sequentially.

Each config is a separate subprocess launch of path1_train.py so a single-config
crash doesn't take down the rest of the grid. Total expected wall time: 1-2h
on 36-core CPU.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path


logger = logging.getLogger(__name__)


# 12 hyperparam configs (3 num_leaves × 2 lr × 2 min_data_in_leaf)
CONFIGS = []
for nl in (31, 63, 127):
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
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--out-root", default=Path("runs/sl_path1"), type=Path)
    ap.add_argument("--num-iterations", type=int, default=2000)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0,
                    help="If > 0, only run the first N (config, seed) combos. For testing.")
    args = ap.parse_args(argv)

    py = sys.executable
    train_script = Path(__file__).resolve().parent / "path1_train.py"

    combos = [(c, s) for c in CONFIGS for s in SEEDS]
    if args.limit > 0:
        combos = combos[: args.limit]
    logger.info("grid: %d configs × %d seeds = %d combos", len(CONFIGS), len(SEEDS), len(combos))

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
            "--out", str(out),
            "--seed", str(s),
            "--num-leaves", str(c["num_leaves"]),
            "--learning-rate", str(c["learning_rate"]),
            "--min-data-in-leaf", str(c["min_data_in_leaf"]),
            "--num-iterations", str(args.num_iterations),
            "--early-stopping-rounds", str(args.early_stopping_rounds),
        ]
        rc = subprocess.run(cmd, cwd=Path.cwd()).returncode
        if rc == 0:
            n_ok += 1
        else:
            n_fail += 1
            logger.error("[%d/%d] FAIL %s (rc=%d)", i + 1, len(combos), name, rc)

    elapsed = time.time() - t_start
    logger.info("grid done in %.0fs: ok=%d skip=%d fail=%d", elapsed, n_ok, n_skip, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
