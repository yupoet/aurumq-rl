"""Path D parallel grid: 3-way concurrent LightGBM × n_jobs=8 = 24 cores.

Same combos as path_d_grid_safe (24 = 2 nl × 2 lr × 2 mdl × 3 seeds, no nl127).
Skips combos with existing results.json. ~30 min wall on 13 remaining runs.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock


logger = logging.getLogger(__name__)


CONFIGS = []
for nl in (31, 63):
    for lr in (0.03, 0.05):
        for mdl in (50, 100):
            CONFIGS.append({"num_leaves": nl, "learning_rate": lr, "min_data_in_leaf": mdl})

SEEDS = (42, 43, 44)

_print_lock = Lock()


def _config_name(c: dict, seed: int) -> str:
    return (
        f"nl{c['num_leaves']}_lr{int(c['learning_rate']*1000):03d}_"
        f"mdl{c['min_data_in_leaf']}_seed{seed}"
    )


def _run_one(idx: int, total: int, c: dict, seed: int, args) -> tuple[str, int]:
    name = _config_name(c, seed)
    out = args.out_root / name
    if (out / "results.json").exists():
        with _print_lock:
            logger.info("[%d/%d] skip %s", idx, total, name)
        return name, -1  # skip sentinel
    with _print_lock:
        logger.info("[%d/%d] BEGIN %s", idx, total, name)
    train_script = Path(__file__).resolve().parent / "path1_train.py"
    cmd = [
        sys.executable, str(train_script),
        "--bundle", str(args.bundle),
        "--feature-panel", args.feature_panel,
        "--out", str(out),
        "--seed", str(seed),
        "--num-leaves", str(c["num_leaves"]),
        "--learning-rate", str(c["learning_rate"]),
        "--min-data-in-leaf", str(c["min_data_in_leaf"]),
        "--num-iterations", str(args.num_iterations),
        "--early-stopping-rounds", str(args.early_stopping_rounds),
        "--train-start", args.train_start,
        "--train-end", args.train_end,
        "--n-jobs", str(args.n_jobs),
    ]
    log_path = out.parent / f"{name}.log"
    out.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=Path.cwd(), stdout=fh, stderr=subprocess.STDOUT).returncode
    with _print_lock:
        if rc == 0:
            logger.info("[%d/%d] DONE  %s", idx, total, name)
        else:
            logger.error("[%d/%d] FAIL  %s rc=%d (log %s)", idx, total, name, rc, log_path)
    return name, rc


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
    ap.add_argument("--workers", type=int, default=3, help="Number of parallel LightGBM trainings")
    args = ap.parse_args(argv)

    combos = [(c, s) for c in CONFIGS for s in SEEDS]
    total = len(combos)
    logger.info("path D parallel grid: %d combos × workers=%d × n_jobs=%d (=%d cores)",
                total, args.workers, args.n_jobs, args.workers * args.n_jobs)

    t_start = time.time()
    n_ok = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_run_one, i + 1, total, c, s, args) for i, (c, s) in enumerate(combos)]
        for f in as_completed(futs):
            _, rc = f.result()
            if rc == -1:
                n_skip += 1
            elif rc == 0:
                n_ok += 1
            else:
                n_fail += 1

    elapsed = time.time() - t_start
    logger.info("path D parallel grid done in %.0fs: ok=%d skip=%d fail=%d",
                elapsed, n_ok, n_skip, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
