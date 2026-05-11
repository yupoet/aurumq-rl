"""Ablation: train Path 1 best-config (nl31, lr=0.030, mdl=100) on multiple
train-window starts (2020, 2022) to build a sample-size learning curve.

Combined with existing Path 1 short (2023-2024) and Path 1 long (2018-2024)
this gives a 4-point curve at {2y, 3y, 5y, 7y} train windows.
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
_lock = Lock()

# Best-config for Path 1 long (per per-run scoreboard: nl31_lr030_mdl100)
CONFIG = {"num_leaves": 31, "learning_rate": 0.030, "min_data_in_leaf": 100}
SEEDS = (42, 43, 44)
WINDOWS = [
    ("2020-01-02", "2024-12-04", "5y_2020"),
    ("2022-01-02", "2024-12-04", "3y_2022"),
]


def _run(name, train_start, train_end, seed, args):
    out = args.out_root / f"{name}_seed{seed}"
    if (out / "results.json").exists():
        with _lock: logger.info("skip %s_seed%d (done)", name, seed)
        return name, -1
    with _lock: logger.info("BEGIN %s_seed%d  train=%s..%s", name, seed, train_start, train_end)
    train_script = Path(__file__).resolve().parent / "path1_train.py"
    cmd = [
        sys.executable, str(train_script),
        "--bundle", str(args.bundle),
        "--feature-panel", args.feature_panel,
        "--out", str(out),
        "--seed", str(seed),
        "--num-leaves", str(CONFIG["num_leaves"]),
        "--learning-rate", str(CONFIG["learning_rate"]),
        "--min-data-in-leaf", str(CONFIG["min_data_in_leaf"]),
        "--num-iterations", str(args.num_iterations),
        "--early-stopping-rounds", str(args.early_stopping_rounds),
        "--n-jobs", str(args.n_jobs),
        "--train-start", train_start,
        "--train-end", train_end,
    ]
    out.mkdir(parents=True, exist_ok=True)
    log_path = out.parent / f"{name}_seed{seed}.log"
    with open(log_path, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    with _lock:
        if rc == 0: logger.info("DONE  %s_seed%d", name, seed)
        else: logger.error("FAIL  %s_seed%d rc=%d", name, seed, rc)
    return name, rc


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070_long", type=Path)
    ap.add_argument("--feature-panel", default="feature_target_long_raw.parquet")
    ap.add_argument("--out-root", default=Path("runs/sl_path1_ablation"), type=Path)
    ap.add_argument("--num-iterations", type=int, default=2000)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    combos = [(name, ts, te, s) for ts, te, name in WINDOWS for s in SEEDS]
    logger.info("ablation: %d combos × workers=%d × n_jobs=%d", len(combos), args.workers, args.n_jobs)

    t0 = time.time()
    n_ok = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_run, n, ts, te, s, args) for (n, ts, te, s) in combos]
        for f in as_completed(futs):
            _, rc = f.result()
            if rc == -1: n_skip += 1
            elif rc == 0: n_ok += 1
            else: n_fail += 1
    logger.info("ablation done in %.0fs ok=%d skip=%d fail=%d", time.time()-t0, n_ok, n_skip, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
