"""Path 2 parallel grid: 3-way concurrent CatBoost+XGB × n_jobs=8.

Same combos as path2_grid (36 = 6 cb + 6 xgb × 3 seeds). Skips combos with
existing results.json. ~3-4h wall on long panel with 3 parallel × 8 cores.
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


CATBOOST_CONFIGS = []
for depth in (6, 8, 10):
    for lr in (0.03, 0.05):
        CATBOOST_CONFIGS.append({"depth": depth, "learning_rate": lr})

XGBOOST_CONFIGS = []
for md in (6, 8, 10):
    for lr in (0.03, 0.05):
        XGBOOST_CONFIGS.append({"max_depth": md, "learning_rate": lr})

SEEDS = (42, 43, 44)
_print_lock = Lock()


def _name(model_class: str, c: dict, seed: int) -> str:
    if model_class == "catboost":
        return f"cb_d{c['depth']}_lr{int(c['learning_rate']*1000):03d}_seed{seed}"
    return f"xgb_d{c['max_depth']}_lr{int(c['learning_rate']*1000):03d}_seed{seed}"


def _run_one(idx: int, total: int, mc: str, c: dict, seed: int, args) -> tuple[str, int]:
    name = _name(mc, c, seed)
    out = args.out_root / name
    if (out / "results.json").exists():
        with _print_lock:
            logger.info("[%d/%d] skip %s", idx, total, name)
        return name, -1
    with _print_lock:
        logger.info("[%d/%d] BEGIN %s", idx, total, name)
    here = Path(__file__).resolve().parent
    train_script = here / f"path2_train_{mc}.py"
    cmd = [
        sys.executable, str(train_script),
        "--bundle", str(args.bundle),
        "--feature-panel", args.feature_panel,
        "--out", str(out),
        "--seed", str(seed),
        "--learning-rate", str(c["learning_rate"]),
        "--num-iterations", str(args.num_iterations),
        "--early-stopping-rounds", str(args.early_stopping_rounds),
        "--n-jobs", str(args.n_jobs),
        "--train-start", args.train_start,
        "--train-end", args.train_end,
    ]
    if mc == "catboost":
        cmd += ["--depth", str(c["depth"])]
    else:
        cmd += ["--max-depth", str(c["max_depth"])]
    out.mkdir(parents=True, exist_ok=True)
    log_path = out.parent / f"{name}.log"
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
    ap.add_argument("--out-root", default=Path("runs/sl_path2_long"), type=Path)
    ap.add_argument("--train-start", default="2018-01-02")
    ap.add_argument("--train-end", default="2024-12-04")
    ap.add_argument("--num-iterations", type=int, default=2000)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--model-classes", nargs="+", default=("catboost", "xgboost"),
                    choices=("catboost", "xgboost"))
    args = ap.parse_args(argv)

    combos: list[tuple[str, dict, int]] = []
    if "catboost" in args.model_classes:
        for c in CATBOOST_CONFIGS:
            for s in SEEDS:
                combos.append(("catboost", c, s))
    if "xgboost" in args.model_classes:
        for c in XGBOOST_CONFIGS:
            for s in SEEDS:
                combos.append(("xgboost", c, s))
    total = len(combos)
    logger.info("path2 parallel grid: %d combos × workers=%d × n_jobs=%d",
                total, args.workers, args.n_jobs)

    t_start = time.time()
    n_ok = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_run_one, i + 1, total, mc, c, s, args)
                for i, (mc, c, s) in enumerate(combos)]
        for f in as_completed(futs):
            _, rc = f.result()
            if rc == -1: n_skip += 1
            elif rc == 0: n_ok += 1
            else: n_fail += 1

    elapsed = time.time() - t_start
    logger.info("path2 parallel grid done in %.0fs: ok=%d skip=%d fail=%d",
                elapsed, n_ok, n_skip, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
