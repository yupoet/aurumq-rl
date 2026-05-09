"""Path 3 — TabNet grid (small: 4 configs × 2 seeds = 8 runs on 4070 GPU).

DL on tabular has high variance and unclear ROI vs GBDT. Conservative grid
to bound runtime — total ~1-2h on 4070.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path


logger = logging.getLogger(__name__)


# 4 configs: 2 architecture sizes × 2 step counts
TABNET_CONFIGS = [
    {"n_d": 32, "n_a": 32, "n_steps": 3, "gamma": 1.5, "learning_rate": 2e-2},
    {"n_d": 32, "n_a": 32, "n_steps": 5, "gamma": 1.5, "learning_rate": 2e-2},
    {"n_d": 64, "n_a": 64, "n_steps": 3, "gamma": 1.5, "learning_rate": 1e-2},
    {"n_d": 64, "n_a": 64, "n_steps": 5, "gamma": 1.3, "learning_rate": 1e-2},
]

SEEDS = (42, 43)


def _name(c: dict, seed: int) -> str:
    return f"tn_d{c['n_d']}_s{c['n_steps']}_g{int(c['gamma']*10):02d}_seed{seed}"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--feature-panel", default="feature_panel_clean.parquet")
    ap.add_argument("--out-root", default=Path("runs/sl_path3"), type=Path)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    py = sys.executable
    train_script = Path(__file__).resolve().parent / "path3_train_tabnet.py"

    combos = [(c, s) for c in TABNET_CONFIGS for s in SEEDS]
    if args.limit > 0:
        combos = combos[: args.limit]
    logger.info("path3 tabnet grid: %d combos (%d configs × %d seeds)",
                len(combos), len(TABNET_CONFIGS), len(SEEDS))

    t_start = time.time()
    n_ok = n_skip = n_fail = 0
    for i, (c, s) in enumerate(combos):
        name = _name(c, s)
        out = args.out_root / name
        if (out / "results.json").exists():
            logger.info("[%d/%d] skip %s", i + 1, len(combos), name)
            n_skip += 1
            continue
        logger.info("[%d/%d] BEGIN %s", i + 1, len(combos), name)
        cmd = [
            py, str(train_script),
            "--bundle", str(args.bundle),
            "--feature-panel", args.feature_panel,
            "--out", str(out),
            "--seed", str(s),
            "--n-d", str(c["n_d"]),
            "--n-a", str(c["n_a"]),
            "--n-steps", str(c["n_steps"]),
            "--gamma", str(c["gamma"]),
            "--learning-rate", str(c["learning_rate"]),
            "--max-epochs", str(args.max_epochs),
            "--patience", str(args.patience),
            "--batch-size", str(args.batch_size),
        ]
        rc = subprocess.run(cmd, cwd=Path.cwd()).returncode
        if rc == 0:
            n_ok += 1
        else:
            n_fail += 1
            logger.error("[%d/%d] FAIL %s (rc=%d)", i + 1, len(combos), name, rc)

    elapsed = time.time() - t_start
    logger.info("path3 grid done in %.0fs: ok=%d skip=%d fail=%d", elapsed, n_ok, n_skip, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
