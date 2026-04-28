#!/usr/bin/env python3
"""Run PPO with each reward_type and dump parallel run dirs.

This is a thin driver around scripts/train.py + scripts/eval_backtest.py
that produces a directory structure consumable by the web frontend's
"compare runs" view.

Usage
-----
    python scripts/compare_rewards.py \\
        --total-timesteps 50000 \\
        --data-path data/synthetic_demo.parquet \\
        --start-date 2022-01-03 --train-end 2023-06-30 \\
        --val-start 2023-07-01 --val-end 2023-12-01 \\
        --top-level-out runs/compare_rewards
"""
from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

REWARD_TYPES = ("return", "sharpe", "sortino", "mean_variance")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-timesteps", type=int, default=50_000)
    p.add_argument("--data-path", default="data/synthetic_demo.parquet")
    p.add_argument("--start-date", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--val-start", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--top-level-out", required=True, type=Path)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--universe-filter", default="all_a")
    return p.parse_args(argv)


def _run_one(reward_type: str, args: argparse.Namespace, run_dir: Path) -> int:
    train_cmd = [
        PYTHON, "scripts/train.py",
        "--algorithm", "PPO",
        "--env-type", "portfolio_weight",
        "--reward-type", reward_type,
        "--total-timesteps", str(args.total_timesteps),
        "--data-path", str(args.data_path),
        "--start-date", args.start_date,
        "--end-date", args.train_end,
        "--universe-filter", args.universe_filter,
        "--n-envs", str(args.n_envs),
        "--vec-normalize",
        "--learning-rate-schedule", "linear",
        "--out-dir", str(run_dir),
    ]
    print(f"\n=== [{reward_type}] training ===")
    print(" ".join(train_cmd))
    rc = subprocess.run(train_cmd, cwd=_root).returncode
    if rc != 0:
        return rc

    eval_cmd = [
        PYTHON, "scripts/eval_backtest.py",
        "--run-dir", str(run_dir),
        "--data-path", str(args.data_path),
        "--val-start", args.val_start,
        "--val-end", args.val_end,
        "--top-k", str(args.top_k),
        "--universe-filter", args.universe_filter,
    ]
    print(f"\n=== [{reward_type}] backtest ===")
    print(" ".join(eval_cmd))
    return subprocess.run(eval_cmd, cwd=_root).returncode


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    top_dir = args.top_level_out / f"compare_rewards_{timestamp}"
    top_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, int]] = []
    for rt in REWARD_TYPES:
        run_dir = top_dir / rt
        rc = _run_one(rt, args, run_dir)
        if rc != 0:
            failures.append((rt, rc))

    print("\n=== summary ===")
    for rt in REWARD_TYPES:
        print(f"  {rt}: {top_dir / rt}")
    if failures:
        print("FAILURES:")
        for rt, rc in failures:
            print(f"  {rt}: exit={rc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
