#!/usr/bin/env python3
"""AurumQ-RL inference entry point.

Loads a trained ONNX policy and outputs the top-k stock picks for a given date.

CPU-only — no PyTorch required. Uses onnxruntime + the FactorPanelLoader to
fetch the latest cross-section from a Parquet file.

Usage
-----
    python scripts/infer.py \\
        --model models/ppo_v1/ \\
        --data data/factor_panel.parquet \\
        --date 2025-12-30 \\
        --top-k 30
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

# Path setup
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="AurumQ-RL inference: top-k stock picks from a trained ONNX policy",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model directory (containing policy.onnx + metadata.json)",
    )
    parser.add_argument(
        "--data",
        default="data/factor_panel.parquet",
        help="Path to factor panel Parquet (default data/factor_panel.parquet)",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Evaluation date (YYYY-MM-DD); must exist in the Parquet",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top picks to output (default 30)",
    )
    parser.add_argument(
        "--universe-filter",
        default="main_board_non_st",
        choices=["all_a", "main_board_non_st", "hs300", "zz500", "zz1000"],
        help="Universe filter (default main_board_non_st)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output JSON path (default stdout)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Days to load before --date (for normalization context, default 30)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry."""
    args = parse_args(argv)

    import numpy as np

    from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter
    from aurumq_rl.inference import RlAgentInference

    eval_date = datetime.date.fromisoformat(args.date)
    start_date = eval_date - datetime.timedelta(days=args.lookback_days)
    universe = UniverseFilter(args.universe_filter)

    # 1) Load agent
    agent = RlAgentInference(args.model)
    print(
        f"[infer] loaded {agent.metadata.algorithm} agent "
        f"(trained for {agent.metadata.training_timesteps:,} steps)",
        file=sys.stderr,
    )

    # 2) Load latest panel slice
    loader = FactorPanelLoader(parquet_path=args.data)
    panel = loader.load_panel(
        start_date=start_date,
        end_date=eval_date,
        n_factors=agent.metadata.factor_count or None,
        forward_period=1,
        universe_filter=universe,
    )

    if eval_date not in panel.dates:
        print(
            f"[ERROR] {eval_date} not found in Parquet (covers {panel.dates[0]} to {panel.dates[-1]})",
            file=sys.stderr,
        )
        return 1

    # Find the time index for eval_date
    t_idx = panel.dates.index(eval_date)
    obs_2d = panel.factor_array[t_idx]  # (n_stocks, n_factors)

    # 3) Run inference
    obs_flat = obs_2d.reshape(-1)

    # Validate shape vs metadata
    expected = agent.metadata.obs_shape
    if obs_flat.shape != tuple(expected):
        # Try padding/truncating to match
        target_dim = expected[0] if len(expected) == 1 else int(np.prod(expected))
        actual = obs_flat.shape[0]
        if actual < target_dim:
            obs_flat = np.concatenate([obs_flat, np.zeros(target_dim - actual, dtype=np.float32)])
        else:
            obs_flat = obs_flat[:target_dim]

    scores = agent.predict(obs_flat, deterministic=True)
    if scores.ndim > 1:
        scores = scores.flatten()

    # 4) Top-k by score
    n = len(panel.stock_codes)
    if scores.shape[0] != n:
        # Truncate or pad scores to match stock count
        if scores.shape[0] > n:
            scores = scores[:n]
        else:
            scores = np.concatenate([scores, np.zeros(n - scores.shape[0], dtype=np.float32)])

    sorted_idx = np.argsort(scores)[::-1]
    top_idx = sorted_idx[: args.top_k]
    picks = [
        {
            "rank": int(i + 1),
            "stock_code": panel.stock_codes[idx],
            "score": float(scores[idx]),
            "weight": 1.0 / args.top_k,
        }
        for i, idx in enumerate(top_idx)
    ]

    output = {
        "agent_metadata": {
            "algorithm": agent.metadata.algorithm,
            "training_timesteps": agent.metadata.training_timesteps,
        },
        "eval_date": str(eval_date),
        "universe_filter": args.universe_filter,
        "n_candidates": int(n),
        "top_k": args.top_k,
        "picks": picks,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[infer] picks written to {out_path}", file=sys.stderr)
    else:
        json.dump(output, sys.stdout, indent=2, ensure_ascii=False)
        print()  # trailing newline

    return 0


if __name__ == "__main__":
    sys.exit(main())
