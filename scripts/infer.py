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

    from aurumq_rl.data_loader import (
        FactorPanelLoader,
        UniverseFilter,
        align_panel_to_training_universe,
        build_tradeable_mask,
    )
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

    # C7: scores are POSITIONAL in the training-time layout — obs rows are
    # metadata stock_codes, obs columns are metadata factor_names. Without
    # both lists the model's outputs cannot be mapped to stocks, so legacy
    # metadata is a hard error (no silent fallback to prefix discovery).
    train_factor_names = getattr(agent.metadata, "factor_names", None)
    train_stock_codes = getattr(agent.metadata, "stock_codes", None)
    if not train_factor_names or not train_stock_codes:
        print(
            "[ERROR] metadata.json is missing 'factor_names' and/or 'stock_codes' "
            "(the training-time input layout). Without them, model scores cannot "
            "be mapped to stock codes and picks would be garbage under universe "
            "drift (C7). Fix: re-export the model from a training run that "
            "records both (scripts/train.py and scripts/train_v2.py write them), "
            "or retrain; there is no safe fallback for legacy exports.",
            file=sys.stderr,
        )
        return 1
    train_factor_names = [str(c) for c in train_factor_names]
    train_stock_codes = [str(c) for c in train_stock_codes]

    # 2) Load latest panel slice with the EXACT training factor columns/order
    loader = FactorPanelLoader(parquet_path=args.data)
    panel = loader.load_panel(
        start_date=start_date,
        end_date=eval_date,
        forward_period=1,
        universe_filter=universe,
        factor_names=train_factor_names,
    )

    if eval_date not in panel.dates:
        print(
            f"[ERROR] {eval_date} not found in Parquet (covers {panel.dates[0]} to {panel.dates[-1]})",
            file=sys.stderr,
        )
        return 1

    # C7: align to the training stock universe (order + count). Stocks in the
    # training list but absent today become NaN/suspended rows (zero obs after
    # z-score) excluded by the eligibility mask below; stocks trading today
    # but unknown to the model are dropped — they have no obs slot and cannot
    # be scored.
    panel, drift = align_panel_to_training_universe(panel, train_stock_codes)
    print(
        f"[infer] aligned to training universe: kept {drift['kept']}/"
        f"{len(train_stock_codes)}, {drift['missing']} missing (zero-padded, "
        f"ineligible), dropped {drift['dropped']} stocks trading today that "
        "the model was not trained on",
        file=sys.stderr,
    )

    # Find the time index for eval_date
    t_idx = panel.dates.index(eval_date)
    obs_2d = panel.factor_array[t_idx]  # (n_train_stocks, n_factors)

    # 3) Run inference. After alignment the obs dim MUST equal the model's
    # expected dim — anything else means the metadata lists do not match this
    # export, so fail loudly (the old tail pad/truncate silently shifted every
    # stock's factor block).
    expected = tuple(agent.metadata.obs_shape)
    target_dim = int(np.prod(expected))
    if obs_2d.size != target_dim:
        print(
            f"[ERROR] obs dim mismatch after alignment: panel gives "
            f"{obs_2d.shape[0]} stocks x {obs_2d.shape[1]} factors = "
            f"{obs_2d.size}, model expects {expected} (= {target_dim}). "
            "metadata stock_codes/factor_names do not match this policy export.",
            file=sys.stderr,
        )
        return 1

    scores = agent.predict(obs_2d.reshape(expected).astype(np.float32), deterministic=True)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    # 4) Top-k by score, over the ALIGNED training stock list
    n = len(panel.stock_codes)
    if scores.shape[0] != n:
        print(
            f"[ERROR] model returned {scores.shape[0]} scores for {n} stocks; "
            "expected one score per training-universe stock.",
            file=sys.stderr,
        )
        return 1

    # C3/C4/C7: eligibility via the SHARED tradeable mask (~ST & ~suspended &
    # IPO gate & ~price-limit), same as training/backtest. Missing (delisted)
    # stocks were marked ST+suspended by the alignment, so they are excluded
    # here; -inf scores are dropped from the picks entirely.
    tradeable = build_tradeable_mask(panel)[t_idx]
    scores = np.where(tradeable, scores, -np.inf)

    sorted_idx = np.argsort(scores)[::-1]
    top_idx = [idx for idx in sorted_idx if np.isfinite(scores[idx])][: args.top_k]
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
