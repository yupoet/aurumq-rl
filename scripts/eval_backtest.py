#!/usr/bin/env python3
"""Run a backtest of a trained ONNX policy on a held-out date window.

Usage
-----
    python scripts/eval_backtest.py \\
        --run-dir runs/ppo_100k \\
        --data-path data/synthetic_demo.parquet \\
        --val-start 2023-06-01 --val-end 2023-12-01 \\
        --top-k 30 \\
        --universe-filter all_a

Writes <run-dir>/backtest.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import numpy as np

from aurumq_rl.backtest import run_backtest
from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--data-path", required=True, type=Path)
    p.add_argument("--val-start", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--universe-filter", default="all_a")
    p.add_argument("--n-factors", type=int, default=None)
    p.add_argument("--forward-period", type=int, default=10)
    p.add_argument("--n-random-simulations", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import onnxruntime as ort

    args = parse_args(argv)
    onnx_path = args.run_dir / "policy.onnx"
    if not onnx_path.exists():
        print(f"[error] {onnx_path} not found", file=sys.stderr)
        return 2

    loader = FactorPanelLoader(parquet_path=args.data_path)
    panel = loader.load_panel(
        start_date=dt.date.fromisoformat(args.val_start),
        end_date=dt.date.fromisoformat(args.val_end),
        n_factors=args.n_factors,
        forward_period=args.forward_period,
        universe_filter=UniverseFilter(args.universe_filter),
    )

    n_dates, n_stocks, n_factors = panel.factor_array.shape
    print(f"[backtest] panel: dates={n_dates} stocks={n_stocks} factors={n_factors}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    obs = panel.factor_array.reshape(n_dates, n_stocks * n_factors).astype(np.float32)
    raw_out = sess.run(None, {input_name: obs})[0]

    if raw_out.ndim == 1:
        out = raw_out.reshape(n_dates, n_stocks)
    elif raw_out.shape == (n_dates, n_stocks):
        out = raw_out
    else:
        out = raw_out.reshape(n_dates, n_stocks)

    print(f"[backtest] predictions shape: {out.shape}")

    result = run_backtest(
        predictions=out,
        returns=panel.return_array,
        top_k=args.top_k,
        n_random_simulations=args.n_random_simulations,
        random_seed=args.seed,
    )

    out_path = args.run_dir / "backtest.json"
    result.to_json(out_path)
    print(f"[backtest] wrote {out_path}")
    print(
        f"[backtest] IC={result.ic:+.4f} IR={result.ic_ir:+.3f} "
        f"top{args.top_k}_Sharpe={result.top_k_sharpe:+.3f} "
        f"vs random p50 {result.random_baseline['p50_sharpe']:+.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
