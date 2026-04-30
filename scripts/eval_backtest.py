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
    import json

    import numpy as np
    import onnxruntime as ort

    from aurumq_rl.backtest import run_backtest_with_series
    from aurumq_rl.data_loader import (
        FactorPanelLoader,
        UniverseFilter,
        align_panel_to_stock_list,
    )

    args = parse_args(argv)
    onnx_path = args.run_dir / "policy.onnx"
    if not onnx_path.exists():
        print(f"[error] {onnx_path} not found", file=sys.stderr)
        return 2

    # Read training metadata first so we know the locked stock universe.
    meta_path = args.run_dir / "metadata.json"
    train_stock_codes: list[str] | None = None
    expected_obs_dim: int | None = None
    feature_group_weights: dict[str, float] | None = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        recorded = meta.get("obs_shape")
        if isinstance(recorded, list) and len(recorded) == 1:
            expected_obs_dim = int(recorded[0])
            print(f"[backtest] obs_shape from metadata: {recorded}")
        sc = meta.get("stock_codes")
        if isinstance(sc, list) and sc:
            train_stock_codes = list(sc)
            print(f"[backtest] training universe: {len(train_stock_codes)} stocks (will align)")
        # Honour per-prefix weights used at training time so OOS sees the
        # same scaling. Missing field -> no weighting (legacy runs).
        fgw = meta.get("feature_group_weights")
        if isinstance(fgw, dict) and fgw:
            try:
                feature_group_weights = {str(k): float(v) for k, v in fgw.items()}
            except (TypeError, ValueError) as e:
                print(
                    f"[backtest] WARN: ignoring malformed feature_group_weights "
                    f"in metadata.json ({e})"
                )
                feature_group_weights = None
            else:
                print(
                    f"[backtest] applying feature_group_weights from metadata: "
                    f"{feature_group_weights}"
                )

    loader = FactorPanelLoader(parquet_path=args.data_path)
    panel = loader.load_panel(
        start_date=dt.date.fromisoformat(args.val_start),
        end_date=dt.date.fromisoformat(args.val_end),
        n_factors=args.n_factors,
        forward_period=args.forward_period,
        universe_filter=UniverseFilter(args.universe_filter),
        feature_group_weights=feature_group_weights,
    )

    raw_n_dates, raw_n_stocks, _ = panel.factor_array.shape
    print(f"[backtest] panel raw: dates={raw_n_dates} stocks={raw_n_stocks}")

    # Align to the training universe (order + count) so the model's fixed
    # observation space matches. Missing stocks become zero-padded rows
    # marked is_st/is_suspended=True; new stocks (in val but not train)
    # are dropped. This is the OOS contract.
    if train_stock_codes is not None:
        raw_codes = set(panel.stock_codes)
        kept = sum(1 for c in train_stock_codes if c in raw_codes)
        missing = len(train_stock_codes) - kept
        dropped = raw_n_stocks - kept
        panel = align_panel_to_stock_list(panel, train_stock_codes)
        print(
            f"[backtest] aligned to training universe: kept {kept}/{len(train_stock_codes)}, "
            f"zero-padded {missing} missing, dropped {dropped} new"
        )

    n_dates, n_stocks, n_factors = panel.factor_array.shape
    print(f"[backtest] panel aligned: dates={n_dates} stocks={n_stocks} factors={n_factors}")

    if expected_obs_dim is None:
        expected_obs_dim = n_stocks * n_factors

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    factor_flat = panel.factor_array.reshape(n_dates, n_stocks * n_factors).astype(np.float32)
    if expected_obs_dim == n_stocks * n_factors:
        obs = factor_flat
    elif expected_obs_dim == n_stocks * (n_factors + 1):
        weights_pad = np.zeros((n_dates, n_stocks), dtype=np.float32)
        obs = np.concatenate([factor_flat, weights_pad], axis=1)
        print("[backtest] padded zero current-weights for portfolio_weight env")
    else:
        print(
            f"[error] obs_dim mismatch: panel gives {n_stocks * n_factors}, "
            f"model expects {expected_obs_dim}",
            file=sys.stderr,
        )
        return 3

    raw_out = sess.run(None, {input_name: obs})[0]

    if raw_out.ndim == 1:
        out = raw_out.reshape(n_dates, n_stocks)
    elif raw_out.shape == (n_dates, n_stocks):
        out = raw_out
    else:
        out = raw_out.reshape(n_dates, n_stocks)

    print(f"[backtest] predictions shape: {out.shape}")

    result, series = run_backtest_with_series(
        predictions=out,
        returns=panel.return_array,
        dates=panel.dates,
        top_k=args.top_k,
        n_random_simulations=args.n_random_simulations,
        random_seed=args.seed,
    )

    out_path = args.run_dir / "backtest.json"
    series_path = args.run_dir / "backtest_series.json"
    result.to_json(out_path)
    series.to_json(series_path)
    print(f"[backtest] wrote {out_path}")
    print(f"[backtest] wrote {series_path}")
    print(
        f"[backtest] IC={result.ic:+.4f} IR={result.ic_ir:+.3f} "
        f"top{args.top_k}_Sharpe={result.top_k_sharpe:+.3f} "
        f"vs random p50 {result.random_baseline['p50_sharpe']:+.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
