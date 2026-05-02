#!/usr/bin/env python3
"""Post-training: load a PerStockEncoderPolicy, run IG + permutation
importance on the OOS panel, write runs/<id>/factor_importance.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import numpy as np
import torch
from stable_baselines3 import PPO

from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter
from aurumq_rl.factor_importance import integrated_gradients, permutation_importance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--data-path", required=True, type=Path)
    p.add_argument("--val-start", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--forward-period", type=int, default=10)
    p.add_argument("--universe-filter", default="main_board_non_st")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--ig-alpha-steps", type=int, default=50)
    p.add_argument("--ig-batch-size", type=int, default=8)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    meta = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    n_factors = int(meta["factor_count"])
    train_factor_names = meta.get("factor_names")
    if isinstance(train_factor_names, list) and train_factor_names:
        train_factor_names = [str(c) for c in train_factor_names]
    else:
        train_factor_names = None
    if isinstance(meta.get("forward_period"), int):
        args.forward_period = int(meta["forward_period"])
        print(f"[importance] forward_period={args.forward_period} (from metadata)")

    loader = FactorPanelLoader(parquet_path=args.data_path)
    panel = loader.load_panel(
        start_date=dt.date.fromisoformat(args.val_start),
        end_date=dt.date.fromisoformat(args.val_end),
        n_factors=n_factors if train_factor_names is None else None,
        forward_period=args.forward_period,
        universe_filter=UniverseFilter(args.universe_filter),
        factor_names=train_factor_names,
    )

    panel_t = torch.from_numpy(panel.factor_array).to("cuda")
    returns_t = torch.from_numpy(panel.return_array).to("cuda")

    final_model_path = next(args.run_dir.glob("*_final.zip"))
    model = PPO.load(str(final_model_path), device="cuda")
    model.policy.eval()

    def score_fn(obs: torch.Tensor) -> torch.Tensor:
        feats = model.policy.features_extractor(obs)
        return model.policy.action_net(feats["per_stock"]).squeeze(-1)

    # IG: take a stratified sample of dates for the batch
    sample_idx = np.linspace(0, panel_t.shape[0] - 1, args.ig_batch_size).astype(int)
    ig_batch = panel_t[sample_idx]   # (B, S, F)
    saliency = integrated_gradients(score_fn, ig_batch, n_alpha_steps=args.ig_alpha_steps).cpu().numpy()
    saliency_per_factor = {name: float(saliency[i]) for i, name in enumerate(panel.factor_names)}

    # Permutation
    perm_out = permutation_importance(
        score_fn=score_fn,
        val_panel=panel_t,
        val_returns=returns_t,
        factor_names=panel.factor_names,
        forward_period=args.forward_period,
        top_k=args.top_k,
        n_seeds=args.n_seeds,
    )

    # Aggregate saliency by prefix
    by_prefix: dict[str, list[float]] = {}
    for name, s in saliency_per_factor.items():
        prefix = name.split("_", 1)[0] if "_" in name else name
        by_prefix.setdefault(prefix, []).append(s)
    for prefix, drops in perm_out.items():
        sals = by_prefix.get(prefix, [])
        if sals:
            drops["saliency_mean"] = float(np.mean(sals))
            drops["saliency_max"] = float(np.max(sals))
            drops["saliency_std"] = float(np.std(sals, ddof=1) if len(sals) > 1 else 0.0)

    output = {
        "method": "integrated_gradients_v1+permutation_v1",
        "panel": str(args.data_path),
        "val_window": f"{args.val_start}..{args.val_end}",
        "saliency_per_factor": saliency_per_factor,
        "importance_per_group": perm_out,
    }
    out_path = args.run_dir / "factor_importance.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[importance] wrote {out_path}")
    print("[importance] top groups by ic_drop_mean:")
    ranked = sorted(perm_out.items(), key=lambda kv: -kv[1]["ic_drop_mean"])
    for prefix, m in ranked[:5]:
        print(f"  {prefix:8s}  ic_drop={m['ic_drop_mean']:+.4f}  n_factors={m['n_factors']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
