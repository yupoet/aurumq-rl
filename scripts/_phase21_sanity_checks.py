#!/usr/bin/env python3
"""Phase 21 architectural sanity checks.

Two checks (third is optional, deferred to Phase 22):
  1. Actor regime ablation — replace regime input with zero / batch-mean /
     date-shuffled and rescore OOS. If all variants give nearly the same
     adj Sharpe as the real regime, the actor isn't using regime.
  2. Leakage delta — diff between regime_real and regime_zero. Reported
     as a single number; the broader bucket-by-bucket leakage analysis is
     deferred to Phase 22.

Usage
-----

    python scripts/_phase21_sanity_checks.py \\
        --run-dir runs/phase21_21a_v2_drop_mkt_seed42 \\
        --checkpoint runs/phase21_21a_v2_drop_mkt_seed42/checkpoints/ppo_249856_steps.zip \\
        --data-path data/factor_panel_combined_short_2023_2026.parquet \\
        --start-date 2025-07-01 --end-date 2026-04-24 \\
        --universe-filter main_board_non_st \\
        --forward-period 10

Writes ``<run-dir>/phase21_sanity_checks.json``.
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
from aurumq_rl.gpu_env import GPUStockPickingEnv  # noqa: F401  (custom_objects)
from aurumq_rl.gpu_rollout_buffer import GPURolloutBuffer
from aurumq_rl.index_dict_rollout_buffer import IndexOnlyDictRolloutBuffer


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_with_regime(
    model,
    panel,
    regime_array_override: np.ndarray | None,
    top_k: int,
    forward_period: int,
) -> dict[str, float | int]:
    """Score every OOS date with a possibly-substituted regime input.

    Returns the trio of adj_sharpe / non_overlap / IC plus n_dates.
    """
    panel_t = torch.from_numpy(panel.factor_array).to("cuda")
    regime_arr = panel.regime_array if regime_array_override is None else regime_array_override
    regime_t = torch.from_numpy(regime_arr.astype(np.float32, copy=False)).to("cuda")
    valid = (
        ~torch.from_numpy(panel.is_st_array).to("cuda")
        & ~torch.from_numpy(panel.is_suspended_array).to("cuda")
        & (torch.from_numpy(panel.days_since_ipo_array).to("cuda") >= 60)
    ).to(dtype=torch.float32)
    n_dates = panel.factor_array.shape[0]

    portfolio_returns: list[float] = []
    ic_values: list[float] = []

    for t in range(n_dates - 1):
        obs_np = {
            "stock": panel_t[t : t + 1].detach().cpu().numpy(),
            "regime": regime_t[t : t + 1].detach().cpu().numpy(),
            "valid_mask": valid[t : t + 1].detach().cpu().numpy(),
        }
        from stable_baselines3.common.utils import obs_as_tensor

        obs_tensor = obs_as_tensor(obs_np, model.policy.device)
        with torch.no_grad():
            actions, _, _ = model.policy.forward(obs_tensor, deterministic=True)
        scores = actions.detach().cpu().numpy()[0]
        v = valid[t].detach().cpu().numpy().astype(bool)
        scores[~v] = -1e9

        if int(v.sum()) < top_k:
            continue
        top_idx = np.argsort(-scores)[:top_k]
        top_returns = panel.return_array[t, top_idx]
        portfolio_returns.append(float(top_returns.mean()))

        # Per-date IC (Spearman ~ Pearson for ranks; we use Pearson)
        if v.sum() >= 2:
            s_v = scores[v]
            r_v = panel.return_array[t, v]
            if np.std(s_v) > 1e-12 and np.std(r_v) > 1e-12:
                c = float(np.corrcoef(s_v, r_v)[0, 1])
                if np.isfinite(c):
                    ic_values.append(c)

    arr = np.asarray(portfolio_returns)
    if len(arr) < 2 or arr.std() < 1e-12:
        return {
            "adj_sharpe": 0.0,
            "non_overlap_sharpe": 0.0,
            "mean_ic": 0.0,
            "n_dates": len(arr),
        }
    annual = np.sqrt(252.0 / max(forward_period, 1))
    adj = float(arr.mean() / arr.std() * annual)

    # Non-overlap: every fp-th date subsample
    sub = arr[::forward_period]
    non_overlap = 0.0
    if len(sub) >= 2 and sub.std() > 1e-12:
        non_overlap = float(sub.mean() / sub.std() * np.sqrt(252.0))

    return {
        "adj_sharpe": adj,
        "non_overlap_sharpe": non_overlap,
        "mean_ic": float(np.mean(ic_values)) if ic_values else 0.0,
        "n_dates": len(arr),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--universe-filter", default="main_board_non_st")
    p.add_argument("--forward-period", type=int, default=10)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/ppo_final.zip. Pass an explicit "
        "checkpoint for the Phase 21A best step.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    meta = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    if "regime_factor_names" not in meta:
        raise RuntimeError(
            f"{args.run_dir / 'metadata.json'} predates Phase 21; cannot run sanity checks."
        )
    stock_factor_names = meta.get("stock_factor_names") or meta.get("factor_names")

    loader = FactorPanelLoader(parquet_path=args.data_path)
    panel = loader.load_panel(
        start_date=dt.date.fromisoformat(args.start_date),
        end_date=dt.date.fromisoformat(args.end_date),
        universe_filter=UniverseFilter(args.universe_filter),
        forward_period=args.forward_period,
        factor_names=stock_factor_names,
    )

    ckpt_path = args.checkpoint or (args.run_dir / "ppo_final.zip")
    print(f"[sanity] loading checkpoint: {ckpt_path}")
    model = PPO.load(
        str(ckpt_path),
        device="cuda",
        custom_objects={
            "rollout_buffer_class": IndexOnlyDictRolloutBuffer,
            "GPURolloutBuffer": GPURolloutBuffer,
        },
    )

    out: dict[str, object] = {}
    print(f"[sanity] scoring with REAL regime ({panel.regime_array.shape[0]} dates)...")
    out["regime_real"] = _score_with_regime(
        model, panel, None, args.top_k, args.forward_period
    )
    print(f"  adj_sharpe = {out['regime_real']['adj_sharpe']:+.3f}")

    print("[sanity] scoring with ZERO regime...")
    zero = np.zeros_like(panel.regime_array)
    out["regime_zero"] = _score_with_regime(
        model, panel, zero, args.top_k, args.forward_period
    )
    print(f"  adj_sharpe = {out['regime_zero']['adj_sharpe']:+.3f}")

    print("[sanity] scoring with BATCH-MEAN regime...")
    mean_vec = panel.regime_array.mean(axis=0, keepdims=True)
    mean_arr = np.broadcast_to(mean_vec, panel.regime_array.shape).astype(np.float32).copy()
    out["regime_batch_mean"] = _score_with_regime(
        model, panel, mean_arr, args.top_k, args.forward_period
    )
    print(f"  adj_sharpe = {out['regime_batch_mean']['adj_sharpe']:+.3f}")

    print("[sanity] scoring with SHUFFLED regime...")
    rng = np.random.default_rng(0)
    perm = rng.permutation(panel.regime_array.shape[0])
    shuffled = panel.regime_array[perm].copy()
    out["regime_shuffled"] = _score_with_regime(
        model, panel, shuffled, args.top_k, args.forward_period
    )
    print(f"  adj_sharpe = {out['regime_shuffled']['adj_sharpe']:+.3f}")

    out["leakage_summary"] = {
        "delta_adj_real_minus_zero": (
            out["regime_real"]["adj_sharpe"] - out["regime_zero"]["adj_sharpe"]
        ),
        "interpretation": (
            "If |delta| < 0.05, regime input is contributing little to actor "
            "outputs (either regime not useful, or stock encoder is leaking "
            "regime info). If |delta| >= 0.10, regime is doing meaningful "
            "work in the actor."
        ),
    }
    print(
        f"[sanity] leakage delta(real - zero) = {out['leakage_summary']['delta_adj_real_minus_zero']:+.3f}"
    )

    out_path = args.run_dir / "phase21_sanity_checks.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[sanity] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
