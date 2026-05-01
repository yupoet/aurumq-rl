#!/usr/bin/env python3
"""GPU-vectorised training entry. Wraps PPO + GPUStockPickingEnv +
PerStockEncoderPolicy. Loads panel once on cuda. Auto-runs factor
importance after training.
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
from stable_baselines3.common.callbacks import CheckpointCallback

from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter
from aurumq_rl.gpu_env import GPUStockPickingEnv
from aurumq_rl.gpu_rollout_buffer import GPURolloutBuffer
from aurumq_rl.policy import PerStockEncoderPolicy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-timesteps", type=int, required=True)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--universe-filter", default="main_board_non_st")
    p.add_argument("--n-factors", type=int, default=None,
                   help="default None = use all available factor cols")
    p.add_argument("--n-envs", type=int, default=12)
    p.add_argument("--episode-length", type=int, default=240)
    p.add_argument("--forward-period", type=int, default=10)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--cost-bps", type=float, default=30.0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--n-steps",
        type=int,
        default=128,
        help=(
            "PPO rollout length per env. Default 128 keeps the SB3 "
            "host-RAM RolloutBuffer at ~6 GB for n_envs=12 / 3014 stocks / "
            "343 factors. Larger values (e.g. 1024) try to allocate ~47 GB "
            "and crash. See spec §5.6(b)."
        ),
    )
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--target-kl", type=float, default=0.20)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--encoder-hidden", default="128,64",
                   help="comma-separated layer sizes for the per-stock MLP hidden layers")
    p.add_argument("--encoder-out-dim", type=int, default=32)
    p.add_argument("--checkpoint-freq", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--rollout-buffer",
        choices=("gpu", "cpu", "index"),
        default="gpu",
        help=(
            "Which rollout buffer to use. 'gpu' (default) keeps every "
            "rollout tensor cuda-resident via GPURolloutBuffer; 'cpu' "
            "falls back to SB3's numpy/host-RAM RolloutBuffer for A/B "
            "comparison. 'index' uses IndexOnlyRolloutBuffer (Phase 9) "
            "which stores t-indices instead of full obs, freeing ~6 GB "
            "of VRAM and unlocking larger n_steps. See spec §5.6 P1/P2."
        ),
    )
    p.add_argument(
        "--tf32",
        action="store_true",
        default=False,
        help=(
            "Enable TF32 matmul on Ampere/Ada GPUs (Phase 14A). Sets "
            "torch.backends.cuda.matmul.allow_tf32=True, "
            "torch.backends.cudnn.allow_tf32=True. Pure speedup on fp32 "
            "GEMM kernels (~1.5-2x); not mathematically identical but "
            "no measurable training-quality impact across ML literature. "
            "Phase 13 confirmed PPO is GEMM-bound (mm+addmm 67%% CUDA)."
        ),
    )
    p.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
        help=(
            "torch.set_float32_matmul_precision setting. 'high' is the "
            "TF32-friendly mode (used in conjunction with --tf32). "
            "'medium' allows fp16 for matmul which we don't want here."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Apply matmul precision settings BEFORE constructing any cuda tensor.
    # (Phase 14A: TF32 / matmul-precision flags.)
    print(f"[train_v2] torch={torch.__version__} cuda={torch.version.cuda} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(args.matmul_precision)
        print(
            f"[train_v2] TF32 ENABLED: matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}, "
            f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}, "
            f"float32_matmul_precision={torch.get_float32_matmul_precision()}"
        )
    else:
        torch.set_float32_matmul_precision(args.matmul_precision)
        print(
            f"[train_v2] TF32 disabled: matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}, "
            f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}, "
            f"float32_matmul_precision={torch.get_float32_matmul_precision()}"
        )

    print(f"[train_v2] loading panel from {args.data_path} ({args.start_date}..{args.end_date})...")
    loader = FactorPanelLoader(parquet_path=args.data_path)
    panel = loader.load_panel(
        start_date=dt.date.fromisoformat(args.start_date),
        end_date=dt.date.fromisoformat(args.end_date),
        n_factors=args.n_factors,
        forward_period=args.forward_period,
        universe_filter=UniverseFilter(args.universe_filter),
    )
    n_dates, n_stocks, n_factors = panel.factor_array.shape
    print(f"[train_v2] panel: dates={n_dates} stocks={n_stocks} factors={n_factors}")

    panel_t = torch.from_numpy(panel.factor_array).to("cuda")
    returns_t = torch.from_numpy(panel.return_array).to("cuda")
    valid_mask = (
        ~torch.from_numpy(panel.is_st_array).to("cuda")
        & ~torch.from_numpy(panel.is_suspended_array).to("cuda")
        & (torch.from_numpy(panel.days_since_ipo_array).to("cuda") >= 60)
    )

    env = GPUStockPickingEnv(
        panel_t, returns_t, valid_mask,
        n_envs=args.n_envs,
        episode_length=args.episode_length,
        forward_period=args.forward_period,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        seed=args.seed,
    )

    encoder_hidden = tuple(int(x) for x in args.encoder_hidden.split(","))
    policy_kwargs = dict(
        encoder_hidden=encoder_hidden,
        encoder_out_dim=args.encoder_out_dim,
    )

    ppo_kwargs: dict = dict(
        policy=PerStockEncoderPolicy,
        env=env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        target_kl=args.target_kl,
        max_grad_norm=args.max_grad_norm,
        verbose=1,
        seed=args.seed,
        device="cuda",
        policy_kwargs=policy_kwargs,
    )
    if args.rollout_buffer == "gpu":
        ppo_kwargs["rollout_buffer_class"] = GPURolloutBuffer
        print("[train_v2] using GPURolloutBuffer (cuda-resident)")
    elif args.rollout_buffer == "index":
        # Pass IndexOnlyRolloutBuffer directly to PPO. Without this, SB3's
        # _setup_model would build the parent GPURolloutBuffer at the
        # full (n_steps, n_envs, n_stocks, n_factors) shape — which at
        # n_steps=1024/n_envs=16/3014/343 = 63 GiB and OOMs on cuda
        # before training starts. IndexOnlyRolloutBuffer's __init__
        # accepts the provider closures as Optional so SB3 can build
        # it; we attach them via .attach_providers() right after.
        from aurumq_rl.index_rollout_buffer import IndexOnlyRolloutBuffer
        ppo_kwargs["rollout_buffer_class"] = IndexOnlyRolloutBuffer
        print("[train_v2] using IndexOnlyRolloutBuffer (lazy obs gather)")
    else:
        print("[train_v2] using SB3 default RolloutBuffer (numpy/host-RAM)")

    model = PPO(**ppo_kwargs)

    if args.rollout_buffer == "index":
        # Provider closures bound to env's panel + last_obs_t. After this
        # the buffer is fully functional for the upcoming model.learn().
        model.rollout_buffer.attach_providers(
            obs_provider=lambda t: env.panel[t],
            obs_index_provider=lambda: env.last_obs_t,
        )
        print("[train_v2] index buffer providers attached")

    callbacks = []
    if args.checkpoint_freq > 0:
        cp_freq_per_env = max(args.checkpoint_freq // args.n_envs, 1)
        callbacks.append(CheckpointCallback(
            save_freq=cp_freq_per_env,
            save_path=str(args.out_dir / "checkpoints"),
            name_prefix="ppo",
        ))

    # Live training metrics → training_metrics.jsonl, which the dashboard
    # tails over SSE for live training-curve panels at
    # http://localhost:3000/runs/<id>. Without this callback the dashboard
    # shows "No metrics yet. N rows total." even after training finishes.
    from aurumq_rl.sb3_callbacks import WandbMetricsCallback
    from aurumq_rl.wandb_integration import WandbLogger

    metrics_path = args.out_dir / "training_metrics.jsonl"
    wandb_logger = WandbLogger(project="aurumq-rl", run_name=args.out_dir.name,
                               mode="disabled")
    callbacks.append(WandbMetricsCallback(
        wandb_logger=wandb_logger,
        jsonl_path=metrics_path,
        log_freq=max(args.n_envs * args.n_steps // 4, 1),
    ))
    print(f"[train_v2] live metrics jsonl: {metrics_path}")

    print(f"[train_v2] training for {args.total_timesteps:,} steps (n_envs={args.n_envs})...")
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks or None)

    final_path = args.out_dir / "ppo_final.zip"
    model.save(str(final_path))
    print(f"[train_v2] final model saved: {final_path}")

    metadata = {
        "algorithm": "PPO",
        "framework": "gpu_v2",
        "policy_class": "PerStockEncoderPolicy",
        "training_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "obs_shape": [n_stocks, n_factors],
        "action_shape": [n_stocks],
        "factor_count": n_factors,
        "stock_codes": panel.stock_codes,
        "factor_names": panel.factor_names,
        "train_start_date": args.start_date,
        "train_end_date": args.end_date,
        "universe": args.universe_filter,
        "encoder_hidden": list(encoder_hidden),
        "encoder_out_dim": args.encoder_out_dim,
        "top_k": args.top_k,
        "forward_period": args.forward_period,
        "rollout_buffer": args.rollout_buffer,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"[train_v2] metadata saved: {args.out_dir / 'metadata.json'}")

    # training_summary.json — what the dashboard's runs index reads
    # (web/lib/runs.ts walkRunDirs requires either training_summary.json
    # or training_metrics.jsonl; without one of these the run is invisible
    # on http://localhost:3000/).
    summary = {
        "algorithm": "PPO",
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "env_type": "gpu_stock_picking",
        "reward_type": "return",
        "universe_filter": args.universe_filter,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "n_factors": n_factors,
        "n_stocks": n_stocks,
        "top_k": args.top_k,
        "out_dir": str(args.out_dir),
        "onnx_path": "",
        "framework": "gpu_v2",
        "policy_class": "PerStockEncoderPolicy",
        "metrics_summary": {},
    }
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"[train_v2] training_summary saved: {args.out_dir / 'training_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
