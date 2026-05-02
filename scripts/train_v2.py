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
        "--unique-date-encoding",
        action="store_true",
        default=False,
        help=(
            "Enable Phase 14B unique-date encoding inside PerStockExtractor: "
            "detect duplicate dates within a PPO mini-batch via hashing "
            "first-stock row, encode each unique date once, broadcast back "
            "via inverse map. Phase 13 measured dup_factor ≈ 2.4 → "
            "encoder fwd+bwd should drop by ~58%%. Numerical equivalence "
            "verified by tests; gradients flow correctly through index "
            "broadcast (PyTorch sums at duplicated indices)."
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
    p.add_argument(
        "--compile-extractor",
        action="store_true",
        default=False,
        help=(
            "torch.compile the PerStockExtractor.forward (Phase 14D). "
            "Produces a fused graph for the per-stock MLP + LayerNorm + "
            "centering + pooling. Spec target: +10-30%% fps on top of TF32. "
            "If graph break warnings flood the log, the speedup is gone — "
            "rerun with --compile-mode default."
        ),
    )
    p.add_argument(
        "--compile-policy",
        action="store_true",
        default=False,
        help=(
            "torch.compile the action/value heads (action_net + value_net). "
            "Independent of --compile-extractor. May give marginal gains; "
            "extractor is the bigger target."
        ),
    )
    p.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
        help=(
            "torch.compile mode. 'reduce-overhead' (default) is best for our "
            "small encoder where Python/launch overhead matters most. "
            "'max-autotune' tries harder but compile takes 30+ seconds and "
            "PyTorch 16 can stall on it."
        ),
    )
    # ----- Phase 15: continuation + LR schedule + factor pruning -----
    p.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Phase 15B/15C: resume PPO training from an existing SB3 zip "
            "(e.g. runs/overnight_1m_phase14c/checkpoints/ppo_600000_steps.zip). "
            "Loads model + optimizer + policy weights, rebinds env, applies "
            "the new --learning-rate (and --lr-schedule), and continues "
            "training without resetting the global step counter. "
            "n_factors / encoder_hidden / encoder_out_dim must match the "
            "loaded zip — use the same data + universe filter."
        ),
    )
    p.add_argument(
        "--lr-schedule",
        choices=("constant", "linear", "cosine"),
        default="constant",
        help=(
            "Phase 15D: learning-rate schedule. 'constant' (default) keeps "
            "--learning-rate fixed; 'linear' / 'cosine' decay from "
            "--learning-rate to --learning-rate * --lr-final-frac across "
            "the run. SB3 accepts a callable f(progress_remaining)->lr; "
            "progress_remaining goes from 1.0 (start) to 0.0 (end)."
        ),
    )
    p.add_argument(
        "--lr-final-frac",
        type=float,
        default=0.1,
        help=(
            "Final fraction of --learning-rate at end of training when "
            "--lr-schedule=linear|cosine. Default 0.1 -> lr decays to 10 pct."
        ),
    )
    p.add_argument(
        "--drop-factor-prefix",
        nargs="+",
        default=None,
        help=(
            "Phase 15E: drop factor columns whose names start with any of "
            "the listed prefixes (e.g. 'mkt_'). Filtering happens after "
            "panel load, before tensors hit cuda. Recorded in metadata."
        ),
    )
    return p.parse_args(argv)


def _make_lr_callable(initial_lr: float, mode: str, final_frac: float):
    """Return either a float or a callable for SB3's learning_rate field."""
    if mode == "constant":
        return initial_lr
    final_lr = initial_lr * final_frac
    if mode == "linear":
        # progress_remaining goes 1.0 -> 0.0 across training
        def _lin(progress_remaining: float) -> float:
            return final_lr + (initial_lr - final_lr) * progress_remaining
        return _lin
    if mode == "cosine":
        import math
        def _cos(progress_remaining: float) -> float:
            # When p=1.0 -> initial_lr; when p=0.0 -> final_lr.
            # cos goes from 1 (p=1) to -1 (p=0) using arg pi*(1-p).
            return final_lr + 0.5 * (initial_lr - final_lr) * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))
        return _cos
    raise ValueError(f"unknown lr-schedule mode: {mode!r}")


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

    # Phase 15E: drop factor columns by name prefix.
    dropped_factors: list[str] = []
    if args.drop_factor_prefix:
        prefixes = tuple(args.drop_factor_prefix)
        keep_idx = [i for i, name in enumerate(panel.factor_names)
                    if not name.startswith(prefixes)]
        drop_idx = [i for i, name in enumerate(panel.factor_names)
                    if name.startswith(prefixes)]
        dropped_factors = [panel.factor_names[i] for i in drop_idx]
        if not dropped_factors:
            print(f"[train_v2] WARN: --drop-factor-prefix={prefixes} matched no columns")
        else:
            from aurumq_rl.data_loader import FactorPanel
            panel = FactorPanel(
                factor_array=panel.factor_array[:, :, keep_idx].copy(),
                return_array=panel.return_array,
                pct_change_array=panel.pct_change_array,
                is_st_array=panel.is_st_array,
                is_suspended_array=panel.is_suspended_array,
                days_since_ipo_array=panel.days_since_ipo_array,
                dates=list(panel.dates),
                stock_codes=list(panel.stock_codes),
                factor_names=[panel.factor_names[i] for i in keep_idx],
            )
            n_dates, n_stocks, n_factors = panel.factor_array.shape
            print(f"[train_v2] dropped {len(dropped_factors)} factor cols matching {prefixes}: "
                  f"{dropped_factors[:6]}{'...' if len(dropped_factors) > 6 else ''}")
            print(f"[train_v2] panel after drop: dates={n_dates} stocks={n_stocks} factors={n_factors}")

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
        unique_date=args.unique_date_encoding,
    )

    # Phase 15D: build LR (callable or float)
    lr_input = _make_lr_callable(args.learning_rate, args.lr_schedule, args.lr_final_frac)
    if args.lr_schedule != "constant":
        print(f"[train_v2] LR schedule={args.lr_schedule} initial={args.learning_rate} "
              f"final={args.learning_rate * args.lr_final_frac:.3e}")

    if args.resume_from is not None:
        # Phase 15B/15C: continuation. PPO.load restores policy, optimizer,
        # rollout_buffer_class, and policy_kwargs from the saved zip. We
        # rebind the env (panel may differ from train-time only in date range)
        # and override learning_rate per the new --learning-rate / schedule.
        if not args.resume_from.exists():
            raise FileNotFoundError(f"--resume-from path does not exist: {args.resume_from}")
        print(f"[train_v2] resuming from {args.resume_from}")
        model = PPO.load(str(args.resume_from), env=env, device="cuda")
        # Override LR.
        model.learning_rate = lr_input
        # SB3 builds .lr_schedule from .learning_rate at construction; rebuild
        # it now to honour the new value.
        from stable_baselines3.common.utils import get_schedule_fn as _get_sched
        model.lr_schedule = _get_sched(lr_input)
        # n_epochs / batch_size / target_kl / max_grad_norm are rate-of-update
        # knobs and the user passing them on the resume CLI should take effect.
        model.n_epochs = args.n_epochs
        model.batch_size = args.batch_size
        model.target_kl = args.target_kl
        model.max_grad_norm = args.max_grad_norm
        print(f"[train_v2] resumed: total_timesteps_so_far={model.num_timesteps:,}")
    else:
        ppo_kwargs: dict = dict(
            policy=PerStockEncoderPolicy,
            env=env,
            learning_rate=lr_input,
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

    # IndexOnlyRolloutBuffer always needs providers attached, whether the
    # buffer was created fresh (PPO(...)) or rebuilt during PPO.load().
    if type(model.rollout_buffer).__name__ == "IndexOnlyRolloutBuffer":
        model.rollout_buffer.attach_providers(
            obs_provider=lambda t: env.panel[t],
            obs_index_provider=lambda: env.last_obs_t,
        )
        print("[train_v2] index buffer providers attached")

    # Phase 14D: torch.compile the per-stock encoder and/or policy heads.
    # Done AFTER PPO construction so model.policy.features_extractor /
    # action_net / value_net exist. obs_provider closures call env.panel[t]
    # directly (not via policy), so the index-buffer path is unaffected.
    if args.compile_extractor and torch.cuda.is_available():
        print(f"[train_v2] torch.compile features_extractor (mode={args.compile_mode})...")
        try:
            model.policy.features_extractor = torch.compile(
                model.policy.features_extractor,
                mode=args.compile_mode,
                fullgraph=False,
                dynamic=False,
            )
            print("[train_v2] features_extractor compile call returned (lazy; first forward triggers actual compile)")
        except Exception as e:
            print(f"[train_v2] features_extractor compile FAILED: {e!r}")
            print("[train_v2] continuing without extractor compile")

    if args.compile_policy and torch.cuda.is_available():
        print(f"[train_v2] torch.compile action_net + value_net (mode={args.compile_mode})...")
        try:
            model.policy.action_net = torch.compile(
                model.policy.action_net, mode=args.compile_mode,
                fullgraph=False, dynamic=False,
            )
            model.policy.value_net = torch.compile(
                model.policy.value_net, mode=args.compile_mode,
                fullgraph=False, dynamic=False,
            )
            print("[train_v2] action_net + value_net compile calls returned (lazy)")
        except Exception as e:
            print(f"[train_v2] policy heads compile FAILED: {e!r}")
            print("[train_v2] continuing without policy compile")

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

    if args.resume_from is not None:
        # SB3 _setup_learn adds num_timesteps to total_timesteps when
        # reset_num_timesteps=False, so passing 300_000 here means "train
        # 300k more steps" — counter ends at resume_step + 300k.
        print(f"[train_v2] continuation: training {args.total_timesteps:,} more steps "
              f"(resume_step={model.num_timesteps:,}, end={model.num_timesteps + args.total_timesteps:,})")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks or None,
            reset_num_timesteps=False,
        )
    else:
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
        "lr_schedule": args.lr_schedule,
        "lr_final_frac": args.lr_final_frac if args.lr_schedule != "constant" else None,
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "dropped_factor_prefixes": list(args.drop_factor_prefix) if args.drop_factor_prefix else [],
        "dropped_factor_names": dropped_factors,
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
