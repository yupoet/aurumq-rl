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
    # Phase 13 — PPO SGD perf-probe flags. None of these change training
    # semantics; they only attach instrumentation when --profile-sgd is
    # passed. Default off so existing flow is unchanged.
    p.add_argument(
        "--run-name",
        default=None,
        help=(
            "Run identifier for log/run identification. "
            "Defaults to the basename of --out-dir."
        ),
    )
    p.add_argument(
        "--profile-sgd",
        action="store_true",
        default=False,
        help=(
            "Enable Phase 13 PPO SGD per-stage timing. Adds a few hundred "
            "microseconds of cuda.synchronize() overhead per minibatch "
            "(only for the first --profile-sgd-minibatches batches)."
        ),
    )
    p.add_argument(
        "--profile-sgd-minibatches",
        type=int,
        default=20,
        help="Number of minibatches to instrument with stage timers.",
    )
    p.add_argument(
        "--profile-torch-profiler",
        type=int,
        default=0,
        help=(
            "If > 0, additionally wrap the first N minibatches in "
            "torch.profiler.profile and dump tables + chrome trace to "
            "--profile-output-dir."
        ),
    )
    p.add_argument(
        "--profile-memory",
        action="store_true",
        default=False,
        help="Forwarded to torch.profiler. Records cuda memory usage.",
    )
    p.add_argument(
        "--profile-print-every",
        type=int,
        default=10,
        help="Print running stage summary every N minibatches.",
    )
    p.add_argument(
        "--profile-output-dir",
        type=Path,
        default=None,
        help=(
            "Where to save profiler outputs. Defaults to "
            "<out-dir>/profiler."
        ),
    )
    return p.parse_args(argv)


def _write_phase13_perf_summary(
    *,
    out_dir: Path,
    profile_output_dir: Path,
    model,  # noqa: ANN001 - PPO or ProfiledPPO; both expose what we need
    args: argparse.Namespace,
    run_name: str,
) -> None:
    """Render Phase 13 perf summary to perf_summary.txt + perf_summary.json.

    When --profile-sgd was off, writes a minimal stub explaining how to
    enable profiling. When on, pulls stage_times + top profiler ops off
    the ProfiledPPO instance, runs diagnose_bottleneck +
    recommend_next_phase, and prints/saves the result.
    """
    txt_path = out_dir / "perf_summary.txt"
    json_path = out_dir / "perf_summary.json"
    if not args.profile_sgd:
        stub = (
            "Phase 13 perf-probe was OFF for this run.\n"
            "Re-run with --profile-sgd to capture stage timings.\n"
        )
        txt_path.write_text(stub, encoding="utf-8")
        json_path.write_text(
            json.dumps({"profile_sgd": False, "run_name": run_name}, indent=2),
            encoding="utf-8",
        )
        return

    from aurumq_rl.profiler_utils import (
        diagnose_bottleneck,
        get_sgd_stage_times,
        recommend_next_phase,
    )
    from aurumq_rl.ppo_profiled import write_perf_summary_json

    stage_times = get_sgd_stage_times()
    top_cuda = list(getattr(model, "_profile_top_cuda", []))
    top_cpu = list(getattr(model, "_profile_top_cpu", []))

    diagnosis = diagnose_bottleneck(stage_times, profiler_top_ops_cuda=top_cuda)
    recommendation = recommend_next_phase(diagnosis)

    config = {
        "run_name": run_name,
        "rollout_buffer": args.rollout_buffer,
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "total_timesteps": args.total_timesteps,
        "profile_sgd_minibatches": args.profile_sgd_minibatches,
        "profile_torch_profiler_n": args.profile_torch_profiler,
        "profile_memory": args.profile_memory,
        "profile_output_dir": str(profile_output_dir),
    }

    # Plain-text summary — both printed and saved. Mirrors the structure
    # so the reader can copy-paste from the terminal or open the file.
    lines: list[str] = []
    lines.append("=== Phase 13 PPO SGD Perf Summary ===")
    lines.append("Config:")
    for k, v in config.items():
        lines.append(f"  {k}={v}")
    lines.append("")
    lines.append("Stage timings (ms):")
    for name, values in stage_times.items():
        if not values:
            continue
        mean_ms = sum(values) / len(values)
        last_ms = values[-1]
        total_s = sum(values) / 1000.0
        lines.append(
            f"  {name:30s} mean={mean_ms:7.2f} last={last_ms:7.2f} "
            f"n={len(values):4d} total={total_s:6.2f}s"
        )
    if top_cuda:
        lines.append("")
        lines.append("Top-40 self_cuda ops (us):")
        for name, value in top_cuda[:40]:
            lines.append(f"  {value:12.1f}  {name}")
    if top_cpu:
        lines.append("")
        lines.append("Top-40 self_cpu ops (us):")
        for name, value in top_cpu[:40]:
            lines.append(f"  {value:12.1f}  {name}")
    lines.append("")
    lines.append("Diagnosis:")
    for k, v in diagnosis.items():
        if k == "evidence":
            continue
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"Recommendation: {recommendation}")
    text = "\n".join(lines) + "\n"
    print()
    print(text)
    txt_path.write_text(text, encoding="utf-8")

    write_perf_summary_json(
        json_path,
        config=config,
        stage_times=stage_times,
        top_cuda_ops=top_cuda,
        top_cpu_ops=top_cpu,
        diagnosis=diagnosis,
        recommendation=recommendation,
    )
    print(f"[train_v2] perf summary saved: {txt_path} + {json_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve profile output dir up front; even when --profile-sgd is off
    # we keep the path resolution deterministic so downstream tooling can
    # rely on it. mkdir is deferred to first write so the dir is only
    # created when something is actually written there.
    profile_output_dir = args.profile_output_dir or (args.out_dir / "profiler")

    # --run-name defaults to basename of --out-dir to keep the existing
    # dashboard/run-id behaviour stable.
    run_name = args.run_name or args.out_dir.name

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

    # Phase 13: switch to ProfiledPPO when --profile-sgd is on. Behaves
    # identically to SB3's PPO when the flag is off; the only reason to
    # always use ProfiledPPO would be if we wanted profiling on by
    # default — we don't. Keeping the conditional preserves the existing
    # codepath byte-for-byte for non-profile runs.
    if args.profile_sgd:
        from aurumq_rl.ppo_profiled import ProfiledPPO
        ppo_kwargs.update(
            profile_sgd=True,
            profile_sgd_minibatches=args.profile_sgd_minibatches,
            profile_torch_profiler_n=args.profile_torch_profiler,
            profile_memory=args.profile_memory,
            profile_print_every=args.profile_print_every,
            profile_output_dir=profile_output_dir,
        )
        print(
            "[train_v2] PHASE 13 perf-probe ON: "
            f"minibatches={args.profile_sgd_minibatches} "
            f"torch_profiler_n={args.profile_torch_profiler} "
            f"output_dir={profile_output_dir}"
        )
        model = ProfiledPPO(**ppo_kwargs)
    else:
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

    print(f"[train_v2] training for {args.total_timesteps:,} steps (n_envs={args.n_envs}, run={run_name})...")
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks or None)

    # Phase 13: emit perf summary (text + JSON) after training. Always
    # write the empty file when --profile-sgd is off so downstream tools
    # have a stable artefact to look for; the body just notes "profiling
    # was off". When --profile-sgd is on, render full diagnosis +
    # recommendation.
    _write_phase13_perf_summary(
        out_dir=args.out_dir,
        profile_output_dir=profile_output_dir,
        model=model,
        args=args,
        run_name=run_name,
    )

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
