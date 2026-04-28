#!/usr/bin/env python3
"""AurumQ-RL training entry point.

WARNING: do NOT run actual training on a low-resource machine.
Training requires ~3GB RAM for PyTorch alone. Consumer GPUs (RTX 4070+)
or rented cloud GPUs (RTX 4090 / A10) are recommended.

Pipeline
--------
1. Load factor panel from Parquet (or synthetic data via --smoke-test).
2. Construct StockPickingEnv (or PortfolioWeightEnv for continuous weights).
3. Initialize PPO / A2C / SAC with optional SubprocVecEnv parallelism.
4. Train + checkpoint at intervals (with optional wandb upload).
5. Export ONNX (policy.onnx + metadata.json).
6. Save training summary JSONL.

Usage
-----
    # Smoke test (CPU only, no training)
    python scripts/train.py --smoke-test --out-dir /tmp/smoke

    # Real training on a single GPU
    python scripts/train.py \\
        --algorithm PPO \\
        --total-timesteps 1000000 \\
        --data-path data/factor_panel.parquet \\
        --start-date 2023-01-01 \\
        --end-date 2025-06-30 \\
        --n-envs 6 \\
        --out-dir models/ppo_v1
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

# Path setup
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))


# Constants
DEFAULT_CHECKPOINT_INTERVAL: int = 100_000
SMOKE_TEST_STEPS: int = 1_000
SMOKE_N_STOCKS: int = 50
SMOKE_N_FACTORS: int = 8
SMOKE_N_DATES: int = 60


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AurumQ-RL training entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Algorithm
    parser.add_argument(
        "--algorithm",
        choices=["PPO", "A2C", "SAC"],
        default="PPO",
        help="RL algorithm (default PPO)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total training steps (default 1M)",
    )

    # Data
    parser.add_argument(
        "--data-path",
        default="data/factor_panel.parquet",
        help="Path to input Parquet (default data/factor_panel.parquet)",
    )
    parser.add_argument(
        "--start-date",
        default="2023-01-01",
        help="Training start date (default 2023-01-01)",
    )
    parser.add_argument(
        "--end-date",
        default="2025-06-30",
        help="Training end date (default 2025-06-30)",
    )
    parser.add_argument(
        "--universe-filter",
        default="main_board_non_st",
        choices=["all_a", "main_board_non_st", "hs300", "zz500", "zz1000"],
        help="Universe filter mode (default main_board_non_st)",
    )

    # Environment
    parser.add_argument(
        "--env-type",
        choices=["stock_picking", "portfolio_weight"],
        default="stock_picking",
        help="Environment type (default stock_picking)",
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=64,
        help="Number of factor dims (default 64; auto-truncated to available columns)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of stocks to pick per step (default 30)",
    )
    parser.add_argument(
        "--forward-period",
        type=int,
        default=10,
        help="Forward-return window in trading days (default 10)",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=30.0,
        help="One-side trading cost in bps (default 30)",
    )

    # PortfolioWeight env extras
    parser.add_argument(
        "--reward-type",
        choices=["return", "sharpe", "sortino", "mean_variance"],
        default="return",
        help="Reward type (portfolio_weight env only, default return)",
    )
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=0.05,
        help="Max single-stock position pct (default 0.05)",
    )
    parser.add_argument(
        "--max-industry-pct",
        type=float,
        default=0.30,
        help="Max single-industry position pct (default 0.30)",
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=1.0,
        help="Mean-variance lambda (default 1.0)",
    )

    # Optimizer
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Initial learning rate (default 3e-4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default 42)",
    )
    parser.add_argument(
        "--vec-normalize",
        action="store_true",
        help=(
            "Wrap VecEnv with stable_baselines3.common.vec_env.VecNormalize "
            "to normalize observations and rewards. Stats are saved to "
            "<out-dir>/vec_normalize.pkl alongside the final model."
        ),
    )
    parser.add_argument(
        "--learning-rate-schedule",
        choices=["constant", "linear", "cosine"],
        default="constant",
        help=(
            "Learning rate schedule (default constant). 'linear' decays "
            "linearly from --learning-rate to 0 across training. 'cosine' "
            "decays as 0.5 * lr * (1 + cos(pi * (1 - p))) for p in [0,1]."
        ),
    )
    parser.add_argument(
        "--policy-kwargs-json",
        default="{}",
        help=(
            "JSON string passed as policy_kwargs to the SB3 algorithm. "
            'Example: \'{"net_arch": [512, 256], "activation_fn": "relu"}\'. '
            "activation_fn strings ('relu', 'tanh', 'elu', 'gelu') are mapped "
            "to torch.nn classes."
        ),
    )

    # Parallelism
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel env workers (default 1; use 6-8 for GPU training)",
    )
    parser.add_argument(
        "--vec-env-method",
        choices=["spawn", "fork"],
        default="spawn",
        help="SubprocVecEnv start method (default spawn)",
    )

    # wandb
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb tracking (default offline mode, no upload)",
    )
    parser.add_argument(
        "--wandb-online",
        action="store_true",
        help="Upload wandb data to cloud (requires wandb login)",
    )
    parser.add_argument(
        "--wandb-project",
        default="aurumq-rl",
        help="wandb project name (default aurumq-rl)",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="wandb run name (default auto-generated)",
    )

    # Checkpoints
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help=f"Checkpoint frequency in steps (default {DEFAULT_CHECKPOINT_INTERVAL})",
    )

    # Output
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for checkpoints / ONNX / metrics",
    )

    # Modes
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Smoke test mode: synthetic data + 1k steps, no real training",
    )

    return parser.parse_args(argv)


def _make_wandb_logger(args: argparse.Namespace, out_dir: Path) -> Any:
    """Build a WandbLogger from CLI args."""
    from aurumq_rl.wandb_integration import WandbLogger

    if not args.wandb:
        return WandbLogger(mode="disabled")

    mode = "online" if args.wandb_online else "offline"

    config: dict = {
        "algorithm": args.algorithm,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "n_factors": args.n_factors,
        "top_k": args.top_k,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "env_type": args.env_type,
        "reward_type": args.reward_type,
    }

    return WandbLogger(
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config=config,
        mode=mode,
        out_dir=out_dir,
    )


def make_env(
    rank: int,
    env_type: str,
    panel: Any,
    args: argparse.Namespace,
) -> Any:
    """Factory for SubprocVecEnv worker initialization."""
    seed = args.seed

    def _init() -> Any:
        if env_type == "stock_picking":
            from aurumq_rl.env import StockPickingConfig, StockPickingEnv

            config = StockPickingConfig(
                start_date=datetime.date.fromisoformat(args.start_date),
                end_date=datetime.date.fromisoformat(args.end_date),
                n_factors=panel.factor_array.shape[2],
                top_k=args.top_k,
                forward_period=args.forward_period,
                cost_bps=args.cost_bps,
            )
            env = StockPickingEnv(
                config=config,
                factor_panel=panel.factor_array,
                return_panel=panel.return_array,
                pct_change_panel=panel.pct_change_array,
                is_st_panel=panel.is_st_array,
                is_suspended_panel=panel.is_suspended_array,
                days_since_ipo_panel=panel.days_since_ipo_array,
                stock_codes=panel.stock_codes,
            )
        else:
            from aurumq_rl.portfolio_weight_env import (
                PortfolioWeightConfig,
                PortfolioWeightEnv,
            )

            config = PortfolioWeightConfig(
                start_date=datetime.date.fromisoformat(args.start_date),
                end_date=datetime.date.fromisoformat(args.end_date),
                n_factors=panel.factor_array.shape[2],
                forward_period=args.forward_period,
                reward_type=args.reward_type,
                risk_aversion=args.risk_aversion,
                cost_bps=args.cost_bps,
                max_position_pct=args.max_position_pct,
                max_industry_pct=args.max_industry_pct,
            )
            env = PortfolioWeightEnv(
                config=config,
                factor_panel=panel.factor_array,
                return_panel=panel.return_array,
                pct_change_panel=panel.pct_change_array,
                is_st_panel=panel.is_st_array,
                is_suspended_panel=panel.is_suspended_array,
                days_since_ipo_panel=panel.days_since_ipo_array,
            )

        env.reset(seed=seed + rank)
        return env

    return _init


def run_smoke_test(args: argparse.Namespace) -> int:
    """Smoke test using synthetic data."""
    import numpy as np

    from aurumq_rl.data_loader import FactorPanelLoader
    from aurumq_rl.metrics import (
        TrainingMetrics,
        append_metrics,
        load_metrics,
        summarize_metrics,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[smoke] generating synthetic panel...")
    panel = FactorPanelLoader.build_synthetic(
        n_dates=SMOKE_N_DATES,
        n_stocks=SMOKE_N_STOCKS,
        n_factors=SMOKE_N_FACTORS,
        forward_period=args.forward_period,
        seed=args.seed,
    )
    print(
        f"[smoke] panel: factors={panel.factor_array.shape}, "
        f"returns={panel.return_array.shape}"
    )

    assert panel.factor_array.ndim == 3
    assert panel.return_array.ndim == 2
    assert len(panel.dates) == SMOKE_N_DATES
    assert len(panel.stock_codes) == SMOKE_N_STOCKS
    assert len(panel.factor_names) == SMOKE_N_FACTORS

    # Mock training loop (no PyTorch)
    metrics_path = out_dir / "training_metrics.jsonl"
    n_steps = SMOKE_TEST_STEPS
    rng = np.random.default_rng(42)

    print(f"[smoke] simulating {n_steps} training steps...")
    for step in range(0, n_steps, 100):
        m = TrainingMetrics(
            timestep=step,
            episode_reward_mean=float(rng.uniform(-0.01, 0.05)),
            policy_loss=float(rng.uniform(0.001, 0.1)),
            value_loss=float(rng.uniform(0.01, 0.5)),
            entropy=float(rng.uniform(0.5, 2.0)),
            explained_variance=float(rng.uniform(0.0, 0.8)),
            learning_rate=3e-4,
            fps=int(rng.integers(100, 500)),
            algorithm=args.algorithm,
        )
        append_metrics(metrics_path, m)

    loaded = load_metrics(metrics_path)
    summary = summarize_metrics(loaded)

    summary_path = out_dir / "smoke_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "smoke_test",
                "status": "ok",
                "n_steps": n_steps,
                "panel_shape": {
                    "n_dates": panel.factor_array.shape[0],
                    "n_stocks": panel.factor_array.shape[1],
                    "n_factors": panel.factor_array.shape[2],
                },
                "metrics_summary": summary,
                "out_dir": str(out_dir),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[smoke] done — summary: {summary_path}")
    print(f"[smoke] metrics: {metrics_path} ({len(loaded)} records)")
    return 0


def run_training(args: argparse.Namespace) -> int:
    """Real training entry — requires PyTorch + SB3."""
    try:
        import torch  # noqa: F401
        from stable_baselines3 import A2C, PPO, SAC
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as e:
        print(
            f"[ERROR] missing dependency: {e}\n"
            "Training requires the [train] extra. Install with:\n"
            "  pip install aurumq-rl[train]\n"
            "On low-resource machines, use --smoke-test instead.",
            file=sys.stderr,
        )
        return 1

    from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter
    from aurumq_rl.metrics import load_metrics, summarize_metrics
    from aurumq_rl.onnx_export import export_sb3_policy_to_onnx
    from aurumq_rl.sb3_callbacks import (
        CheckpointArtifactCallback,
        WandbMetricsCallback,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = datetime.date.fromisoformat(args.start_date)
    end_date = datetime.date.fromisoformat(args.end_date)
    universe = UniverseFilter(args.universe_filter)

    # 1) Load panel
    print(f"[train] loading panel from {args.data_path} ({start_date}..{end_date})...")
    loader = FactorPanelLoader(parquet_path=args.data_path)
    panel = loader.load_panel(
        start_date=start_date,
        end_date=end_date,
        n_factors=args.n_factors,
        forward_period=args.forward_period,
        universe_filter=universe,
    )
    print(
        f"[train] panel: factors={panel.factor_array.shape}, "
        f"returns={panel.return_array.shape}, "
        f"factor_names={panel.factor_names[:5]}..."
    )

    # 2) Build a single-env first to validate via check_env
    single_env_init = make_env(0, args.env_type, panel, args)
    single_env = single_env_init()
    print("[train] validating environment...")
    check_env(single_env, warn=True)
    single_env.close()

    # 3) Build VecEnv
    n_envs = args.n_envs
    env_fns = [make_env(i, args.env_type, panel, args) for i in range(n_envs)]

    if n_envs > 1:
        print(f"[train] using SubprocVecEnv n_envs={n_envs} method={args.vec_env_method}...")
        vec_env = SubprocVecEnv(env_fns, start_method=args.vec_env_method)
    else:
        print("[train] using DummyVecEnv n_envs=1...")
        vec_env = DummyVecEnv(env_fns)

    if args.vec_normalize:
        from stable_baselines3.common.vec_env import VecNormalize

        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        print("[train] VecNormalize wrapped (norm_obs=True, norm_reward=True)")

    # 4) wandb
    wandb_logger = _make_wandb_logger(args, out_dir)

    # 5) Create model
    print(f"[train] initializing {args.algorithm} agent...")
    algo_cls = {"PPO": PPO, "A2C": A2C, "SAC": SAC}[args.algorithm]
    model = algo_cls(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(out_dir / "tb_logs"),
    )

    # 6) Callbacks
    metrics_path = out_dir / "training_metrics.jsonl"
    checkpoint_freq_per_env = max(args.checkpoint_freq // n_envs, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq_per_env,
        save_path=str(out_dir / "checkpoints"),
        name_prefix=args.algorithm.lower(),
    )
    wandb_metrics_cb = WandbMetricsCallback(
        wandb_logger=wandb_logger,
        jsonl_path=metrics_path,
        log_freq=max(1000 // n_envs, 1),
    )
    wandb_artifact_cb = CheckpointArtifactCallback(
        wandb_logger=wandb_logger,
        save_path=out_dir / "checkpoints",
        name_prefix=args.algorithm.lower(),
        save_freq=checkpoint_freq_per_env,
        artifact_type="model",
    )

    # 7) Train
    print(f"[train] training for {args.total_timesteps:,} steps (n_envs={n_envs})...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, wandb_metrics_cb, wandb_artifact_cb],
        progress_bar=True,
    )

    if args.vec_normalize:
        stats_path = out_dir / "vec_normalize.pkl"
        vec_env.save(str(stats_path))
        print(f"[train] VecNormalize stats saved: {stats_path}")

    vec_env.close()

    # 8) Save final model
    final_model_path = out_dir / f"{args.algorithm.lower()}_final.zip"
    model.save(str(final_model_path))
    print(f"[train] final model saved: {final_model_path}")

    wandb_logger.log_artifact(
        path=final_model_path,
        name=f"{args.algorithm.lower()}_final",
        artifact_type="model",
        metadata={"total_timesteps": args.total_timesteps},
    )

    # 9) Export ONNX
    n_stocks = panel.factor_array.shape[1]
    n_factors = panel.factor_array.shape[2]

    if args.env_type == "stock_picking":
        obs_shape = (n_stocks * n_factors,)
    else:
        obs_shape = (n_stocks * (n_factors + 1),)
    print(f"[train] exporting ONNX (obs_shape={obs_shape})...")

    metrics_data = load_metrics(metrics_path)
    summary = summarize_metrics(metrics_data)
    final_reward = summary.get("final_reward")

    onnx_path = export_sb3_policy_to_onnx(
        model_path=final_model_path,
        output_dir=out_dir,
        obs_shape=obs_shape,
        training_timesteps=args.total_timesteps,
        final_reward=final_reward,
        extra_metadata={
            "universe": args.universe_filter,
            "env_type": args.env_type,
            "reward_type": args.reward_type,
            "top_k": args.top_k,
            "factor_count": n_factors,
        },
    )
    print(f"[train] ONNX exported: {onnx_path}")

    wandb_logger.log_artifact(
        path=onnx_path,
        name=f"{args.algorithm.lower()}_policy_onnx",
        artifact_type="model",
    )

    # 10) Summary
    summary_path = out_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "algorithm": args.algorithm,
                "total_timesteps": args.total_timesteps,
                "n_envs": n_envs,
                "env_type": args.env_type,
                "reward_type": args.reward_type,
                "universe_filter": args.universe_filter,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "n_factors": n_factors,
                "n_stocks": n_stocks,
                "top_k": args.top_k,
                "out_dir": str(out_dir),
                "onnx_path": str(onnx_path),
                "metrics_summary": summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[train] training complete — summary: {summary_path}")

    wandb_logger.finish()
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry."""
    args = parse_args(argv)
    if args.smoke_test:
        print("[aurumq-rl] smoke test mode (synthetic data, no real training)")
        return run_smoke_test(args)
    print("[aurumq-rl] full training mode — ensure you are on a GPU machine")
    return run_training(args)


if __name__ == "__main__":
    sys.exit(main())
