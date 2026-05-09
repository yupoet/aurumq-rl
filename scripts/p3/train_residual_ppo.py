"""P3 train entry — Hybrid PPO Residual on 4070.

Usage:
    # Smoke test (100k step, ~1.5h)
    python scripts/p3/train_residual_ppo.py --bundle ./data_p3 --smoke 100000

    # Full training (300k step, 5-7h, after smoke passes)
    python scripts/p3/train_residual_ppo.py --bundle ./data_p3 --total 300000 [--resume]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from aurumq_rl.p3 import (
    P3Bundle,
    ResidualPPOEnv,
    ResidualPerStockPolicy,
    load_bundle,
)
from aurumq_rl.p3.policy import (
    ACTION_RANGE,
    LOG_STD_ANNEAL_STEPS,
    LOG_STD_FINAL,
    LOG_STD_FREEZE_STEPS,
    LOG_STD_INIT,
)


logger = logging.getLogger(__name__)


TRAIN_EFF = (date(2023, 1, 3),  date(2024, 12, 4))
VAL_EFF   = (date(2025, 1, 1),  date(2025, 6, 4))
H1        = (date(2025, 7, 1),  date(2025, 9, 30))


class LogStdAnnealCallback(BaseCallback):
    """ALGORITHM_SPEC v2 §6: ln(std) frozen for 100k, then linear anneal."""

    def __init__(self, freeze_steps: int, anneal_steps: int, log_std_init: float, log_std_final: float):
        super().__init__()
        self.freeze_steps = freeze_steps
        self.anneal_steps = anneal_steps
        self.init = log_std_init
        self.final = log_std_final

    def _on_rollout_end(self) -> None:
        step = self.num_timesteps
        if step < self.freeze_steps:
            target = self.init
        else:
            progress = min((step - self.freeze_steps) / max(self.anneal_steps, 1), 1.0)
            target = self.init + (self.final - self.init) * progress
        with torch.no_grad():
            if hasattr(self.model.policy, "log_std"):
                self.model.policy.log_std.data.fill_(target)
        if self.logger is not None:
            self.logger.record("custom/log_std_target", target)
            self.logger.record("custom/std_actual", math.exp(target))

    def _on_step(self) -> bool:
        return True


class DeltaSaturationCallback(BaseCallback):
    """Track Δ saturation (>= 0.9 * ACTION_RANGE) per rollout."""

    def __init__(self):
        super().__init__()
        self._sat_history = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            if "saturation_fraction" in info:
                self._sat_history.append(info["saturation_fraction"])
        return True

    def _on_rollout_end(self) -> None:
        if self._sat_history:
            self.logger.record("custom/delta_saturation_mean", float(np.mean(self._sat_history)))
            self.logger.record("custom/delta_saturation_max", float(np.max(self._sat_history)))
            self._sat_history = []


def make_env(bundle, lo, hi, seed):
    def _f():
        env = ResidualPPOEnv(bundle, start_date=lo, end_date=hi, top_k=50, seed=seed)
        return env
    return _f


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="./data_p3")
    parser.add_argument("--smoke", type=int, default=0, help="if > 0, train this many step then stop")
    parser.add_argument("--total", type=int, default=300_000)
    parser.add_argument("--out", default="runs/p3_residual")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=2048)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load bundle
    logger.info("Loading bundle from %s ...", args.bundle)
    bundle = load_bundle(args.bundle)
    logger.info("Bundle: T=%d S=%d F=%d", bundle.n_dates, bundle.n_stocks, bundle.n_features)

    # 2. Vec env
    seeds = [args.seed + i for i in range(args.n_envs)]
    vec_env = DummyVecEnv([make_env(bundle, *TRAIN_EFF, s) for s in seeds])
    vec_env = VecMonitor(vec_env)

    # 3. Build PPO with residual policy
    total_steps = args.smoke if args.smoke > 0 else args.total
    learning_rate = 1e-4

    if args.resume and (out_dir / "ppo_latest.zip").exists():
        logger.info("Resuming from %s", out_dir / "ppo_latest.zip")
        model = PPO.load(str(out_dir / "ppo_latest.zip"), env=vec_env)
    else:
        model = PPO(
            policy=ResidualPerStockPolicy,
            env=vec_env,
            learning_rate=learning_rate,
            n_steps=args.n_steps,
            batch_size=4096,
            n_epochs=10,
            gamma=0.95,
            gae_lambda=0.95,
            clip_range=0.2,
            target_kl=0.03,
            vf_coef=0.5,
            ent_coef=0.01,
            max_grad_norm=0.5,
            seed=args.seed,
            device="cuda",
            verbose=1,
        )

        # Δ zero-init self-test
        sample_obs = torch.from_numpy(vec_env.reset()).to(model.device)
        try:
            model.policy.self_test_zero_init(sample_obs)
        except AssertionError as exc:
            logger.error("Δ zero-init failed: %s", exc)
            return 2

    # 4. Callbacks
    callbacks = [
        LogStdAnnealCallback(LOG_STD_FREEZE_STEPS, LOG_STD_ANNEAL_STEPS, LOG_STD_INIT, LOG_STD_FINAL),
        DeltaSaturationCallback(),
        CheckpointCallback(save_freq=25_000 // max(args.n_envs, 1), save_path=str(out_dir),
                           name_prefix="ppo_ckpt"),
    ]

    # 5. Train
    mode = "smoke" if args.smoke > 0 else "full"
    logger.info("Starting %s training: total_timesteps=%d, n_envs=%d, n_steps=%d",
                mode, total_steps, args.n_envs, args.n_steps)
    model.learn(total_timesteps=total_steps, callback=callbacks, reset_num_timesteps=not args.resume)
    final_path = out_dir / ("ppo_smoke.zip" if mode == "smoke" else "ppo_final.zip")
    model.save(str(final_path))
    model.save(str(out_dir / "ppo_latest.zip"))
    logger.info("Saved %s", final_path)

    # 6. Quick eval on H1
    eval_env = ResidualPPOEnv(bundle, *H1, top_k=50, seed=999)
    obs, _ = eval_env.reset()
    rewards = []
    sat_fractions = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = eval_env.step(action)
        rewards.append(r)
        sat_fractions.append(info["saturation_fraction"])
        if term or trunc:
            break
    logger.info(
        "H1 eval: mean_reward=%.5f std=%.5f delta_saturation_mean=%.4f n_steps=%d",
        float(np.mean(rewards)), float(np.std(rewards)),
        float(np.mean(sat_fractions)), len(rewards),
    )

    summary = {
        "mode": mode,
        "total_timesteps_target": total_steps,
        "h1_mean_reward": float(np.mean(rewards)),
        "h1_std_reward": float(np.std(rewards)),
        "delta_saturation_mean": float(np.mean(sat_fractions)),
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "seed": args.seed,
    }
    with (out_dir / "training_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Done. Summary: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
