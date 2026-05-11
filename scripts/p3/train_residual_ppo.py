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
from stable_baselines3.common.vec_env import VecMonitor

from aurumq_rl.index_rollout_buffer import IndexOnlyRolloutBuffer
from aurumq_rl.p3 import (
    P3Bundle,
    ResidualPPOEnv,
    ResidualPerStockPolicy,
    load_bundle,
)
from aurumq_rl.p3.gpu_env import ResidualGPUEnv
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
    parser.add_argument("--n-steps", type=int, default=128,
                        help="ALGORITHM_SPEC v2 §5 asked for 2048 but at S=5517 the IndexOnly "
                             "gather at SGD time would still need (batch_size, S, F+2) cuda "
                             "= 15.7 GiB at batch_size=4096. 26F-v3 used n_steps=128 / "
                             "batch_size=512 successfully on the same 4070. Override with "
                             "--n-steps if running on a bigger card.")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="See --n-steps note. batch_size=512 at fp16 panel ≈ 1.96 GiB cuda "
                             "gather, fits 4070 alongside model + optim.")
    parser.add_argument("--panel-dtype", choices=["fp16", "fp32"], default="fp16",
                        help="cuda obs panel dtype; fp16 halves VRAM (~3 GiB) on 4070")
    # ── Spec-deviation overrides (auto-tuned 2026-05-09 on 4070) ────
    # ALGORITHM_SPEC v2 fixed target_kl=0.03. In practice that triggers
    # SB3's "Early stopping at step 0 due to reaching max kl" on every PPO
    # update because zero-init action_net + log_std=ln(0.3) noise produces
    # a large KL on the very first SGD step. Result: n_updates accumulates
    # at ~1 per iteration instead of n_epochs * batches_per_epoch, and
    # ep_rew_mean stays flat (PPO doesn't learn). Standard PPO recipes use
    # 0.05-0.10 or no target_kl. Default raised to 0.10; pass --target-kl
    # to override (e.g. 0.03 to reproduce spec faithfully).
    parser.add_argument("--target-kl", type=float, default=0.10,
                        help="ALGORITHM_SPEC v2 said 0.03 but PPO never updated under that. "
                             "Default 0.10 lets SGD make progress; override to compare.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--log-std-init-override", type=float, default=None,
                        help="ln(std). Spec is ln(0.3)=-1.204; pass smaller (e.g. -2.3 for "
                             "std=0.1) to reduce action_dim noise that drives target_kl violations.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load bundle
    logger.info("Loading bundle from %s ...", args.bundle)
    bundle = load_bundle(args.bundle)
    logger.info("Bundle: T=%d S=%d F=%d", bundle.n_dates, bundle.n_stocks, bundle.n_features)

    # 2. Vec env: GPU-resident panel + IndexOnlyRolloutBuffer pattern.
    # SB3's default RolloutBuffer would allocate (n_steps, n_envs, n_stocks, F+2)
    # × float32 ≈ 234 GiB at the spec config (2048×16×5517×347). The IndexOnly
    # buffer stores only t-indices and lazy-gathers obs from env.panel at SGD
    # time, shrinking buffer storage by ~6 orders of magnitude.
    panel_dtype = torch.float16 if args.panel_dtype == "fp16" else torch.float32
    vec_env = ResidualGPUEnv(
        bundle,
        start_date=TRAIN_EFF[0],
        end_date=TRAIN_EFF[1],
        n_envs=args.n_envs,
        top_k=50,
        lambda_logit=1.0,
        action_range=ACTION_RANGE,
        device="cuda",
        panel_dtype=panel_dtype,
    )
    vec_env = VecMonitor(vec_env)

    # 3. Build PPO with residual policy
    total_steps = args.smoke if args.smoke > 0 else args.total
    learning_rate = args.learning_rate

    if args.resume and (out_dir / "ppo_latest.zip").exists():
        logger.info("Resuming from %s", out_dir / "ppo_latest.zip")
        model = PPO.load(str(out_dir / "ppo_latest.zip"), env=vec_env)
    else:
        model = PPO(
            policy=ResidualPerStockPolicy,
            env=vec_env,
            learning_rate=learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.95,
            gae_lambda=0.95,
            clip_range=0.2,
            target_kl=args.target_kl,
            vf_coef=0.5,
            ent_coef=args.ent_coef,
            max_grad_norm=0.5,
            seed=args.seed,
            device="cuda",
            verbose=1,
            rollout_buffer_class=IndexOnlyRolloutBuffer,
        )

        # Attach IndexOnly providers — must be done after PPO.__init__ since
        # SB3 _setup_model() builds the buffer with no extra kwargs.
        # The underlying VecEnv (unwrapped from VecMonitor) holds the panel.
        underlying = vec_env.venv if hasattr(vec_env, "venv") else vec_env
        model.rollout_buffer.attach_providers(
            obs_provider=lambda t: underlying.panel[t],
            obs_index_provider=lambda: underlying.last_obs_t,
        )
        logger.info("IndexOnlyRolloutBuffer providers attached")

        # log_std init override (post-policy-build). The policy already set
        # log_std to LOG_STD_INIT in _build(); override here when the user
        # passes --log-std-init-override (auto-tuning escape hatch).
        if args.log_std_init_override is not None:
            with torch.no_grad():
                if hasattr(model.policy, "log_std"):
                    model.policy.log_std.data.fill_(args.log_std_init_override)
            logger.info("log_std init override: %.4f (std=%.3f)",
                        args.log_std_init_override, math.exp(args.log_std_init_override))

        # Δ zero-init self-test
        sample_obs = torch.from_numpy(vec_env.reset()).to(model.device)
        try:
            model.policy.self_test_zero_init(sample_obs)
        except AssertionError as exc:
            logger.error("Δ zero-init failed: %s", exc)
            return 2

    # 4. Callbacks. log_std init follows --log-std-init-override if provided
    # (auto-tuning escape hatch), else falls back to spec value LOG_STD_INIT.
    log_std_init_actual = (
        args.log_std_init_override if args.log_std_init_override is not None else LOG_STD_INIT
    )
    callbacks = [
        LogStdAnnealCallback(LOG_STD_FREEZE_STEPS, LOG_STD_ANNEAL_STEPS, log_std_init_actual, LOG_STD_FINAL),
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

    # 6. Quick eval on H1 — use ResidualGPUEnv with same panel_dtype/clipping
    # as training. Mixing train (clipped fp16) with eval (paris's gym.Env, no
    # clip) produces NaN scores: gtja_017 reaches 3.4e38 (fp32 max) and the
    # policy was never trained to consume those magnitudes.
    eval_env = ResidualGPUEnv(
        bundle, start_date=H1[0], end_date=H1[1],
        n_envs=1, top_k=50, lambda_logit=1.0, action_range=ACTION_RANGE,
        device="cuda", panel_dtype=panel_dtype,
    )
    obs = eval_env.reset()
    rewards = []
    sat_fractions = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        eval_env.step_async(action)
        obs, r_arr, dones, infos = eval_env.step_wait()
        rewards.append(float(r_arr[0]))
        sat_fractions.append(float(infos[0]["saturation_fraction"]))
        if dones[0]:
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
