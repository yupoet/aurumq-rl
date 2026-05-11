"""Numerical-equivalence gate before P3 4070 training.

Compares ResidualGPUEnv (cuda VecEnv) vs ResidualPPOEnv (cpu gym.Env)
on identical inputs:

  - Bundle data: feature_panel, p_baseline, realized, market, in_universe
  - Same start_date, end_date, top_k, lambda_logit, action_range
  - Walks 50 random actions; checks reward/info per step

The gym.Env reference cannot run PPO at the spec config (RolloutBuffer
OOM). The parity test runs ONE env step at a time, so memory is fine.
What we verify here is the *math* — same action at same t produces the
same reward in both implementations. PPO mechanics (rollout, GAE,
clipping) are SB3's, unchanged between the two envs.

Exit 0 on parity, non-zero on first divergence.
"""
from __future__ import annotations

import logging
import sys
from datetime import date

import numpy as np
import torch

from aurumq_rl.p3 import ResidualPPOEnv, load_bundle
from aurumq_rl.p3.gpu_env import ResidualGPUEnv


logger = logging.getLogger(__name__)


# Tighter window for parity check — we only need a handful of steps.
PARITY_START = date(2023, 1, 3)
PARITY_END = date(2023, 6, 30)
TOP_K = 50
LAMBDA = 1.0
ACTION_RANGE = 0.2
N_STEPS = 50


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    bundle = load_bundle("data/p3_4070", verify_manifest=False)
    logger.info("Bundle loaded: T=%d S=%d F=%d", bundle.n_dates, bundle.n_stocks, bundle.n_features)

    # Reference: gym.Env on CPU
    ref_env = ResidualPPOEnv(
        bundle, start_date=PARITY_START, end_date=PARITY_END,
        top_k=TOP_K, lambda_logit=LAMBDA, action_range=ACTION_RANGE, seed=0,
    )
    # Probe: VecEnv on cuda, n_envs=1 for direct comparison.
    # Use fp32 panel here so we match the gym.Env reference math byte-for-byte.
    # Production training uses fp16 panel + extreme-value clipping (a small
    # deviation from the gym.Env path that prevents fp16 cast → +inf →
    # nan_to_num → 0 collapse for outlier features like gtja_017 max=3.4e38).
    test_env = ResidualGPUEnv(
        bundle, start_date=PARITY_START, end_date=PARITY_END,
        n_envs=1, top_k=TOP_K, lambda_logit=LAMBDA, action_range=ACTION_RANGE,
        device="cuda", panel_dtype=torch.float32,
    )

    ref_obs, _ = ref_env.reset()
    test_obs = test_env.reset()
    test_obs_0 = test_obs[0]  # (S, F+2) — strip env axis

    logger.info("ref obs shape=%s test obs shape=%s", ref_obs.shape, test_obs_0.shape)
    if ref_obs.shape != test_obs_0.shape:
        logger.error("OBS SHAPE MISMATCH")
        return 1

    # Compare initial obs (allow fp16 quantisation tolerance)
    diff = np.abs(ref_obs - test_obs_0)
    logger.info("init obs diff: max=%.6e mean=%.6e (fp16 quant tolerance ~1e-3)",
                diff.max(), diff.mean())
    if diff.max() > 0.1:
        logger.error("INIT OBS DIVERGES > 0.1; not just fp16 quant")
        return 2

    # Walk N_STEPS random actions, compare reward + info per step
    rng = np.random.default_rng(42)
    max_reward_diff = 0.0
    n_in_uni_mismatches = 0
    sat_diffs = []
    rewards_ref = []
    rewards_test = []
    for step_i in range(N_STEPS):
        # Sample action in [-action_range, action_range]
        action_1d = rng.uniform(-ACTION_RANGE, ACTION_RANGE, size=(bundle.n_stocks,)).astype(np.float32)

        # Reference step
        ref_obs, ref_r, ref_term, _, ref_info = ref_env.step(action_1d)

        # Test step (broadcast to (n_envs=1, S))
        test_env.step_async(action_1d[None, :])
        test_obs, test_r_arr, test_done, test_infos = test_env.step_wait()
        test_r = float(test_r_arr[0])
        test_info = test_infos[0]

        rdiff = abs(ref_r - test_r)
        max_reward_diff = max(max_reward_diff, rdiff)
        if ref_info["n_in_universe"] != test_info["n_in_universe"]:
            n_in_uni_mismatches += 1
        sat_diffs.append(abs(ref_info["saturation_fraction"] - test_info["saturation_fraction"]))
        rewards_ref.append(ref_r)
        rewards_test.append(test_r)
        if step_i < 5 or rdiff > 1e-4:
            logger.info(
                "  step %02d  ref_r=%+.6f  test_r=%+.6f  Δr=%.2e  ref_n_uni=%d test_n_uni=%d",
                step_i, ref_r, test_r, rdiff,
                ref_info["n_in_universe"], test_info["n_in_universe"],
            )
        if ref_term:
            logger.info("ref_env terminated at step %d, stopping", step_i)
            break

    logger.info(
        "PARITY: max|Δreward|=%.2e  max|Δsat_frac|=%.2e  n_uni mismatches=%d",
        max_reward_diff, max(sat_diffs) if sat_diffs else 0.0, n_in_uni_mismatches,
    )
    logger.info(
        "  reward stats: ref mean=%+.6f std=%.6f / test mean=%+.6f std=%.6f",
        float(np.mean(rewards_ref)), float(np.std(rewards_ref)),
        float(np.mean(rewards_test)), float(np.std(rewards_test)),
    )

    # Tolerance: reward diffs come from fp16 panel quantisation propagating
    # into the score computation (logit_p in fp16 vs fp64 in numpy). Keep
    # 1e-3 tolerance since the obs panel itself only enters via the policy
    # input, not the reward path — reward path uses fp32 p_baseline_cuda.
    # But topk argpartition in numpy vs topk in cuda may select slightly
    # different ties when scores are very close. Allow 1e-3.
    if max_reward_diff > 1e-3:
        logger.error("REWARD DIVERGES > 1e-3")
        return 3
    if n_in_uni_mismatches > 0:
        logger.error("n_in_universe diverges in %d steps", n_in_uni_mismatches)
        return 4

    logger.info("PARITY PASS — gym.Env and VecEnv produce equivalent rewards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
