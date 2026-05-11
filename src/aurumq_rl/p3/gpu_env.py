"""GPU-vectorised version of ResidualPPOEnv.

Mirrors src/aurumq_rl/gpu_env.py:GPUStockPickingEnv pattern: a single
SB3 VecEnv with the panel resident on cuda, per-env state as small
cuda long tensors. Pairs with IndexOnlyRolloutBuffer to keep the
PPO rollout buffer at ~bytes-per-step (storing only t-indices) instead
of the (n_steps, n_envs, n_stocks, F+2) float32 explosion that tripped
the default RolloutBuffer at 234 GiB on the 4070.

Step semantics are byte-equivalent to ResidualPPOEnv (CPU gym.Env)
under matched seeds — verified by ``scripts/p3/parity_check.py``.
"""
from __future__ import annotations

import logging
from datetime import date

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv


logger = logging.getLogger(__name__)


def _logit_t(p: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


class ResidualGPUEnv(VecEnv):
    """Single-process VecEnv for residual PPO with cuda-resident panel.

    Per (env, t) the obs is ``concat([feature_panel[t], p_baseline[t],
    rank_pct[t]], dim=-1)`` of shape ``(n_stocks, F+2)``. We pre-build
    this concatenation once on cuda; ``last_obs_t`` and ``self.t`` are
    cuda long tensors of shape ``(n_envs,)`` so the rollout buffer can
    record a snapshot per add().

    All envs walk the same ``[start_idx, end_idx)`` window in lockstep
    (matches the original DummyVecEnv-of-gym.Env semantics where seeds
    only affected per-env RNG never read by step()). For real PPO
    training this gives n_envs identical trajectories which still helps
    PPO via 16x larger rollout per iter.
    """

    def __init__(
        self,
        bundle,                      # P3Bundle
        start_date: date,
        end_date: date,
        n_envs: int = 16,
        top_k: int = 50,
        lambda_logit: float = 1.0,
        action_range: float = 0.2,
        device: str = "cuda",
        panel_dtype: torch.dtype = torch.float16,
    ) -> None:
        all_dates = bundle.trade_dates
        start_idx = next(i for i, d in enumerate(all_dates) if d >= start_date)
        end_idx = next((i for i, d in enumerate(all_dates) if d > end_date), len(all_dates))
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.n_stocks = bundle.n_stocks
        self.n_features = bundle.n_features
        self.top_k = top_k
        self.lambda_logit = lambda_logit
        self.action_range = action_range
        self.device = torch.device(device)

        # Pre-build the obs panel on cuda. fp16 halves VRAM:
        # (T=804, S=5517, F+2=347) × 2 bytes ≈ 3.08 GiB on a 4070.
        # PerStockEncoderPolicy already handles fp16→fp32 cast.
        #
        # Panel values include unsanitized factor outputs (gtja_017 reaches
        # 3.4e38, alpha_083 ±3e4, gtja_011 ±1e10) which overflow fp16
        # (max 65504). Build in fp32, sanitize (nan→0 + clip ±FP16_MAX),
        # THEN cast to fp16. Without this, fp16 cast turns extreme valid
        # values into +inf which post-nan_to_num go to 0 — divergence vs
        # the gym.Env reference path. ALGORITHM_SPEC v2 §5 says fp32 feature
        # panel; this is the 4070-fit substitute.
        F = bundle.n_features
        feats = torch.from_numpy(bundle.feature_panel).to(self.device, dtype=torch.float32)
        p = torch.from_numpy(bundle.p_baseline).to(self.device, dtype=torch.float32).unsqueeze(-1)
        r = torch.from_numpy(bundle.rank_pct_baseline).to(self.device, dtype=torch.float32).unsqueeze(-1)
        obs_panel = torch.cat([feats, p, r], dim=-1)
        del feats, p, r
        # Sanitize in fp32 first.
        obs_panel = torch.nan_to_num(obs_panel, nan=0.0, posinf=0.0, neginf=0.0)
        if panel_dtype == torch.float16:
            FP16_MAX = 65504.0
            obs_panel = obs_panel.clamp_(-FP16_MAX * 0.99, FP16_MAX * 0.99)
        obs_panel = obs_panel.to(dtype=panel_dtype)
        self.panel = obs_panel  # IndexOnlyRolloutBuffer reads via env.panel[t]

        # Reward / score inputs (fp32 for numerical accuracy at top_k).
        self.p_baseline_cuda = torch.from_numpy(bundle.p_baseline).to(self.device, dtype=torch.float32)
        self.realized_cuda = torch.from_numpy(bundle.realized_pct_t_plus_1).to(self.device, dtype=torch.float32)
        self.market_cuda = torch.from_numpy(bundle.market_pct_t_plus_1).to(self.device, dtype=torch.float32)
        self.in_universe_cuda = torch.from_numpy(bundle.in_universe).to(self.device, dtype=torch.bool)

        # Per-env state on cuda
        self.t = torch.full((n_envs,), self.start_idx, dtype=torch.long, device=self.device)
        # last_obs_t mirrors the t-index of the obs most recently emitted.
        # IndexOnlyRolloutBuffer.add() reads this snapshot to record the
        # panel slice that produced the obs, instead of materialising it.
        self.last_obs_t = torch.full((n_envs,), self.start_idx, dtype=torch.long, device=self.device)
        self._pending_action: torch.Tensor | None = None

        obs_dim_per_stock = F + 2
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_stocks, obs_dim_per_stock),
            dtype=np.float32,
        )
        action_space = gym.spaces.Box(
            low=-action_range, high=action_range,
            shape=(self.n_stocks,),
            dtype=np.float32,
        )
        super().__init__(num_envs=n_envs, observation_space=observation_space, action_space=action_space)

        logger.info(
            "ResidualGPUEnv: T_window=[%d,%d) S=%d F+2=%d n_envs=%d panel=%s",
            start_idx, end_idx, self.n_stocks, obs_dim_per_stock, n_envs,
            tuple(self.panel.shape),
        )

    # ── SB3 VecEnv abstract methods ───────────────────────────────────

    def reset(self):
        self.t.fill_(self.start_idx)
        self.last_obs_t = self.t.clone()
        return self._gather_obs(self.t)

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_action = torch.from_numpy(actions).to(self.device, dtype=torch.float32)

    def step_wait(self):
        action = self._pending_action
        self._pending_action = None
        assert action is not None, "step_wait called without step_async"

        # Snapshot t for this step: the action was decided based on obs at
        # last_obs_t (which equals self.t prior to advance). Reward is from
        # realized_pct[last_obs_t], not from the new t. This mirrors the
        # gym.Env semantics where reward at step t is from realized[t].
        t_now = self.t.clone()  # (n_envs,) long

        # 1. Clip delta
        delta = action.clamp(-self.action_range, self.action_range)            # (n_envs, S)

        # 2. Score = logit(p_baseline) + λ·delta, mask out-of-universe
        p = self.p_baseline_cuda[t_now]                                         # (n_envs, S)
        in_uni = self.in_universe_cuda[t_now]                                   # (n_envs, S)
        # Replace NaN p with 0.5 so logit→0 (matches gym.Env behavior)
        p_safe = torch.where(torch.isnan(p), torch.tensor(0.5, device=self.device), p)
        score = _logit_t(p_safe) + self.lambda_logit * delta                    # (n_envs, S)
        score = torch.where(in_uni, score, torch.tensor(float("-inf"), device=self.device))

        # 3. Top-k per env
        k = min(self.top_k, self.n_stocks)
        topk_vals, topk_idx = torch.topk(score, k=k, dim=-1)                    # (n_envs, k)

        # 4. Realized excess return
        # Gather realized[t_now, topk_idx_per_env]
        r_full = self.realized_cuda[t_now]                                      # (n_envs, S)
        r_topk = torch.gather(r_full, 1, topk_idx)                              # (n_envs, k)
        r_topk = torch.where(torch.isnan(r_topk), torch.zeros_like(r_topk), r_topk)
        m = self.market_cuda[t_now]                                             # (n_envs,)
        m = torch.where(torch.isnan(m), torch.zeros_like(m), m)
        # If a stock got -inf score, it's not in universe — but topk may still
        # pick it when n_uni < top_k (unavoidable). Match gym.Env which doesn't
        # filter further; the reward ends up averaging some nan-protected zero.
        reward = (r_topk - m.unsqueeze(-1)).mean(dim=-1)                        # (n_envs,)
        reward_np = reward.detach().cpu().numpy().astype(np.float32)

        # 5. Saturation diagnostic per env
        delta_abs = delta.abs()
        sat_frac = (delta_abs >= 0.9 * self.action_range).float().mean(dim=-1)
        n_uni = in_uni.sum(dim=-1)
        infos = []
        for i in range(self.num_envs):
            infos.append({
                "delta_abs_mean": float(delta_abs[i].mean().item()),
                "delta_abs_p95": float(torch.quantile(delta_abs[i], 0.95).item()),
                "saturation_fraction": float(sat_frac[i].item()),
                "n_in_universe": int(n_uni[i].item()),
            })

        # 6. Advance t; if hit end_idx, reset to start_idx (auto-reset like SB3 VecEnv)
        self.t = self.t + 1
        done_mask = self.t >= self.end_idx
        if done_mask.any():
            self.t = torch.where(done_mask, torch.full_like(self.t, self.start_idx), self.t)
        dones_np = done_mask.detach().cpu().numpy().astype(bool)

        # On reset, SB3 expects the next obs to be the post-reset obs.
        self.last_obs_t = self.t.clone()
        next_obs = self._gather_obs(self.t)
        return next_obs, reward_np, dones_np, infos

    # ── VecEnv plumbing ───────────────────────────────────────────────

    def _gather_obs(self, t_idx: torch.Tensor) -> np.ndarray:
        # Materialise (n_envs, S, F+2) by indexing the panel; cast to fp32.
        # NOTE: this is the obs SB3 receives in numpy form. PPO's actual SGD
        # uses the IndexOnlyRolloutBuffer which gathers via env.panel[t] in
        # cuda fp16 → policy upcasts. So this numpy round-trip only matters
        # for SB3 internals (e.g. obs sample broadcast in initial setup).
        return self.panel[t_idx].to(dtype=torch.float32).cpu().numpy()

    def close(self) -> None:
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        return [None] * self.num_envs

    def get_attr(self, attr_name: str, indices=None):
        return [getattr(self, attr_name, None)] * self.num_envs

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        setattr(self, attr_name, value)

    def seed(self, seed=None):
        return [seed] * self.num_envs


__all__ = ["ResidualGPUEnv"]
