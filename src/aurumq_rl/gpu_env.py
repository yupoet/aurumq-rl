"""GPU-vectorised stock-picking environment.

Inherits from stable_baselines3.common.vec_env.VecEnv (NOT
gymnasium.vector.VectorEnv). All n_envs share a single panel
tensor on cuda; per-env state is a time-index vector also on
cuda. step_wait() is a single batched tensor op.

See docs/superpowers/specs/2026-05-01-gpu-rl-framework-design.md §5.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv


class GPUStockPickingEnv(VecEnv):
    """Single-process VecEnv with the panel resident on cuda."""

    def __init__(
        self,
        panel: torch.Tensor,        # (T, S, F) fp32 cuda
        returns: torch.Tensor,      # (T, S)    fp32 cuda
        valid_mask: torch.Tensor,   # (T, S)    bool cuda
        n_envs: int,
        episode_length: int = 240,
        forward_period: int = 10,
        top_k: int = 30,
        cost_bps: float = 30.0,
        turnover_coef: float = 0.0,
        device: str = "cuda",
        seed: int | None = None,
    ) -> None:
        if panel.device.type != "cuda":
            raise ValueError("panel must be a cuda tensor")
        if panel.shape[0] != returns.shape[0] or panel.shape[1] != returns.shape[1]:
            raise ValueError("panel and returns date/stock dims must match")
        if panel.shape[:2] != valid_mask.shape:
            raise ValueError("panel and valid_mask date/stock dims must match")

        self.panel = panel
        self.returns = returns
        self.valid_mask = valid_mask
        self.n_dates, self.n_stocks, self.n_factors = panel.shape
        self.episode_length = episode_length
        self.forward_period = forward_period
        self.top_k = top_k
        self.cost_bps = cost_bps
        self.turnover_coef = turnover_coef
        self.device = torch.device(device)
        self._rng = torch.Generator(device=self.device)
        if seed is not None:
            self._rng.manual_seed(seed)

        # Per-env state, all on cuda
        self.t = torch.zeros(n_envs, dtype=torch.long, device=self.device)
        self.steps_done = torch.zeros(n_envs, dtype=torch.long, device=self.device)
        self.episode_returns = torch.zeros(n_envs, dtype=torch.float32, device=self.device)
        self.prev_top_idx = torch.zeros(n_envs, top_k, dtype=torch.long, device=self.device)
        self._pending_action: torch.Tensor | None = None

        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_stocks, self.n_factors),
            dtype=np.float32,
        )
        action_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_stocks,),
            dtype=np.float32,
        )
        super().__init__(num_envs=n_envs, observation_space=observation_space, action_space=action_space)

    # SB3 VecEnv abstract methods --------------------------------------

    def reset(self):
        self._sample_starts(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self.steps_done.zero_()
        self.episode_returns.zero_()
        self.prev_top_idx.zero_()
        return self._current_obs()

    def step_async(self, actions):
        self._pending_action = self._coerce_action(actions)

    def step_wait(self):
        assert self._pending_action is not None, "step_async must be called before step_wait"
        action = self._pending_action
        self._pending_action = None

        # 1. mask invalid stocks (they can never enter top-K)
        action = action.masked_fill(~self.valid_mask[self.t], float("-inf"))
        # 2. top-K
        top_idx = torch.topk(action, k=self.top_k, dim=-1).indices  # (n_envs, K)
        # 3. forward returns gathered for the K picked stocks
        fwd_t = (self.t + self.forward_period).clamp(max=self.n_dates - 1)
        fwd_rets = self.returns[fwd_t].gather(1, top_idx)            # (n_envs, K)
        rewards = fwd_rets.mean(dim=-1) - self.cost_bps / 1e4
        # 4. turnover penalty (Jaccard-style)
        if self.turnover_coef > 0.0:
            overlap = torch.zeros_like(rewards)
            for i in range(self.num_envs):
                overlap[i] = float(
                    len(set(top_idx[i].tolist()) & set(self.prev_top_idx[i].tolist()))
                )
            jaccard_dist = 1.0 - overlap / float(self.top_k)
            rewards = rewards - self.turnover_coef * jaccard_dist
        self.prev_top_idx = top_idx
        self.episode_returns += rewards

        # 5. advance time
        self.t = self.t + 1
        self.steps_done = self.steps_done + 1
        dones = self.steps_done >= self.episode_length

        # 6. for done envs, build SB3 episode info, then auto-reset
        infos: list[dict] = [{} for _ in range(self.num_envs)]
        if bool(dones.any().item()):
            for i in dones.nonzero(as_tuple=True)[0].tolist():
                infos[i]["episode"] = {
                    "r": float(self.episode_returns[i].item()),
                    "l": int(self.steps_done[i].item()),
                }
            self._reset_done_envs(dones)

        obs = self._current_obs()
        return obs, rewards.detach().cpu().numpy().astype(np.float32), dones.detach().cpu().numpy(), infos

    def _reset_done_envs(self, dones: torch.Tensor) -> None:
        self._sample_starts(dones)
        self.steps_done = torch.where(
            dones,
            torch.zeros_like(self.steps_done),
            self.steps_done,
        )
        self.episode_returns = torch.where(
            dones,
            torch.zeros_like(self.episode_returns),
            self.episode_returns,
        )
        # Zero prev_top_idx for done envs only
        self.prev_top_idx[dones] = 0

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices=None):
        # Most common SB3 internal asks: 'render_mode', 'spec'
        if attr_name in {"render_mode", "spec"}:
            return [None] * self._indices_count(indices)
        raise NotImplementedError(f"get_attr({attr_name!r}) not supported")

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        raise NotImplementedError(f"set_attr({attr_name!r}) not supported")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError(f"env_method({method_name!r}) not supported")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self._indices_count(indices)

    def seed(self, seed=None):
        if seed is not None:
            self._rng.manual_seed(seed)
        return [seed] * self.num_envs

    # Helpers ----------------------------------------------------------

    def _coerce_action(self, actions):
        if isinstance(actions, np.ndarray):
            return torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        return actions.to(self.device, dtype=torch.float32)

    def _current_obs(self) -> torch.Tensor:
        return self.panel[self.t]

    def _sample_starts(self, mask: torch.Tensor) -> None:
        max_start = self.n_dates - self.episode_length - self.forward_period
        if max_start <= 0:
            raise ValueError(
                f"panel too short: n_dates={self.n_dates} episode_length="
                f"{self.episode_length} forward_period={self.forward_period}"
            )
        new_starts = torch.randint(
            low=0, high=max_start + 1,
            size=(int(mask.sum().item()),),
            generator=self._rng, device=self.device,
        )
        self.t[mask] = new_starts

    def _indices_count(self, indices) -> int:
        if indices is None:
            return self.num_envs
        if isinstance(indices, int):
            return 1
        return len(indices)
