"""GPU-vectorised stock-picking environment.

Inherits from stable_baselines3.common.vec_env.VecEnv (NOT
gymnasium.vector.VectorEnv). All n_envs share a single panel
tensor on cuda; per-env state is a time-index vector also on
cuda. step_wait() is a single batched tensor op.

See docs/superpowers/specs/2026-05-01-gpu-rl-framework-design.md §5.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv


class GPUStockPickingEnv(VecEnv):
    """Single-process VecEnv with the panel resident on cuda."""

    def __init__(
        self,
        panel: torch.Tensor,  # (T, S, F) fp32 cuda
        returns: torch.Tensor,  # (T, S)    fp32 cuda
        valid_mask: torch.Tensor,  # (T, S)    bool cuda
        n_envs: int,
        episode_length: int = 240,
        forward_period: int = 10,
        top_k: int = 30,
        cost_bps: float = 30.0,
        turnover_coef: float = 0.0,
        device: str = "cuda",
        seed: int | None = None,
        # Phase 22: optional reward override. If hold_returns is provided,
        # the env uses it instead of `returns` for reward (per-stock realized
        # hold return under signal-exit). When None, falls back to V1 behaviour
        # (mean of N-day forward log return). Indexing is the same: row t is
        # the reward for action taken at t.
        hold_returns: torch.Tensor | None = None,  # (T, S) fp32 cuda
    ) -> None:
        # Residency guard: the panel must live on the env device. With the
        # default device="cuda" this keeps the original protection against
        # accidentally-CPU panels in training; device="cpu" allows the
        # masking/reward logic to be tested without CUDA.
        expected_device = torch.device(device).type
        if panel.device.type != expected_device:
            raise ValueError(f"panel must be a {expected_device} tensor")
        if panel.shape[0] != returns.shape[0] or panel.shape[1] != returns.shape[1]:
            raise ValueError("panel and returns date/stock dims must match")
        if panel.shape[:2] != valid_mask.shape:
            raise ValueError("panel and valid_mask date/stock dims must match")
        if hold_returns is not None:
            if hold_returns.device.type != expected_device:
                raise ValueError(f"hold_returns must be a {expected_device} tensor")
            if hold_returns.shape != returns.shape:
                raise ValueError(
                    f"hold_returns shape {tuple(hold_returns.shape)} must match "
                    f"returns shape {tuple(returns.shape)}"
                )

        self.panel = panel
        self.returns = returns
        # Phase 22: when hold_returns provided, the env uses it as the reward
        # source. Otherwise falls back to `returns` (legacy 10d forward mean).
        self.hold_returns = hold_returns
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
        # ``last_obs_t`` mirrors the t-index of the obs most recently emitted
        # by ``reset()`` / ``step_wait()``. The IndexOnlyRolloutBuffer reads
        # this snapshot inside ``add()`` to record which panel slice produced
        # the obs, instead of materialising the obs itself. See
        # src/aurumq_rl/index_rollout_buffer.py.
        self.last_obs_t = torch.zeros(n_envs, dtype=torch.long, device=self.device)
        self._pending_action: torch.Tensor | None = None

        observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_stocks, self.n_factors),
            dtype=np.float32,
        )
        action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_stocks,),
            dtype=np.float32,
        )
        super().__init__(
            num_envs=n_envs, observation_space=observation_space, action_space=action_space
        )

    # SB3 VecEnv abstract methods --------------------------------------

    def reset(self):
        self._sample_starts(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self.steps_done.zero_()
        self.episode_returns.zero_()
        self.prev_top_idx.zero_()
        # Snapshot the t-index of the obs we are about to emit so the
        # index-only rollout buffer can reference it without copying obs.
        self.last_obs_t = self.t.clone()
        return self._obs_for_sb3()

    def step_async(self, actions):
        self._pending_action = self._coerce_action(actions)

    def step_wait(self):
        assert self._pending_action is not None, "step_async must be called before step_wait"
        action = self._pending_action
        self._pending_action = None

        # Phase 22 fix: snapshot last_obs_t to the CURRENT t (the t of the obs
        # SB3 just passed to policy.forward). SB3 calls rollout_buffer.add(
        # self._last_obs) AFTER env.step but BEFORE updating self._last_obs to
        # the new obs, so the buffer must store the t that produced the OLD
        # obs. The previous end-of-step-wait update wrote the post-advance t,
        # corrupting evaluate_actions during PPO updates.
        self.last_obs_t = self.t.clone()

        # 1. mask invalid stocks (they can never enter top-K).
        #    PARITY RULE (C4): valid_mask is built by the caller
        #    (scripts/train_v2.py via data_loader.build_tradeable_mask) with
        #    the same semantics as the CPU env's _apply_trading_mask —
        #    ~ST & ~suspended & IPO gate & neither limit-up NOR limit-down
        #    at the decision date, plus finite forward returns.
        action = action.masked_fill(~self.valid_mask[self.t], float("-inf"))
        # 2. top-K
        top_scores, top_idx = torch.topk(action, k=self.top_k, dim=-1)  # (n_envs, K)
        # 3. realized return. Phase 22: prefer hold_returns (per-stock realized
        #    hold_return under MA5/MA10 signal exit, capped at 5 days) when
        #    provided; else fall back to V1's 10d forward mean. Both use the
        #    same indexing: row t is the realized return for action at t.
        return_source = self.hold_returns if self.hold_returns is not None else self.returns
        fwd_rets = return_source[self.t].gather(1, top_idx)  # (n_envs, K)
        # NaN guard: missing (date, stock) cells carry NaN forward returns.
        # valid_mask normally excludes them, but when < top_k stocks are
        # valid on a date, topk can still select a masked stock — treat its
        # return as 0 (the pre-C2 semantics) instead of poisoning the mean.
        fwd_rets = torch.nan_to_num(fwd_rets, nan=0.0)
        # C4 fix: when a date has fewer than top_k valid stocks, topk pads
        # with -inf-scored (invalid) picks whose REAL returns would otherwise
        # enter the mean. Zero those picks and average over real picks only
        # (clamp avoids 0/0 on dates with no valid stock → reward = -cost).
        real_pick = torch.isfinite(top_scores)  # (n_envs, K)
        fwd_rets = fwd_rets * real_pick
        n_real = real_pick.sum(dim=-1).clamp(min=1)
        rewards = fwd_rets.sum(dim=-1) / n_real - self.cost_bps / 1e4
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

        # NOTE: do NOT update self.last_obs_t here. It was already snapshotted
        # at the top of step_wait to the t of the obs SB3 just consumed.
        # Re-setting it post-advance would re-introduce the off-by-one bug
        # that PPO update sees inconsistent log_probs across rollout vs eval.
        obs = self._obs_for_sb3()
        return (
            obs,
            rewards.detach().cpu().numpy().astype(np.float32),
            dones.detach().cpu().numpy(),
            infos,
        )

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

    def _obs_for_sb3(self) -> np.ndarray:
        """Convert the cuda obs tensor to a numpy array for SB3's VecEnv contract.

        SB3's ``obs_as_tensor`` only handles ``np.ndarray`` and ``dict`` — see
        ``stable_baselines3.common.utils.obs_as_tensor``. The spec's assumption
        that a cuda tensor is a no-op (§5.4) is incorrect against SB3 2.8.0,
        so we convert at the VecEnv boundary. Internal state (``self.panel``)
        stays on cuda; only the return value of ``reset()`` / ``step_wait()``
        is materialised as numpy.
        """
        return self._current_obs().detach().cpu().numpy()

    def _sample_starts(self, mask: torch.Tensor) -> None:
        max_start = self.n_dates - self.episode_length - self.forward_period
        if max_start <= 0:
            raise ValueError(
                f"panel too short: n_dates={self.n_dates} episode_length="
                f"{self.episode_length} forward_period={self.forward_period}"
            )
        new_starts = torch.randint(
            low=0,
            high=max_start + 1,
            size=(int(mask.sum().item()),),
            generator=self._rng,
            device=self.device,
        )
        self.t[mask] = new_starts

    def _indices_count(self, indices) -> int:
        if indices is None:
            return self.num_envs
        if isinstance(indices, int):
            return 1
        return len(indices)
