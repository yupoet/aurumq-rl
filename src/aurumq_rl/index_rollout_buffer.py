"""Index-only PPO rollout buffer.

Subclass of :class:`GPURolloutBuffer` (Phase 8) that stores
``(t, env_idx)`` integer indices into the cuda panel instead of the
materialised observations themselves. Obs tensors are lazy-gathered at
SGD time via a caller-supplied ``obs_provider`` closure.

Why this matters (spec §5.6 P2): at the smoke-test config
(n_steps=128, n_envs=12, n_stocks=3014, n_factors=343 fp32) the obs
tensor alone is 6.35 GB. Phase 8 moved that storage to cuda but did
not shrink it, so n_steps=1024 still cannot fit on a 12 GB card. By
keeping only the t-index per (step, env), the buffer shrinks to ~200 MB
even at n_steps=1024 / n_envs=16 / 3014 stocks. The actual obs is
re-materialised inside :meth:`_get_samples` via ``panel[t]``, which is
a single cuda gather and roughly free at SGD scale.

Numerical equivalence with Phase 8's GPURolloutBuffer is asserted in
the test suite — the GAE math is unchanged; only the obs storage
strategy differs.
"""
from __future__ import annotations

from collections.abc import Callable, Generator

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from aurumq_rl.gpu_rollout_buffer import GPURolloutBuffer


class IndexOnlyRolloutBuffer(GPURolloutBuffer):
    """PPO rollout buffer that stores t-indices instead of full obs.

    The obs tensor is reconstructed lazily at SGD time via two closures
    bound at construction:

    - ``obs_provider(t_indices: LongTensor) -> Tensor`` materialises the
      obs for a flat batch of t-indices. In the AurumQ-RL training loop
      this is ``lambda t: env.panel[t]`` — a single cuda gather.
    - ``obs_index_provider() -> LongTensor`` returns the
      ``(n_envs,)`` long tensor of t-indices for the obs currently held
      by the env (i.e. the one that will be added next). In our case
      this is ``lambda: env.last_obs_t``.

    The buffer ignores the ``obs`` argument to :meth:`add`; it reads
    the t-snapshot from ``obs_index_provider()`` instead. SB3's caller
    still passes a numpy obs (it doesn't know we've changed the
    storage strategy) and we accept-and-discard it for compatibility.
    """

    # The parent's annotation declares ``observations: th.Tensor`` — we
    # don't allocate that tensor here, so type-checkers should treat it
    # as absent. The runtime check in tests asserts the substitution.
    t_buffer: th.Tensor  # type: ignore[assignment]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
        *,
        obs_provider: Callable[[th.Tensor], th.Tensor],
        obs_index_provider: Callable[[], th.Tensor],
    ) -> None:
        # Stash the closures BEFORE calling super().__init__, because
        # the parent's __init__ ends with a self.reset() call which we
        # override and which (via the parent BaseBuffer chain) does not
        # need the closures itself, but we want them available if any
        # subclass override needs them.
        self._obs_provider = obs_provider
        self._obs_index_provider = obs_index_provider
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )

    # ------------------------------------------------------------------
    # Storage allocation
    # ------------------------------------------------------------------

    def reset(self) -> None:  # type: ignore[override]
        """Allocate t_buffer instead of the (huge) observations tensor.

        All other tensors (actions, rewards, values, log_probs,
        episode_starts, advantages, returns) are allocated identically
        to Phase 8's ``GPURolloutBuffer.reset``. Action storage still
        dominates (~200 MB at 1024×16×3014 fp32), but obs storage —
        previously 6 GB+ — drops to (1024×16×8 B) ≈ 130 KB.
        """
        device = self.device
        act_dtype = self._np_to_torch_dtype(self.action_space.dtype)

        # Replaces ``self.observations``. Each entry is the t-index that
        # produced the obs we *would* have stored.
        self.t_buffer = th.zeros(
            (self.buffer_size, self.n_envs),
            dtype=th.long, device=device,
        )
        # ``observations`` deliberately not allocated. Some external
        # tooling may probe ``hasattr(buf, 'observations')`` — set it to
        # None so attribute lookup returns a clean sentinel rather than
        # AttributeError.
        self.observations = None  # type: ignore[assignment]

        self.actions = th.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=act_dtype, device=device,
        )
        self.rewards = th.zeros((self.buffer_size, self.n_envs),
                                dtype=th.float32, device=device)
        self.returns = th.zeros((self.buffer_size, self.n_envs),
                                dtype=th.float32, device=device)
        self.episode_starts = th.zeros((self.buffer_size, self.n_envs),
                                       dtype=th.float32, device=device)
        self.values = th.zeros((self.buffer_size, self.n_envs),
                               dtype=th.float32, device=device)
        self.log_probs = th.zeros((self.buffer_size, self.n_envs),
                                  dtype=th.float32, device=device)
        self.advantages = th.zeros((self.buffer_size, self.n_envs),
                                   dtype=th.float32, device=device)
        self.generator_ready = False
        # BaseBuffer.reset() body, inlined (matches GPURolloutBuffer.reset).
        self.pos = 0
        self.full = False

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add(  # type: ignore[override]
        self,
        obs,  # noqa: ARG002 - intentionally ignored; kept for SB3 contract
        action,
        reward,
        episode_start,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """Store one transition. ``obs`` is IGNORED.

        Instead of copying the obs into a slot, we snapshot the env's
        ``last_obs_t`` (via ``obs_index_provider``) and store that.
        SB3 still hands us a numpy obs because it doesn't know we
        changed the storage strategy; that's fine — we drop it.
        """
        if log_prob.dim() == 0:
            log_prob = log_prob.reshape(-1, 1)

        device = self.device
        # Snapshot the t-index for the obs SB3 just emitted. The env
        # guarantees ``last_obs_t`` matches the obs it returned from the
        # most recent reset()/step_wait() call.
        t_now = self._obs_index_provider()
        if not isinstance(t_now, th.Tensor):
            t_now = th.as_tensor(np.asarray(t_now), dtype=th.long, device=device)
        else:
            t_now = t_now.to(device=device, dtype=th.long)
        self.t_buffer[self.pos].copy_(t_now)

        action_t = self._as_tensor(action, device, self.actions.dtype)
        action_t = action_t.reshape((self.n_envs, self.action_dim))
        self.actions[self.pos].copy_(action_t)
        self.rewards[self.pos].copy_(
            self._as_tensor(reward, device, self.rewards.dtype)
        )
        self.episode_starts[self.pos].copy_(
            self._as_tensor(episode_start, device, self.episode_starts.dtype)
        )
        self.values[self.pos].copy_(value.detach().to(device).flatten())
        self.log_probs[self.pos].copy_(log_prob.detach().to(device).flatten())

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def get(  # type: ignore[override]
        self, batch_size: int | None = None,
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        total = self.buffer_size * self.n_envs
        indices = np.random.permutation(total)

        if not self.generator_ready:
            # Same swap_and_flatten as parent, but operate on
            # ``t_buffer`` instead of ``observations``. ``actions`` and
            # ``log_probs`` follow exactly the same path the parent
            # uses; values/returns/advantages cuda mirrors are kept on
            # _values_cuda / _returns_cuda / _advantages_cuda by
            # compute_returns_and_advantage() (inherited).
            self.t_buffer = self.swap_and_flatten_torch(self.t_buffer)
            for name in ("actions", "log_probs"):
                self.__dict__[name] = self.swap_and_flatten_torch(
                    self.__dict__[name]
                )
            self._values_cuda = self.swap_and_flatten_torch(self._values_cuda)
            self._returns_cuda = self.swap_and_flatten_torch(self._returns_cuda)
            self._advantages_cuda = self.swap_and_flatten_torch(self._advantages_cuda)
            self.generator_ready = True

        if batch_size is None:
            batch_size = total

        idx_cuda = th.as_tensor(indices, dtype=th.long, device=self.device)
        start = 0
        while start < total:
            yield self._get_samples(idx_cuda[start : start + batch_size])
            start += batch_size

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray | th.Tensor,
        env: VecNormalize | None = None,
    ) -> RolloutBufferSamples:
        # Accept numpy or torch indices; our get() always passes torch.
        if isinstance(batch_inds, np.ndarray):
            batch_inds = th.as_tensor(batch_inds, dtype=th.long,
                                      device=self.device)
        else:
            batch_inds = batch_inds.to(self.device, dtype=th.long)

        # Re-materialise the obs lazily by gathering the panel at the
        # planted t-indices. ``t_buffer`` is already swap_and_flatten'd
        # by get(), so ``self.t_buffer[batch_inds]`` is shape (B,).
        # The provider returns (B, n_stocks, n_factors) on cuda.
        # squeeze(-1) handles ``swap_and_flatten_torch``'s unsqueeze for
        # 2-D inputs: t_buffer is (n_steps, n_envs); the helper
        # unsqueezes to (n_steps, n_envs, 1) before transpose+reshape,
        # giving us a flattened (n_steps*n_envs, 1) we need to squeeze
        # before passing to the provider.
        t_idx = self.t_buffer[batch_inds]
        if t_idx.dim() == 2 and t_idx.shape[-1] == 1:
            t_idx = t_idx.squeeze(-1)
        obs = self._obs_provider(t_idx)

        values_cuda = getattr(self, "_values_cuda", self.values)
        returns_cuda = getattr(self, "_returns_cuda", self.returns)
        advantages_cuda = getattr(self, "_advantages_cuda", self.advantages)
        return RolloutBufferSamples(
            observations=obs,
            actions=self.actions[batch_inds].to(dtype=th.float32),
            old_values=values_cuda[batch_inds].flatten(),
            old_log_prob=self.log_probs[batch_inds].flatten(),
            advantages=advantages_cuda[batch_inds].flatten(),
            returns=returns_cuda[batch_inds].flatten(),
        )
