"""CUDA-resident PPO rollout buffer.

Drop-in replacement for ``stable_baselines3.common.buffers.RolloutBuffer``
that keeps every tensor on the GPU. Eliminates the per-step
cuda -> cpu -> numpy -> cpu -> cuda round-trip the SB3 default buffer pays
when used with a GPU vec-env (~50 MB / step / direction at our shapes).

Phase 8 of the AurumQ-RL GPU framework. See spec §5.6 P1.

Notes:
- Subclass overrides ``reset``, ``add``, ``compute_returns_and_advantage``,
  ``get`` and ``_get_samples``. Everything else is inherited.
- ``__init__`` signature is identical to the parent so SB3's
  ``OnPolicyAlgorithm._setup_model`` can construct it via
  ``rollout_buffer_class=GPURolloutBuffer`` without further changes.
- ``add()`` accepts both numpy arrays and torch tensors -- SB3 hands us a
  mix because the VecEnv boundary still goes through numpy (see spec §5.5).
- ``compute_returns_and_advantage`` ports the parent's GAE loop to torch
  (still on cuda); it takes ``dones`` as numpy because that's what
  ``OnPolicyAlgorithm.collect_rollouts`` passes.
- Numerical equivalence with the parent is asserted in the test suite
  (1e-4 fp32 tolerance).
"""
from __future__ import annotations

from collections.abc import Generator

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class GPURolloutBuffer(RolloutBuffer):
    """PPO rollout buffer with all tensors resident on cuda.

    See module docstring for context. The shape and semantics of every
    tensor mirror the parent ``RolloutBuffer``; only the storage backend
    (cuda torch tensor) differs.
    """

    # Re-declare attribute types so static analysers see torch tensors,
    # not the parent's numpy-typed ndarray declarations.
    observations: th.Tensor  # type: ignore[assignment]
    actions: th.Tensor  # type: ignore[assignment]
    rewards: th.Tensor  # type: ignore[assignment]
    advantages: th.Tensor  # type: ignore[assignment]
    returns: th.Tensor  # type: ignore[assignment]
    episode_starts: th.Tensor  # type: ignore[assignment]
    log_probs: th.Tensor  # type: ignore[assignment]
    values: th.Tensor  # type: ignore[assignment]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        # The parent __init__ calls self.reset() at the end, which our
        # override handles. We just need to make sure the resolved device
        # is cuda before any allocation happens.
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )
        # ``self.device`` is set by BaseBuffer via ``get_device``.
        if self.device.type != "cuda":
            raise ValueError(
                "GPURolloutBuffer requires cuda; got "
                f"device={self.device!r}. Pass device='cuda' (or 'auto' on "
                "a cuda-equipped host), or use the default RolloutBuffer."
            )

    # ------------------------------------------------------------------
    # Storage allocation
    # ------------------------------------------------------------------

    def reset(self) -> None:  # type: ignore[override]
        """Allocate all storage tensors directly on cuda.

        We do NOT call ``super().reset()`` (which would re-allocate as
        numpy). Instead we replicate ``BaseBuffer.reset`` by hand
        (``pos = 0``, ``full = False``).
        """
        device = self.device
        # Use float32 for everything except actions (which follow the
        # action_space dtype, mirroring the parent). This matches what
        # the policy expects on the forward pass.
        obs_dtype = self._np_to_torch_dtype(self.observation_space.dtype)
        act_dtype = self._np_to_torch_dtype(self.action_space.dtype)

        self.observations = th.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=obs_dtype, device=device,
        )
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
        # BaseBuffer.reset() body, inlined:
        self.pos = 0
        self.full = False

    @staticmethod
    def _np_to_torch_dtype(np_dtype) -> th.dtype:
        """Translate the numpy dtype carried on a gym Box space to torch.

        Falls back to float32 for object/None dtypes (rare for our spaces).
        """
        if np_dtype is None:
            return th.float32
        try:
            return th.from_numpy(np.zeros((), dtype=np_dtype)).dtype
        except TypeError:
            return th.float32

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add(  # type: ignore[override]
        self,
        obs,
        action,
        reward,
        episode_start,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """Store one transition. Accepts numpy or torch input.

        SB3's on-policy loop hands us:
            obs, action, reward, episode_start: numpy arrays
            value, log_prob: cuda torch tensors
        but we accept torch for everything to stay flexible (e.g. tests
        that pre-convert).
        """
        if log_prob.dim() == 0:
            log_prob = log_prob.reshape(-1, 1)

        if isinstance(self.observation_space, spaces.Discrete):
            # Match parent's reshape for discrete obs spaces. Box spaces
            # (our case) skip this branch entirely.
            if isinstance(obs, np.ndarray):
                obs = obs.reshape((self.n_envs, *self.obs_shape))
            else:
                obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Move into pre-allocated slot. ``copy_`` will dtype-cast as needed
        # (float32 obs into float32 storage; numpy bools into float32 for
        # episode_starts via ``as_tensor``).
        device = self.device
        self.observations[self.pos].copy_(
            self._as_tensor(obs, device, self.observations.dtype)
        )
        # Reshape action to (n_envs, action_dim) the same way the parent does.
        action_t = self._as_tensor(action, device, self.actions.dtype)
        action_t = action_t.reshape((self.n_envs, self.action_dim))
        self.actions[self.pos].copy_(action_t)
        self.rewards[self.pos].copy_(
            self._as_tensor(reward, device, self.rewards.dtype)
        )
        self.episode_starts[self.pos].copy_(
            self._as_tensor(episode_start, device, self.episode_starts.dtype)
        )
        # values / log_prob already cuda; flatten to (n_envs,).
        self.values[self.pos].copy_(value.detach().to(device).flatten())
        self.log_probs[self.pos].copy_(log_prob.detach().to(device).flatten())

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    @staticmethod
    def _as_tensor(x, device: th.device, dtype: th.dtype) -> th.Tensor:
        """Promote ndarray-or-tensor into a cuda tensor of the given dtype.

        Skips host->device copy when ``x`` is already a tensor on the
        target device with the right dtype.
        """
        if isinstance(x, th.Tensor):
            if x.device == device and x.dtype == dtype:
                return x
            return x.to(device=device, dtype=dtype)
        return th.as_tensor(np.asarray(x), device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_returns_and_advantage(  # type: ignore[override]
        self, last_values: th.Tensor, dones: np.ndarray | th.Tensor,
    ) -> None:
        """Compute lambda-return and GAE(lambda), all on cuda.

        Direct port of the parent's loop; same numerics, just torch ops.
        Rewritten as a reverse Python loop on ``buffer_size`` (typically
        128); a fully-vectorised cumulative-product variant would be a
        Phase 9 micro-opt.
        """
        device = self.device
        # last_values: (n_envs,) cuda tensor.
        last_values = last_values.detach().to(device).flatten()

        # dones may arrive as numpy from on_policy_algorithm.collect_rollouts;
        # accept torch too for unit-test convenience.
        if isinstance(dones, th.Tensor):
            dones_t = dones.to(device=device, dtype=th.float32)
        else:
            dones_t = th.as_tensor(np.asarray(dones).astype(np.float32),
                                   device=device, dtype=th.float32)

        last_gae_lam = th.zeros(self.n_envs, dtype=th.float32, device=device)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones_t
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        # TD(lambda) target.
        self.returns = self.advantages + self.values

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @staticmethod
    def swap_and_flatten_torch(t: th.Tensor) -> th.Tensor:  # noqa: D401
        """Torch analogue of ``BaseBuffer.swap_and_flatten``.

        ``(n_steps, n_envs, *F) -> (n_steps * n_envs, *F)`` while keeping
        the per-env order. ``transpose + reshape`` requires
        ``contiguous()`` because torch's transpose returns a non-contig
        view and reshape can't merge non-contig dims for arbitrary shapes.
        """
        if t.dim() < 3:
            t = t.unsqueeze(-1)
        n_steps, n_envs = t.shape[0], t.shape[1]
        rest = t.shape[2:]
        return t.transpose(0, 1).contiguous().reshape(n_steps * n_envs, *rest)

    def get(  # type: ignore[override]
        self, batch_size: int | None = None,
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        total = self.buffer_size * self.n_envs
        # np.random.permutation keeps determinism aligned with the parent.
        indices = np.random.permutation(total)

        if not self.generator_ready:
            for name in (
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ):
                self.__dict__[name] = self.swap_and_flatten_torch(
                    self.__dict__[name]
                )
            self.generator_ready = True

        if batch_size is None:
            batch_size = total

        # Pre-compute the cuda index tensor once per call.
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
        # Accept either numpy or torch indices; SB3 internals only ever
        # pass numpy via sample(), but our get() uses a cuda LongTensor.
        if isinstance(batch_inds, np.ndarray):
            batch_inds = th.as_tensor(batch_inds, dtype=th.long,
                                      device=self.device)
        else:
            batch_inds = batch_inds.to(self.device, dtype=th.long)

        # All tensors already swap_and_flatten'd by get().
        return RolloutBufferSamples(
            observations=self.observations[batch_inds],
            # Match parent: cast actions to float32 for the loss (this is
            # a no-op when actions are already float32, which they are for
            # our Box action space).
            actions=self.actions[batch_inds].to(dtype=th.float32),
            old_values=self.values[batch_inds].flatten(),
            old_log_prob=self.log_probs[batch_inds].flatten(),
            advantages=self.advantages[batch_inds].flatten(),
            returns=self.returns[batch_inds].flatten(),
        )
