"""Phase 21 IndexOnlyDictRolloutBuffer.

Subclass of stable_baselines3.common.buffers.DictRolloutBuffer that stores
(t, env_idx) indices into the cuda panel/regime/valid_mask tensors instead
of the full Dict observations. Materialises obs lazily at SGD time via
three caller-supplied providers (one per Dict key) plus an
obs_index_provider that reports the env's last_obs_t tensor.

Memory profile: at n_steps=1024, n_envs=16, 3014 stocks, 343 factors fp32
the full DictRolloutBuffer would require ~64 GiB of obs storage. This buffer
replaces that with a (1024, 16) long tensor (~130 KiB) and re-materialises
the obs inside _get_samples via a single cuda index_select gather.
"""
from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.type_aliases import DictRolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class IndexOnlyDictRolloutBuffer(DictRolloutBuffer):
    """Dict rollout buffer that stores t-indices and gathers obs lazily.

    Instead of allocating full obs arrays for all (steps × envs) frames,
    this buffer stores only a ``(buffer_size, n_envs)`` long tensor of
    t-indices (``t_buffer``). At SGD time, ``_get_samples`` re-materialises
    the Dict obs via three caller-supplied closures:

    - ``stock_provider(t: LongTensor) -> Tensor`` shape ``(B, S, F_stock)``
    - ``regime_provider(t: LongTensor) -> Tensor`` shape ``(B, R)``
    - ``mask_provider(t: LongTensor) -> Tensor`` shape ``(B, S)``
    - ``obs_index_provider() -> LongTensor`` shape ``(n_envs,)``

    The buffer ignores the ``obs`` argument to :meth:`add`; it reads the
    t-snapshot from ``obs_index_provider()`` instead.

    Providers can be supplied at construction time via keyword arguments or
    later via :meth:`attach_providers` (needed when SB3's ``_setup_model``
    builds the buffer without extra kwargs).
    """

    t_buffer: th.Tensor  # type: ignore[assignment]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
        *,
        stock_provider: Callable[[th.Tensor], th.Tensor] | None = None,
        regime_provider: Callable[[th.Tensor], th.Tensor] | None = None,
        mask_provider: Callable[[th.Tensor], th.Tensor] | None = None,
        obs_index_provider: Callable[[], th.Tensor] | None = None,
    ) -> None:
        # Store providers before super().__init__ because the parent's
        # __init__ calls reset() which we override. Providers are set here
        # so they are available if reset() ever queries them (they don't
        # currently, but guarding order of operations is cleaner).
        self._stock_provider = stock_provider
        self._regime_provider = regime_provider
        self._mask_provider = mask_provider
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
    # Provider attachment
    # ------------------------------------------------------------------

    def attach_providers(
        self,
        stock_provider: Callable[[th.Tensor], th.Tensor],
        regime_provider: Callable[[th.Tensor], th.Tensor],
        mask_provider: Callable[[th.Tensor], th.Tensor],
        obs_index_provider: Callable[[], th.Tensor],
    ) -> None:
        """Attach panel-gather closures after construction.

        Required when the buffer is built by SB3's ``_setup_model`` (which
        does not forward extra kwargs). Must be called once before the first
        ``model.learn()`` invocation.

        :param stock_provider: ``(B,) LongTensor -> (B, S, F) cuda Tensor``
        :param regime_provider: ``(B,) LongTensor -> (B, R) cuda Tensor``
        :param mask_provider: ``(B,) LongTensor -> (B, S) cuda Tensor``
        :param obs_index_provider: ``() -> (n_envs,) LongTensor`` — returns
            the current t-index for each env (i.e. ``env.last_obs_t``).
        """
        self._stock_provider = stock_provider
        self._regime_provider = regime_provider
        self._mask_provider = mask_provider
        self._obs_index_provider = obs_index_provider

    # ------------------------------------------------------------------
    # Storage allocation
    # ------------------------------------------------------------------

    def reset(self) -> None:  # type: ignore[override]
        """Allocate ``t_buffer`` in place of the (huge) per-key obs arrays.

        All other bookkeeping tensors (rewards, values, returns, etc.) are
        allocated as numpy arrays so that the inherited
        ``compute_returns_and_advantage`` (which runs the GAE loop on CPU)
        works without modification.  They are converted to torch at SGD
        time inside :meth:`get`.

        ``t_buffer`` and ``actions`` are the only allocations stored as
        torch tensors because they are needed on CUDA at gather time.
        """
        device = self.device

        # t-index storage — replaces per-key obs arrays.
        self.t_buffer = th.zeros(
            (self.buffer_size, self.n_envs),
            dtype=th.long,
            device=device,
        )

        # Sentinel-out the per-key obs arrays; they are never populated.
        self.observations = {  # type: ignore[assignment]
            k: None for k in self.observation_space.spaces.keys()
        }

        # Actions stored as torch so they stay on CUDA for _get_samples.
        self.actions = th.zeros(  # type: ignore[assignment]
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=th.float32,
            device=device,
        )

        # GAE bookkeeping stored as numpy (parent's compute_returns_and_advantage
        # does CPU arithmetic on these arrays directly).
        self.rewards = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.returns = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.episode_starts = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.values = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.log_probs = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.advantages = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )

        self.generator_ready = False
        self.pos = 0
        self.full = False

    # ------------------------------------------------------------------
    # Add (obs is IGNORED; we snapshot last_obs_t instead)
    # ------------------------------------------------------------------

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],  # noqa: ARG002 — intentionally ignored
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """Store one transition; ``obs`` is discarded.

        Instead of copying obs into per-key arrays, we snapshot the env's
        ``last_obs_t`` tensor (via ``obs_index_provider``) into
        ``t_buffer[pos]``.  SB3 still passes a numpy obs dict because it
        does not know about our storage strategy; we accept and drop it.

        :raises RuntimeError: if providers have not been attached yet.
        """
        if self._obs_index_provider is None:
            raise RuntimeError(
                "IndexOnlyDictRolloutBuffer.add(): providers not attached. "
                "Call attach_providers() before model.learn()."
            )

        if log_prob.dim() == 0:
            log_prob = log_prob.reshape(-1, 1)

        device = self.device

        # Snapshot the t-index for the obs SB3 just emitted.
        t_now = self._obs_index_provider()
        if not isinstance(t_now, th.Tensor):
            t_now = th.as_tensor(np.asarray(t_now), dtype=th.long, device=device)
        else:
            t_now = t_now.to(device=device, dtype=th.long)
        self.t_buffer[self.pos].copy_(t_now)

        # Actions stored directly as torch on device.
        action_t = _as_tensor(action, device, th.float32)
        action_t = action_t.reshape((self.n_envs, self.action_dim))
        self.actions[self.pos].copy_(action_t)

        # GAE bookkeeping stored as numpy (matches parent's layout).
        self.rewards[self.pos] = np.asarray(reward, dtype=np.float32)
        self.episode_starts[self.pos] = np.asarray(episode_start, dtype=np.float32)
        self.values[self.pos] = value.detach().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.detach().cpu().numpy().flatten()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def get(  # type: ignore[override]
        self,
        batch_size: int | None = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        """Yield ``DictRolloutBufferSamples`` mini-batches.

        On the first call, ``t_buffer`` and ``actions`` are swap-and-flattened
        (axes 0 and 1 transposed and merged), and the numpy bookkeeping
        arrays are converted to flat torch tensors on device.
        """
        assert self.full, "rollout buffer not full"
        total = self.buffer_size * self.n_envs
        indices = np.random.permutation(total)

        if not self.generator_ready:
            # Flatten t_buffer and actions (CUDA tensors).
            self.t_buffer = _swap_and_flatten_torch(self.t_buffer)
            self.actions = _swap_and_flatten_torch(self.actions)

            # Convert numpy GAE arrays to flat CUDA tensors.
            for name in ("log_probs", "values", "returns", "advantages"):
                arr = self.__dict__[name]  # numpy (buffer_size, n_envs)
                flat = th.as_tensor(
                    arr.swapaxes(0, 1).reshape(-1),
                    dtype=th.float32,
                    device=device_of(self),
                )
                self.__dict__[name] = flat

            self.generator_ready = True

        if batch_size is None:
            batch_size = total

        idx_cuda = th.as_tensor(indices, dtype=th.long, device=device_of(self))
        start = 0
        while start < total:
            yield self._get_samples(idx_cuda[start : start + batch_size])
            start += batch_size

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray | th.Tensor,
        env: VecNormalize | None = None,
    ) -> DictRolloutBufferSamples:
        """Materialise Dict obs for ``batch_inds`` via provider closures.

        :param batch_inds: flat indices into the swap-and-flattened buffer
            (shape ``(B,)``).
        :param env: ignored (no VecNormalize support for GPU buffers).
        :raises RuntimeError: if providers have not been attached.
        """
        dev = device_of(self)

        if isinstance(batch_inds, np.ndarray):
            batch_inds = th.as_tensor(batch_inds, dtype=th.long, device=dev)
        else:
            batch_inds = batch_inds.to(dev, dtype=th.long)

        if any(
            p is None
            for p in (self._stock_provider, self._regime_provider, self._mask_provider)
        ):
            raise RuntimeError(
                "IndexOnlyDictRolloutBuffer._get_samples(): providers not "
                "attached. Call attach_providers() before model.learn()."
            )

        # Gather t-indices for this batch, squeeze trailing dim if needed.
        t_idx = self.t_buffer[batch_inds]
        if t_idx.dim() == 2 and t_idx.shape[-1] == 1:
            t_idx = t_idx.squeeze(-1)

        observations: dict[str, th.Tensor] = {
            "stock": self._stock_provider(t_idx),
            "regime": self._regime_provider(t_idx),
            "valid_mask": self._mask_provider(t_idx),
        }

        return DictRolloutBufferSamples(
            observations=observations,
            actions=self.actions[batch_inds].to(dtype=th.float32),
            old_values=self.values[batch_inds].flatten(),
            old_log_prob=self.log_probs[batch_inds].flatten(),
            advantages=self.advantages[batch_inds].flatten(),
            returns=self.returns[batch_inds].flatten(),
        )


# ------------------------------------------------------------------
# Module-level helpers (private)
# ------------------------------------------------------------------

def _swap_and_flatten_torch(arr: th.Tensor) -> th.Tensor:
    """Swap axes 0 (buffer_size) and 1 (n_envs) then flatten to 1-D or
    (n_steps * n_envs, ...) for higher-rank tensors.

    Mirrors ``BaseBuffer.swap_and_flatten`` but operates on torch tensors
    so the result stays on the original device without a host round-trip.
    """
    shape = arr.shape
    if arr.dim() < 3:
        return arr.transpose(0, 1).reshape(shape[0] * shape[1])
    return arr.transpose(0, 1).reshape(shape[0] * shape[1], *shape[2:])


def _as_tensor(x: Any, device: th.device, dtype: th.dtype) -> th.Tensor:
    """Convert numpy array or torch tensor to a tensor on *device*."""
    if isinstance(x, th.Tensor):
        return x.to(device=device, dtype=dtype)
    return th.as_tensor(np.asarray(x), device=device, dtype=dtype)


def device_of(buf: "IndexOnlyDictRolloutBuffer") -> th.device:
    """Return the torch device the buffer was constructed with."""
    return buf.device
