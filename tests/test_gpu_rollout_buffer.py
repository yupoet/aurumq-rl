"""Tests for src/aurumq_rl/gpu_rollout_buffer.py.

Covers:
- Storage residency (everything on cuda after reset).
- ``add()`` accepts numpy + torch input interchangeably.
- GAE parity with SB3's CPU RolloutBuffer (1e-4 fp32 tol).
- ``get()`` yields cuda RolloutBufferSamples with correct shapes.
- VRAM footprint sanity at the smoke configuration.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

from aurumq_rl.gpu_rollout_buffer import GPURolloutBuffer

cuda = pytest.mark.skipif(not th.cuda.is_available(), reason="cuda required")


def _build_spaces(n_stocks: int = 50, n_factors: int = 20):
    obs_space = spaces.Box(low=-np.inf, high=np.inf,
                           shape=(n_stocks, n_factors), dtype=np.float32)
    act_space = spaces.Box(low=0.0, high=1.0,
                           shape=(n_stocks,), dtype=np.float32)
    return obs_space, act_space


def _fill_buffer(buf, *, n_steps: int, n_envs: int, n_stocks: int,
                 n_factors: int, seed: int = 0) -> tuple[
                     list[np.ndarray], np.ndarray]:
    """Fill buffer with deterministic synthetic transitions.

    Returns (last_values_cpu, dones_cpu) for compute_returns_and_advantage.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_steps):
        obs = rng.standard_normal((n_envs, n_stocks, n_factors)).astype(np.float32)
        action = rng.uniform(0.0, 1.0, size=(n_envs, n_stocks)).astype(np.float32)
        reward = rng.standard_normal(n_envs).astype(np.float32) * 0.01
        episode_start = (rng.random(n_envs) < 0.05).astype(np.float32)
        value = th.as_tensor(rng.standard_normal(n_envs).astype(np.float32),
                             device="cuda" if th.cuda.is_available() else "cpu")
        log_prob = th.as_tensor(rng.standard_normal(n_envs).astype(np.float32),
                                device="cuda" if th.cuda.is_available() else "cpu")
        buf.add(obs, action, reward, episode_start, value, log_prob)
    last_values = th.as_tensor(rng.standard_normal(n_envs).astype(np.float32),
                               device="cuda" if th.cuda.is_available() else "cpu")
    dones = (rng.random(n_envs) < 0.1).astype(bool)
    return last_values, dones


@cuda
def test_buffer_residency_on_cuda():
    obs_space, act_space = _build_spaces(20, 8)
    buf = GPURolloutBuffer(
        buffer_size=16, observation_space=obs_space, action_space=act_space,
        device="cuda", n_envs=4, gae_lambda=0.95, gamma=0.99,
    )
    for name in ("observations", "actions", "rewards", "returns",
                 "episode_starts", "values", "log_probs", "advantages"):
        t = getattr(buf, name)
        assert isinstance(t, th.Tensor), f"{name} should be torch.Tensor"
        assert t.device.type == "cuda", f"{name} on {t.device}, expected cuda"


def test_cpu_device_raises():
    obs_space, act_space = _build_spaces(10, 4)
    with pytest.raises(ValueError, match="GPURolloutBuffer requires cuda"):
        GPURolloutBuffer(
            buffer_size=8, observation_space=obs_space, action_space=act_space,
            device="cpu", n_envs=2,
        )


@cuda
def test_add_accepts_numpy_and_torch():
    obs_space, act_space = _build_spaces(15, 6)
    n_steps, n_envs = 4, 3
    rng = np.random.default_rng(123)

    obs = rng.standard_normal((n_envs, 15, 6)).astype(np.float32)
    action = rng.uniform(0.0, 1.0, size=(n_envs, 15)).astype(np.float32)
    reward = rng.standard_normal(n_envs).astype(np.float32)
    episode_start = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    value = th.as_tensor(rng.standard_normal(n_envs).astype(np.float32),
                         device="cuda")
    log_prob = th.as_tensor(rng.standard_normal(n_envs).astype(np.float32),
                            device="cuda")

    buf_np = GPURolloutBuffer(n_steps, obs_space, act_space, device="cuda",
                              n_envs=n_envs)
    buf_th = GPURolloutBuffer(n_steps, obs_space, act_space, device="cuda",
                              n_envs=n_envs)

    buf_np.add(obs, action, reward, episode_start, value, log_prob)
    buf_th.add(
        th.as_tensor(obs, device="cuda"),
        th.as_tensor(action, device="cuda"),
        th.as_tensor(reward, device="cuda"),
        th.as_tensor(episode_start, device="cuda"),
        value,
        log_prob,
    )

    for name in ("observations", "actions", "rewards", "episode_starts",
                 "values", "log_probs"):
        a = getattr(buf_np, name)[0]
        b = getattr(buf_th, name)[0]
        assert th.allclose(a, b, atol=1e-6), f"{name} mismatch np vs torch input"


@cuda
def test_gae_matches_cpu_baseline():
    obs_space, act_space = _build_spaces(20, 8)
    n_steps, n_envs = 32, 4
    n_stocks, n_factors = 20, 8
    gae_lambda, gamma = 0.95, 0.99

    cpu_buf = RolloutBuffer(
        buffer_size=n_steps, observation_space=obs_space,
        action_space=act_space, device="cpu", n_envs=n_envs,
        gae_lambda=gae_lambda, gamma=gamma,
    )
    gpu_buf = GPURolloutBuffer(
        buffer_size=n_steps, observation_space=obs_space,
        action_space=act_space, device="cuda", n_envs=n_envs,
        gae_lambda=gae_lambda, gamma=gamma,
    )

    rng = np.random.default_rng(42)
    # Pre-generate the full transition stream so the two buffers see
    # IDENTICAL inputs (same arrays, same draws).
    transitions = []
    for _ in range(n_steps):
        transitions.append((
            rng.standard_normal((n_envs, n_stocks, n_factors)).astype(np.float32),
            rng.uniform(0.0, 1.0, size=(n_envs, n_stocks)).astype(np.float32),
            rng.standard_normal(n_envs).astype(np.float32) * 0.01,
            (rng.random(n_envs) < 0.05).astype(np.float32),
            rng.standard_normal(n_envs).astype(np.float32),  # value (cpu numpy)
            rng.standard_normal(n_envs).astype(np.float32),  # log_prob
        ))
    last_vals_np = rng.standard_normal(n_envs).astype(np.float32)
    dones = (rng.random(n_envs) < 0.1).astype(bool)

    for obs, act, rew, eps, val, lp in transitions:
        cpu_buf.add(
            obs, act, rew, eps,
            th.as_tensor(val, device="cpu"),
            th.as_tensor(lp, device="cpu"),
        )
        gpu_buf.add(
            obs, act, rew, eps,
            th.as_tensor(val, device="cuda"),
            th.as_tensor(lp, device="cuda"),
        )

    cpu_buf.compute_returns_and_advantage(
        last_values=th.as_tensor(last_vals_np, device="cpu"), dones=dones,
    )
    gpu_buf.compute_returns_and_advantage(
        last_values=th.as_tensor(last_vals_np, device="cuda"), dones=dones,
    )

    cpu_adv = cpu_buf.advantages
    cpu_ret = cpu_buf.returns
    gpu_adv = gpu_buf.advantages.detach().cpu().numpy()
    gpu_ret = gpu_buf.returns.detach().cpu().numpy()

    np.testing.assert_allclose(gpu_adv, cpu_adv, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(gpu_ret, cpu_ret, atol=1e-4, rtol=1e-4)


@cuda
def test_get_yields_cuda_samples():
    obs_space, act_space = _build_spaces(12, 5)
    n_steps, n_envs = 16, 4
    n_stocks, n_factors = 12, 5
    buf = GPURolloutBuffer(n_steps, obs_space, act_space, device="cuda",
                           n_envs=n_envs, gae_lambda=0.95, gamma=0.99)
    last_values, dones = _fill_buffer(buf, n_steps=n_steps, n_envs=n_envs,
                                      n_stocks=n_stocks, n_factors=n_factors,
                                      seed=7)
    buf.compute_returns_and_advantage(last_values=last_values, dones=dones)

    batch_size = 8
    total = n_steps * n_envs
    seen = 0
    batches = list(buf.get(batch_size=batch_size))
    assert len(batches) == total // batch_size
    for batch in batches:
        for name in ("observations", "actions", "old_values", "old_log_prob",
                     "advantages", "returns"):
            t = getattr(batch, name)
            assert t.device.type == "cuda", f"{name} on {t.device}"
        assert batch.observations.shape == (batch_size, n_stocks, n_factors)
        assert batch.actions.shape == (batch_size, n_stocks)
        assert batch.old_values.shape == (batch_size,)
        assert batch.old_log_prob.shape == (batch_size,)
        assert batch.advantages.shape == (batch_size,)
        assert batch.returns.shape == (batch_size,)
        seen += batch.observations.shape[0]
    assert seen == total


@cuda
def test_buffer_size_estimate():
    """At the smoke-test config (n_steps=128, n_envs=12, 50x20 obs) the
    GPU buffer should comfortably fit in <1 GB of VRAM. This sanity-checks
    the dominant term: obs storage = n_steps * n_envs * n_stocks * n_factors * 4 B.
    """
    obs_space, act_space = _build_spaces(50, 20)
    n_steps, n_envs = 128, 12
    th.cuda.empty_cache()
    th.cuda.reset_peak_memory_stats()
    base = th.cuda.memory_allocated()

    buf = GPURolloutBuffer(n_steps, obs_space, act_space, device="cuda",
                           n_envs=n_envs, gae_lambda=0.95, gamma=0.99)

    after = th.cuda.memory_allocated()
    bytes_used = after - base
    one_gb = 1024 ** 3
    assert bytes_used < one_gb, (
        f"GPURolloutBuffer at smoke config used {bytes_used / 1e6:.1f} MB "
        f">= 1 GB (expected obs term only ~6 MB)"
    )
    # And confirm the obs tensor is the dominant chunk:
    obs_bytes = buf.observations.element_size() * buf.observations.numel()
    assert obs_bytes < 50 * 1024 * 1024, (
        f"obs tensor {obs_bytes / 1e6:.1f} MB looks too big at smoke config"
    )
