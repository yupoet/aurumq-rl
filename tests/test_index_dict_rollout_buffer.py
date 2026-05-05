"""Phase 21 IndexOnlyDictRolloutBuffer: stores t-indices, gathers Dict obs lazily."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from aurumq_rl.index_dict_rollout_buffer import IndexOnlyDictRolloutBuffer

CUDA_OK = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_OK, reason="CUDA required for buffer tests")


def _make_buffer(buffer_size=8, n_envs=2, S=4, F=3, R=8):
    obs_space = gym.spaces.Dict({
        "stock": gym.spaces.Box(-np.inf, np.inf, (S, F), dtype=np.float32),
        "regime": gym.spaces.Box(-np.inf, np.inf, (R,), dtype=np.float32),
        "valid_mask": gym.spaces.Box(0.0, 1.0, (S,), dtype=np.float32),
    })
    act_space = gym.spaces.Box(0.0, 1.0, (S,), dtype=np.float32)
    return IndexOnlyDictRolloutBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=act_space,
        device="cuda",
        gae_lambda=1.0,
        gamma=0.99,
        n_envs=n_envs,
    )


def test_buffer_storage_is_t_indices_only():
    buf = _make_buffer(buffer_size=8, n_envs=2, S=4, F=3, R=8)
    buf.reset()
    assert buf.t_buffer.shape == (8, 2)
    assert buf.t_buffer.dtype == torch.long
    # observations dict either None or all values None (sentinel-out)
    assert buf.observations is None or all(v is None for v in buf.observations.values())


def test_buffer_add_and_get_roundtrip():
    T, S, F, R = 30, 4, 3, 8
    panel = torch.randn(T, S, F, device="cuda")
    regime = torch.randn(T, R, device="cuda")
    valid = torch.ones(T, S, device="cuda")
    last_t = torch.zeros(2, dtype=torch.long, device="cuda")

    buf = _make_buffer()
    buf.reset()
    buf.attach_providers(
        stock_provider=lambda t: panel.index_select(0, t),
        regime_provider=lambda t: regime.index_select(0, t),
        mask_provider=lambda t: valid.index_select(0, t),
        obs_index_provider=lambda: last_t,
    )

    for step in range(8):
        last_t.copy_(torch.tensor([step, step + 10], dtype=torch.long, device="cuda"))
        obs = {
            "stock": np.zeros((2, S, F), dtype=np.float32),
            "regime": np.zeros((2, R), dtype=np.float32),
            "valid_mask": np.ones((2, S), dtype=np.float32),
        }
        action = np.random.uniform(0, 1, (2, S)).astype(np.float32)
        reward = np.array([0.1, 0.2], dtype=np.float32)
        episode_start = np.array([0.0, 0.0], dtype=np.float32)
        value = torch.tensor([0.0, 0.0], device="cuda")
        log_prob = torch.tensor([-1.0, -1.0], device="cuda")
        buf.add(obs, action, reward, episode_start, value, log_prob)

    last_values = torch.zeros(2, device="cuda")
    dones = np.zeros(2, dtype=np.float32)
    buf.compute_returns_and_advantage(last_values, dones)

    samples = next(buf.get(batch_size=4))
    assert isinstance(samples.observations, dict)
    assert samples.observations["stock"].shape == (4, S, F)
    assert samples.observations["regime"].shape == (4, R)
    assert samples.observations["valid_mask"].shape == (4, S)
    assert samples.actions.shape == (4, S)


def test_buffer_raises_without_providers():
    buf = _make_buffer()
    buf.reset()
    obs = {
        "stock": np.zeros((2, 4, 3), dtype=np.float32),
        "regime": np.zeros((2, 8), dtype=np.float32),
        "valid_mask": np.ones((2, 4), dtype=np.float32),
    }
    with pytest.raises(RuntimeError, match="not attached"):
        buf.add(
            obs,
            np.zeros((2, 4), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            torch.zeros(2, device="cuda"),
            torch.zeros(2, device="cuda"),
        )
