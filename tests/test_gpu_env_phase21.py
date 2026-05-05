"""Phase 21 GPUStockPickingEnv: Dict obs, regime tensor plumbing."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from aurumq_rl.gpu_env import GPUStockPickingEnv

CUDA_OK = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_OK, reason="CUDA required for gpu_env tests")


def _make_env(T=120, S=8, F=5, R=8, n_envs=4, episode_length=20):
    panel = torch.randn(T, S, F, device="cuda")
    regime = torch.randn(T, R, device="cuda")
    returns = torch.randn(T, S, device="cuda") * 0.01
    valid = torch.ones(T, S, dtype=torch.bool, device="cuda")
    env = GPUStockPickingEnv(
        panel=panel, regime=regime, returns=returns, valid_mask=valid,
        n_envs=n_envs, episode_length=episode_length,
        forward_period=5, top_k=3, cost_bps=0.0, seed=0,
    )
    return env, (T, S, F, R, n_envs)


def test_observation_space_is_dict_with_three_keys():
    env, (T, S, F, R, n_envs) = _make_env()
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert set(env.observation_space.spaces.keys()) == {"stock", "regime", "valid_mask"}
    assert env.observation_space["stock"].shape == (S, F)
    assert env.observation_space["regime"].shape == (R,)
    assert env.observation_space["valid_mask"].shape == (S,)
    assert env.observation_space["stock"].dtype == np.float32
    assert env.observation_space["regime"].dtype == np.float32
    assert env.observation_space["valid_mask"].dtype == np.float32
    assert env.observation_space["valid_mask"].low.min() == 0.0
    assert env.observation_space["valid_mask"].high.max() == 1.0


def test_reset_returns_dict_with_correct_shapes():
    env, (T, S, F, R, n_envs) = _make_env()
    obs = env.reset()
    assert isinstance(obs, dict)
    assert obs["stock"].shape == (n_envs, S, F)
    assert obs["regime"].shape == (n_envs, R)
    assert obs["valid_mask"].shape == (n_envs, S)
    assert obs["stock"].dtype == np.float32
    assert obs["regime"].dtype == np.float32
    assert obs["valid_mask"].dtype == np.float32


def test_step_wait_returns_dict_obs():
    env, (T, S, F, R, n_envs) = _make_env()
    env.reset()
    actions = np.random.uniform(0, 1, size=(n_envs, S)).astype(np.float32)
    env.step_async(actions)
    obs, rewards, dones, infos = env.step_wait()
    assert isinstance(obs, dict)
    assert obs["stock"].shape == (n_envs, S, F)
    assert obs["regime"].shape == (n_envs, R)
    assert obs["valid_mask"].shape == (n_envs, S)
    assert rewards.shape == (n_envs,)
    assert dones.shape == (n_envs,)


def test_valid_mask_passes_through_from_panel_input():
    panel = torch.randn(20, 4, 3, device="cuda")
    regime = torch.randn(20, 8, device="cuda")
    returns = torch.zeros(20, 4, device="cuda")
    valid = torch.ones(20, 4, dtype=torch.bool, device="cuda")
    valid[:, 1] = False  # Stock 1 untradeable everywhere
    env = GPUStockPickingEnv(
        panel=panel, regime=regime, returns=returns, valid_mask=valid,
        n_envs=2, episode_length=10, forward_period=2, top_k=1,
        cost_bps=0.0, seed=0,
    )
    obs = env.reset()
    assert (obs["valid_mask"][:, 1] == 0.0).all()
    assert (obs["valid_mask"][:, 0] == 1.0).all()


def test_last_obs_t_unchanged_semantics():
    env, _ = _make_env()
    env.reset()
    t0 = env.last_obs_t.clone()
    actions = np.random.uniform(0, 1, size=(env.num_envs, env.n_stocks)).astype(np.float32)
    env.step_async(actions)
    env.step_wait()
    assert env.last_obs_t.shape == t0.shape
    assert env.last_obs_t.dtype == torch.long
