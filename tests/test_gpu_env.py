"""Tests for src/aurumq_rl/gpu_env.py."""
from __future__ import annotations

import numpy as np
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

from aurumq_rl.gpu_env import GPUStockPickingEnv
from tests._synthetic_panel import make_synthetic_panel


def _panel_to_cuda(syn, device="cuda"):
    panel = torch.from_numpy(syn.factor_array).to(device)
    returns = torch.from_numpy(syn.return_array).to(device)
    valid_mask = torch.ones(panel.shape[:2], dtype=torch.bool, device=device)
    return panel, returns, valid_mask


@cuda
def test_env_residency_on_cuda():
    syn = make_synthetic_panel(n_dates=60, n_stocks=50, n_factors=20)
    panel, returns, valid_mask = _panel_to_cuda(syn)
    env = GPUStockPickingEnv(panel, returns, valid_mask, n_envs=4)
    assert env.panel.device.type == "cuda"
    assert env.returns.device.type == "cuda"
    assert env.valid_mask.device.type == "cuda"
    assert env.t.device.type == "cuda"
    assert env.num_envs == 4


@cuda
def test_reset_returns_correct_shape_and_dtype():
    syn = make_synthetic_panel()
    panel, returns, valid_mask = _panel_to_cuda(syn)
    env = GPUStockPickingEnv(panel, returns, valid_mask, n_envs=3,
                             episode_length=30, forward_period=5, seed=42)
    obs = env.reset()
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (3, 50, 20)         # (n_envs, n_stocks, n_factors)
    assert obs.dtype == torch.float32
    assert obs.device.type == "cuda"
    # Each env got an independently sampled start
    starts = env.t.cpu().tolist()
    assert all(0 <= s for s in starts)


@cuda
def test_step_returns_obs_rewards_dones_infos():
    syn = make_synthetic_panel(n_dates=120)
    panel, returns, valid_mask = _panel_to_cuda(syn)
    env = GPUStockPickingEnv(panel, returns, valid_mask, n_envs=2, episode_length=50,
                             forward_period=5, top_k=10, cost_bps=0.0, seed=0)
    env.reset()
    actions = np.random.default_rng(0).standard_normal((2, 50)).astype(np.float32)
    env.step_async(actions)
    obs, rewards, dones, infos = env.step_wait()

    assert isinstance(obs, torch.Tensor) and obs.shape == (2, 50, 20)
    assert isinstance(rewards, np.ndarray) and rewards.shape == (2,) and rewards.dtype == np.float32
    assert isinstance(dones, np.ndarray) and dones.shape == (2,) and dones.dtype == bool
    assert isinstance(infos, list) and len(infos) == 2
