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
    env = GPUStockPickingEnv(
        panel, returns, valid_mask, n_envs=3, episode_length=30, forward_period=5, seed=42
    )
    obs = env.reset()
    # SB3 VecEnv contract requires numpy obs; internal panel stays on cuda.
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (3, 50, 20)  # (n_envs, n_stocks, n_factors)
    assert obs.dtype == np.float32
    assert env.panel.device.type == "cuda"  # internal residency unchanged
    # Each env got an independently sampled start
    starts = env.t.cpu().tolist()
    assert all(0 <= s for s in starts)


@cuda
def test_step_returns_obs_rewards_dones_infos():
    syn = make_synthetic_panel(n_dates=120)
    panel, returns, valid_mask = _panel_to_cuda(syn)
    env = GPUStockPickingEnv(
        panel,
        returns,
        valid_mask,
        n_envs=2,
        episode_length=50,
        forward_period=5,
        top_k=10,
        cost_bps=0.0,
        seed=0,
    )
    env.reset()
    actions = np.random.default_rng(0).standard_normal((2, 50)).astype(np.float32)
    env.step_async(actions)
    obs, rewards, dones, infos = env.step_wait()

    assert isinstance(obs, np.ndarray) and obs.shape == (2, 50, 20)
    assert isinstance(rewards, np.ndarray) and rewards.shape == (2,) and rewards.dtype == np.float32
    assert isinstance(dones, np.ndarray) and dones.shape == (2,) and dones.dtype == bool
    assert isinstance(infos, list) and len(infos) == 2


@cuda
def test_auto_reset_on_episode_end():
    syn = make_synthetic_panel(n_dates=60)
    panel, returns, valid_mask = _panel_to_cuda(syn)
    env = GPUStockPickingEnv(
        panel, returns, valid_mask, n_envs=1, episode_length=5, forward_period=2, top_k=5, seed=0
    )
    env.reset()
    initial_t = env.t.clone()
    actions = np.zeros((1, 50), dtype=np.float32)
    last_info = None
    last_done = False
    for _ in range(6):
        env.step_async(actions)
        _, _, dones, infos = env.step_wait()
        if dones[0]:
            last_done = True
            last_info = infos[0]
            break
    assert last_done, "episode_length=5 should fire done within 6 steps"
    assert "episode" in last_info, "done env must populate info['episode']"
    assert {"r", "l"} <= last_info["episode"].keys()
    assert env.steps_done[0].item() == 0, "steps_done resets after auto-reset"
    # New start was sampled (very likely different)
    assert env.t[0].item() != initial_t[0].item()


@cuda
def test_vecenv_required_methods():
    syn = make_synthetic_panel()
    panel, returns, valid_mask = _panel_to_cuda(syn)
    env = GPUStockPickingEnv(panel, returns, valid_mask, n_envs=2)
    assert env.get_attr("render_mode") == [None, None]
    assert env.env_is_wrapped(object) == [False, False]
    env.close()


@cuda
def test_nan_returns_do_not_poison_rewards():
    """C2: NaN forward returns (missing cells) must not produce NaN rewards."""
    syn = make_synthetic_panel(n_dates=60, n_stocks=50, n_factors=20)
    ret = syn.return_array.copy()
    ret[:] = np.nan  # worst case: every selected stock has a NaN return
    panel = torch.from_numpy(syn.factor_array).to("cuda")
    returns = torch.from_numpy(ret).to("cuda")
    valid_mask = torch.ones(panel.shape[:2], dtype=torch.bool, device="cuda")
    env = GPUStockPickingEnv(
        panel,
        returns,
        valid_mask,
        n_envs=2,
        episode_length=10,
        forward_period=5,
        top_k=5,
        cost_bps=0.0,
        seed=0,
    )
    env.reset()
    env.step_async(np.zeros((2, 50), dtype=np.float32))
    _, rewards, _, _ = env.step_wait()
    assert np.isfinite(rewards).all(), "NaN returns must be nan-guarded to 0"


# ---------------------------------------------------------------------------
# CPU-device tests (C4). GPUStockPickingEnv accepts device="cpu" so the
# masking / reward logic is testable without CUDA; the cuda-residency guard
# still applies for the default device="cuda".
# ---------------------------------------------------------------------------


def _make_cpu_env(returns_np, valid_np, top_k, n_stocks, cost_bps=0.0, turnover_coef=0.0):
    n_dates = returns_np.shape[0]
    panel = torch.zeros((n_dates, n_stocks, 3), dtype=torch.float32)
    returns = torch.from_numpy(returns_np.astype(np.float32))
    valid = torch.from_numpy(valid_np)
    env = GPUStockPickingEnv(
        panel,
        returns,
        valid,
        n_envs=1,
        episode_length=10,
        forward_period=5,
        top_k=top_k,
        cost_bps=cost_bps,
        turnover_coef=turnover_coef,
        device="cpu",
        seed=0,
    )
    env.reset()
    env.t.zero_()  # deterministic start at t=0
    env.last_obs_t = env.t.clone()
    return env


def test_default_device_still_requires_cuda_tensor():
    panel = torch.zeros((30, 5, 3))
    returns = torch.zeros((30, 5))
    valid = torch.ones((30, 5), dtype=torch.bool)
    with pytest.raises(ValueError, match="cuda"):
        GPUStockPickingEnv(panel, returns, valid, n_envs=1)


def test_cpu_device_construction_and_step():
    n_dates, n_stocks = 30, 6
    returns_np = np.zeros((n_dates, n_stocks), dtype=np.float32)
    valid_np = np.ones((n_dates, n_stocks), dtype=np.bool_)
    env = _make_cpu_env(returns_np, valid_np, top_k=3, n_stocks=n_stocks)
    env.step_async(np.zeros((1, n_stocks), dtype=np.float32))
    obs, rewards, dones, infos = env.step_wait()
    assert obs.shape == (1, n_stocks, 3)
    assert np.isfinite(rewards).all()


def test_invalid_stock_never_credited_in_reward():
    """C4: a stock invalid at t (e.g. closed limit-up) must not enter the
    top-K reward even when the policy scores it highest."""
    n_dates, n_stocks = 30, 6
    returns_np = np.full((n_dates, n_stocks), 0.01, dtype=np.float32)
    returns_np[0, 0] = 1.0  # huge return on the invalid stock
    valid_np = np.ones((n_dates, n_stocks), dtype=np.bool_)
    valid_np[0, 0] = False  # limit-up at decision date t=0
    env = _make_cpu_env(returns_np, valid_np, top_k=2, n_stocks=n_stocks)
    action = np.zeros((1, n_stocks), dtype=np.float32)
    action[0, 0] = 1.0  # policy loves the invalid stock
    env.step_async(action)
    _, rewards, _, _ = env.step_wait()
    # Reward = mean over 2 valid picks (0.01 each), NOT (1.0 + 0.01) / 2.
    assert rewards[0] == pytest.approx(0.01, abs=1e-6)


def test_fewer_valid_than_top_k_uses_only_valid_picks():
    """C4 adjacent bug: when valid_count < top_k, torch.topk pads with -inf
    picks; their (real, possibly large) returns must NOT enter the mean."""
    n_dates, n_stocks = 30, 5
    returns_np = np.full((n_dates, n_stocks), 1.0, dtype=np.float32)
    returns_np[0, 0] = 0.02
    returns_np[0, 1] = 0.04
    valid_np = np.zeros((n_dates, n_stocks), dtype=np.bool_)
    valid_np[0, :2] = True  # only 2 valid stocks; top_k=4
    valid_np[1:] = True
    env = _make_cpu_env(returns_np, valid_np, top_k=4, n_stocks=n_stocks)
    env.step_async(np.zeros((1, n_stocks), dtype=np.float32))
    _, rewards, _, _ = env.step_wait()
    # Mean over the 2 REAL picks only: (0.02 + 0.04) / 2 = 0.03.
    # The old code averaged over top_k=4 including two invalid stocks
    # whose returns are 1.0 → (0.02 + 0.04 + 1.0 + 1.0) / 4 = 0.515.
    assert rewards[0] == pytest.approx(0.03, abs=1e-6)


def test_turnover_penalty_ignores_padding_picks():
    """C4 follow-up: when valid_count < top_k, the padded -inf picks in
    top_idx / prev_top_idx must not create spurious Jaccard overlap that
    erases the turnover penalty."""
    n_dates, n_stocks = 30, 6
    returns_np = np.zeros((n_dates, n_stocks), dtype=np.float32)
    valid_np = np.zeros((n_dates, n_stocks), dtype=np.bool_)
    valid_np[0, [0, 1]] = True  # t=0: only stocks {0, 1} valid; top_k=4
    valid_np[1, [2, 3]] = True  # t=1: portfolio flips completely to {2, 3}
    valid_np[2:] = True
    env = _make_cpu_env(returns_np, valid_np, top_k=4, n_stocks=n_stocks, turnover_coef=1.0)
    action = np.zeros((1, n_stocks), dtype=np.float32)
    env.step_async(action)
    _, rewards_1, _, _ = env.step_wait()
    # First step: no previous holdings → full turnover → -coef * 1.0.
    assert rewards_1[0] == pytest.approx(-1.0, abs=1e-6)
    env.step_async(action)
    _, rewards_2, _, _ = env.step_wait()
    # Real picks changed completely ({0,1} → {2,3}): full turnover again.
    # The old code compared padded index SETS ({0,1,2,3} vs {2,3,0,1}) →
    # spurious overlap 4 → zero turnover penalty.
    assert rewards_2[0] == pytest.approx(-1.0, abs=1e-6)


def test_zero_valid_stocks_yields_zero_reward_minus_cost():
    n_dates, n_stocks = 30, 4
    returns_np = np.full((n_dates, n_stocks), 1.0, dtype=np.float32)
    valid_np = np.ones((n_dates, n_stocks), dtype=np.bool_)
    valid_np[0] = False  # nothing tradeable at t=0
    env = _make_cpu_env(returns_np, valid_np, top_k=2, n_stocks=n_stocks, cost_bps=30.0)
    env.step_async(np.zeros((1, n_stocks), dtype=np.float32))
    _, rewards, _, _ = env.step_wait()
    assert rewards[0] == pytest.approx(-30.0 / 1e4, abs=1e-7)


@cuda
def test_sb3_ppo_one_rollout():
    """Smoke: SB3 PPO can collect one rollout against our VecEnv without crashing."""
    from stable_baselines3 import PPO

    syn = make_synthetic_panel(n_dates=120, n_stocks=20, n_factors=8)
    panel, returns, valid_mask = _panel_to_cuda(syn)
    env = GPUStockPickingEnv(
        panel, returns, valid_mask, n_envs=4, episode_length=30, forward_period=5, top_k=4, seed=0
    )
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, verbose=0, device="cuda")
    model.learn(total_timesteps=256)
    # If we got here, collect_rollouts + train both worked
