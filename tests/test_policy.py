"""Tests for src/aurumq_rl/policy.py and feature_extractor.py."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

from aurumq_rl.feature_extractor import PerStockExtractor


def _obs_space(n_stocks=50, n_factors=20):
    return gym.spaces.Box(-np.inf, np.inf, (n_stocks, n_factors), dtype=np.float32)


def test_extractor_output_shapes():
    ext = PerStockExtractor(_obs_space(50, 20), hidden=(128, 64), out_dim=32)
    obs = torch.randn(4, 50, 20)
    out = ext(obs)
    assert out["per_stock"].shape == (4, 50, 32)
    assert out["pooled"].shape == (4, 32)


def test_extractor_param_count_under_budget():
    ext = PerStockExtractor(_obs_space(3014, 343), hidden=(128, 64), out_dim=32)
    n_params = sum(p.numel() for p in ext.parameters())
    assert n_params <= 100_000, f"extractor has {n_params} params, budget is 100K"


def test_extractor_permutation_equivariance_per_stock():
    ext = PerStockExtractor(_obs_space(50, 20), hidden=(64,), out_dim=8)
    ext.eval()
    obs = torch.randn(2, 50, 20)
    pi = torch.randperm(50)
    out_a = ext(obs[:, pi])["per_stock"]
    out_b = ext(obs)["per_stock"][:, pi]
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_extractor_pool_invariance():
    ext = PerStockExtractor(_obs_space(50, 20), hidden=(64,), out_dim=8)
    ext.eval()
    obs = torch.randn(2, 50, 20)
    pi = torch.randperm(50)
    pooled_a = ext(obs[:, pi])["pooled"]
    pooled_b = ext(obs)["pooled"]
    assert torch.allclose(pooled_a, pooled_b, atol=1e-5)


from aurumq_rl.policy import PerStockEncoderPolicy


def _make_policy(n_stocks=50, n_factors=20, lr=1e-3):
    obs_space = _obs_space(n_stocks, n_factors)
    act_space = gym.spaces.Box(0.0, 1.0, (n_stocks,), dtype=np.float32)
    policy = PerStockEncoderPolicy(
        obs_space, act_space, lr_schedule=lambda _: lr,
        encoder_hidden=(64,), encoder_out_dim=16, value_hidden=(32,),
    )
    return policy


def test_policy_param_count_under_budget():
    policy = _make_policy(n_stocks=3014, n_factors=343)
    n_params = sum(p.numel() for p in policy.parameters())
    assert n_params <= 100_000, f"policy has {n_params} params, budget is 100K"


def test_policy_forward_shapes():
    policy = _make_policy(n_stocks=50, n_factors=20)
    obs = torch.randn(4, 50, 20)
    actions, values, log_probs = policy(obs)
    assert actions.shape == (4, 50)
    assert values.shape == (4,)
    assert log_probs.shape == (4,)


def test_policy_action_equivariance_in_eval():
    """Action mean must permute the same way the input does."""
    policy = _make_policy(n_stocks=50, n_factors=20)
    policy.eval()
    obs = torch.randn(3, 50, 20)
    pi = torch.randperm(50)
    # Forward with permuted input vs forward then permute output
    feats_a = policy._features(obs[:, pi])
    feats_b = policy._features(obs)
    scores_a = policy.action_net(feats_a["per_stock"]).squeeze(-1)
    scores_b = policy.action_net(feats_b["per_stock"]).squeeze(-1)
    assert torch.allclose(scores_a, scores_b[:, pi], atol=1e-5)


def test_policy_value_invariance():
    policy = _make_policy(n_stocks=50, n_factors=20)
    policy.eval()
    obs = torch.randn(3, 50, 20)
    pi = torch.randperm(50)
    v_a = policy.predict_values(obs[:, pi])
    v_b = policy.predict_values(obs)
    assert torch.allclose(v_a, v_b, atol=1e-5)


@cuda
def test_policy_bf16_autocast_finite():
    policy = _make_policy(n_stocks=50, n_factors=20).to("cuda")
    obs = torch.randn(2, 50, 20, device="cuda")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        actions, values, log_probs = policy(obs)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(values).all()
    assert torch.isfinite(log_probs).all()
