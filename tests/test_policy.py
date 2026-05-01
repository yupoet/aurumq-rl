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
    # pooled is concat(market_mean, opportunity_max) → 2 * out_dim
    assert out["pooled"].shape == (4, 64)
    assert ext.pooled_dim == 64


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


def test_policy_optimizer_tracks_all_trainable_params():
    """P0 regression: rebuilt optimizer must see every trainable parameter."""
    policy = _make_policy(n_stocks=50, n_factors=20)
    opt_ids = {id(p) for g in policy.optimizer.param_groups for p in g["params"]}
    missing = [
        name for name, p in policy.named_parameters()
        if p.requires_grad and id(p) not in opt_ids
    ]
    assert missing == [], (
        f"optimizer is missing {len(missing)} trainable parameters: {missing}"
    )


def test_extractor_dual_pooling_shape_and_mean_zero():
    """P1: pooled is concat(market_mean, opportunity_max), per_stock is centered."""
    ext = PerStockExtractor(_obs_space(50, 20), hidden=(64,), out_dim=8)
    obs = torch.randn(2, 50, 20)
    out = ext(obs)
    assert out["per_stock"].shape == (2, 50, 8)
    # cross-section centering — per-batch per_stock has zero mean over stock axis
    assert out["per_stock"].mean(dim=1).abs().max().item() < 1e-5
    # pooled = [market_mean, opportunity_max], so 2*out_dim
    assert out["pooled"].shape == (2, 16)
    assert torch.isfinite(out["per_stock"]).all()
    assert torch.isfinite(out["pooled"]).all()


def test_policy_value_net_input_dim_matches_pooled():
    """P1: value_net's first Linear must accept 2*encoder_out_dim, not encoder_out_dim."""
    policy = _make_policy(n_stocks=50, n_factors=20)
    first_linear = next(m for m in policy.value_net.modules() if isinstance(m, torch.nn.Linear))
    assert first_linear.in_features == 2 * policy._encoder_out_dim


def test_extractor_unique_date_no_dups_matches_vanilla():
    """With all-distinct obs, unique path output equals vanilla within fp32 rounding."""
    torch.manual_seed(0)
    ext_v = PerStockExtractor(_obs_space(50, 20), hidden=(64,), out_dim=8, unique_date=False)
    ext_u = PerStockExtractor(_obs_space(50, 20), hidden=(64,), out_dim=8, unique_date=True)
    # share weights
    ext_u.load_state_dict(ext_v.state_dict())
    obs = torch.randn(4, 50, 20)
    # rows are all distinct (random) → unique path falls through to vanilla
    with torch.no_grad():
        out_v = ext_v(obs)
        out_u = ext_u(obs)
    assert torch.allclose(out_v["per_stock"], out_u["per_stock"], atol=1e-5)
    assert torch.allclose(out_v["pooled"], out_u["pooled"], atol=1e-5)


def test_extractor_unique_date_with_dups_matches_vanilla():
    """With duplicate dates (rows 0 and 2 identical), unique path output matches vanilla."""
    torch.manual_seed(1)
    ext_v = PerStockExtractor(_obs_space(50, 20), hidden=(64,), out_dim=8, unique_date=False)
    ext_u = PerStockExtractor(_obs_space(50, 20), hidden=(64,), out_dim=8, unique_date=True)
    ext_u.load_state_dict(ext_v.state_dict())
    obs = torch.randn(4, 50, 20)
    obs[2] = obs[0].clone()  # row 2 = row 0 (duplicate date)
    obs[3] = obs[1].clone()  # row 3 = row 1
    with torch.no_grad():
        out_v = ext_v(obs)
        out_u = ext_u(obs)
    # output rows should match between paths
    assert torch.allclose(out_v["per_stock"], out_u["per_stock"], atol=1e-5)
    assert torch.allclose(out_v["pooled"], out_u["pooled"], atol=1e-5)
    # rows 0,2 should be identical in both paths (since input rows are identical)
    assert torch.allclose(out_u["per_stock"][0], out_u["per_stock"][2], atol=1e-5)


def test_extractor_unique_date_gradient_flows():
    """Backward through unique path produces non-zero grads on every parameter."""
    torch.manual_seed(2)
    ext = PerStockExtractor(_obs_space(50, 20), hidden=(64,), out_dim=8, unique_date=True)
    obs = torch.randn(4, 50, 20)
    obs[2] = obs[0].clone()  # introduce a duplicate
    out = ext(obs)
    loss = out["per_stock"].pow(2).mean() + out["pooled"].pow(2).mean()
    loss.backward()
    for name, p in ext.named_parameters():
        assert p.grad is not None, f"{name}: grad is None"
        assert p.grad.abs().max().item() > 0, f"{name}: grad is all-zero"
