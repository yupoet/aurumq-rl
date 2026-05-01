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
