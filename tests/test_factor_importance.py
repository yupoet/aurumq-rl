"""Tests for src/aurumq_rl/factor_importance.py."""
from __future__ import annotations

import numpy as np
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

from aurumq_rl.factor_importance import (
    integrated_gradients,
    per_date_cross_section_shuffle,
    permutation_importance,
)


def _linear_score_fn(weights: torch.Tensor):
    """Closure: scores = obs @ weights, where weights is (n_factors,)."""
    def _fn(obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, n_stocks, n_factors)
        return (obs * weights).sum(dim=-1)
    return _fn


def test_ig_recovers_planted_weight_direction():
    """A linear score with weight=10 on factor 3 should saliency-rank factor 3 first."""
    n_factors = 6
    weights = torch.zeros(n_factors)
    weights[3] = 10.0
    score_fn = _linear_score_fn(weights)
    panel_batch = torch.randn(4, 12, n_factors)
    sal = integrated_gradients(score_fn, panel_batch, n_alpha_steps=20)
    assert sal.shape == (n_factors,)
    assert sal.argmax().item() == 3


def test_per_date_cross_section_shuffle_preserves_marginal():
    panel = torch.randn(8, 30, 5)
    shuffled = per_date_cross_section_shuffle(panel, cols=[1, 3], seed=0)
    # Shape preserved
    assert shuffled.shape == panel.shape
    # Untouched columns identical
    assert torch.allclose(shuffled[:, :, 0], panel[:, :, 0])
    assert torch.allclose(shuffled[:, :, 2], panel[:, :, 2])
    assert torch.allclose(shuffled[:, :, 4], panel[:, :, 4])
    # Per-date marginal of touched cols preserved (sorted values match per date)
    for t in range(panel.shape[0]):
        for c in (1, 3):
            assert torch.allclose(
                torch.sort(shuffled[t, :, c]).values,
                torch.sort(panel[t, :, c]).values,
            )


def test_per_date_shuffle_deterministic_with_seed():
    panel = torch.randn(4, 20, 5)
    a = per_date_cross_section_shuffle(panel, [1, 3], seed=42)
    b = per_date_cross_section_shuffle(panel, [1, 3], seed=42)
    assert torch.equal(a, b)


def test_permutation_importance_identifies_planted_group():
    """Plant the signal in alpha_003 only — `alpha` group must dominate."""
    n_dates, n_stocks, n_factors = 30, 40, 8
    rng = torch.Generator().manual_seed(0)
    panel = torch.randn(n_dates, n_stocks, n_factors, generator=rng)
    weights = torch.zeros(n_factors)
    weights[3] = 5.0
    # Synth returns: depend on factor_3 of each stock, plus noise
    returns = (panel * weights).sum(dim=-1) + 0.01 * torch.randn(n_dates, n_stocks, generator=rng)
    factor_names = [f"alpha_{i:03d}" if i < 4 else f"gtja_{i:03d}" for i in range(n_factors)]
    score_fn = _linear_score_fn(weights)
    out = permutation_importance(
        score_fn, panel, returns, factor_names=factor_names,
        forward_period=2, top_k=5, n_seeds=3, base_seed=0,
    )
    assert "alpha" in out
    assert "gtja" in out
    # alpha group should have strictly larger ic_drop than gtja
    assert out["alpha"]["ic_drop_mean"] > out["gtja"]["ic_drop_mean"]
