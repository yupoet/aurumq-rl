"""Phase 21 feature extractor: PerStockEncoderV2, RegimeEncoder, masked_mean."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from aurumq_rl.feature_extractor import (
    PerStockEncoderV2,
    RegimeEncoder,
    masked_mean,
)


def test_per_stock_encoder_v2_shape():
    enc = PerStockEncoderV2(n_factors=10, hidden=(32, 16), out_dim=8)
    x = torch.randn(4, 12, 10)
    out = enc(x)
    assert out.shape == (4, 12, 8)


def test_per_stock_encoder_v2_layer_norm_active():
    enc = PerStockEncoderV2(n_factors=10, hidden=(32, 16), out_dim=8)
    x = torch.randn(4, 12, 10) * 100.0
    out = enc(x)
    # LayerNorm over last dim → per-row mean ~0, std ~1 (with affine init=1/0)
    assert out.std(dim=-1).mean().item() == pytest.approx(1.0, abs=0.5)


def test_per_stock_encoder_v2_grad_flows():
    enc = PerStockEncoderV2(n_factors=4, hidden=(8,), out_dim=2)
    x = torch.randn(2, 3, 4, requires_grad=True)
    out = enc(x).sum()
    out.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_regime_encoder_shape():
    re = RegimeEncoder(regime_dim=8, hidden=64, out_dim=16)
    x = torch.randn(4, 8)
    out = re(x)
    assert out.shape == (4, 16)


def test_regime_encoder_layer_norm_active():
    re = RegimeEncoder(regime_dim=8, hidden=64, out_dim=16)
    x = torch.randn(4, 8) * 100.0
    out = re(x)
    assert out.std(dim=-1).mean().item() == pytest.approx(1.0, abs=0.5)


def test_masked_mean_correctness():
    x = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
    ])  # (2, 3, 2)
    mask = torch.tensor([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ])
    expected = torch.tensor([
        [(1 + 3) / 2, (2 + 4) / 2],
        [(30 + 50) / 2, (40 + 60) / 2],
    ])
    out = masked_mean(x, mask)
    torch.testing.assert_close(out, expected)


def test_masked_mean_zero_mask_does_not_explode():
    x = torch.randn(2, 3, 4)
    mask = torch.zeros(2, 3)
    out = masked_mean(x, mask)
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()


def test_masked_mean_grad_flows():
    x = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    masked_mean(x, mask).sum().backward()
    assert x.grad is not None
