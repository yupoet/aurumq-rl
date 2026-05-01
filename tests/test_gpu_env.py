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
