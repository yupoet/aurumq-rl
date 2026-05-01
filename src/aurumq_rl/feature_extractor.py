"""Per-stock features extractor with shared MLP across stocks (Deep Sets)."""
from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class PerStockExtractor(BaseFeaturesExtractor):
    """Apply a shared MLP per stock; return both per-stock and pooled features.

    Returned features are a TensorDict-like dict (we use a regular dict because
    SB3 doesn't insist on a Tensor return — see PerStockEncoderPolicy.forward).

    Output keys:
      - "per_stock":  (B, n_stocks, out_dim) — used by action head
      - "pooled":     (B,         out_dim) — mean-pool across stocks, used by value head
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        hidden: tuple[int, ...] = (128, 64),
        out_dim: int = 32,
    ):
        # observation_space.shape = (n_stocks, n_factors)
        n_stocks, n_factors = observation_space.shape
        # SB3's BaseFeaturesExtractor stores features_dim — set it to per-stock out_dim
        # (we won't use SB3's mlp_extractor split anyway, see policy.py)
        super().__init__(observation_space, features_dim=out_dim)
        self.n_stocks = n_stocks
        self.n_factors = n_factors
        self.out_dim = out_dim

        layers: list[nn.Module] = []
        prev = n_factors
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        # obs: (B, n_stocks, n_factors)
        b, s, f = obs.shape
        flat = obs.reshape(b * s, f)
        encoded = self.mlp(flat).reshape(b, s, self.out_dim)
        pooled = encoded.mean(dim=1)
        return {"per_stock": encoded, "pooled": pooled}
