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
      - "per_stock":  (B, n_stocks, out_dim) — cross-section centered embeddings,
                      used by the action head. Per-batch mean over stocks is ~0,
                      so the actor scores stocks relative to today's market.
      - "pooled":     (B, 2 * out_dim) — concat(market_mean, opportunity_max),
                      used by the value head. ``market_mean`` is the LayerNorm'd
                      stock-axis mean BEFORE centering (i.e. today's market
                      baseline); ``opportunity_max`` is the per-feature max over
                      the centered embeddings (the strongest cross-section
                      deviation per channel).

    The market-mean is computed BEFORE centering on purpose: computing it AFTER
    centering yields ~0 (centering subtracts that very mean) and starves the
    value head of any market-level signal — that was a latent bug in the
    earlier dual-pooling attempt.
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
        # Pooled features (consumed by the value head) are concat of two
        # out_dim-sized vectors. Expose for caller introspection.
        self.pooled_dim = 2 * out_dim

        layers: list[nn.Module] = []
        prev = n_factors
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        # obs: (B, n_stocks, n_factors)
        b, s, f = obs.shape
        flat = obs.reshape(b * s, f)
        normed = self.norm(self.mlp(flat).reshape(b, s, self.out_dim))

        # market_mean is computed BEFORE centering on purpose (see class docstring).
        market_mean = normed.mean(dim=1)
        centered = normed - market_mean.unsqueeze(1)

        opportunity_max = centered.max(dim=1).values
        pooled = torch.cat([market_mean, opportunity_max], dim=-1)

        return {"per_stock": centered, "pooled": pooled}
