"""PerStockEncoderPolicy — SB3 ActorCriticPolicy with Deep-Sets feature
extraction. Permutation-equivariant action head, permutation-invariant
value head.

We override forward / _predict / evaluate_actions / predict_values
because SB3's default ActorCriticPolicy assumes a flat features tensor
and an mlp_extractor that splits it into (latent_pi, latent_vf). Our
extractor returns a dict and the two heads need different shapes.
"""
from __future__ import annotations

from functools import partial
from typing import Any

import gymnasium as gym
import torch
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from aurumq_rl.feature_extractor import PerStockExtractor


class _IdentityMlpExtractor(nn.Module):
    """No-op standin for SB3's mlp_extractor — we don't use the split."""

    def __init__(self, features_dim: int) -> None:
        super().__init__()
        self.latent_dim_pi = features_dim
        self.latent_dim_vf = features_dim

    def forward(self, features):  # not actually called
        return features, features

    def forward_actor(self, features):
        return features

    def forward_critic(self, features):
        return features


class PerStockEncoderPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args: Any,
        encoder_hidden: tuple[int, ...] = (128, 64),
        encoder_out_dim: int = 32,
        value_hidden: tuple[int, ...] = (64,),
        **kwargs: Any,
    ):
        # Hand the kwargs to SB3 with a custom features_extractor
        kwargs["features_extractor_class"] = PerStockExtractor
        kwargs["features_extractor_kwargs"] = {
            "hidden": encoder_hidden, "out_dim": encoder_out_dim,
        }
        kwargs["share_features_extractor"] = True
        # Disable SB3's mlp_extractor splits with empty net_arch
        kwargs.setdefault("net_arch", dict(pi=[], vf=[]))
        self._encoder_out_dim = encoder_out_dim
        self._value_hidden = value_hidden
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = _IdentityMlpExtractor(features_dim=self._encoder_out_dim)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # Override action_net and value_net with per-stock-aware heads
        n_stocks = self.action_space.shape[0]
        self.action_net = nn.Linear(self._encoder_out_dim, 1)  # per-stock score

        # Value head input is concat(market_mean, opportunity_max) — see
        # PerStockExtractor.forward. So the first Linear must accept
        # 2 * encoder_out_dim, not encoder_out_dim.
        pooled_dim = 2 * self._encoder_out_dim
        layers: list[nn.Module] = []
        prev = pooled_dim
        for h in self._value_hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.value_net = nn.Sequential(*layers)
        # Re-init action distribution + log_std
        self.action_dist = DiagGaussianDistribution(n_stocks)
        self.log_std = nn.Parameter(torch.full((n_stocks,), -0.69, dtype=torch.float32))  # ~log(0.5)

        if getattr(self, "ortho_init", False):
            self.action_net.apply(partial(self.init_weights, gain=0.01))
            self.value_net.apply(partial(self.init_weights, gain=1.0))

        # CRITICAL: rebuild optimizer so it tracks the post-replacement
        # action_net / value_net / log_std parameters. Without this, only
        # parameters present at super()._build() time get optimized — which
        # was the latent bug surfaced by the Phase 9 200k mid-test (OOS
        # metrics bit-identical at every checkpoint because action_mean was
        # frozen at random init, only log_std was drifting).
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _features(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.features_extractor(obs)

    def forward(self, obs, deterministic: bool = False):
        feats = self._features(obs)
        scores = self.action_net(feats["per_stock"]).squeeze(-1)  # (B, S)
        values = self.value_net(feats["pooled"]).squeeze(-1)      # (B,)
        distribution = self.action_dist.proba_distribution(scores, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)
        return actions, values, log_probs

    def evaluate_actions(self, obs, actions):
        feats = self._features(obs)
        scores = self.action_net(feats["per_stock"]).squeeze(-1)
        values = self.value_net(feats["pooled"]).squeeze(-1)
        distribution = self.action_dist.proba_distribution(scores, self.log_std)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_probs, entropy

    def predict_values(self, obs):
        feats = self._features(obs)
        return self.value_net(feats["pooled"]).squeeze(-1)

    def _predict(self, obs, deterministic: bool = False):
        feats = self._features(obs)
        scores = self.action_net(feats["per_stock"]).squeeze(-1)
        distribution = self.action_dist.proba_distribution(scores, self.log_std)
        return distribution.get_actions(deterministic=deterministic)
