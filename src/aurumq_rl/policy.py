"""Phase 21 PerStockEncoderPolicyV2 — split-head SB3 ActorCriticPolicy.

Architecture
------------

    obs : Dict { stock:(S,F), regime:(R,), valid_mask:(S,) }
        ├── stock  ─── PerStockEncoderV2 ─── stock_emb (B,S,D)
        ├── regime ─── RegimeEncoder      ─── regime_emb (B,R')
        └── (broadcast) regime_b = expand(regime_emb, S) → (B,S,R')
        head_in = concat(stock_emb, regime_b, dim=-1) → (B, S, D+R')

        actor:  Linear(D+R'→1) → mask invalid → Normal(loc, exp(log_std))
        critic: per-stock value MLP (D+R'→H→H) → masked_mean → Linear(H→1)

The action space stays ``Box(0,1,(S,))`` so the env's existing top-K
selection is preserved. The distribution is a per-stock Normal whose loc
is hard-masked at invalid positions; the env will never pick those stocks
because top-K argsort sees ``-1e9`` scores.
"""
from __future__ import annotations

from functools import partial
from typing import Any

import gymnasium as gym
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from aurumq_rl.feature_extractor import (
    PerStockEncoderV2,
    RegimeEncoder,
    masked_mean,
)


class _IdentityFeatures(BaseFeaturesExtractor):
    """Stand-in for SB3's features_extractor — V2 doesn't use it.

    SB3's ``ActorCriticPolicy.__init__`` instantiates a features_extractor and
    calls ``features_extractor.features_dim`` in several code paths. We inherit
    from ``BaseFeaturesExtractor`` with ``features_dim=1`` to satisfy that
    contract, and return the Dict obs unchanged so any incidental call to
    ``extract_features`` doesn't crash.
    """

    def __init__(self, observation_space: gym.spaces.Space) -> None:
        # features_dim=1: BaseFeaturesExtractor asserts > 0; 1 is the minimum.
        # The MlpExtractor will be built with feature_dim=1 and net_arch=dict(pi=[],vf=[])
        # producing latent_dim_pi=latent_dim_vf=1. Parent _build then creates
        # action_net/value_net of that width — our _build immediately overwrites them.
        super().__init__(observation_space, features_dim=1)

    def forward(self, obs):
        return obs  # pass the dict straight through; our methods bypass this


class PerStockEncoderPolicyV2(ActorCriticPolicy):
    """Custom ActorCriticPolicy with split-head architecture.

    The parent's mlp_extractor / action_net / value_net machinery is bypassed
    entirely. We override forward / evaluate_actions / get_distribution /
    predict_values to do the split-head computation ourselves.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Box,
        lr_schedule,
        *args: Any,
        encoder_hidden: tuple[int, ...] = (128, 64),
        encoder_out_dim: int = 32,
        regime_encoder_hidden: int = 64,
        regime_encoder_out_dim: int = 16,
        critic_token_hidden: int = 64,
        **kwargs: Any,
    ) -> None:
        # Use our identity features_extractor to keep parent happy.
        kwargs["features_extractor_class"] = _IdentityFeatures
        kwargs["features_extractor_kwargs"] = {}
        kwargs["share_features_extractor"] = True
        kwargs.setdefault("net_arch", dict(pi=[], vf=[]))

        n_stocks = action_space.shape[0]
        f_stock = observation_space["stock"].shape[1]
        regime_dim = observation_space["regime"].shape[0]

        # Save for _build (called by super().__init__())
        self._encoder_hidden = encoder_hidden
        self._encoder_out_dim = encoder_out_dim
        self._regime_encoder_hidden = regime_encoder_hidden
        self._regime_encoder_out_dim = regime_encoder_out_dim
        self._critic_token_hidden = critic_token_hidden
        self._n_stocks = n_stocks
        self._f_stock = f_stock
        self._regime_dim = regime_dim

        super().__init__(
            observation_space, action_space, lr_schedule, *args, **kwargs
        )

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        head_in_dim = self._encoder_out_dim + self._regime_encoder_out_dim

        self.stock_encoder = PerStockEncoderV2(
            n_factors=self._f_stock,
            hidden=self._encoder_hidden,
            out_dim=self._encoder_out_dim,
        )
        self.regime_encoder = RegimeEncoder(
            regime_dim=self._regime_dim,
            hidden=self._regime_encoder_hidden,
            out_dim=self._regime_encoder_out_dim,
        )
        self.actor_head = nn.Linear(head_in_dim, 1)
        self.value_token_mlp = nn.Sequential(
            nn.Linear(head_in_dim, self._critic_token_hidden),
            nn.ReLU(),
            nn.Linear(self._critic_token_hidden, self._critic_token_hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(self._critic_token_hidden, 1)

        # Per-stock log_std for the Normal action distribution. Initialised at
        # log(0.5) ≈ -0.69 to mirror the V1 PerStockEncoderPolicy default.
        # This OVERWRITES the log_std created by the parent DiagGaussianDistribution
        # path (which used latent_dim=1 → wrong shape). Shape must be (S,).
        self.log_std = nn.Parameter(
            torch.full((self._n_stocks,), -0.69, dtype=torch.float32)
        )

        if getattr(self, "ortho_init", False):
            self.actor_head.apply(partial(self.init_weights, gain=0.01))
            self.value_head.apply(partial(self.init_weights, gain=1.0))
            for m in self.value_token_mlp.modules():
                if isinstance(m, nn.Linear):
                    self.init_weights(m, gain=1.0)

        # CRITICAL: rebuild the optimizer to track the freshly-added
        # parameters. Without this, super()._build()'s optimizer only sees
        # the parent's (empty) mlp_extractor + action_net + value_net and
        # our stock_encoder / regime_encoder / heads stay at random init.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    # ----------------------- Shared compute -----------------------

    def _shared_forward(self, obs):
        """Apply both encoders and concat. Returns (head_in, valid_mask_bool)."""
        stock_x = obs["stock"]
        regime_x = obs["regime"]
        valid_mask = obs["valid_mask"].to(dtype=torch.bool)

        stock_emb = self.stock_encoder(stock_x)                 # (B, S, D)
        regime_emb = self.regime_encoder(regime_x)              # (B, R')
        b, s, _ = stock_emb.shape
        regime_b = regime_emb.unsqueeze(1).expand(-1, s, -1)    # (B, S, R')
        head_in = torch.cat([stock_emb, regime_b], dim=-1)      # (B, S, D+R')
        return head_in, valid_mask

    def _logits(self, head_in, valid_mask):
        logits = self.actor_head(head_in).squeeze(-1)            # (B, S)
        return logits.masked_fill(~valid_mask, -1e9)

    def _value(self, head_in, valid_mask):
        tokens = self.value_token_mlp(head_in)                  # (B, S, H)
        pooled = masked_mean(tokens, valid_mask.to(dtype=tokens.dtype))
        return self.value_head(pooled).squeeze(-1)              # (B,)

    def _make_distribution(self, head_in, valid_mask):
        loc = self._logits(head_in, valid_mask)
        scale = self.log_std.exp().expand_as(loc)
        return torch.distributions.Normal(loc, scale)

    # ----------------------- SB3 contract -----------------------

    def forward(self, obs, deterministic: bool = False):
        head_in, valid_mask = self._shared_forward(obs)
        if not valid_mask.any(dim=1).all():
            raise RuntimeError(
                "empty valid_mask: every sample needs at least one valid stock"
            )
        dist = self._make_distribution(head_in, valid_mask)
        actions = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(actions).sum(dim=-1)            # (B,)
        values = self._value(head_in, valid_mask)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        head_in, valid_mask = self._shared_forward(obs)
        if not valid_mask.any(dim=1).all():
            raise RuntimeError(
                "empty valid_mask in evaluate_actions: every sample needs at least one valid stock"
            )
        dist = self._make_distribution(head_in, valid_mask)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self._value(head_in, valid_mask)
        return values, log_prob, entropy

    def get_distribution(self, obs):
        head_in, valid_mask = self._shared_forward(obs)
        if not valid_mask.any(dim=1).all():
            raise RuntimeError("empty valid_mask in get_distribution")
        return self._make_distribution(head_in, valid_mask)

    def predict_values(self, obs):
        head_in, valid_mask = self._shared_forward(obs)
        return self._value(head_in, valid_mask)

    def _predict(self, obs, deterministic: bool = False):
        head_in, valid_mask = self._shared_forward(obs)
        dist = self._make_distribution(head_in, valid_mask)
        return dist.mean if deterministic else dist.rsample()
