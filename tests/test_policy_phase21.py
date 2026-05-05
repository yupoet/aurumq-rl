"""Phase 21 PerStockEncoderPolicyV2: Dict obs, Box action, hard mask, true b2."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from aurumq_rl.feature_extractor import (
    PerStockEncoderV2,
    RegimeEncoder,
    masked_mean,
)
from aurumq_rl.policy import PerStockEncoderPolicyV2


def _make_obs_space(S=8, F=5, R=8):
    return gym.spaces.Dict({
        "stock": gym.spaces.Box(-np.inf, np.inf, (S, F), dtype=np.float32),
        "regime": gym.spaces.Box(-np.inf, np.inf, (R,), dtype=np.float32),
        "valid_mask": gym.spaces.Box(0.0, 1.0, (S,), dtype=np.float32),
    })


def _make_action_space(S=8):
    return gym.spaces.Box(0.0, 1.0, (S,), dtype=np.float32)


def _make_obs_tensors(B=2, S=8, F=5, R=8, mask=None):
    return {
        "stock": torch.randn(B, S, F),
        "regime": torch.randn(B, R),
        "valid_mask": torch.ones(B, S) if mask is None else mask,
    }


def _build_policy(S=8, F=5, R=8):
    obs_space = _make_obs_space(S, F, R)
    act_space = _make_action_space(S)
    lr = lambda _: 1e-4
    return PerStockEncoderPolicyV2(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lr,
        encoder_hidden=(32, 16),
        encoder_out_dim=8,
        regime_encoder_hidden=16,
        regime_encoder_out_dim=4,
        critic_token_hidden=16,
    )


def test_policy_constructs_with_dict_obs_space():
    p = _build_policy()
    assert isinstance(p.stock_encoder, PerStockEncoderV2)
    assert isinstance(p.regime_encoder, RegimeEncoder)
    assert hasattr(p, "log_std")
    assert p.log_std.shape == (8,)


def test_forward_returns_action_value_logprob():
    torch.manual_seed(0)
    p = _build_policy()
    obs = _make_obs_tensors()
    actions, values, log_prob = p.forward(obs, deterministic=False)
    assert actions.shape == (2, 8)
    assert values.shape == (2,)
    assert log_prob.shape == (2,)


def test_forward_deterministic_returns_loc():
    torch.manual_seed(0)
    p = _build_policy()
    obs = _make_obs_tensors()
    actions, _, _ = p.forward(obs, deterministic=True)
    actions2, _, _ = p.forward(obs, deterministic=True)
    torch.testing.assert_close(actions, actions2)


def test_evaluate_actions_consistent_logprob_with_forward():
    torch.manual_seed(0)
    p = _build_policy()
    obs = _make_obs_tensors()
    actions, _, log_prob_fwd = p.forward(obs, deterministic=False)
    values, log_prob_eval, _ = p.evaluate_actions(obs, actions)
    torch.testing.assert_close(log_prob_fwd, log_prob_eval, rtol=1e-5, atol=1e-6)


def test_invalid_stocks_get_neg_inf_logits():
    torch.manual_seed(0)
    p = _build_policy()
    mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
    ], dtype=torch.float32)
    obs = _make_obs_tensors(mask=mask)
    dist = p.get_distribution(obs)
    loc = dist.loc
    assert (loc[0, 4:] <= -1e8).all()
    assert (loc[0, :4] > -1e8).all()
    assert (loc[1, :4] <= -1e8).all()
    assert (loc[1, 4:] > -1e8).all()


def test_empty_mask_raises():
    p = _build_policy()
    mask = torch.zeros(2, 8)
    obs = _make_obs_tensors(mask=mask)
    with pytest.raises(RuntimeError, match="empty valid_mask"):
        p.forward(obs)


def test_critic_uses_true_b2_not_b1():
    """Different regime → different value (regime path is real, not degenerate)."""
    torch.manual_seed(42)
    p = _build_policy()
    obs_a = _make_obs_tensors()
    obs_b = {k: v.clone() for k, v in obs_a.items()}
    obs_b["regime"] = obs_a["regime"] + 1.0
    v_a = p.predict_values(obs_a)
    v_b = p.predict_values(obs_b)
    assert not torch.allclose(v_a, v_b, rtol=1e-3, atol=1e-3)


def test_predict_values_returns_b_shape():
    p = _build_policy()
    obs = _make_obs_tensors()
    v = p.predict_values(obs)
    assert v.shape == (2,)
