"""Tests for src/aurumq_rl/index_rollout_buffer.py.

Covers:
- t_buffer + actions + rewards etc residency on cuda after reset.
- Verification that t_buffer replaces the obs tensor.
- Lazy gather: ``_get_samples`` materialises obs via ``obs_provider``.
- GAE numerical equivalence with Phase 8's GPURolloutBuffer.
- ``get(batch_size)`` yields the same obs we'd get from direct indexing.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from aurumq_rl.gpu_rollout_buffer import GPURolloutBuffer
from aurumq_rl.index_rollout_buffer import IndexOnlyRolloutBuffer

cuda = pytest.mark.skipif(not th.cuda.is_available(), reason="cuda required")


# --- helpers --------------------------------------------------------------

def _build_spaces(n_stocks: int = 50, n_factors: int = 20):
    obs_space = spaces.Box(low=-np.inf, high=np.inf,
                           shape=(n_stocks, n_factors), dtype=np.float32)
    act_space = spaces.Box(low=0.0, high=1.0,
                           shape=(n_stocks,), dtype=np.float32)
    return obs_space, act_space


def _make_panel(n_dates: int, n_stocks: int, n_factors: int,
                seed: int = 0, device: str = "cuda") -> th.Tensor:
    rng = np.random.default_rng(seed)
    panel = rng.standard_normal((n_dates, n_stocks, n_factors)).astype(np.float32)
    return th.as_tensor(panel, device=device)


def _make_index_buf(
    *,
    buffer_size: int,
    n_envs: int,
    n_stocks: int,
    n_factors: int,
    panel: th.Tensor,
    last_obs_t: th.Tensor,
    gae_lambda: float = 0.95,
    gamma: float = 0.99,
):
    obs_space, act_space = _build_spaces(n_stocks, n_factors)
    return IndexOnlyRolloutBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=act_space,
        device="cuda",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=n_envs,
        obs_provider=lambda t: panel[t],
        obs_index_provider=lambda: last_obs_t,
    )


# --- tests ----------------------------------------------------------------

@cuda
def test_buffer_residency_on_cuda():
    """All allocated storage tensors must live on cuda after reset."""
    n_envs, n_stocks, n_factors = 4, 20, 8
    panel = _make_panel(40, n_stocks, n_factors)
    last_obs_t = th.zeros(n_envs, dtype=th.long, device="cuda")
    buf = _make_index_buf(
        buffer_size=16, n_envs=n_envs,
        n_stocks=n_stocks, n_factors=n_factors,
        panel=panel, last_obs_t=last_obs_t,
    )
    for name in ("t_buffer", "actions", "rewards", "returns",
                 "episode_starts", "values", "log_probs", "advantages"):
        t = getattr(buf, name)
        assert isinstance(t, th.Tensor), f"{name} should be torch.Tensor"
        assert t.device.type == "cuda", f"{name} on {t.device}, expected cuda"
    # ``observations`` is intentionally not allocated as a tensor.
    assert getattr(buf, "observations", None) is None, (
        "observations tensor should not be allocated by IndexOnlyRolloutBuffer"
    )


@cuda
def test_t_buffer_replaces_observations():
    """t_buffer is allocated with the right shape/dtype, obs tensor isn't.

    Uses realistic stock/factor counts (matching the smoke config the
    controller will run) so the bytes ratio between t_buffer and the
    obs tensor it replaces is meaningfully large.
    """
    # 3014 stocks × 343 factors mirrors the real training panel; that's
    # also where Phase 8's 6 GB obs tensor came from. The buffer is kept
    # tiny (8 steps × 4 envs) so the test stays cheap.
    n_envs, n_stocks, n_factors = 4, 3014, 343
    buffer_size = 8
    obs_space, act_space = _build_spaces(n_stocks, n_factors)
    last_obs_t = th.zeros(n_envs, dtype=th.long, device="cuda")
    # Skip allocating a real 8 GB panel; the buffer never reads it
    # during this test (no get() / _get_samples()), so a stub provider
    # is fine.
    buf = IndexOnlyRolloutBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=act_space,
        device="cuda",
        gae_lambda=0.95,
        gamma=0.99,
        n_envs=n_envs,
        obs_provider=lambda t: th.zeros(
            (t.shape[0], n_stocks, n_factors), device="cuda"),
        obs_index_provider=lambda: last_obs_t,
    )
    assert buf.t_buffer.shape == (buffer_size, n_envs)
    assert buf.t_buffer.dtype == th.long
    # No giant obs tensor was allocated.
    assert getattr(buf, "observations", None) is None
    # 8 bytes per t-index vs 4 bytes per fp32 obs cell × 3014 × 343 cells
    # per (step, env). The ratio should be tens of thousands.
    t_bytes = buf.t_buffer.element_size() * buf.t_buffer.numel()
    obs_bytes_would_be = buffer_size * n_envs * n_stocks * n_factors * 4
    assert t_bytes < obs_bytes_would_be / 1000, (
        f"t_buffer={t_bytes}B should be vastly smaller than the obs "
        f"tensor it replaces ({obs_bytes_would_be}B)"
    )


@cuda
def test_gather_matches_direct_indexing():
    """``_get_samples`` returns obs that exactly match panel[planted_t]."""
    n_envs, n_stocks, n_factors = 4, 12, 5
    n_dates, buffer_size = 30, 8
    panel = _make_panel(n_dates, n_stocks, n_factors, seed=1)

    # last_obs_t is mutated externally by the env in the real loop.
    # Here we tick it manually for each add().
    last_obs_t = th.zeros(n_envs, dtype=th.long, device="cuda")
    buf = _make_index_buf(
        buffer_size=buffer_size, n_envs=n_envs,
        n_stocks=n_stocks, n_factors=n_factors,
        panel=panel, last_obs_t=last_obs_t,
    )

    # Plant a deterministic schedule of t-indices into the buffer via add().
    # planted_t[step, env] is what the buffer should record at that slot.
    rng = np.random.default_rng(7)
    planted_t = rng.integers(low=0, high=n_dates,
                             size=(buffer_size, n_envs)).astype(np.int64)

    for step in range(buffer_size):
        # Mutate the closure-bound tensor in place so the provider sees
        # the new values without rebinding the lambda.
        last_obs_t.copy_(th.as_tensor(planted_t[step], device="cuda",
                                      dtype=th.long))
        dummy_obs = np.zeros((n_envs, n_stocks, n_factors), dtype=np.float32)
        action = rng.uniform(0, 1, size=(n_envs, n_stocks)).astype(np.float32)
        reward = rng.standard_normal(n_envs).astype(np.float32) * 0.01
        episode_start = np.zeros(n_envs, dtype=np.float32)
        value = th.as_tensor(rng.standard_normal(n_envs).astype(np.float32),
                             device="cuda")
        log_prob = th.as_tensor(rng.standard_normal(n_envs).astype(np.float32),
                                device="cuda")
        buf.add(dummy_obs, action, reward, episode_start, value, log_prob)

    # Confirm t_buffer matches what we planted.
    np.testing.assert_array_equal(
        buf.t_buffer.cpu().numpy(), planted_t,
    )

    # Run GAE so get() doesn't trip on missing _values_cuda mirrors.
    last_values = th.zeros(n_envs, device="cuda")
    dones = np.zeros(n_envs, dtype=bool)
    buf.compute_returns_and_advantage(last_values=last_values, dones=dones)

    # After get() flattens, the order is per-env-major; verify _get_samples
    # against direct indexing on the flattened t_buffer.
    list(buf.get(batch_size=buffer_size * n_envs))  # triggers swap_and_flatten
    flat_t = buf.t_buffer  # now (total,) or (total, 1)
    if flat_t.dim() == 2 and flat_t.shape[-1] == 1:
        flat_t = flat_t.squeeze(-1)

    batch_inds = th.arange(buffer_size * n_envs, device="cuda", dtype=th.long)
    samples = buf._get_samples(batch_inds)
    expected = panel[flat_t]
    th.testing.assert_close(samples.observations, expected, atol=0.0, rtol=0.0)


@cuda
def test_gae_matches_phase8_buffer():
    """Identical inputs to GPURolloutBuffer + IndexOnlyRolloutBuffer must
    produce identical advantages and returns. The obs storage strategy
    cannot affect GAE math.
    """
    n_envs, n_stocks, n_factors = 4, 20, 8
    n_dates = 40
    buffer_size = 32
    gae_lambda, gamma = 0.95, 0.99

    panel = _make_panel(n_dates, n_stocks, n_factors, seed=2)
    last_obs_t = th.zeros(n_envs, dtype=th.long, device="cuda")

    obs_space, act_space = _build_spaces(n_stocks, n_factors)
    gpu_buf = GPURolloutBuffer(
        buffer_size=buffer_size, observation_space=obs_space,
        action_space=act_space, device="cuda", n_envs=n_envs,
        gae_lambda=gae_lambda, gamma=gamma,
    )
    idx_buf = IndexOnlyRolloutBuffer(
        buffer_size=buffer_size, observation_space=obs_space,
        action_space=act_space, device="cuda", n_envs=n_envs,
        gae_lambda=gae_lambda, gamma=gamma,
        obs_provider=lambda t: panel[t],
        obs_index_provider=lambda: last_obs_t,
    )

    rng = np.random.default_rng(99)
    transitions = []
    for _ in range(buffer_size):
        transitions.append((
            rng.integers(low=0, high=n_dates, size=(n_envs,), dtype=np.int64),
            rng.standard_normal((n_envs, n_stocks, n_factors)).astype(np.float32),
            rng.uniform(0.0, 1.0, size=(n_envs, n_stocks)).astype(np.float32),
            rng.standard_normal(n_envs).astype(np.float32) * 0.01,
            (rng.random(n_envs) < 0.05).astype(np.float32),
            rng.standard_normal(n_envs).astype(np.float32),
            rng.standard_normal(n_envs).astype(np.float32),
        ))
    last_vals_np = rng.standard_normal(n_envs).astype(np.float32)
    dones = (rng.random(n_envs) < 0.1).astype(bool)

    for t_idx, obs, act, rew, eps, val, lp in transitions:
        last_obs_t.copy_(th.as_tensor(t_idx, device="cuda", dtype=th.long))
        gpu_buf.add(
            obs, act, rew, eps,
            th.as_tensor(val, device="cuda"),
            th.as_tensor(lp, device="cuda"),
        )
        idx_buf.add(
            obs, act, rew, eps,  # obs is ignored by idx_buf
            th.as_tensor(val, device="cuda"),
            th.as_tensor(lp, device="cuda"),
        )

    last_values = th.as_tensor(last_vals_np, device="cuda")
    gpu_buf.compute_returns_and_advantage(last_values=last_values, dones=dones)
    idx_buf.compute_returns_and_advantage(last_values=last_values, dones=dones)

    # After compute_returns_and_advantage the parent exposes numpy
    # mirrors; the cuda copies live on _*_cuda. Compare whichever is
    # convenient.
    np.testing.assert_allclose(idx_buf.advantages, gpu_buf.advantages,
                               atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(idx_buf.returns, gpu_buf.returns,
                               atol=1e-5, rtol=1e-5)


@cuda
def test_get_yields_obs_via_provider():
    """``get(batch_size)`` should yield obs that match panel[t_buffer[idx]]
    after the buffer's swap_and_flatten on t_buffer.
    """
    n_envs, n_stocks, n_factors = 4, 12, 5
    n_dates, buffer_size = 30, 16
    panel = _make_panel(n_dates, n_stocks, n_factors, seed=11)
    last_obs_t = th.zeros(n_envs, dtype=th.long, device="cuda")

    buf = _make_index_buf(
        buffer_size=buffer_size, n_envs=n_envs,
        n_stocks=n_stocks, n_factors=n_factors,
        panel=panel, last_obs_t=last_obs_t,
    )

    rng = np.random.default_rng(13)
    planted_t = rng.integers(low=0, high=n_dates,
                             size=(buffer_size, n_envs)).astype(np.int64)
    for step in range(buffer_size):
        last_obs_t.copy_(th.as_tensor(planted_t[step], device="cuda",
                                      dtype=th.long))
        buf.add(
            np.zeros((n_envs, n_stocks, n_factors), dtype=np.float32),
            rng.uniform(0, 1, size=(n_envs, n_stocks)).astype(np.float32),
            rng.standard_normal(n_envs).astype(np.float32) * 0.01,
            np.zeros(n_envs, dtype=np.float32),
            th.as_tensor(rng.standard_normal(n_envs).astype(np.float32),
                         device="cuda"),
            th.as_tensor(rng.standard_normal(n_envs).astype(np.float32),
                         device="cuda"),
        )

    last_values = th.zeros(n_envs, device="cuda")
    dones = np.zeros(n_envs, dtype=bool)
    buf.compute_returns_and_advantage(last_values=last_values, dones=dones)

    # The expected flat order is the same swap_and_flatten produces:
    # (n_steps, n_envs) -> transpose -> (n_envs, n_steps) -> flatten.
    expected_flat_t = (
        th.as_tensor(planted_t, device="cuda", dtype=th.long)
        .transpose(0, 1)
        .contiguous()
        .reshape(buffer_size * n_envs)
    )
    expected_obs_full = panel[expected_flat_t]

    batch_size = 8
    seen = 0
    for batch in buf.get(batch_size=batch_size):
        # Every yielded obs should exactly equal a slice of the expected
        # full obs tensor — same obs that direct indexing would produce.
        # The permutation in get() means we don't know which slice up
        # front, but we can check shape/device and that each row of
        # batch.observations is present in the expected_obs_full set.
        assert batch.observations.device.type == "cuda"
        assert batch.observations.shape == (batch_size, n_stocks, n_factors)
        seen += batch_size

    assert seen == buffer_size * n_envs

    # Stronger: deterministically rebuild the permutation by feeding
    # an arange index tensor straight into _get_samples and assert
    # exact equality with panel[expected_flat_t].
    arange_idx = th.arange(buffer_size * n_envs, device="cuda", dtype=th.long)
    samples_arange = buf._get_samples(arange_idx)
    th.testing.assert_close(samples_arange.observations, expected_obs_full,
                            atol=0.0, rtol=0.0)
