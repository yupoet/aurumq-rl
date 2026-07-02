"""Tests for src/aurumq_rl/vecnorm_eval.py (C8 eval-side VecNormalize stats)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from aurumq_rl.vecnorm_eval import (  # noqa: E402
    VEC_NORMALIZE_FILENAME,
    load_vecnorm_obs_stats,
    resolve_obs_normalizer,
)

N_STOCKS = 6
N_FACTORS = 4
OBS_DIM = N_STOCKS * N_FACTORS


class _FlatEnv(gym.Env):
    """Trivial env with a flat obs space (matches scripts/train.py envs)."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (N_STOCKS,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(OBS_DIM, dtype=np.float32), 0.0, True, False, {}


def _save_vecnorm(path: Path, norm_obs: bool = True, clip_obs: float = 10.0) -> VecNormalize:
    vn = VecNormalize(
        DummyVecEnv([_FlatEnv]),
        norm_obs=norm_obs,
        norm_reward=True,
        clip_obs=clip_obs,
    )
    if norm_obs:  # VecNormalize only creates obs_rms when norm_obs=True
        rng = np.random.default_rng(5)
        vn.obs_rms.mean = rng.normal(0.0, 1.5, size=(OBS_DIM,))
        vn.obs_rms.var = rng.uniform(0.5, 3.0, size=(OBS_DIM,))
        vn.obs_rms.count = 500.0
    vn.save(str(path))
    return vn


def test_load_stats_matches_vecnormalize_transform(tmp_path: Path) -> None:
    pkl = tmp_path / VEC_NORMALIZE_FILENAME
    vn = _save_vecnorm(pkl, clip_obs=4.0)
    stats = load_vecnorm_obs_stats(pkl)
    assert stats is not None

    rng = np.random.default_rng(9)
    obs = rng.normal(0.0, 3.0, size=(7, OBS_DIM)).astype(np.float32)
    expected = vn.normalize_obs(obs)  # SB3 reference transform
    got = stats.normalize_obs(obs)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, expected)
    # wide input must have hit the clip bound
    assert np.any(np.abs(expected) >= 4.0)


def test_load_stats_norm_obs_false_returns_none(tmp_path: Path) -> None:
    pkl = tmp_path / VEC_NORMALIZE_FILENAME
    _save_vecnorm(pkl, norm_obs=False)
    assert load_vecnorm_obs_stats(pkl) is None


def test_normalize_obs_reshapes_flat_stats_to_2d_obs(tmp_path: Path) -> None:
    """Flat (S*F,) stats applied to per-stock (dates, S, F) panels."""
    pkl = tmp_path / VEC_NORMALIZE_FILENAME
    vn = _save_vecnorm(pkl)
    stats = load_vecnorm_obs_stats(pkl)
    assert stats is not None

    rng = np.random.default_rng(2)
    panel = rng.normal(0.0, 2.0, size=(5, N_STOCKS, N_FACTORS)).astype(np.float32)
    got = stats.normalize_obs(panel)
    expected = vn.normalize_obs(panel.reshape(5, OBS_DIM)).reshape(5, N_STOCKS, N_FACTORS)
    np.testing.assert_array_equal(got, expected)


def test_normalize_obs_shape_mismatch_raises(tmp_path: Path) -> None:
    pkl = tmp_path / VEC_NORMALIZE_FILENAME
    _save_vecnorm(pkl)
    stats = load_vecnorm_obs_stats(pkl)
    assert stats is not None
    with pytest.raises(ValueError, match="incompatible"):
        stats.normalize_obs(np.zeros((3, OBS_DIM + 1), dtype=np.float32))


def test_resolve_pkl_present_returns_stats(tmp_path: Path) -> None:
    _save_vecnorm(tmp_path / VEC_NORMALIZE_FILENAME)
    stats = resolve_obs_normalizer(tmp_path, metadata={})
    assert stats is not None


def test_resolve_no_pkl_no_metadata_flag_returns_none(tmp_path: Path) -> None:
    assert resolve_obs_normalizer(tmp_path, metadata={}) is None
    assert resolve_obs_normalizer(tmp_path, metadata=None) is None
    assert resolve_obs_normalizer(tmp_path, metadata={"obs_normalized": False}) is None


def test_resolve_metadata_normalized_but_pkl_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="obs_normalized"):
        resolve_obs_normalizer(tmp_path, metadata={"obs_normalized": True})


def test_resolve_metadata_normalized_but_pkl_norm_obs_false_raises(tmp_path: Path) -> None:
    """Inconsistent artifacts (flag says normalized, pkl says not) must not pass silently."""
    _save_vecnorm(tmp_path / VEC_NORMALIZE_FILENAME, norm_obs=False)
    with pytest.raises(FileNotFoundError, match="no usable"):
        resolve_obs_normalizer(tmp_path, metadata={"obs_normalized": True})
