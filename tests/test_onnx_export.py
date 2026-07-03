"""SB3 -> ONNX export parity tests (C6 deterministic export + C8 vecnorm baking).

The heart of C6: the ONNX graph must compute EXACTLY what
``policy.predict(obs, deterministic=True)`` computes (the DiagGaussian mean),
not a stochastic sample traced from ``policy.forward()``.

C8: when a run was trained under VecNormalize, the export must bake the
frozen obs-normalization constants into the graph so raw observations can be
fed at serve time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
onnx = pytest.importorskip("onnx")

import gymnasium as gym  # noqa: E402
import onnxruntime as ort  # noqa: E402
from gymnasium import spaces  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from aurumq_rl.onnx_export import export_sb3_policy_to_onnx, load_metadata  # noqa: E402

# The repo intentionally uses the legacy TorchScript ONNX exporter
# (dynamo=False, see onnx_export.py). torch >= 2.9 deprecation-warns about it
# on every export call; silence exactly those two torch-internal messages so
# the suite stays noise-free without hiding anything else.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:You are using the legacy TorchScript-based ONNX export.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:The feature will be removed. Please remove usage of this function:DeprecationWarning"
    ),
]

OBS_DIM = 24
ACT_DIM = 12

_RANDOM_OPS = {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike"}


class _TinyBoxEnv(gym.Env):
    """Minimal continuous env: random obs, zero reward, 4-step episodes."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), dtype=np.float32)
        self._rng = np.random.default_rng(0)
        self._t = 0

    def _obs(self) -> np.ndarray:
        return self._rng.standard_normal(OBS_DIM).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        return self._obs(), 0.0, self._t >= 4, False, {}


@pytest.fixture(scope="module")
def tiny_ppo(tmp_path_factory: pytest.TempPathFactory) -> tuple[PPO, Path]:
    """Tiny CPU PPO (random init is fine — parity is weight-agnostic)."""
    model = PPO(
        "MlpPolicy",
        _TinyBoxEnv(),
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        seed=0,
        device="cpu",
        policy_kwargs={"net_arch": [16]},
    )
    path = tmp_path_factory.mktemp("tiny_ppo") / "ppo_final.zip"
    model.save(str(path))
    return model, path


def _make_vecnorm(clip_obs: float = 5.0) -> VecNormalize:
    """VecNormalize with non-trivial hand-set running stats."""
    vn = VecNormalize(
        DummyVecEnv([_TinyBoxEnv]),
        norm_obs=True,
        norm_reward=True,
        clip_obs=clip_obs,
    )
    rng = np.random.default_rng(7)
    vn.obs_rms.mean = rng.normal(0.0, 2.0, size=(OBS_DIM,))
    vn.obs_rms.var = rng.uniform(0.25, 4.0, size=(OBS_DIM,))
    vn.obs_rms.count = 1000.0
    return vn


def _assert_parity(sess: ort.InferenceSession, model, obs: np.ndarray, sb3_obs=None) -> None:
    """ONNX(obs) must match policy.predict(sb3_obs or obs, deterministic=True)."""
    sb3_actions, _ = model.policy.predict(obs if sb3_obs is None else sb3_obs, deterministic=True)
    # Precondition: predict() clips to the (-1, 1) action space while the
    # export is the RAW mean — value parity is only meaningful in-bounds.
    assert np.all(np.abs(sb3_actions) < 1.0)
    onnx_actions = sess.run(["action"], {"observation": obs})[0]
    np.testing.assert_allclose(onnx_actions, sb3_actions, rtol=1e-4, atol=1e-5)
    # Downstream consumers rank scores for top-k selection: the ranking must
    # be IDENTICAL between the SB3 eval path and the exported graph.
    assert np.array_equal(np.argsort(-onnx_actions, axis=1), np.argsort(-sb3_actions, axis=1))


# ---------------------------------------------------------------------------
# C6: deterministic export parity
# ---------------------------------------------------------------------------


def test_onnx_matches_sb3_deterministic_predict(tiny_ppo, tmp_path: Path) -> None:
    """RED against the old exporter: it traced the stochastic forward pass."""
    model, model_path = tiny_ppo
    onnx_path = export_sb3_policy_to_onnx(
        model_path=model_path, output_dir=tmp_path, obs_shape=(OBS_DIM,)
    )
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    for _ in range(3):
        obs = rng.standard_normal((8, OBS_DIM)).astype(np.float32)
        _assert_parity(sess, model, obs)


def test_onnx_graph_deterministic_single_output(tiny_ppo, tmp_path: Path) -> None:
    """Graph has no sampling ops, exactly one output named 'action'."""
    _, model_path = tiny_ppo
    onnx_path = export_sb3_policy_to_onnx(
        model_path=model_path, output_dir=tmp_path, obs_shape=(OBS_DIM,)
    )
    graph = onnx.load(str(onnx_path)).graph
    op_types = {node.op_type for node in graph.node}
    assert not (op_types & _RANDOM_OPS), f"sampling ops in graph: {op_types & _RANDOM_OPS}"
    assert [o.name for o in graph.output] == ["action"]

    # Same input twice -> bit-identical output.
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    obs = np.random.default_rng(1).standard_normal((4, OBS_DIM)).astype(np.float32)
    out1 = sess.run(["action"], {"observation": obs})[0]
    out2 = sess.run(["action"], {"observation": obs})[0]
    np.testing.assert_array_equal(out1, out2)


def test_sac_export_deterministic_parity(tmp_path: Path) -> None:
    """SAC deterministic action = tanh(mu); must match predict(deterministic=True)."""
    model = SAC(
        "MlpPolicy",
        _TinyBoxEnv(),
        buffer_size=64,
        seed=0,
        device="cpu",
        policy_kwargs={"net_arch": [16]},
    )
    model_path = tmp_path / "sac_final.zip"
    model.save(str(model_path))
    onnx_path = export_sb3_policy_to_onnx(
        model_path=model_path, output_dir=tmp_path / "out", obs_shape=(OBS_DIM,)
    )
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    obs = np.random.default_rng(3).standard_normal((8, OBS_DIM)).astype(np.float32)
    _assert_parity(sess, model, obs)


# ---------------------------------------------------------------------------
# C8: VecNormalize stats baked into the graph
# ---------------------------------------------------------------------------


def test_vecnorm_baked_parity(tiny_ppo, tmp_path: Path) -> None:
    """ONNX(raw_obs) == policy.predict(vecnorm.normalize_obs(raw_obs), det=True)."""
    model, model_path = tiny_ppo
    vn = _make_vecnorm(clip_obs=5.0)
    onnx_path = export_sb3_policy_to_onnx(
        model_path=model_path,
        output_dir=tmp_path,
        obs_shape=(OBS_DIM,),
        vec_normalize=vn,
    )
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(11)
    # Wide raw obs so the +/-clip_obs clamp is actually exercised.
    raw_obs = rng.normal(0.0, 4.0, size=(8, OBS_DIM)).astype(np.float32)
    norm_obs = vn.normalize_obs(raw_obs)
    assert np.any(np.abs(norm_obs) >= 5.0), "test setup: clipping should trigger"
    _assert_parity(sess, model, raw_obs, sb3_obs=norm_obs)

    meta = load_metadata(tmp_path)
    assert meta["obs_normalized"] is True


def test_metadata_obs_normalized_false_without_stats(tiny_ppo, tmp_path: Path) -> None:
    _, model_path = tiny_ppo
    export_sb3_policy_to_onnx(model_path=model_path, output_dir=tmp_path, obs_shape=(OBS_DIM,))
    meta = load_metadata(tmp_path)
    assert meta["obs_normalized"] is False


def test_vecnorm_shape_mismatch_raises(tiny_ppo, tmp_path: Path) -> None:
    from types import SimpleNamespace

    _, model_path = tiny_ppo
    bad = SimpleNamespace(
        norm_obs=True,
        obs_rms=SimpleNamespace(mean=np.zeros(OBS_DIM + 1), var=np.ones(OBS_DIM + 1)),
        epsilon=1e-8,
        clip_obs=10.0,
    )
    with pytest.raises(ValueError, match="obs"):
        export_sb3_policy_to_onnx(
            model_path=model_path,
            output_dir=tmp_path,
            obs_shape=(OBS_DIM,),
            vec_normalize=bad,
        )
