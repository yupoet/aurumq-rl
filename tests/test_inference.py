"""Tests for ONNX inference (RlAgentInference + metadata)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from aurumq_rl.inference import RlAgentInference, RlAgentMetadata


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_minimal_construction() -> None:
    meta = RlAgentMetadata(
        algorithm="PPO",
        training_timesteps=1000,
        obs_shape=[10, 8],
        action_shape=[10],
    )
    assert meta.algorithm == "PPO"
    assert meta.obs_shape == (10, 8)
    assert meta.action_shape == (10,)
    assert meta.training_timesteps == 1000


def test_metadata_extra_fields_allowed() -> None:
    meta = RlAgentMetadata(
        algorithm="PPO",
        training_timesteps=0,
        obs_shape=[5],
        action_shape=[5],
        custom_x=42,  # type: ignore[call-arg]
    )
    dumped = meta.model_dump()
    assert dumped["custom_x"] == 42


def test_metadata_round_trip_json(tmp_path: Path) -> None:
    meta = RlAgentMetadata(
        algorithm="A2C",
        training_timesteps=5000,
        final_reward=0.123,
        obs_shape=[10],
        action_shape=[10],
        universe="main_board_non_st",
        factor_count=6,
        git_sha="abc1234",
        exported_at=datetime(2024, 1, 1),
    )
    path = tmp_path / "metadata.json"
    path.write_text(meta.model_dump_json(), encoding="utf-8")
    restored = RlAgentMetadata.model_validate_json(path.read_text(encoding="utf-8"))
    assert restored.algorithm == "A2C"
    assert restored.training_timesteps == 5000
    assert restored.obs_shape == (10,)


# ---------------------------------------------------------------------------
# Inference loader (uses a tiny synthetic ONNX model built on the fly)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_onnx_dir(tmp_path: Path) -> Path:
    """Build a hand-crafted ONNX model that maps obs → action via Identity.

    We build the ONNX graph directly with onnx (which is part of onnxruntime's
    ecosystem). If the `onnx` package is not installed, the test using this
    fixture is skipped.
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    obs_dim = 16

    # Build graph: action = observation (1:1 identity)
    obs_input = helper.make_tensor_value_info(
        "observation", TensorProto.FLOAT, ["batch_size", obs_dim]
    )
    action_output = helper.make_tensor_value_info(
        "action", TensorProto.FLOAT, ["batch_size", obs_dim]
    )
    identity_node = helper.make_node(
        "Identity",
        inputs=["observation"],
        outputs=["action"],
    )
    graph = helper.make_graph(
        nodes=[identity_node],
        name="identity",
        inputs=[obs_input],
        outputs=[action_output],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.checker.check_model(model)

    out_dir = tmp_path / "tiny_model"
    out_dir.mkdir()
    onnx.save(model, str(out_dir / "policy.onnx"))

    metadata = {
        "algorithm": "PPO",
        "training_timesteps": 0,
        "obs_shape": [obs_dim],
        "action_shape": [obs_dim],
        "git_sha": "test",
        "framework": "synthetic",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return out_dir


def test_inference_load_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RlAgentInference(model_dir=tmp_path / "no_such_dir")


def test_inference_predict_single(tiny_onnx_dir: Path) -> None:
    agent = RlAgentInference(model_dir=tiny_onnx_dir)
    obs = np.arange(16, dtype=np.float32)
    action = agent.predict(obs)
    assert action.shape == (16,)
    assert action.dtype == np.float32
    # Identity model: action == obs
    np.testing.assert_allclose(action, obs)


def test_inference_predict_batched_obs(tiny_onnx_dir: Path) -> None:
    agent = RlAgentInference(model_dir=tiny_onnx_dir)
    # 1×16 batch
    obs = np.arange(16, dtype=np.float32).reshape(1, 16)
    action = agent.predict(obs)
    assert action.shape == (16,)


def test_inference_shape_mismatch_raises(tiny_onnx_dir: Path) -> None:
    agent = RlAgentInference(model_dir=tiny_onnx_dir)
    with pytest.raises(ValueError, match="shape mismatch"):
        agent.predict(np.zeros(7, dtype=np.float32))  # wrong dim


def test_inference_batch_predict(tiny_onnx_dir: Path) -> None:
    agent = RlAgentInference(model_dir=tiny_onnx_dir)
    obs_list = [
        np.arange(16, dtype=np.float32),
        np.arange(16, dtype=np.float32) * 2.0,
        np.arange(16, dtype=np.float32) * -1.0,
    ]
    actions = agent.batch_predict(obs_list)
    assert len(actions) == 3
    for orig, act in zip(obs_list, actions, strict=True):
        np.testing.assert_allclose(act, orig)


def test_inference_batch_predict_empty(tiny_onnx_dir: Path) -> None:
    agent = RlAgentInference(model_dir=tiny_onnx_dir)
    assert agent.batch_predict([]) == []


def test_inference_metadata_property(tiny_onnx_dir: Path) -> None:
    agent = RlAgentInference(model_dir=tiny_onnx_dir)
    assert agent.metadata.algorithm == "PPO"
    assert agent.metadata.obs_shape == (16,)
