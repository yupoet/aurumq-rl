"""ONNX-based RL agent inference engine.

CPU-only (uses ``CPUExecutionProvider``); does not depend on PyTorch.
A single ``ort.InferenceSession`` is created per agent and reused across calls,
avoiding repeated model-load overhead (~50-200ms each).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# CPU only — no CUDA dependency
_ORT_PROVIDERS = ["CPUExecutionProvider"]


class RlAgentMetadata(BaseModel):
    """ONNX model metadata schema (metadata.json)."""

    algorithm: str
    training_timesteps: int
    final_reward: float | None = None
    obs_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    universe: str = ""
    factor_count: int = 0
    git_sha: str = "unknown"
    exported_at: datetime | None = None

    @field_validator("obs_shape", "action_shape", mode="before")
    @classmethod
    def _coerce_tuple(cls, v: list | tuple) -> tuple[int, ...]:
        return tuple(int(x) for x in v)

    model_config = {"extra": "allow"}


class RlAgentInference:
    """Loads a single RL agent and runs ONNX inference.

    Session reuse
    -------------
    The ``InferenceSession`` is created in ``__init__`` and reused across all
    ``predict()`` / ``batch_predict()`` calls.

    Thread safety
    -------------
    ORT's ``CPUExecutionProvider`` supports multi-threaded inference on the
    same session object.
    """

    def __init__(self, model_dir: Path | str) -> None:
        """Load policy.onnx + metadata.json from a directory.

        Parameters
        ----------
        model_dir:
            Directory containing ``policy.onnx`` and ``metadata.json``.

        Raises
        ------
        FileNotFoundError if either file is missing.
        """
        model_dir = Path(model_dir)
        onnx_path = model_dir / "policy.onnx"
        meta_path = model_dir / "metadata.json"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found: {meta_path}")

        logger.info("Loading ONNX session from %s", onnx_path)
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=_ORT_PROVIDERS,
        )
        self._metadata = RlAgentMetadata.model_validate_json(meta_path.read_text())

        # Cache I/O names
        self._input_name: str = self._session.get_inputs()[0].name
        self._output_name: str = self._session.get_outputs()[0].name

        logger.info(
            "RlAgentInference ready: algorithm=%s obs=%s action=%s",
            self._metadata.algorithm,
            self._metadata.obs_shape,
            self._metadata.action_shape,
        )

    @property
    def metadata(self) -> RlAgentMetadata:
        return self._metadata

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Run inference on a single observation.

        Parameters
        ----------
        observation:
            Shape must match ``metadata.obs_shape``. Accepts 1D or 2D
            (with batch dim). Auto-converted to float32.
        deterministic:
            True: deterministic policy output (default).
            False: preserve any distribution sampling in the model.

        Returns
        -------
        Action array matching ``metadata.action_shape``.
        """
        obs = np.asarray(observation, dtype=np.float32)

        expected = self._metadata.obs_shape
        if obs.shape == expected:
            obs_batched = obs[np.newaxis, ...]
        elif obs.ndim == len(expected) + 1 and obs.shape[1:] == expected:
            obs_batched = obs
        else:
            raise ValueError(f"observation shape mismatch: got {obs.shape}, expected {expected}")

        # ORT prefers 2D input (batch, features)
        batch_size = obs_batched.shape[0]
        obs_flat = obs_batched.reshape(batch_size, -1)

        outputs = self._session.run(
            [self._output_name],
            {self._input_name: obs_flat},
        )
        action = outputs[0][0]
        return action.astype(np.float32)

    def batch_predict(
        self,
        observations: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Batch inference for higher throughput."""
        if not observations:
            return []

        arrays = []
        for i, obs in enumerate(observations):
            arr = np.asarray(obs, dtype=np.float32)
            if arr.shape != self._metadata.obs_shape:
                raise ValueError(
                    f"observations[{i}] shape mismatch: "
                    f"got {arr.shape}, expected {self._metadata.obs_shape}"
                )
            arrays.append(arr)

        batch = np.stack(arrays, axis=0).astype(np.float32)
        n = batch.shape[0]
        batch_flat = batch.reshape(n, -1)

        outputs = self._session.run(
            [self._output_name],
            {self._input_name: batch_flat},
        )
        actions_batch = outputs[0]
        return [actions_batch[i].astype(np.float32) for i in range(len(observations))]


__all__ = ["RlAgentInference", "RlAgentMetadata"]
