"""SB3 model → ONNX export utility.

Converts a trained Stable-Baselines3 model into ONNX format for CPU inference
via :class:`aurumq_rl.inference.RlAgentInference`.

This module can be safely imported without PyTorch installed. The actual
``export_sb3_policy_to_onnx()`` call requires the ``[train]`` extra.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

# Optional dependencies — only required for export, not for import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

try:
    from stable_baselines3 import A2C, PPO, SAC

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = A2C = SAC = None  # type: ignore[assignment,misc]


# Constants
SUPPORTED_ALGORITHMS: frozenset[str] = frozenset({"PPO", "A2C", "SAC"})
ONNX_OPSET_VERSION: int = 17
METADATA_FILENAME: str = "metadata.json"
POLICY_ONNX_FILENAME: str = "policy.onnx"


def _get_git_sha() -> str:
    """Best-effort git short SHA. Returns 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _detect_algorithm(model: Any) -> str:
    """Detect SB3 algorithm name from model class."""
    return type(model).__name__


def _build_metadata(
    algorithm: str,
    training_timesteps: int,
    final_reward: float | None,
    obs_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata.json contents."""
    return {
        "algorithm": algorithm,
        "training_timesteps": training_timesteps,
        "final_reward": final_reward,
        "obs_shape": list(obs_shape),
        "action_shape": list(action_shape),
        "git_sha": _get_git_sha(),
        "onnx_opset": ONNX_OPSET_VERSION,
        "framework": "stable-baselines3",
        **(extra or {}),
    }


def export_sb3_policy_to_onnx(
    model_path: Path,
    output_dir: Path,
    obs_shape: tuple[int, ...],
    training_timesteps: int = 0,
    final_reward: float | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Export a Stable-Baselines3 model to ONNX.

    Parameters
    ----------
    model_path:
        Path to ``model.zip`` saved by ``model.save()``.
    output_dir:
        Output directory; will contain ``policy.onnx`` + ``metadata.json``.
    obs_shape:
        Observation shape, e.g. ``(n_stocks * n_factors,)``.
    training_timesteps:
        Total training steps (recorded in metadata).
    final_reward:
        Final ``episode_reward_mean`` (recorded in metadata).
    extra_metadata:
        Additional fields to merge into metadata.

    Returns
    -------
    Path to ``policy.onnx``.

    Raises
    ------
    ImportError if PyTorch / SB3 not installed.
    FileNotFoundError if ``model_path`` is missing.
    ValueError for unsupported algorithms.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch not installed. export_sb3_policy_to_onnx() requires "
            "the [train] extra. Install with: pip install aurumq-rl[train]"
        )

    if not SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 not installed. Install with: pip install aurumq-rl[train]"
        )

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_sb3_model(model_path)
    algorithm = _detect_algorithm(model)

    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm {algorithm!r}. Supported: {sorted(SUPPORTED_ALGORITHMS)}"
        )

    onnx_path = output_dir / POLICY_ONNX_FILENAME
    _export_policy_onnx(model, onnx_path, obs_shape)

    action_shape = _get_action_shape(model)

    metadata = _build_metadata(
        algorithm=algorithm,
        training_timesteps=training_timesteps,
        final_reward=final_reward,
        obs_shape=obs_shape,
        action_shape=action_shape,
        extra=extra_metadata,
    )
    meta_path = output_dir / METADATA_FILENAME
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return onnx_path


def _load_sb3_model(model_path: Path) -> Any:
    """Load SB3 model, inferring algorithm from filename."""
    name = model_path.stem.upper()
    if "A2C" in name:
        return A2C.load(str(model_path))
    if "SAC" in name:
        return SAC.load(str(model_path))
    try:
        return PPO.load(str(model_path))
    except Exception:
        try:
            return A2C.load(str(model_path))
        except Exception:
            return SAC.load(str(model_path))


def _export_policy_onnx(
    model: Any,
    onnx_path: Path,
    obs_shape: tuple[int, ...],
) -> None:
    """Export the policy network using torch.onnx.export."""
    policy = model.policy
    policy.set_training_mode(False)
    policy = policy.to("cpu")

    dummy_obs = torch.zeros(1, *obs_shape, dtype=torch.float32)

    torch.onnx.export(
        policy,
        dummy_obs,
        str(onnx_path),
        opset_version=ONNX_OPSET_VERSION,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
        export_params=True,
        dynamo=False,
    )


def _get_action_shape(model: Any) -> tuple[int, ...]:
    """Extract action_space shape from SB3 model."""
    try:
        return tuple(model.action_space.shape)
    except AttributeError:
        return (1,)


METADATA_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "algorithm",
        "training_timesteps",
        "final_reward",
        "obs_shape",
        "action_shape",
        "git_sha",
        "onnx_opset",
        "framework",
    }
)


def validate_metadata(metadata: dict[str, Any]) -> list[str]:
    """Return list of missing required keys (empty = valid)."""
    return sorted(k for k in METADATA_REQUIRED_KEYS if k not in metadata)


def load_metadata(output_dir: Path) -> dict[str, Any]:
    """Load metadata.json from an export directory."""
    meta_path = output_dir / METADATA_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "TORCH_AVAILABLE",
    "SB3_AVAILABLE",
    "SUPPORTED_ALGORITHMS",
    "ONNX_OPSET_VERSION",
    "export_sb3_policy_to_onnx",
    "validate_metadata",
    "load_metadata",
]
