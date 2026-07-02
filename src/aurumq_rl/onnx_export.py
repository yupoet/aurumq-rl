"""SB3 model → ONNX export utility.

Converts a trained Stable-Baselines3 model into ONNX format for CPU inference
via :class:`aurumq_rl.inference.RlAgentInference`.

This module can be safely imported without PyTorch installed. The actual
``export_sb3_policy_to_onnx()`` call requires the ``[train]`` extra.
"""

from __future__ import annotations

import json
import pickle
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

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
    vec_normalize: Any | None = None,
) -> Path:
    """Export a Stable-Baselines3 model to ONNX.

    C6: the graph computes the DETERMINISTIC action (the distribution mean;
    exactly what ``policy.predict(obs, deterministic=True)`` returns), not a
    stochastic sample. The raw mean is NOT clipped to the action space:
    downstream consumers rank scores for top-k selection and ranking is
    clip-invariant.

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
    vec_normalize:
        C8: optional ``VecNormalize`` instance (or path to a
        ``vec_normalize.pkl``) whose obs stats are baked into the graph as
        frozen constants: ``clip((obs - mean) / sqrt(var + eps), +/-clip_obs)``.
        The exported model then accepts RAW observations. Recorded as
        ``"obs_normalized": true`` in metadata.json.

    Returns
    -------
    Path to ``policy.onnx``.

    Raises
    ------
    ImportError if PyTorch / SB3 not installed.
    FileNotFoundError if ``model_path`` is missing.
    ValueError for unsupported algorithms or mismatched vec_normalize stats.
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

    norm_stats = _extract_obs_norm_stats(vec_normalize, obs_shape)

    onnx_path = output_dir / POLICY_ONNX_FILENAME
    _export_policy_onnx(model, onnx_path, obs_shape, norm_stats)

    action_shape = _get_action_shape(model)

    # C6/C8 provenance: the graph outputs the raw (unclipped) deterministic
    # action mean; obs_normalized says whether a VecNormalize front is baked
    # into the graph (feed RAW obs when true).
    export_fields: dict[str, Any] = {
        "action_output": "deterministic_mean_raw",
        "obs_normalized": norm_stats is not None,
    }
    if norm_stats is not None:
        export_fields["obs_norm_source"] = norm_stats["source"]
        export_fields["obs_norm_clip_obs"] = norm_stats["clip_obs"]
        export_fields["obs_norm_epsilon"] = norm_stats["epsilon"]

    metadata = _build_metadata(
        algorithm=algorithm,
        training_timesteps=training_timesteps,
        final_reward=final_reward,
        obs_shape=obs_shape,
        action_shape=action_shape,
        extra={**(extra_metadata or {}), **export_fields},
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


def _extract_obs_norm_stats(
    vec_normalize: Any | None,
    obs_shape: tuple[int, ...],
) -> dict[str, Any] | None:
    """Pull frozen obs-normalization constants from a VecNormalize (C8).

    Accepts a live ``VecNormalize`` instance or a path to a
    ``vec_normalize.pkl``. Returns ``None`` when no normalization applies
    (no object given, or it was created with ``norm_obs=False``).
    """
    if vec_normalize is None:
        return None

    vn = vec_normalize
    source = "VecNormalize (in-memory, train-time obs_rms)"
    if isinstance(vn, (str, Path)):
        source = f"VecNormalize pickle: {vn}"
        with Path(vn).open("rb") as f:
            vn = pickle.load(f)

    if not getattr(vn, "norm_obs", False):
        return None

    rms = vn.obs_rms
    if isinstance(rms, dict):
        raise ValueError("Dict observation spaces are not supported for ONNX export")

    mean = np.asarray(rms.mean, dtype=np.float64)
    var = np.asarray(rms.var, dtype=np.float64)
    expected = int(np.prod(obs_shape))
    if mean.size != expected or var.size != expected:
        raise ValueError(
            f"VecNormalize obs stats size {mean.size} does not match "
            f"obs_shape {obs_shape} (= {expected} elements)"
        )
    return {
        "mean": mean.reshape(obs_shape),
        "var": var.reshape(obs_shape),
        "epsilon": float(vn.epsilon),
        "clip_obs": float(vn.clip_obs),
        "source": source,
    }


def _build_deterministic_export_module(policy: Any, norm_stats: dict[str, Any] | None) -> Any:
    """Wrap a policy in an nn.Module whose forward(obs) is the DETERMINISTIC action.

    C6: SB3's ``policy.forward()`` samples from the action distribution, so
    tracing it bakes Gaussian noise into the graph. This wrapper instead
    computes the distribution MODE — numerically identical to
    ``policy._predict(obs, deterministic=True)``:

    - ActorCriticPolicy (PPO/A2C) with a DiagGaussian head: mirrors
      ``get_distribution()`` — pi features -> ``mlp_extractor.forward_actor``
      -> ``action_net`` = the distribution mean (``DiagGaussianDistribution
      .mode()``). No Distribution object is constructed, so nothing stochastic
      (and no tracer warnings) enters the graph.
    - SAC: ``tanh(mu)`` (``SquashedDiagGaussianDistribution.mode()``).
      CAVEAT: this equals ``policy.predict(deterministic=True)`` only for
      (-1, 1) action bounds, where SB3's ``unscale_action`` is the identity.
      Other Box bounds differ by that affine rescale (monotonic, so top-k
      ranking is unaffected).

    The raw mean is intentionally NOT clipped to the action space (see
    ``export_sb3_policy_to_onnx``).

    C8: when ``norm_stats`` is given, a VecNormalize front is baked in as
    frozen float32 constants:
    ``obs = clamp((obs - mean) / sqrt(var + eps), -clip_obs, +clip_obs)``.
    """
    from stable_baselines3.common.distributions import DiagGaussianDistribution
    from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy

    is_actor_critic = isinstance(policy, ActorCriticPolicy)
    if is_actor_critic:
        if not isinstance(policy.action_dist, DiagGaussianDistribution):
            raise ValueError(
                "Deterministic ONNX export only supports DiagGaussian action "
                f"distributions, got {type(policy.action_dist).__name__}"
            )
    elif not hasattr(policy, "actor"):
        raise ValueError(
            f"Unsupported policy structure for deterministic export: {type(policy).__name__}"
        )

    class DeterministicPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy = policy
            self.normalize_obs = norm_stats is not None
            if norm_stats is not None:
                self.register_buffer(
                    "obs_mean",
                    torch.as_tensor(norm_stats["mean"], dtype=torch.float32),
                )
                std = np.sqrt(norm_stats["var"] + norm_stats["epsilon"])
                self.register_buffer(
                    "obs_std", torch.as_tensor(std, dtype=torch.float32)
                )
                self.clip_obs = float(norm_stats["clip_obs"])

        def forward(self, obs: Any) -> Any:
            if self.normalize_obs:
                obs = torch.clamp(
                    (obs - self.obs_mean) / self.obs_std, -self.clip_obs, self.clip_obs
                )
            if is_actor_critic:
                features = BasePolicy.extract_features(
                    self.policy, obs, self.policy.pi_features_extractor
                )
                latent_pi = self.policy.mlp_extractor.forward_actor(features)
                return self.policy.action_net(latent_pi)
            # SAC actor: mode of the squashed Gaussian = tanh(mu)
            actor = self.policy.actor
            mean_actions, _log_std, dist_kwargs = actor.get_action_dist_params(obs)
            if dist_kwargs:
                raise ValueError("gSDE SAC actors are not supported for ONNX export")
            return torch.tanh(mean_actions)

    return DeterministicPolicy()


def _export_policy_onnx(
    model: Any,
    onnx_path: Path,
    obs_shape: tuple[int, ...],
    norm_stats: dict[str, Any] | None = None,
) -> None:
    """Export the deterministic policy head using torch.onnx.export."""
    policy = model.policy
    policy.set_training_mode(False)
    policy = policy.to("cpu")

    export_module = _build_deterministic_export_module(policy, norm_stats)
    export_module.eval()

    dummy_obs = torch.zeros(1, *obs_shape, dtype=torch.float32)

    torch.onnx.export(
        export_module,
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
