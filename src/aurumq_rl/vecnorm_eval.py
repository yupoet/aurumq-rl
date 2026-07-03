"""VecNormalize observation-stats helpers for eval/inference (C8).

A model trained under ``VecNormalize(norm_obs=True)`` only ever saw
normalized observations; evaluating or serving it on raw observations is a
silent train/serve skew. This module loads the frozen obs stats from the
``vec_normalize.pkl`` that scripts/train.py saves next to the checkpoint and
exposes the exact transform VecNormalize applied at train time:

    obs_n = clip((obs - mean) / sqrt(var + eps), -clip_obs, +clip_obs)

Loading the pickle requires ``stable_baselines3`` to be importable (the
pickle contains a ``VecNormalize`` instance); the eval scripts that use this
module already depend on it.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

VEC_NORMALIZE_FILENAME: str = "vec_normalize.pkl"


@dataclass(frozen=True)
class VecNormObsStats:
    """Frozen obs-normalization constants extracted from a VecNormalize."""

    mean: np.ndarray  # float64, observation_space shape
    var: np.ndarray  # float64, observation_space shape
    epsilon: float
    clip_obs: float

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply VecNormalize's obs transform; returns float32.

        ``obs`` is either ``(*obs_shape)`` or batched ``(batch, *obs_shape)``.
        A flat/2D layout mismatch between the stats and the observations
        (e.g. stats saved on a flattened space, obs given per-stock) is
        resolved by reshaping the stats when the element counts agree.
        """
        obs = np.asarray(obs)
        stats_shape = self.mean.shape
        nd = len(stats_shape)
        if obs.ndim >= nd and obs.shape[obs.ndim - nd :] == stats_shape:
            mean, var = self.mean, self.var
        elif obs.ndim >= 2 and int(np.prod(obs.shape[1:])) == self.mean.size:
            mean = self.mean.reshape(obs.shape[1:])
            var = self.var.reshape(obs.shape[1:])
        else:
            raise ValueError(
                f"observation shape {obs.shape} is incompatible with "
                f"VecNormalize stats shape {stats_shape}"
            )
        # Same numerics as VecNormalize._normalize_obs + normalize_obs:
        # float64 arithmetic, clip, then cast to float32.
        normalized = np.clip(
            (obs - mean) / np.sqrt(var + self.epsilon), -self.clip_obs, self.clip_obs
        )
        return normalized.astype(np.float32)


def load_vecnorm_obs_stats(pkl_path: Path | str) -> VecNormObsStats | None:
    """Load obs-normalization stats from a ``vec_normalize.pkl``.

    Unpickles the ``VecNormalize`` object directly: its ``__getstate__``
    drops the wrapped venv, and ``VecNormalize.load()`` only adds a
    ``set_venv()`` call that a pure obs transform does not need.

    Returns ``None`` when the pickle was saved with ``norm_obs=False``
    (no obs transform was applied at train time).
    """
    with Path(pkl_path).open("rb") as f:
        vn = pickle.load(f)
    if not getattr(vn, "norm_obs", False):
        return None
    rms = vn.obs_rms
    if isinstance(rms, dict):
        raise ValueError("Dict observation spaces are not supported")
    return VecNormObsStats(
        mean=np.asarray(rms.mean, dtype=np.float64),
        var=np.asarray(rms.var, dtype=np.float64),
        epsilon=float(vn.epsilon),
        clip_obs=float(vn.clip_obs),
    )


def resolve_obs_normalizer(
    run_dir: Path | str,
    metadata: dict[str, Any] | None,
) -> VecNormObsStats | None:
    """Decide the eval-time obs transform for a checkpoint in ``run_dir``.

    - ``vec_normalize.pkl`` present -> its obs stats (``None`` if it was
      saved with ``norm_obs=False``).
    - ``metadata`` says the model was trained on normalized obs
      (``obs_normalized: true``) but no usable stats found -> hard
      ``FileNotFoundError``; silently evaluating on raw obs was defect C8.
    - otherwise -> ``None`` (raw observations).

    NOTE: for ONNX exports with the stats baked into the graph
    (``obs_normalized: true`` in metadata), feed RAW observations and do NOT
    call this — the graph already normalizes.
    """
    run_dir = Path(run_dir)
    pkl_path = run_dir / VEC_NORMALIZE_FILENAME
    stats = load_vecnorm_obs_stats(pkl_path) if pkl_path.exists() else None
    if stats is None and metadata and metadata.get("obs_normalized"):
        raise FileNotFoundError(
            "metadata says this model was trained with VecNormalize "
            f"(obs_normalized: true) but no usable {VEC_NORMALIZE_FILENAME} "
            f"was found in {run_dir} — refusing to evaluate on raw "
            "observations (C8)."
        )
    return stats


__all__ = [
    "VEC_NORMALIZE_FILENAME",
    "VecNormObsStats",
    "load_vecnorm_obs_stats",
    "resolve_obs_normalizer",
]
