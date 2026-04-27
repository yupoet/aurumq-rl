"""Training metrics JSONL reader/writer.

Provides realtime-append + batch-load of training metrics in JSONL format.
This module has no PyTorch / gymnasium dependency; safe to import in any
environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TrainingMetrics(BaseModel):
    """Snapshot of training indicators at a single checkpoint.

    Attributes
    ----------
    timestep:           Global training step.
    episode_reward_mean: Mean reward over recent N episodes.
    policy_loss:        Policy/actor network loss.
    value_loss:         Value/critic network loss.
    entropy:            Policy entropy (higher = more exploration).
    explained_variance: Value-function explained variance (closer to 1 = better).
    learning_rate:      Current LR (supports dynamic schedules).
    fps:                Steps per second.
    algorithm:          PPO / A2C / SAC.
    extra:              Additional algorithm-specific metrics.
    """

    timestep: int = Field(ge=0)
    episode_reward_mean: float
    policy_loss: float
    value_loss: float
    entropy: float
    explained_variance: float
    learning_rate: float = Field(gt=0)
    fps: int = Field(ge=0)
    algorithm: str = "PPO"
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("algorithm")
    @classmethod
    def _validate_algorithm(cls, v: str) -> str:
        allowed = {"PPO", "A2C", "SAC"}
        if v not in allowed:
            raise ValueError(f"algorithm must be one of {allowed}, got {v!r}")
        return v

    model_config = {"extra": "allow"}


def append_metrics(metrics_path: Path, metrics: TrainingMetrics) -> None:
    """Append one TrainingMetrics record to a JSONL file.

    Auto-creates parent directories. Safe to call concurrently from a single
    process; for multi-process scenarios, use a lock.
    """
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    line = metrics.model_dump_json()
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_metrics(metrics_path: Path) -> list[TrainingMetrics]:
    """Load all training metrics from JSONL.

    Empty/missing files return []. Malformed lines are silently skipped.
    Results are sorted by timestep ascending.
    """
    if not metrics_path.exists():
        return []

    results: list[TrainingMetrics] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                results.append(TrainingMetrics.model_validate(data))
            except (json.JSONDecodeError, ValueError):
                continue

    results.sort(key=lambda m: m.timestep)
    return results


def summarize_metrics(metrics: list[TrainingMetrics]) -> dict[str, Any]:
    """Generate a summary dict for metadata.json."""
    if not metrics:
        return {
            "total_timesteps": 0,
            "best_reward": None,
            "final_reward": None,
            "mean_fps": None,
            "n_checkpoints": 0,
        }

    rewards = [m.episode_reward_mean for m in metrics]
    return {
        "total_timesteps": metrics[-1].timestep,
        "best_reward": max(rewards),
        "final_reward": rewards[-1],
        "mean_fps": int(sum(m.fps for m in metrics) / len(metrics)),
        "n_checkpoints": len(metrics),
        "algorithm": metrics[-1].algorithm,
    }


__all__ = [
    "TrainingMetrics",
    "append_metrics",
    "load_metrics",
    "summarize_metrics",
]
