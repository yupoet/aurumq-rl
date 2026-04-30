"""Custom Stable-Baselines3 callbacks for wandb + JSONL co-logging."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


# Optional dependency — SB3 only needed for training
try:
    from stable_baselines3.common.callbacks import BaseCallback

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    BaseCallback = object  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from aurumq_rl.wandb_integration import WandbLogger


if SB3_AVAILABLE:

    class WandbMetricsCallback(BaseCallback):  # type: ignore[misc]
        """Push SB3 internal logger metrics to wandb + JSONL every log_freq steps.

        Parameters
        ----------
        wandb_logger:
            WandbLogger instance (no-op when disabled).
        jsonl_path:
            JSONL file path for offline metric records.
        log_freq:
            Push frequency in timesteps. Default 1000.
        """

        def __init__(
            self,
            wandb_logger: WandbLogger,
            jsonl_path: Path,
            log_freq: int = 1000,
        ) -> None:
            super().__init__(verbose=0)
            self._wandb = wandb_logger
            self._jsonl = Path(jsonl_path)
            self._log_freq = log_freq
            self._t_start: float | None = None
            self._steps_at_start: int = 0

        def _on_step(self) -> bool:
            if self.num_timesteps % self._log_freq != 0:
                return True

            raw_metrics: dict[str, Any] = dict(self.logger.name_to_value)
            if not raw_metrics:
                return True

            self._wandb.log_metrics(raw_metrics, step=self.num_timesteps)
            self._append_jsonl(raw_metrics)
            return True

        def _append_jsonl(self, metrics: dict[str, Any]) -> None:
            """Append metrics to JSONL file.

            Writes a record that conforms to the canonical
            :class:`aurumq_rl.metrics.TrainingMetrics` schema (so
            ``summarize_metrics`` can read it back), with the raw SB3 keys
            preserved under ``extra`` for debugging.
            """
            self._jsonl.parent.mkdir(parents=True, exist_ok=True)

            def _f(key: str, default: float = 0.0) -> float:
                v = metrics.get(key)
                if v is None:
                    return default
                try:
                    return float(v.item() if hasattr(v, "item") else v)
                except (TypeError, ValueError):
                    return default

            algo_name = type(self.model).__name__ if self.model is not None else "PPO"
            if algo_name not in {"PPO", "A2C", "SAC"}:
                algo_name = "PPO"

            # Compute fps from elapsed wall time. SB3 only emits time/fps in
            # the rollout-summary frame, so most callback flushes wouldn't
            # see it and would default to 0 → mean_fps = 0 in summary.
            if self._t_start is None:
                self._t_start = time.monotonic()
                self._steps_at_start = self.num_timesteps
                fps_now = 0
            else:
                elapsed = time.monotonic() - self._t_start
                steps_since_start = self.num_timesteps - self._steps_at_start
                fps_now = int(steps_since_start / elapsed) if elapsed > 0 else 0

            sb3_fps = _f("time/fps")
            record_fps = int(sb3_fps) if sb3_fps > 0 else fps_now

            record = {
                "timestep": self.num_timesteps,
                "episode_reward_mean": _f("rollout/ep_rew_mean"),
                "policy_loss": _f("train/policy_gradient_loss") or _f("train/loss"),
                "value_loss": _f("train/value_loss"),
                "entropy": -_f("train/entropy_loss"),  # SB3 reports entropy_loss = -E
                "explained_variance": _f("train/explained_variance"),
                "learning_rate": _f("train/learning_rate", default=1e-9),
                "fps": record_fps,
                "algorithm": algo_name,
                "extra": metrics,
            }

            try:
                with self._jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            except OSError:
                pass

    class CheckpointArtifactCallback(BaseCallback):  # type: ignore[misc]
        """Save checkpoint locally and upload to wandb artifact registry.

        Parameters
        ----------
        wandb_logger:
            WandbLogger (no-op when disabled — local save still happens).
        save_path:
            Local checkpoint directory.
        name_prefix:
            Filename prefix (e.g. "ppo").
        save_freq:
            Save frequency in timesteps. Default 100k.
        artifact_type:
            Wandb artifact type label.
        """

        def __init__(
            self,
            wandb_logger: WandbLogger,
            save_path: Path,
            name_prefix: str = "model",
            save_freq: int = 100_000,
            artifact_type: str = "model",
        ) -> None:
            super().__init__(verbose=0)
            self._wandb = wandb_logger
            self._save_path = Path(save_path)
            self._name_prefix = name_prefix
            self._save_freq = save_freq
            self._artifact_type = artifact_type

        def _on_step(self) -> bool:
            if self.num_timesteps % self._save_freq != 0:
                return True

            self._save_path.mkdir(parents=True, exist_ok=True)
            ckpt_name = f"{self._name_prefix}_{self.num_timesteps}_steps"
            ckpt_path = self._save_path / f"{ckpt_name}.zip"

            try:
                self.model.save(str(ckpt_path))
            except Exception:
                return True

            self._wandb.log_artifact(
                path=ckpt_path,
                name=ckpt_name,
                artifact_type=self._artifact_type,
                metadata={"timestep": self.num_timesteps},
            )

            return True

else:
    # Stubs when SB3 not installed
    class WandbMetricsCallback:  # type: ignore[no-redef]
        """Placeholder when stable-baselines3 is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "stable-baselines3 not installed. Install with: pip install aurumq-rl[train]"
            )

    class CheckpointArtifactCallback:  # type: ignore[no-redef]
        """Placeholder when stable-baselines3 is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "stable-baselines3 not installed. Install with: pip install aurumq-rl[train]"
            )


__all__ = [
    "SB3_AVAILABLE",
    "WandbMetricsCallback",
    "CheckpointArtifactCallback",
]
