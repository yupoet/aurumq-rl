"""Phase 23 — episodes → per-(t, j) target_quality reward signal.

Given a list of detected ``MainWaveEpisode``, produce a (T, S) numpy array
``target_quality[t, j]`` for the env's reward and a separate ``proximity[t, j]``
encoding for eval. The goal: deciding to buy stock j on day t should get
strong positive reward iff a real main wave starts soon after t.

Decision-day convention: if an episode E has ``t_start = T_s``, the user's
"buy decision" happens on day ``t = T_s - 1`` (T-1 ideal). We extend credit
to T-1, T-2, T-3 with decreasing weights:

    proximity_weight = [1.0, 0.6, 0.3]  for k = 1, 2, 3

Quality of an episode is shaped by 4 dimensions:

    quality(E) = peak_return(E)
                 × duration_factor(E)        # clip(duration / 10, 0.5, 1.5)
                 × smoothness_factor(E)      # 1 - clip(max_dd / peak_return, 0, 1)

Per-(t, j) target:

    target_quality[t, j] = max over episodes E in stock j with t_start(E)
                                 in [t+1, t+L]:
        quality(E) * proximity_weight[t_start(E) - (t+1)]

We use ``max`` (not sum) so the target stays bounded. If multiple
episodes match, we take the best. If none, target = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .main_wave_episodes import MainWaveEpisode


@dataclass(frozen=True)
class TargetConfig:
    """Knobs for episode → target conversion."""

    proximity_lookahead: int = 3   # T-1, T-2, T-3 valid; T-4+ no credit
    proximity_weights: tuple[float, ...] = (1.0, 0.6, 0.3)
    duration_factor_baseline: float = 10.0     # duration / baseline → clip([0.5, 1.5])
    duration_factor_min: float = 0.5
    duration_factor_max: float = 1.5


@dataclass
class MainWaveTargets:
    """Per-(t, j) target signals from episodes."""

    target_quality: np.ndarray              # (T, S) float, env reward source
    proximity: np.ndarray                   # (T, S) int8 in {0, 1, 2, 3} — best (smallest k) hit, 0=miss
    episode_id_at_proximity: np.ndarray     # (T, S) int32, idx into episodes list, -1 if miss
    config: TargetConfig


def quality_of_episode(e: MainWaveEpisode, cfg: TargetConfig | None = None) -> float:
    """Composite per-episode quality used as the reward magnitude."""
    cfg = cfg or TargetConfig()
    duration_factor = float(np.clip(
        e.duration / cfg.duration_factor_baseline,
        cfg.duration_factor_min,
        cfg.duration_factor_max,
    ))
    smoothness_factor = 1.0 - float(np.clip(
        e.max_dd_during / max(e.peak_return, 1e-6), 0.0, 1.0
    ))
    return float(e.peak_return) * duration_factor * smoothness_factor


def build_target_quality(
    episodes: list[MainWaveEpisode],
    n_dates: int,
    n_stocks: int,
    cfg: TargetConfig | None = None,
) -> MainWaveTargets:
    """Convert a list of episodes into per-(t, j) target arrays.

    Parameters
    ----------
    episodes : list[MainWaveEpisode]
    n_dates, n_stocks : int
    cfg : TargetConfig

    Returns
    -------
    MainWaveTargets
    """
    cfg = cfg or TargetConfig()
    L = cfg.proximity_lookahead
    assert len(cfg.proximity_weights) >= L, (
        f"proximity_weights must have at least {L} entries"
    )

    target_quality = np.zeros((n_dates, n_stocks), dtype=np.float32)
    proximity = np.zeros((n_dates, n_stocks), dtype=np.int8)
    ep_id = np.full((n_dates, n_stocks), -1, dtype=np.int32)

    for ep_idx, e in enumerate(episodes):
        q = quality_of_episode(e, cfg)
        # decision day t for proximity k means t_start - t = k → t = t_start - k
        for k in range(1, L + 1):
            t = e.t_start - k
            if t < 0 or t >= n_dates:
                continue
            j = e.stock_idx
            weighted = float(q * cfg.proximity_weights[k - 1])
            if weighted > target_quality[t, j]:
                target_quality[t, j] = weighted
                proximity[t, j] = k
                ep_id[t, j] = ep_idx

    return MainWaveTargets(
        target_quality=target_quality,
        proximity=proximity,
        episode_id_at_proximity=ep_id,
        config=cfg,
    )


__all__ = [
    "TargetConfig",
    "MainWaveTargets",
    "build_target_quality",
    "quality_of_episode",
]
