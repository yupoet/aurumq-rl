"""Tests for training metrics JSONL roundtrip + summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aurumq_rl.metrics import (
    TrainingMetrics,
    append_metrics,
    load_metrics,
    summarize_metrics,
)


def _make_metric(timestep: int, reward: float = 0.0, fps: int = 100) -> TrainingMetrics:
    return TrainingMetrics(
        timestep=timestep,
        episode_reward_mean=reward,
        policy_loss=0.01,
        value_loss=0.05,
        entropy=1.0,
        explained_variance=0.5,
        learning_rate=3e-4,
        fps=fps,
        algorithm="PPO",
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_metrics_validates_algorithm() -> None:
    with pytest.raises(ValidationError):
        TrainingMetrics(
            timestep=0,
            episode_reward_mean=0.0,
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            explained_variance=0.0,
            learning_rate=1e-4,
            fps=10,
            algorithm="UNKNOWN_ALG",  # rejected
        )


def test_metrics_negative_timestep_invalid() -> None:
    with pytest.raises(ValidationError):
        TrainingMetrics(
            timestep=-1,
            episode_reward_mean=0.0,
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            explained_variance=0.0,
            learning_rate=1e-4,
            fps=10,
        )


def test_metrics_zero_lr_invalid() -> None:
    with pytest.raises(ValidationError):
        TrainingMetrics(
            timestep=0,
            episode_reward_mean=0.0,
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            explained_variance=0.0,
            learning_rate=0.0,  # gt=0
            fps=10,
        )


def test_metrics_extra_fields_allowed() -> None:
    m = TrainingMetrics(
        timestep=10,
        episode_reward_mean=0.1,
        policy_loss=0.0,
        value_loss=0.0,
        entropy=0.0,
        explained_variance=0.0,
        learning_rate=1e-4,
        fps=10,
        custom_field="hello",  # type: ignore[call-arg]
    )
    # Extra fields land in __pydantic_extra__ thanks to model_config={'extra':'allow'}
    assert hasattr(m, "custom_field") or "custom_field" in m.model_dump()


# ---------------------------------------------------------------------------
# JSONL append/load roundtrip
# ---------------------------------------------------------------------------


def test_append_and_load_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "metrics.jsonl"
    expected = [_make_metric(t, reward=0.001 * t, fps=100 + t) for t in (0, 100, 200)]
    for m in expected:
        append_metrics(p, m)

    loaded = load_metrics(p)
    assert len(loaded) == 3
    for original, restored in zip(expected, loaded, strict=True):
        assert restored.timestep == original.timestep
        assert restored.fps == original.fps
        assert restored.episode_reward_mean == pytest.approx(original.episode_reward_mean)


def test_load_metrics_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_metrics(tmp_path / "nope.jsonl") == []


def test_load_metrics_skips_malformed_lines(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    good = _make_metric(50)
    p.write_text(
        "not-json\n"
        + good.model_dump_json()
        + "\n"
        + '{"missing": "fields"}\n'
        + "\n",
        encoding="utf-8",
    )
    loaded = load_metrics(p)
    assert len(loaded) == 1
    assert loaded[0].timestep == 50


def test_load_metrics_sorted_by_timestep(tmp_path: Path) -> None:
    p = tmp_path / "metrics.jsonl"
    for t in (300, 100, 200, 0):
        append_metrics(p, _make_metric(t))
    loaded = load_metrics(p)
    assert [m.timestep for m in loaded] == [0, 100, 200, 300]


def test_append_metrics_creates_parent_dir(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "deeper" / "metrics.jsonl"
    append_metrics(nested, _make_metric(1))
    assert nested.exists()


# ---------------------------------------------------------------------------
# summarize_metrics
# ---------------------------------------------------------------------------


def test_summarize_empty() -> None:
    summary = summarize_metrics([])
    assert summary["total_timesteps"] == 0
    assert summary["best_reward"] is None
    assert summary["final_reward"] is None
    assert summary["n_checkpoints"] == 0


def test_summarize_picks_best_and_final() -> None:
    metrics = [
        _make_metric(0, reward=0.1, fps=100),
        _make_metric(100, reward=0.5, fps=200),  # best
        _make_metric(200, reward=0.3, fps=150),  # final
    ]
    summary = summarize_metrics(metrics)
    assert summary["total_timesteps"] == 200
    assert summary["best_reward"] == pytest.approx(0.5)
    assert summary["final_reward"] == pytest.approx(0.3)
    assert summary["mean_fps"] == 150
    assert summary["n_checkpoints"] == 3
    assert summary["algorithm"] == "PPO"


def test_summarize_serializable(tmp_path: Path) -> None:
    """Smoke check — summary should be JSON-dumpable."""
    metrics = [_make_metric(t, reward=0.01 * t) for t in (0, 100)]
    summary = summarize_metrics(metrics)
    out = tmp_path / "summary.json"
    out.write_text(json.dumps(summary), encoding="utf-8")
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["total_timesteps"] == 100
