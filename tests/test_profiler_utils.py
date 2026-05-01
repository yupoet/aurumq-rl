"""Tests for src/aurumq_rl/profiler_utils.py (Phase 13).

These tests intentionally avoid touching cuda — the cuda.synchronize()
calls inside :func:`cuda_stage` are guarded by ``torch.cuda.is_available()``
so the helper works correctly on CPU-only hosts.

Coverage:
- ``cuda_stage(enabled=False)`` is a no-op (no entry recorded).
- ``cuda_stage(enabled=True)`` records a positive ms value.
- ``print_sgd_stage_times`` does not crash on an empty dict.
- ``diagnose_bottleneck`` returns a safe default for empty input.
- ``diagnose_bottleneck`` flags ``cpu_to_gpu_copy_bottleneck`` when the
  cpu_to_gpu_copy stage dominates.
- ``recommend_next_phase`` returns a non-empty string.
"""
from __future__ import annotations

import time

import pytest

from aurumq_rl.profiler_utils import (
    cuda_stage,
    diagnose_bottleneck,
    get_sgd_stage_times,
    print_sgd_stage_times,
    recommend_next_phase,
    reset_sgd_stage_times,
)


@pytest.fixture(autouse=True)
def _reset_stage_times():
    """Ensure each test starts with an empty stage-times dict."""
    reset_sgd_stage_times()
    yield
    reset_sgd_stage_times()


def test_cuda_stage_disabled_is_noop() -> None:
    """When enabled=False the context must not record anything."""
    with cuda_stage("zero_grad", enabled=False):
        time.sleep(0.001)
    times = get_sgd_stage_times()
    assert "zero_grad" not in times
    assert times == {}


def test_cuda_stage_records_time() -> None:
    """When enabled=True a positive ms value must be recorded."""
    with cuda_stage("backward", enabled=True):
        # Small sleep so the recorded ms is comfortably > 0 even on a
        # fast clock.
        time.sleep(0.005)
    times = get_sgd_stage_times()
    assert "backward" in times
    assert len(times["backward"]) == 1
    assert times["backward"][0] > 0.0


def test_print_sgd_stage_times_no_crash_when_empty(capsys: pytest.CaptureFixture) -> None:
    """Calling print_sgd_stage_times with no recorded stages must not raise."""
    # Pre-condition: empty dict (autouse fixture handles it)
    print_sgd_stage_times()
    captured = capsys.readouterr()
    # No header should be emitted because the dict is empty.
    assert "PPO SGD stage timing" not in captured.out


def test_diagnose_bottleneck_empty_returns_safe_default() -> None:
    """Passing an empty stage_times dict returns all flags False."""
    diag = diagnose_bottleneck({})
    assert diag["cpu_to_gpu_copy_bottleneck"] is False
    assert diag["gather_index_bottleneck"] is False
    assert diag["forward_dominated"] is False
    assert diag["backward_dominated"] is False
    assert diag["optimizer_dominated"] is False
    assert diag["python_overhead_dominated"] is False
    assert "evidence" in diag


def test_diagnose_bottleneck_cpu_gpu_copy_flagged() -> None:
    """cpu_to_gpu_copy at 50% of iter time should flag the bottleneck."""
    stage_times = {
        # 50 ms / 100 ms total = 50% — well above the 5% threshold.
        "cpu_to_gpu_copy": [50.0, 50.0, 50.0],
        "forward_eval_actions": [25.0, 25.0, 25.0],
        "backward": [15.0, 15.0, 15.0],
        "optimizer_step": [5.0, 5.0, 5.0],
        "cuda_tail_sync": [5.0, 5.0, 5.0],
    }
    diag = diagnose_bottleneck(stage_times)
    assert diag["cpu_to_gpu_copy_bottleneck"] is True
    # Sanity: forward_dominated should be False (25% < 40%).
    assert diag["forward_dominated"] is False


def test_diagnose_bottleneck_forward_and_backward_flags() -> None:
    """Forward at 50% and backward at 45% should both be flagged."""
    stage_times = {
        "cpu_to_gpu_copy": [1.0, 1.0],
        "forward_eval_actions": [50.0, 50.0],
        "backward": [45.0, 45.0],
        "optimizer_step": [2.0, 2.0],
        "cuda_tail_sync": [2.0, 2.0],
    }
    diag = diagnose_bottleneck(stage_times)
    assert diag["forward_dominated"] is True
    assert diag["backward_dominated"] is True
    # Optimizer at ~2% should not flag (threshold is 15%).
    assert diag["optimizer_dominated"] is False


def test_recommend_next_phase_returns_nonempty_string() -> None:
    """recommend_next_phase must always produce a useful string."""
    # Empty diag → guidance to re-run.
    msg_empty = recommend_next_phase({})
    assert isinstance(msg_empty, str)
    assert len(msg_empty) > 0

    # cpu_to_gpu_copy bottleneck → Phase 14 hint.
    diag = diagnose_bottleneck({
        "cpu_to_gpu_copy": [50.0, 50.0],
        "forward_eval_actions": [10.0, 10.0],
        "backward": [10.0, 10.0],
        "optimizer_step": [5.0, 5.0],
        "cuda_tail_sync": [5.0, 5.0],
    })
    msg = recommend_next_phase(diag)
    assert isinstance(msg, str)
    assert len(msg) > 0
    assert "Phase 14" in msg
