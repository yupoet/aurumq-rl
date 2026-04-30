"""Unit tests for the GPU sampler callback.

These tests do NOT require an NVIDIA GPU. They exercise the no-op fallback
path and patch ``pynvml`` with fakes for the happy path.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest


# Make src/ importable.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(json.loads(s))
    return out


def test_gpu_sampler_no_pynvml_falls_back_silently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When pynvml is unavailable, the callback must be a silent no-op."""
    from aurumq_rl import gpu_monitor

    monkeypatch.setattr(gpu_monitor, "NVML_AVAILABLE", False)
    monkeypatch.setattr(gpu_monitor, "pynvml", None)

    cb = gpu_monitor.GpuSamplerCallback(
        jsonl_path=tmp_path / "gpu.jsonl",
        sample_interval_seconds=0.0,
    )
    cb.num_timesteps = 0
    cb._on_training_start()

    # Drive _on_step a few times — must never raise.
    for i in range(5):
        cb.num_timesteps = (i + 1) * 100
        assert cb._on_step() is True

    cb._on_training_end()

    # No file should be written when disabled.
    assert not (tmp_path / "gpu.jsonl").exists()

    # And we should have warned exactly once.
    out = capsys.readouterr().out
    assert out.count("[gpu_monitor]") == 1


def _fake_pynvml() -> types.SimpleNamespace:
    """Build a minimal fake pynvml namespace."""

    class _Util:
        def __init__(self, gpu: int) -> None:
            self.gpu = gpu

    class _Mem:
        def __init__(self, used_mb: int, total_mb: int) -> None:
            self.used = used_mb * 1024 * 1024
            self.total = total_mb * 1024 * 1024

    state = {"util": 42, "used_mb": 1320, "total_mb": 12282, "temp": 48, "power_mw": 85_300}

    fake = types.SimpleNamespace()
    fake.NVML_TEMPERATURE_GPU = 0
    fake._state = state

    def nvmlInit() -> None:
        return None

    def nvmlShutdown() -> None:
        return None

    def nvmlDeviceGetHandleByIndex(idx: int) -> object:
        return f"handle-{idx}"

    def nvmlDeviceGetName(handle: object) -> str:
        return "FakeGPU 9000"

    def nvmlDeviceGetMemoryInfo(handle: object) -> _Mem:
        return _Mem(state["used_mb"], state["total_mb"])

    def nvmlDeviceGetUtilizationRates(handle: object) -> _Util:
        return _Util(state["util"])

    def nvmlDeviceGetTemperature(handle: object, kind: int) -> int:
        return state["temp"]

    def nvmlDeviceGetPowerUsage(handle: object) -> int:
        return state["power_mw"]

    fake.nvmlInit = nvmlInit
    fake.nvmlShutdown = nvmlShutdown
    fake.nvmlDeviceGetHandleByIndex = nvmlDeviceGetHandleByIndex
    fake.nvmlDeviceGetName = nvmlDeviceGetName
    fake.nvmlDeviceGetMemoryInfo = nvmlDeviceGetMemoryInfo
    fake.nvmlDeviceGetUtilizationRates = nvmlDeviceGetUtilizationRates
    fake.nvmlDeviceGetTemperature = nvmlDeviceGetTemperature
    fake.nvmlDeviceGetPowerUsage = nvmlDeviceGetPowerUsage
    return fake


def test_gpu_sampler_writes_jsonl_when_pynvml_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a fake NVML, every triggered _on_step appends one JSON line."""
    from aurumq_rl import gpu_monitor

    fake = _fake_pynvml()
    monkeypatch.setattr(gpu_monitor, "NVML_AVAILABLE", True)
    monkeypatch.setattr(gpu_monitor, "pynvml", fake)

    out_path = tmp_path / "nested" / "gpu.jsonl"
    cb = gpu_monitor.GpuSamplerCallback(
        jsonl_path=out_path,
        sample_interval_seconds=0.0,  # never throttle
    )
    cb.num_timesteps = 0
    cb._on_training_start()

    n_steps = 7
    for i in range(n_steps):
        cb.num_timesteps = (i + 1) * 100
        assert cb._on_step() is True

    cb._on_training_end()

    rows = _read_jsonl(out_path)
    assert len(rows) == n_steps
    first = rows[0]
    assert first["util_pct"] == 42
    assert first["mem_used_mb"] == 1320
    assert first["mem_total_mb"] == 12282
    assert first["temp_c"] == 48
    assert first["power_w"] == pytest.approx(85.3)
    assert first["device_name"] == "FakeGPU 9000"
    assert first["timestep"] == 100
    # ISO 8601 UTC timestamp present.
    assert isinstance(first["timestamp"], str) and "T" in first["timestamp"]


def test_gpu_sampler_throttles_by_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long interval must drop most _on_step calls."""
    from aurumq_rl import gpu_monitor

    fake = _fake_pynvml()
    monkeypatch.setattr(gpu_monitor, "NVML_AVAILABLE", True)
    monkeypatch.setattr(gpu_monitor, "pynvml", fake)

    out_path = tmp_path / "gpu.jsonl"
    cb = gpu_monitor.GpuSamplerCallback(
        jsonl_path=out_path,
        sample_interval_seconds=10.0,  # effectively unreachable in this test
    )
    cb.num_timesteps = 0
    cb._on_training_start()

    n_calls = 50
    for i in range(n_calls):
        cb.num_timesteps = i + 1
        cb._on_step()

    cb._on_training_end()

    rows = _read_jsonl(out_path)
    # First call writes one sample (no prior timestamp); subsequent calls
    # within the 10s window should all be throttled.
    assert len(rows) == 1
    assert len(rows) < n_calls
