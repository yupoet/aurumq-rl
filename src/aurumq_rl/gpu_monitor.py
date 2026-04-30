"""GPU sampler callback for SB3 training runs.

Samples NVIDIA GPU utilization, memory, temperature, and power draw at a
fixed interval and appends them to a JSONL file. Becomes a silent no-op when
``pynvml`` is not installed or no GPU is present, so training is never
blocked by missing infrastructure.
"""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Any

from aurumq_rl.sb3_callbacks import SB3_AVAILABLE, _json_default

try:
    import pynvml  # type: ignore[import-not-found]

    NVML_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on machines without pynvml
    NVML_AVAILABLE = False
    pynvml = None  # type: ignore[assignment]


if SB3_AVAILABLE:
    from stable_baselines3.common.callbacks import BaseCallback
else:  # pragma: no cover - tests stub SB3 in
    BaseCallback = object  # type: ignore[assignment,misc]


class GpuSamplerCallback(BaseCallback):  # type: ignore[misc]
    """Sample GPU stats periodically and append them to ``gpu.jsonl``.

    Parameters
    ----------
    jsonl_path:
        Output JSONL path (typically ``runs/<id>/gpu.jsonl``). Parent
        directories are created on first write.
    sample_interval_seconds:
        Minimum wall-clock seconds between samples. Default 2s — sampling
        every SB3 step would be wasteful.
    device_index:
        NVML device index to monitor. Default 0 (first GPU).
    """

    def __init__(
        self,
        jsonl_path: Path,
        sample_interval_seconds: float = 2.0,
        device_index: int = 0,
    ) -> None:
        if SB3_AVAILABLE:
            super().__init__(verbose=0)
        self._jsonl = Path(jsonl_path)
        self._interval = float(sample_interval_seconds)
        self._device_index = int(device_index)

        self._handle: Any = None
        self._device_name: str | None = None
        self._mem_total_mb: int | None = None
        self._last_sample_t: float | None = None
        self._fallback_warned: bool = False
        self._enabled: bool = False

    # ------------------------------------------------------------------
    # SB3 lifecycle hooks
    # ------------------------------------------------------------------
    def _on_training_start(self) -> None:
        """Initialize NVML and pick the requested device."""
        if not NVML_AVAILABLE:
            self._warn_fallback_once(
                "pynvml not installed — GPU monitoring disabled. "
                "Install with: pip install pynvml"
            )
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            self._device_name = str(name)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._mem_total_mb = int(mem_info.total // (1024 * 1024))
            self._enabled = True
        except Exception as exc:  # pragma: no cover - hardware-dependent
            self._warn_fallback_once(
                f"NVML init failed ({exc!r}); GPU monitoring disabled."
            )
            self._handle = None
            self._enabled = False

    def _on_step(self) -> bool:
        if not self._enabled:
            # No-op fast path — don't even check the clock.
            return True
        now = time.monotonic()
        if self._last_sample_t is not None and (now - self._last_sample_t) < self._interval:
            return True
        self._last_sample_t = now

        sample = self._collect_sample()
        if sample is not None:
            self._append_jsonl(sample)
        return True

    def _on_training_end(self) -> None:
        if self._enabled and NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            self._enabled = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _collect_sample(self) -> dict[str, Any] | None:
        if self._handle is None:
            return None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU
            )
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                power_w: float = float(power_mw) / 1000.0
            except Exception:
                power_w = 0.0
        except Exception as exc:  # pragma: no cover - hardware-dependent
            self._warn_fallback_once(
                f"NVML sample failed ({exc!r}); disabling further sampling."
            )
            self._enabled = False
            return None

        timestep = int(getattr(self, "num_timesteps", 0) or 0)
        record: dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "timestep": timestep,
            "util_pct": int(util.gpu),
            "mem_used_mb": int(mem.used // (1024 * 1024)),
            "mem_total_mb": int(mem.total // (1024 * 1024)),
            "temp_c": int(temp),
            "power_w": round(power_w, 2),
        }
        if self._device_name is not None:
            record["device_name"] = self._device_name
        return record

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        try:
            self._jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self._jsonl.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
                )
        except OSError:
            # Never break training because of a logging IO error.
            pass

    def _warn_fallback_once(self, msg: str) -> None:
        if self._fallback_warned:
            return
        self._fallback_warned = True
        # Use plain print so warnings are visible in the training console
        # without depending on logging configuration.
        print(f"[gpu_monitor] {msg}")


__all__ = ["GpuSamplerCallback", "NVML_AVAILABLE"]
