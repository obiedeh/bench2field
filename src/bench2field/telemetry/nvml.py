"""Discrete NVIDIA GPU telemetry via NVML (RTX 5090 and similar).

NVML reports the GPU's own power draw only, so the channel is `power_gpu_w`.
It is not board power and must not be compared with a Jetson's
`power_board_w`, which covers the whole module.
"""

from __future__ import annotations

from typing import Any

from .base import TelemetrySampler


class NvmlSampler(TelemetrySampler):
    name = "nvml"

    @staticmethod
    def unavailable_reason() -> str | None:
        """None if NVML can be used, otherwise why not."""
        try:
            import pynvml  # type: ignore
        except ImportError:
            return "pynvml is not installed (pip install 'bench2field[nvml]')"
        try:
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() == 0:
                return "NVML reports no GPUs"
        except Exception as exc:
            return f"NVML failed to initialise ({exc!r})"
        return None

    @classmethod
    def available(cls) -> bool:
        return cls.unavailable_reason() is None

    def __init__(self, interval_s: float = 0.5, index: int = 0) -> None:
        super().__init__(interval_s)
        import pynvml  # type: ignore

        self._nv = pynvml
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(index)

    def read_once(self) -> dict[str, float]:
        nv, h = self._nv, self._h
        util = nv.nvmlDeviceGetUtilizationRates(h)
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        return {
            "power_gpu_w": nv.nvmlDeviceGetPowerUsage(h) / 1000.0,
            "temp_gpu_c": float(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
            "gpu_util_pct": float(util.gpu),
            "gpu_mem_util_pct": float(util.memory),
            "gpu_mem_used_mb": mem.used / 2**20,
            "sm_clock_mhz": float(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)),
        }

    def describe(self) -> dict[str, Any]:
        nv, h = self._nv, self._h
        name = nv.nvmlDeviceGetName(h)
        return {
            "telemetry": self.name,
            "gpu": name.decode() if isinstance(name, bytes) else name,
            "driver": str(nv.nvmlSystemGetDriverVersion()),
        }

    def close(self) -> None:
        try:
            self._nv.nvmlShutdown()
        except Exception:
            pass
