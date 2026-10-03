"""Discrete NVIDIA GPU telemetry via NVML (RTX 5090 and similar).

NVML reports the GPU's own power draw only, so the channel is `power_gpu_w`.
It is not board power and must not be compared with a Jetson's
`power_board_w`, which covers the whole module.
"""

from __future__ import annotations

from typing import Any

from .base import TelemetrySampler


def read_channels(nv: Any, h: Any) -> dict[str, float]:
    """One NVML sample for device handle `h`, leaving out what the device does
    not implement: Jetson Thor's NVML has no memory info and no clock info,
    and Jetson Orin's answers nothing but the device name. Any other NVML
    error propagates."""
    unsupported = getattr(nv, "NVMLError_NotSupported", ())
    out: dict[str, float] = {}

    def read(channels) -> None:
        try:
            out.update(channels())
        except unsupported:
            pass

    def utilisation() -> dict[str, float]:
        util = nv.nvmlDeviceGetUtilizationRates(h)
        return {"gpu_util_pct": float(util.gpu), "gpu_mem_util_pct": float(util.memory)}

    read(lambda: {"power_gpu_w": nv.nvmlDeviceGetPowerUsage(h) / 1000.0})
    read(lambda: {"temp_gpu_c": float(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))})
    read(utilisation)
    read(lambda: {"gpu_mem_used_mb": nv.nvmlDeviceGetMemoryInfo(h).used / 2**20})
    read(lambda: {"sm_clock_mhz": float(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM))})
    return out


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
            if not read_channels(pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)):
                return "NVML answers no telemetry query on this device (as on Jetson Orin)"
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
        return read_channels(self._nv, self._h)

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
