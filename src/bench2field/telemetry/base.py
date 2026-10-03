"""Telemetry sampler interface.

A sampler polls a platform source in a background thread while a tier runs,
then returns a summary dict (p50 and peak for every numeric channel). Each
platform implements read_once(); the threading and summarising live here.
"""

from __future__ import annotations

import shutil
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np


class TelemetryWarning(UserWarning):
    """A run is about to proceed without the telemetry this machine should have."""


class TelemetrySampler:
    name = "base"

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self._samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- platform hooks -------------------------------------------------
    def read_once(self) -> dict[str, float]:
        """Return one flat sample: channel name -> value (W, C, %, MB...)."""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"telemetry": self.name}

    def close(self) -> None:
        pass

    # --- shared machinery -----------------------------------------------
    def read_max_temp_c(self) -> float | None:
        try:
            s = self.read_once()
        except Exception:
            return None
        temps = [v for k, v in s.items() if k.startswith("temp_") and k.endswith("_c")]
        return max(temps) if temps else None

    def start(self) -> None:
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._samples.append({"t": time.monotonic(), **self.read_once()})
            except Exception as exc:  # telemetry must never kill a run
                self._samples.append({"t": time.monotonic(), "error": 1.0})
                self._last_error = repr(exc)
            self._stop.wait(self.interval_s)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return summarize(self._samples)

    @property
    def samples(self) -> list[dict[str, float]]:
        return list(self._samples)


def summarize(samples: list[dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_samples": len(samples)}
    keys = sorted({k for s in samples for k in s if k != "t"})
    for k in keys:
        vals = np.array([s[k] for s in samples if k in s], dtype=float)
        if vals.size:
            out[k] = {"p50": float(np.median(vals)), "peak": float(vals.max())}
    return out


class NullSampler(TelemetrySampler):
    name = "none"

    def __init__(self, interval_s: float = 0.5, reason: str = "") -> None:
        super().__init__(interval_s)
        self.reason = reason

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {"telemetry": self.name}
        if self.reason:
            info["telemetry_missing"] = self.reason
        return info

    def read_once(self) -> dict[str, float]:
        return {}

    def start(self) -> None:
        pass

    def stop(self) -> dict[str, Any]:
        return {"n_samples": 0}


def _is_jetson() -> bool:
    return Path("/etc/nv_tegra_release").exists()


def _has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None or Path("/proc/driver/nvidia/version").exists()


def auto_sampler(interval_s: float = 0.5) -> TelemetrySampler:
    """Pick the best sampler for this machine: tegrastats on Jetson, NVML on
    discrete NVIDIA, rocm-smi on AMD.

    If none can be used the run still goes ahead, but with a TelemetryWarning
    saying what is missing and the reason recorded in the report: a run with
    no power or temperature data should never look like a normal one.
    """
    from .nvml import NvmlSampler
    from .rocm_smi import RocmSmiSampler
    from .tegrastats import TegrastatsSampler

    for cls in (TegrastatsSampler, NvmlSampler, RocmSmiSampler):
        if cls.available():
            return cls(interval_s=interval_s)

    if _is_jetson():
        reason = "this is a Jetson but tegrastats is not on PATH"
    elif _has_nvidia_gpu():
        reason = f"an NVIDIA GPU is present but NVML is unusable: {NvmlSampler.unavailable_reason()}"
    else:
        reason = "no tegrastats, NVML or rocm-smi on this machine"
    warnings.warn(
        f"NO TELEMETRY: {reason}. This run will record no power, temperature or "
        "utilisation, and budget gates that need them will be INCOMPLETE.",
        TelemetryWarning, stacklevel=2,
    )
    return NullSampler(interval_s=interval_s, reason=reason)
