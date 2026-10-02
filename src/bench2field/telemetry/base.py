"""Telemetry sampler interface.

A sampler polls a platform source in a background thread while a tier runs,
then returns a summary dict (p50 and peak for every numeric channel). Each
platform implements read_once(); the threading and summarising live here.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np


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

    def read_once(self) -> dict[str, float]:
        return {}

    def start(self) -> None:
        pass

    def stop(self) -> dict[str, Any]:
        return {"n_samples": 0}


def auto_sampler(interval_s: float = 0.5) -> TelemetrySampler:
    """Pick the best sampler for this machine: tegrastats on Jetson, NVML on
    discrete NVIDIA, rocm-smi on AMD, otherwise none."""
    from .rocm_smi import RocmSmiSampler
    from .nvml import NvmlSampler
    from .tegrastats import TegrastatsSampler

    for cls in (TegrastatsSampler, NvmlSampler, RocmSmiSampler):
        if cls.available():
            return cls(interval_s=interval_s)
    return NullSampler(interval_s=interval_s)
