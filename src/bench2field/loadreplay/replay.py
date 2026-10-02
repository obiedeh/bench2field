"""Replay a recorded field-load profile on the bench.

Each stressor reproduces one kind of background load and can run alone, which
is what makes gap attribution possible: idle, then +thermal, +cpu, +membw,
+gpu one at a time, then all together.

Stressors are duty-cycled to hit the profile's median utilisation. When the
sampler reports the same channel (tegrastats on Jetson reports CPU, GPU and
EMC load), a proportional controller closes the loop; otherwise the duty
cycle is open-loop and the replay must be checked with replay_validity()
before its results are trusted.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from collections.abc import Iterable
from typing import Any

import numpy as np

from ..telemetry.base import NullSampler, TelemetrySampler
from .record import LoadProfile

PERIOD_S = 0.05  # duty-cycle window


def _duty_loop(work, duty: Any, stop: Any) -> None:  # runs in a worker process
    while not stop.is_set():
        d = min(max(duty.value, 0.0), 1.0)
        t0 = time.perf_counter()
        busy_until = t0 + d * PERIOD_S
        while time.perf_counter() < busy_until:
            work()
        rest = PERIOD_S - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)


def _cpu_work() -> None:
    x = 0
    for i in range(2000):
        x += i * i


def _membw_worker(duty: Any, stop: Any, mb: int) -> None:
    src = np.ones(mb * 2**20 // 8)
    dst = np.empty_like(src)

    def work() -> None:
        np.copyto(dst, src)

    _duty_loop(work, duty, stop)


def _cpu_worker(duty: Any, stop: Any) -> None:
    _duty_loop(_cpu_work, duty, stop)


class Stressor:
    """One duty-cycled load, optionally steered by a telemetry channel."""

    def __init__(self, name: str, channel: str | None, target_pct: float) -> None:
        self.name, self.channel, self.target_pct = name, channel, target_pct
        self.duty = mp.Value("d", target_pct / 100.0)
        self.stop_evt = mp.Event()
        self.procs: list[mp.Process] = []

    def start(self) -> None:
        raise NotImplementedError

    def adjust(self, measured_pct: float, gain: float = 0.3) -> None:
        err = (self.target_pct - measured_pct) / 100.0
        self.duty.value = min(max(self.duty.value + gain * err, 0.0), 1.0)

    def stop(self) -> None:
        self.stop_evt.set()
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


class CpuStressor(Stressor):
    def __init__(self, target_pct: float, workers: int | None = None) -> None:
        super().__init__("cpu", "cpu_util_mean_pct", target_pct)
        self.workers = workers or mp.cpu_count()

    def start(self) -> None:
        self.procs = [mp.Process(target=_cpu_worker, args=(self.duty, self.stop_evt), daemon=True)
                      for _ in range(self.workers)]
        for p in self.procs:
            p.start()


class MemBwStressor(Stressor):
    def __init__(self, target_pct: float, workers: int = 2, buffer_mb: int = 256) -> None:
        super().__init__("membw", "emc_util_pct", target_pct)
        self.workers, self.buffer_mb = workers, buffer_mb

    def start(self) -> None:
        self.procs = [mp.Process(target=_membw_worker, args=(self.duty, self.stop_evt, self.buffer_mb),
                                 daemon=True) for _ in range(self.workers)]
        for p in self.procs:
            p.start()


class GpuStressor(Stressor):
    """Duty-cycled matmuls on the GPU. Needs PyTorch with CUDA or ROCm (HIP
    builds expose the same torch.cuda API)."""

    def __init__(self, target_pct: float, size: int = 4096) -> None:
        super().__init__("gpu", "gpu_util_pct", target_pct)
        self.size = size
        self._thread: threading.Thread | None = None
        self._stop_t = threading.Event()

    def start(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("GpuStressor needs a CUDA or ROCm build of PyTorch")
        a = torch.randn(self.size, self.size, device="cuda", dtype=torch.float16)

        def work() -> None:
            (a @ a).sum().item()

        def loop() -> None:
            while not self._stop_t.is_set():
                d = min(max(self.duty.value, 0.0), 1.0)
                t0 = time.perf_counter()
                while time.perf_counter() < t0 + d * PERIOD_S:
                    work()
                rest = PERIOD_S - (time.perf_counter() - t0)
                if rest > 0:
                    time.sleep(rest)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_t.set()
        if self._thread:
            self._thread.join(timeout=5)


def build_stressors(profile: LoadProfile, only: Iterable[str] | None = None) -> list[Stressor]:
    """Stressors for the profile's targets; `only` restricts to named ones
    (cpu, membw, gpu) for single-stressor attribution runs."""
    wanted = set(only) if only is not None else {"cpu", "membw", "gpu"}
    t = profile.targets
    out: list[Stressor] = []
    if "cpu" in wanted and "cpu_util_mean_pct" in t:
        out.append(CpuStressor(t["cpu_util_mean_pct"]))
    if "membw" in wanted and "emc_util_pct" in t:
        out.append(MemBwStressor(t["emc_util_pct"]))
    if "gpu" in wanted and "gpu_util_pct" in t:
        out.append(GpuStressor(t["gpu_util_pct"]))
    return out


class Replay:
    """Context manager: start the stressors, optionally pre-soak to the
    profile's temperature, keep steering while the benchmark runs."""

    def __init__(self, profile: LoadProfile, sampler: TelemetrySampler | None = None,
                 only: Iterable[str] | None = None, thermal_soak: bool = True,
                 soak_timeout_s: float = 600.0) -> None:
        self.profile = profile
        self.sampler = sampler or NullSampler()
        self.only = None if only is None else list(only)
        self.thermal_soak = thermal_soak and (self.only is None or "thermal" in self.only)
        self.soak_timeout_s = soak_timeout_s
        self.stressors: list[Stressor] = []
        self._steer_stop = threading.Event()
        self._steer: threading.Thread | None = None
        self.soak_reached_c: float | None = None

    def __enter__(self) -> Replay:
        only = None if self.only is None else [s for s in self.only if s != "thermal"]
        self.stressors = build_stressors(self.profile, only)
        for s in self.stressors:
            s.start()
        if self.thermal_soak and self.profile.soak_temp_c is not None:
            self.soak_reached_c = self._soak(self.profile.soak_temp_c)
        self._steer = threading.Thread(target=self._steer_loop, daemon=True)
        self._steer.start()
        return self

    def _soak(self, target_c: float) -> float | None:
        # Heat with whatever stressors exist (or a full-duty CPU load if none)
        # until the hottest sensor reaches the profile's field temperature.
        heater = None if self.stressors else CpuStressor(100.0)
        if heater:
            heater.start()
        deadline = time.monotonic() + self.soak_timeout_s
        temp = self.sampler.read_max_temp_c()
        while temp is not None and temp < target_c and time.monotonic() < deadline:
            time.sleep(2.0)
            temp = self.sampler.read_max_temp_c()
        if heater:
            heater.stop()
        return temp

    def _steer_loop(self) -> None:
        while not self._steer_stop.wait(1.0):
            try:
                reading = self.sampler.read_once()
            except Exception:
                continue
            for s in self.stressors:
                if s.channel and s.channel in reading:
                    s.adjust(reading[s.channel])

    def __exit__(self, *exc: object) -> None:
        self._steer_stop.set()
        for s in self.stressors:
            s.stop()

    def describe(self) -> dict[str, Any]:
        return {
            "replay_profile": self.profile.name,
            "replay_stressors": [s.name for s in self.stressors],
            "replay_thermal_soak": self.thermal_soak,
            "replay_soak_reached_c": self.soak_reached_c,
        }
