"""Replay a recorded field-load profile on the bench.

Each stressor reproduces one kind of background load and can run alone, which
is what makes gap attribution possible: idle, then +thermal, +cpu, +membw,
+gpu one at a time, then all together.

Stressors are duty-cycled to hit the profile's median utilisation. The
profile was recorded without the model under test, so the duty cycles are
calibrated the same way: before the benchmark starts, with the model idle, a
proportional controller steers each stressor until the sampler reports the
profile's target, and the duty cycles are then frozen for the run. Steering
during the run would count the model's own load towards the target and back
the stressors off by that much.

Where the sampler does not report a stressor's channel (NVML has no CPU or
memory-controller load), the duty cycle stays at its open-loop starting value
and the replay must be checked with replay_validity() before its results are
trusted. Either way the calibrated duty and the utilisation it achieved are
recorded in the report.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from ..telemetry.base import NullSampler, TelemetrySampler
from .record import LoadProfile

PERIOD_S = 0.05  # duty-cycle window

# Every stressor runs in its own spawned process. Spawn, not fork: the
# benchmark process already holds runtime thread pools and usually a CUDA
# context by the time replay starts, and neither survives a fork. It also
# means the GPU stressor gets a CUDA context of its own, so the GPU
# time-slices between it and the model the way it does between two processes
# on the robot, and the stressor never holds the benchmark's GIL.
_CTX = mp.get_context("spawn")


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


def _gpu_worker(duty: Any, stop: Any, size: int, conn: Any) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("this PyTorch build has no usable CUDA or ROCm device")
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)

        def work() -> None:
            (a @ a).sum().item()  # .item() waits for the GPU, so duty is real busy time

        work()
        conn.send(("ready", torch.cuda.get_device_name(0)))
    except BaseException as exc:  # report to the parent instead of dying silently
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        return
    _duty_loop(work, duty, stop)


class Stressor:
    """One duty-cycled load, optionally steered by a telemetry channel."""

    def __init__(self, name: str, channel: str | None, target_pct: float) -> None:
        self.name, self.channel, self.target_pct = name, channel, target_pct
        self.duty = _CTX.Value("d", target_pct / 100.0)
        self.stop_evt = _CTX.Event()
        self.procs: list[Any] = []

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
        self.procs = [_CTX.Process(target=_cpu_worker, args=(self.duty, self.stop_evt), daemon=True)
                      for _ in range(self.workers)]
        for p in self.procs:
            p.start()


class MemBwStressor(Stressor):
    def __init__(self, target_pct: float, workers: int = 2, buffer_mb: int = 256) -> None:
        super().__init__("membw", "emc_util_pct", target_pct)
        self.workers, self.buffer_mb = workers, buffer_mb

    def start(self) -> None:
        self.procs = [_CTX.Process(target=_membw_worker, args=(self.duty, self.stop_evt, self.buffer_mb),
                                   daemon=True) for _ in range(self.workers)]
        for p in self.procs:
            p.start()


class GpuStressor(Stressor):
    """Duty-cycled matmuls on the GPU, in a process of its own with its own
    CUDA context. Needs PyTorch with CUDA or ROCm (HIP builds expose the same
    torch.cuda API). start() returns once the worker has run its first matmul,
    and raises if it could not."""

    def __init__(self, target_pct: float, size: int = 4096, start_timeout_s: float = 120.0) -> None:
        super().__init__("gpu", "gpu_util_pct", target_pct)
        self.size = size
        self.start_timeout_s = start_timeout_s
        self.device: str | None = None

    def start(self) -> None:
        parent, child = _CTX.Pipe(duplex=False)
        proc = _CTX.Process(target=_gpu_worker, args=(self.duty, self.stop_evt, self.size, child),
                            daemon=True)
        proc.start()
        child.close()
        try:
            if not parent.poll(self.start_timeout_s):
                raise RuntimeError(f"GPU stressor did not start within {self.start_timeout_s:g} s")
            try:
                status, detail = parent.recv()
            except EOFError:
                raise RuntimeError("GPU stressor process died while starting") from None
            if status != "ready":
                raise RuntimeError(f"GPU stressor could not start: {detail}")
        except RuntimeError:
            proc.terminate()
            proc.join(timeout=5)
            raise
        finally:
            parent.close()
        self.procs, self.device = [proc], detail


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


def calibrate(
    stressors: Iterable[Stressor],
    read: Callable[[], dict[str, float]],
    wait: Callable[[], None],
    max_steps: int = 30,
    tol_pct: float = 2.0,
    hold: int = 3,
    verify_steps: int = 5,
) -> dict[str, dict[str, Any]]:
    """Steer each stressor to its target with the model idle, then freeze.

    Each step waits, takes one reading and adjusts every stressor whose
    channel the reading contains. Steering stops once all of them have been
    within `tol_pct` points of target for `hold` consecutive readings, or
    after `max_steps`. The duty cycles are then left alone for `verify_steps`
    more readings, whose median is the utilisation the frozen duty achieves.
    """
    stressors = list(stressors)

    def reading() -> dict[str, float]:
        wait()
        try:
            return read()
        except Exception:
            return {}

    steered: set[str] = set()
    in_tol = 0
    for _ in range(max_steps):
        r = reading()
        live = [s for s in stressors if s.channel and s.channel in r]
        if not live:
            break
        steered |= {s.name for s in live}
        if all(abs(r[s.channel] - s.target_pct) <= tol_pct for s in live):
            in_tol += 1
            if in_tol >= hold:
                break
            continue
        in_tol = 0
        for s in live:
            s.adjust(r[s.channel])

    seen: dict[str, list[float]] = {s.name: [] for s in stressors}
    for _ in range(verify_steps if steered else 0):
        r = reading()
        for s in stressors:
            if s.channel and s.channel in r:
                seen[s.name].append(r[s.channel])

    out: dict[str, dict[str, Any]] = {}
    for s in stressors:
        achieved = float(np.median(seen[s.name])) if seen[s.name] else None
        out[s.name] = {
            "channel": s.channel,
            "target_pct": s.target_pct,
            "duty": float(s.duty.value),
            "achieved_pct": achieved,
            "steered": s.name in steered,
            "converged": achieved is not None and abs(achieved - s.target_pct) <= tol_pct,
        }
    return out


class Replay:
    """Context manager: start the stressors, optionally pre-soak to the
    profile's temperature, calibrate the duty cycles with the model idle, then
    hold them fixed while the benchmark runs."""

    def __init__(self, profile: LoadProfile, sampler: TelemetrySampler | None = None,
                 only: Iterable[str] | None = None, thermal_soak: bool = True,
                 soak_timeout_s: float = 600.0, calibrate_steps: int = 30,
                 calibrate_step_s: float = 1.0) -> None:
        self.profile = profile
        self.sampler = sampler or NullSampler()
        self.only = None if only is None else list(only)
        self.thermal_soak = thermal_soak and (self.only is None or "thermal" in self.only)
        self.soak_timeout_s = soak_timeout_s
        self.calibrate_steps = calibrate_steps
        self.calibrate_step_s = calibrate_step_s
        self.stressors: list[Stressor] = []
        self.soak_reached_c: float | None = None
        self.calibration: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> Replay:
        only = None if self.only is None else [s for s in self.only if s != "thermal"]
        self.stressors = build_stressors(self.profile, only)
        started: list[Stressor] = []
        try:
            for s in self.stressors:
                s.start()
                started.append(s)
        except BaseException:
            for s in started:  # do not leave earlier stressors loading the machine
                s.stop()
            raise
        if self.thermal_soak and self.profile.soak_temp_c is not None:
            self.soak_reached_c = self._soak(self.profile.soak_temp_c)
        self.calibration = calibrate(
            self.stressors, self.sampler.read_once,
            wait=lambda: time.sleep(self.calibrate_step_s),
            max_steps=self.calibrate_steps,
        )
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

    def __exit__(self, *exc: object) -> None:
        for s in self.stressors:
            s.stop()

    def describe(self) -> dict[str, Any]:
        return {
            "replay_profile": self.profile.name,
            "replay_stressors": [s.name for s in self.stressors],
            "replay_thermal_soak": self.thermal_soak,
            "replay_soak_reached_c": self.soak_reached_c,
            # Per stressor: the frozen duty cycle and the utilisation it
            # achieved with the model idle (None if the sampler has no channel).
            "replay_calibration": self.calibration,
        }
