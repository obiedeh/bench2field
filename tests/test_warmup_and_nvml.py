"""Warmup pacing, and the real NVML behaviour that made it necessary."""

import json
from pathlib import Path

import numpy as np
import pytest

from bench2field.runner import run_tier
from bench2field.telemetry.base import summarize
from bench2field.telemetry.nvml import NvmlSampler

FIXTURE = Path(__file__).parent / "fixtures" / "nvml_5090_after_flatout_warmup.json"
NVML_CHANNELS = {"power_gpu_w", "temp_gpu_c", "gpu_util_pct", "gpu_mem_util_pct",
                 "gpu_mem_used_mb", "sm_clock_mhz"}


def test_real_nvml_capture_shows_warmup_burst_bleeding_into_the_tier():
    """Captured on the RTX 5090: a 30 Hz tier measured straight after a
    flat-out warmup. NVML keeps reporting the burst for about three seconds,
    so the tier's summary claimed a 350 W, 91% peak for a 70 W, 2% load."""
    samples = json.loads(FIXTURE.read_text())["samples"]
    assert all(NVML_CHANNELS <= s.keys() for s in samples)
    early = [s for s in samples if s["t"] < 3.0]
    steady = [s for s in samples if s["t"] >= 5.0]
    steady_w = np.median([s["power_gpu_w"] for s in steady])
    assert 60 < steady_w < 80 and max(s["gpu_util_pct"] for s in steady) <= 5
    assert max(s["power_gpu_w"] for s in early) > 4 * steady_w
    assert max(s["gpu_util_pct"] for s in early) > 80
    # What a report would have said about this tier:
    summary = summarize([{k: v for k, v in s.items() if k != "t"} for s in samples])
    assert summary["power_gpu_w"]["peak"] > 4 * steady_w


class FakeTime:
    def __init__(self, infer_s):
        self.t, self.infer_s, self.starts = 0.0, infer_s, []

    def clock(self):
        return self.t

    def sleep(self, d):
        self.t += d

    def infer(self, _):
        self.starts.append(self.t)
        self.t += self.infer_s


def test_warmup_runs_at_the_tier_rate_not_flat_out():
    ft = FakeTime(0.001)  # 1 ms inference: flat out would be ~1000 calls per second
    t = run_tier(ft.infer, lambda i: None, target_hz=30, duration_s=1.0, deadline_ms=33.3,
                 warmup_s=2.0, clock=ft.clock, sleep=ft.sleep)
    warm = ft.starts[:-t.latency.n]
    assert len(warm) == 60  # 2 s at 30 Hz
    assert np.diff(warm) == pytest.approx(1 / 30)
    assert t.latency.n == 30 and t.deadline_misses == 0


def test_slow_first_warmup_call_is_not_followed_by_a_catch_up_burst():
    """A TensorRT engine build makes the first call take seconds; the frames
    that fell due meanwhile must not then run back to back."""
    ft = FakeTime(0.001)
    first = {"done": False}

    def infer(x):
        ft.starts.append(ft.t)
        ft.t += 0.001 if first["done"] else 5.0
        first["done"] = True

    run_tier(infer, lambda i: None, target_hz=100, duration_s=0.1, deadline_ms=10.0,
             warmup_s=0.5, clock=ft.clock, sleep=ft.sleep)
    warm = ft.starts[:50]
    assert min(np.diff(warm[1:])) == pytest.approx(0.01)


def test_no_warmup_means_no_warmup_calls():
    ft = FakeTime(0.001)
    t = run_tier(ft.infer, lambda i: None, target_hz=100, duration_s=0.1, deadline_ms=10.0,
                 warmup_s=0.0, clock=ft.clock, sleep=ft.sleep)
    assert len(ft.starts) == t.latency.n == 10


def test_nvml_sampler_on_real_hardware():
    """Runs only where NVML works: the channels the device supports are
    present and physically sane. Memory and clock are optional (Jetson Thor
    does not report them)."""
    if not NvmlSampler.available():
        pytest.skip(f"NVML unavailable: {NvmlSampler.unavailable_reason()}")
    s = NvmlSampler()
    try:
        r = s.read_once()
        required = {"power_gpu_w", "temp_gpu_c", "gpu_util_pct", "gpu_mem_util_pct"}
        assert required <= set(r) <= NVML_CHANNELS
        assert all(isinstance(v, float) for v in r.values())
        assert 1 < r["power_gpu_w"] < 1000 and 0 < r["temp_gpu_c"] < 110
        assert 0 <= r["gpu_util_pct"] <= 100 and 0 <= r["gpu_mem_util_pct"] <= 100
        assert r.get("gpu_mem_used_mb", 1) > 0 and r.get("sm_clock_mhz", 1) > 0
        d = s.describe()
        assert isinstance(d["gpu"], str) and d["gpu"] and isinstance(d["driver"], str)
        assert s.read_max_temp_c() == pytest.approx(r["temp_gpu_c"], abs=5)
    finally:
        s.close()
