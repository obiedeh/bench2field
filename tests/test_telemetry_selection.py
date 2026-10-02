"""Sampler selection: missing telemetry is announced, never silent; samplers
are closed after a run; NVML power is GPU power, not board power."""

import sys
import types

import pytest

from bench2field import cli
from bench2field.telemetry import NullSampler, TelemetryWarning, auto_sampler
from bench2field.telemetry import base as tbase
from bench2field.telemetry.nvml import NvmlSampler
from bench2field.telemetry.rocm_smi import RocmSmiSampler
from bench2field.telemetry.tegrastats import TegrastatsSampler


def fake_pynvml(calls=None):
    """The slice of pynvml the sampler uses, with fixed readings."""
    calls = calls if calls is not None else []
    m = types.ModuleType("pynvml")
    m.NVML_TEMPERATURE_GPU, m.NVML_CLOCK_SM = 0, 1
    m.nvmlInit = lambda: calls.append("init")
    m.nvmlShutdown = lambda: calls.append("shutdown")
    m.nvmlDeviceGetCount = lambda: 1
    m.nvmlDeviceGetHandleByIndex = lambda i: f"gpu{i}"
    m.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=37, memory=12)
    m.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=2 * 2**30, total=32 * 2**30)
    m.nvmlDeviceGetPowerUsage = lambda h: 123456  # milliwatts
    m.nvmlDeviceGetTemperature = lambda h, sensor: 51
    m.nvmlDeviceGetClockInfo = lambda h, clock: 2400
    m.nvmlDeviceGetName = lambda h: b"Fake GPU"
    m.NVMLError = type("NVMLError", (Exception,), {})
    m.NVMLError_NotSupported = type("NVMLError_NotSupported", (m.NVMLError,), {})
    m.NVMLError_GpuIsLost = type("NVMLError_GpuIsLost", (m.NVMLError,), {})
    m.nvmlSystemGetDriverVersion = lambda: "1.2.3"
    return m


@pytest.fixture
def no_samplers(monkeypatch):
    for cls in (TegrastatsSampler, NvmlSampler, RocmSmiSampler):
        monkeypatch.setattr(cls, "available", staticmethod(lambda: False))
    monkeypatch.setattr(tbase, "_is_jetson", lambda: False)
    monkeypatch.setattr(tbase, "_has_nvidia_gpu", lambda: False)


def test_nvidia_gpu_without_pynvml_warns_and_is_recorded(no_samplers, monkeypatch):
    monkeypatch.setattr(tbase, "_has_nvidia_gpu", lambda: True)
    monkeypatch.setitem(sys.modules, "pynvml", None)  # makes `import pynvml` fail
    with pytest.warns(TelemetryWarning, match="NO TELEMETRY.*pynvml is not installed"):
        s = auto_sampler()
    assert isinstance(s, NullSampler)
    assert "pynvml is not installed" in s.describe()["telemetry_missing"]


def test_jetson_without_tegrastats_warns(no_samplers, monkeypatch):
    monkeypatch.setattr(tbase, "_is_jetson", lambda: True)
    with pytest.warns(TelemetryWarning, match="Jetson but tegrastats is not on PATH"):
        assert isinstance(auto_sampler(), NullSampler)


def test_machine_with_no_telemetry_source_warns(no_samplers):
    with pytest.warns(TelemetryWarning, match="no tegrastats, NVML or rocm-smi"):
        auto_sampler()


def test_explicit_null_sampler_does_not_claim_anything_is_missing():
    assert NullSampler().describe() == {"telemetry": "none"}


def test_nvml_power_is_gpu_power_and_close_shuts_nvml_down(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml(calls))
    assert NvmlSampler.available()
    s = NvmlSampler()
    r = s.read_once()
    assert r["power_gpu_w"] == pytest.approx(123.456)
    assert "power_board_w" not in r and [k for k in r if k.startswith("power_")] == ["power_gpu_w"]
    assert r["temp_gpu_c"] == 51 and r["gpu_util_pct"] == 37 and r["gpu_mem_used_mb"] == 2048
    assert s.describe() == {"telemetry": "nvml", "gpu": "Fake GPU", "driver": "1.2.3"}
    s.close()
    assert calls[-1] == "shutdown"


def test_nvml_channels_a_device_does_not_support_are_left_out(monkeypatch):
    """Jetson AGX Thor: NVML answers power, temperature and utilisation but
    raises NotSupported for memory info and clock info
    (bringup/thor/nvml_supported_calls.txt)."""
    nv = fake_pynvml()

    def not_supported(*args):
        raise nv.NVMLError_NotSupported()

    nv.nvmlDeviceGetMemoryInfo = nv.nvmlDeviceGetClockInfo = not_supported
    monkeypatch.setitem(sys.modules, "pynvml", nv)
    r = NvmlSampler().read_once()
    assert set(r) == {"power_gpu_w", "temp_gpu_c", "gpu_util_pct", "gpu_mem_util_pct"}


def test_other_nvml_errors_are_not_swallowed(monkeypatch):
    nv = fake_pynvml()

    def lost(*args):
        raise nv.NVMLError_GpuIsLost()

    nv.nvmlDeviceGetPowerUsage = lost
    monkeypatch.setitem(sys.modules, "pynvml", nv)
    with pytest.raises(nv.NVMLError_GpuIsLost):
        NvmlSampler().read_once()


def test_record_load_closes_its_sampler(monkeypatch, tmp_path):
    closed = []

    class Sampler(NullSampler):
        def close(self):
            closed.append(True)

    monkeypatch.setattr("bench2field.telemetry.auto_sampler", lambda interval_s=0.5: Sampler())
    assert cli.main(["record-load", "p", "--duration", "0.01", "--out", str(tmp_path / "p.json")]) == 0
    assert closed == [True]
