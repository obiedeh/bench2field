"""Stressor processes: spawned, separate from the benchmark, and loud when
they cannot start."""

import os

import pytest

from bench2field.loadreplay import replay as replay_mod
from bench2field.loadreplay.record import LoadProfile
from bench2field.loadreplay.replay import CpuStressor, GpuStressor, Replay, Stressor


def test_cpu_stressor_runs_in_spawned_processes_and_stops():
    s = CpuStressor(50.0, workers=2)
    s.start()
    try:
        assert len(s.procs) == 2 and all(p.is_alive() for p in s.procs)
        assert all(p.pid != os.getpid() for p in s.procs)
        assert type(s.procs[0]).__name__ == "SpawnProcess"
    finally:
        s.stop()
    assert not any(p.is_alive() for p in s.procs)


def test_gpu_stressor_start_failure_is_raised_in_the_parent():
    # A negative matrix size fails wherever this runs; so does a missing
    # PyTorch or a missing GPU. None of them may pass silently.
    s = GpuStressor(50.0, size=-1, start_timeout_s=120)
    with pytest.raises(RuntimeError, match="GPU stressor could not start"):
        s.start()
    assert s.procs == []


def test_replay_stops_started_stressors_when_a_later_one_fails(monkeypatch):
    class Ok(Stressor):
        stopped = False

        def start(self):
            pass

        def stop(self):
            self.stopped = True

    class Broken(Stressor):
        def start(self):
            raise RuntimeError("GPU stressor could not start: no device")

    ok, broken = Ok("cpu", "cpu_util_mean_pct", 40.0), Broken("gpu", "gpu_util_pct", 50.0)
    monkeypatch.setattr(replay_mod, "build_stressors", lambda profile, only=None: [ok, broken])
    prof = LoadProfile("p", 1.0, 1.0, {}, [], targets={})
    with pytest.raises(RuntimeError, match="could not start"):
        Replay(prof).__enter__()
    assert ok.stopped


def test_gpu_stressor_has_its_own_process_and_cuda_context():
    """Real hardware only."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA or ROCm device")
    s = GpuStressor(50.0)
    s.start()
    try:
        (proc,) = s.procs
        assert proc.is_alive() and proc.pid != os.getpid() and s.device
        # The benchmark process's own PyTorch never touched the GPU ...
        assert not torch.cuda.is_initialized()
        # ... and the driver lists the worker as a separate GPU client.
        try:
            import pynvml
        except ImportError:
            return
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        pids = {p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h)}
        assert proc.pid in pids
    finally:
        s.stop()
    assert not proc.is_alive()
