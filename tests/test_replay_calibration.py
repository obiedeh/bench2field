"""Calibrate-then-freeze: stressor duty cycles are steered to the profile
targets with the model idle, then held fixed for the run."""

import pytest

from bench2field.loadreplay import replay as replay_mod
from bench2field.loadreplay.record import LoadProfile
from bench2field.loadreplay.replay import Replay, Stressor, calibrate
from bench2field.telemetry.base import TelemetrySampler


class FakeStressor(Stressor):
    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class Plant(TelemetrySampler):
    """A machine where utilisation = gain * duty, plus whatever the model adds."""

    def __init__(self, stressors, gain, model_load_pct=0.0):
        super().__init__()
        self.stressors, self.gain, self.model_load_pct = stressors, gain, model_load_pct

    def read_once(self):
        return {s.channel: min(100.0, self.gain[s.name] * s.duty.value * 100 + self.model_load_pct)
                for s in self.stressors if s.name in self.gain}


def test_calibrate_steers_to_target_then_reports_frozen_duty():
    cpu = FakeStressor("cpu", "cpu_util_mean_pct", 40.0)
    plant = Plant([cpu], {"cpu": 0.8})  # open-loop 40% duty would only give 32%
    cal = calibrate([cpu], plant.read_once, wait=lambda: None)["cpu"]
    assert cal["steered"] and cal["converged"]
    assert cal["achieved_pct"] == pytest.approx(40.0, abs=2.0)
    assert cal["duty"] == cpu.duty.value == pytest.approx(0.5, abs=0.03)
    assert cal["target_pct"] == 40.0 and cal["channel"] == "cpu_util_mean_pct"


def test_calibrate_leaves_unobserved_channel_open_loop():
    cpu = FakeStressor("cpu", "cpu_util_mean_pct", 40.0)
    membw = FakeStressor("membw", "emc_util_pct", 30.0)
    plant = Plant([cpu, membw], {"cpu": 1.25})  # sampler has no EMC channel
    cal = calibrate([cpu, membw], plant.read_once, wait=lambda: None)
    assert cal["cpu"]["converged"] and cal["cpu"]["duty"] == pytest.approx(0.32, abs=0.03)
    assert cal["membw"] == {"channel": "emc_util_pct", "target_pct": 30.0, "duty": 0.30,
                            "achieved_pct": None, "steered": False, "converged": False}


def test_calibrate_reports_a_target_it_cannot_reach():
    gpu = FakeStressor("gpu", "gpu_util_pct", 90.0)
    plant = Plant([gpu], {"gpu": 0.5})  # full duty only reaches 50%
    cal = calibrate([gpu], plant.read_once, wait=lambda: None)["gpu"]
    assert cal["steered"] and not cal["converged"]
    assert cal["duty"] == 1.0 and cal["achieved_pct"] == pytest.approx(50.0)


def test_calibrate_survives_a_failing_sampler():
    cpu = FakeStressor("cpu", "cpu_util_mean_pct", 40.0)

    def broken():
        raise OSError("sampler died")

    cal = calibrate([cpu], broken, wait=lambda: None)["cpu"]
    assert not cal["steered"] and cal["duty"] == pytest.approx(0.40)


def test_replay_freezes_duty_while_the_model_runs(monkeypatch):
    cpu = FakeStressor("cpu", "cpu_util_mean_pct", 40.0)
    plant = Plant([cpu], {"cpu": 0.8})
    monkeypatch.setattr(replay_mod, "build_stressors", lambda profile, only=None: [cpu])
    prof = LoadProfile("rover", 1.0, 1.0, {}, [], targets={"cpu_util_mean_pct": 40.0})
    with Replay(prof, plant, calibrate_step_s=0.0) as rp:
        frozen = cpu.duty.value
        plant.model_load_pct = 25.0  # the benchmark starts and adds its own CPU load
        assert plant.read_once()["cpu_util_mean_pct"] > 60.0
        assert not hasattr(rp, "_steer")  # nothing left running that could react to it
        d = rp.describe()
    assert cpu.duty.value == frozen == d["replay_calibration"]["cpu"]["duty"]
    assert d["replay_calibration"]["cpu"]["achieved_pct"] == pytest.approx(40.0, abs=2.0)
    assert cpu.started and cpu.stopped
