"""tegrastats and nvpmodel output captured on real boards (bringup/<board>/)."""

import re
import subprocess
import types
from pathlib import Path

import pytest

from bench2field.telemetry import tegrastats
from bench2field.telemetry.tegrastats import parse_tegrastats_line, read_nvpmodel

BRINGUP = Path(__file__).resolve().parents[1] / "bringup"
THOR_CAPTURES = sorted((BRINGUP / "thor").glob("tegrastats_*.txt"))
THOR_KEYS = {
    "ram_used_mb", "ram_total_mb", "cpu_util_mean_pct", "cpu_util_max_pct", "cpu_cores_online",
    "temp_cpu_c", "temp_tj_c", "temp_soc012_c", "temp_gpu_c", "temp_soc345_c",
    "power_vdd_gpu_w", "power_vdd_cpu_soc_mss_w", "power_vin_sys_5v0_w", "power_vin_w",
    "power_board_w",
}


def test_thor_captures_exist():
    assert [p.name for p in THOR_CAPTURES] == [
        "tegrastats_cuda_load.txt", "tegrastats_idle.txt", "tegrastats_trt_load.txt"]


@pytest.mark.parametrize("capture", THOR_CAPTURES, ids=lambda p: p.stem)
def test_every_real_thor_line_parses_completely(capture):
    lines = capture.read_text().splitlines()
    assert len(lines) == 20
    for line in lines:
        s = parse_tegrastats_line(line)
        assert set(s) == THOR_KEYS, line
        assert s["cpu_cores_online"] == 14 and s["ram_total_mb"] == 125772
        assert 0 <= s["cpu_util_mean_pct"] <= s["cpu_util_max_pct"] <= 100
        assert all(20 < s[k] < 105 for k in s if k.startswith("temp_"))
        # Every field tegrastats printed was captured: 5 temperatures, 4 rails.
        assert len(re.findall(r"@[\d.]+C\b", line)) == 5 and line.count("mW/") == 4
        # VIN is the board total: it is what power_board_w reports, and it is
        # larger than the three sub-rails put together.
        assert s["power_board_w"] == s["power_vin_w"]
        assert s["power_vin_w"] > s["power_vdd_gpu_w"] + s["power_vdd_cpu_soc_mss_w"] + s["power_vin_sys_5v0_w"]


def test_thor_tegrastats_has_no_gpu_or_memory_controller_load():
    """Not even under GPU load: replay cannot steer GPU or memory bandwidth there."""
    for capture in THOR_CAPTURES:
        text = capture.read_text()
        assert "GR3D_FREQ" not in text and "EMC_FREQ" not in text
        for line in text.splitlines():
            s = parse_tegrastats_line(line)
            assert "gpu_util_pct" not in s and "emc_util_pct" not in s


def test_board_power_is_not_guessed_for_an_unconfirmed_rail():
    s = parse_tegrastats_line("RAM 1/2MB CPU [1%@1] tj@40C VDD_SOMETHING 5000mW/5000mW")
    assert s["power_vdd_something_w"] == 5.0 and "power_board_w" not in s


def test_nvpmodel_mode_from_real_thor_output(monkeypatch):
    captured = (BRINGUP / "thor" / "nvpmodel.txt").read_text()
    assert captured == "NV Power Mode: 120W\n1\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=captured))
    assert read_nvpmodel() == "NV Power Mode: 120W"


def test_nvpmodel_missing_or_silent_is_none(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("nvpmodel")

    monkeypatch.setattr(subprocess, "run", missing)
    assert read_nvpmodel() is None
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=""))
    assert read_nvpmodel() is None
    assert tegrastats.BOARD_POWER_RAILS == ("VIN",)
