"""CPU frequency and governor are read from cpufreq sysfs, never written."""

import time

import pytest

from bench2field.telemetry.cpufreq import CpuFreqSampler, cpu_dirs, describe_cpufreq, read_cpufreq


def fake_sysfs(root, cores):
    """cores: {index: (cur_khz, governor, min_khz, max_khz)}; plus entries that must be ignored."""
    for i, (cur, gov, lo, hi) in cores.items():
        f = root / f"cpu{i}" / "cpufreq"
        f.mkdir(parents=True)
        (f / "scaling_cur_freq").write_text(f"{cur}\n")
        (f / "scaling_governor").write_text(f"{gov}\n")
        (f / "scaling_min_freq").write_text(f"{lo}\n")
        (f / "scaling_max_freq").write_text(f"{hi}\n")
    (root / "cpufreq").mkdir()          # the policy directory, not a core
    (root / "cpu6").mkdir()             # an offline core has no cpufreq directory
    (root / "cpuidle").mkdir()
    return root


def test_reads_every_online_core_in_numeric_order(tmp_path):
    root = fake_sysfs(tmp_path, {0: (1984000, "schedutil", 729600, 1984000), 10: (729600, "schedutil", 729600, 1984000),
                                 2: (1190400, "schedutil", 729600, 1984000)})
    assert [d.name for d in cpu_dirs(root)] == ["cpu0", "cpu2", "cpu10"]
    assert read_cpufreq(root) == {"cpu0": 1984000, "cpu2": 1190400, "cpu10": 729600}
    info = describe_cpufreq(root)
    assert info["cpu2"] == {"governor": "schedutil", "min_khz": 729600, "max_khz": 1984000}


def test_missing_cpufreq_is_empty_not_an_error(tmp_path):
    assert read_cpufreq(tmp_path) == {} and describe_cpufreq(tmp_path) == {}
    s = CpuFreqSampler(interval_s=0.01, root=tmp_path)
    s.start()
    time.sleep(0.03)
    out = s.stop()
    assert out["n_samples"] == 0 and out["mean_khz"] is None and out["series"] == []


def test_sampler_summarises_per_core_and_across_cores(tmp_path):
    root = fake_sysfs(tmp_path, {0: (1984000, "schedutil", 729600, 1984000), 1: (729600, "schedutil", 729600, 1984000)})
    s = CpuFreqSampler(interval_s=0.01, root=root)
    s.start()
    time.sleep(0.05)
    (root / "cpu1" / "cpufreq" / "scaling_cur_freq").write_text("1984000\n")  # the governor raises core 1
    time.sleep(0.05)
    out = s.stop()
    assert out["n_samples"] >= 4
    assert out["per_core"]["cpu0"] == {"p50_khz": 1984000.0, "min_khz": 1984000.0, "max_khz": 1984000.0}
    assert out["per_core"]["cpu1"]["min_khz"] == 729600.0 and out["per_core"]["cpu1"]["max_khz"] == 1984000.0
    assert out["mean_khz"]["min"] == pytest.approx((1984000 + 729600) / 2) and out["mean_khz"]["max"] == 1984000.0
    assert 0.5 < out["share_at_max"] < 1.0
    assert out["cores"]["cpu1"]["governor"] == "schedutil"
    ts = [t for t, _ in out["series"]]
    assert ts == sorted(ts) and len(out["series"]) == out["n_samples"]


def test_reading_never_writes(tmp_path):
    root = fake_sysfs(tmp_path, {0: (1984000, "schedutil", 729600, 1984000)})
    before = {p: p.read_text() for p in root.rglob("*") if p.is_file()}
    read_cpufreq(root); describe_cpufreq(root)
    s = CpuFreqSampler(interval_s=0.01, root=root); s.start(); time.sleep(0.03); s.stop()
    assert {p: p.read_text() for p in root.rglob("*") if p.is_file()} == before
