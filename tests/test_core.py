import json

import pytest

from bench2field.loadreplay.record import profile_from_samples
from bench2field.metrics import attribute_gap, field_retention, latency_stats, replay_validity
from bench2field.runner import RunConfig, run, run_tier
from bench2field.schema import RunReport, TierResult, Variant
from bench2field.telemetry.base import summarize
from bench2field.telemetry.rocm_smi import parse_rocm_smi_json
from bench2field.telemetry.tegrastats import parse_tegrastats_line
from bench2field.verdict import Budget, evaluate

BASE = Variant("det", "onnxruntime", "TensorrtExecutionProvider", "fp32")
OPT = Variant("det", "onnxruntime", "TensorrtExecutionProvider", "int8", "ptq")


def report(variant, env, p95, hz=30.0, misses=0, telemetry=None, accuracy=None, service_p95=None):
    """A one-tier report whose response p95 is `p95` (service time too, unless given)."""
    resp = latency_stats([p95 * 0.8] * 94 + [p95] * 6)
    svc = resp if service_p95 is None else latency_stats([service_p95 * 0.8] * 94 + [service_p95] * 6)
    return RunReport(variant, env,
                     [TierResult(hz, hz, 10.0, 33.3, misses, svc, telemetry or {}, response=resp)],
                     accuracy=accuracy or {})


# --- metrics ---------------------------------------------------------------
def test_field_retention_partial_gain():
    r = field_retention(report(BASE, "bench-idle", 9.0), report(OPT, "bench-idle", 3.0),
                        report(BASE, "field", 12.0), report(OPT, "field", 6.0), 30.0)
    assert r.bench_speedup == pytest.approx(3.0, rel=1e-3)
    assert r.field_speedup == pytest.approx(2.0, rel=1e-3)
    assert r.retention == pytest.approx(0.5, rel=1e-3)  # (2-1)/(3-1)


def test_field_retention_undefined_without_bench_gain():
    r = field_retention(report(BASE, "bench-idle", 3.0), report(OPT, "bench-idle", 3.0),
                        report(BASE, "field", 4.0), report(OPT, "field", 4.0), 30.0)
    assert r.retention is None


def test_field_retention_rejects_mismatched_variants():
    with pytest.raises(ValueError):
        field_retention(report(BASE, "bench-idle", 9), report(OPT, "bench-idle", 3),
                        report(OPT, "field", 12), report(OPT, "field", 6), 30.0)


def test_attribution_keeps_unexplained_residual():
    idle, field = report(BASE, "bench-idle", 10.0), report(BASE, "field", 20.0)
    runs = [("thermal", report(BASE, "bench-replay:rover+thermal", 13.0)),
            ("membw", report(BASE, "bench-replay:rover+membw", 15.0))]
    at = attribute_gap(idle, field, runs, 30.0)
    assert at.total_gap_ms == pytest.approx(10.0, rel=1e-3)
    assert at.components_ms["membw"] == pytest.approx(5.0, rel=1e-3)
    assert at.interaction_ms == pytest.approx(2.0, rel=1e-2)
    assert sum(at.shares().values()) == pytest.approx(1.0)


def test_replay_validity():
    ok = replay_validity(report(BASE, "bench-replay:rover", 10.5), report(BASE, "field", 10.0), 30.0)
    bad = replay_validity(report(BASE, "bench-replay:rover", 13.0), report(BASE, "field", 10.0), 30.0)
    assert ok.valid and not bad.valid


# --- runner and schema -----------------------------------------------------
def test_runner_counts_deadline_misses_and_roundtrips(tmp_path):
    calls = {"n": 0}

    def infer(_):
        calls["n"] += 1

    tier = run_tier(infer, lambda i: None, target_hz=200, duration_s=0.2, deadline_ms=50)
    assert tier.latency.n == 40 and tier.deadline_misses == 0

    rep = run(BASE, "bench-idle", infer, lambda i: None,
              RunConfig(tiers_hz=[100.0], duration_s=0.1, warmup_s=0.0))
    loaded = RunReport.load(rep.save(tmp_path / "r.json"))
    assert loaded.variant.key == BASE.key and loaded.tier(100.0).latency.n == 10


def test_runner_flags_slow_inference_as_misses():
    import time

    tier = run_tier(lambda _: time.sleep(0.004), lambda i: None, target_hz=100,
                    duration_s=0.1, deadline_ms=2.0)
    assert tier.deadline_misses == tier.latency.n


def test_environment_label_is_enforced():
    with pytest.raises(ValueError):
        run(BASE, "lab", lambda _: None, lambda i: None, RunConfig(tiers_hz=[10.0], duration_s=0.1))


# --- telemetry parsers -----------------------------------------------------
ORIN = ("RAM 3072/7620MB (lfb 1x4MB) SWAP 0/3810MB (cached 0MB) CPU [12%@729,5%@729,off,off,"
        "3%@729,0%@729] EMC_FREQ 3%@2133 GR3D_FREQ 41%@[305] cpu@45.5C soc2@43.25C gpu@44C "
        "tj@45.5C VDD_IN 4520mW/4520mW VDD_CPU_GPU_CV 480mW/480mW VDD_SOC 1440mW/1440mW")
THOR = "RAM 9000/125770MB CPU [1%@972,0%@972] GR3D_FREQ 0% tj@41.0C VIN 24212mW/24212mW"


def test_tegrastats_orin_line():
    s = parse_tegrastats_line(ORIN)
    assert s["power_vdd_in_w"] == pytest.approx(4.52)
    assert s["temp_tj_c"] == 45.5 and s["gpu_util_pct"] == 41 and s["emc_util_pct"] == 3
    assert s["cpu_cores_online"] == 4 and s["ram_used_mb"] == 3072


def test_tegrastats_thor_line():
    s = parse_tegrastats_line(THOR)
    assert s["power_vin_w"] == pytest.approx(24.212) and s["temp_tj_c"] == 41.0


def test_rocm_smi_parser():
    payload = {"card0": {"Current Socket Graphics Package Power (W)": "212.0",
                         "Temperature (Sensor junction) (C)": "61.0",
                         "GPU use (%)": "87", "GPU Memory Allocated (VRAM%)": "40"}}
    s = parse_rocm_smi_json(payload)
    assert s["power_gpu_w"] == 212.0 and s["temp_junction_c"] == 61.0 and s["gpu_util_pct"] == 87


def test_load_profile_targets():
    samples = [{"t": float(i), **parse_tegrastats_line(ORIN)} for i in range(5)]
    prof = profile_from_samples("rover-slam", samples, 1.0)
    assert prof.targets["emc_util_pct"] == 3 and prof.soak_temp_c == 45.5
    assert prof.duration_s == 4.0


# --- verdict ---------------------------------------------------------------
def test_verdict_go_nogo_and_no_data():
    tel = summarize([{"t": 0, "power_vin_w": 12.0, "temp_tj_c": 70.0}])
    rep = report(BASE, "field", 20.0, telemetry=tel, accuracy={"map50": 0.61})
    b = Budget("rover", 30.0, p99_ms=25.0, max_miss_rate=0.001, max_power_w=15.0,
               power_channel="power_vin_w", max_temp_c=85.0, min_accuracy={"map50": 0.6})
    assert evaluate(rep, b).status == "GO"
    b.max_power_w = 10.0
    assert evaluate(rep, b).status == "NO-GO"
    b2 = Budget("rover", 30.0, min_accuracy={"mota": 0.5})
    assert evaluate(rep, b2).status == "INCOMPLETE"


def test_budget_yaml_loads():
    b = Budget.from_yaml("configs/budgets/rover_perception.yaml")
    assert b.target_hz == 30 and b.p99_ms == 25.0
