"""b2f report: the data layer that turns a case study's committed files into
numbers, and a smoke test of the page it renders."""

import json
from pathlib import Path

import pytest

from bench2field import cli
from bench2field.metrics import latency_stats
from bench2field.report import load_case_study, render
from bench2field.report.data import parse_topic_hz
from bench2field.schema import RunReport, TierResult, Variant

ROOT = Path(__file__).resolve().parents[1]
CASE_STUDY = ROOT / "case_studies" / "01_perception_detector"


def make_report(variant, hz_list, p95_by_hz, repeat, order, nvpmodel="NV Power Mode: 120W", late_at=None,
                power_key="power_board_w"):
    tiers = []
    for hz in hz_list:
        p95 = p95_by_hz[hz]
        lat = latency_stats([p95 * 0.8] * 94 + [p95] * 6)
        late = lat.n if late_at and hz in late_at else 0  # every processed frame late
        tiers.append(TierResult(hz, hz * (0.8 if late else 1.0), 10.0, 33.3, late, lat,
                                {power_key: {"p50": 10.0 + hz / 10, "peak": 20.0},
                                 "temp_tj_c": {"p50": 50.0 + repeat, "peak": 52.0 + repeat},
                                 "gpu_util_pct": {"p50": 40.0, "peak": 90.0}},
                                response=lat))
    return RunReport(variant, "bench-idle", tiers, platform={
        "host": "x", "nvpmodel": nvpmodel, "onnxruntime": "1.24.0", "tensorrt": "10.7.0", "cudnn": "9.3.0",
        "cuda_runtime": "12.6", "allow_spinning": False, "sweep": {"name": "s", "label": variant.technique or variant.precision,
                                                                     "repeat": repeat, "order": order},
        "background": {"containers_running": [], "top_processes": [], "stopped_for_this_run": []},
        "git": {"commit": "c" * 40, "dirty": False, "root": "/r"}})


def write_sweep(d: Path, name: str, variants: dict, hz_list, repeats=3, complete=True, late_at=None):
    d = d / name
    d.mkdir(parents=True)
    runs = []
    order = 0
    for r in range(1, repeats + 1):
        for label, (variant, p95_by_hz) in variants.items():
            order += 1
            rep = make_report(variant, hz_list, p95_by_hz, r, order, late_at=late_at)
            rep.save(d / f"{label}_r{r}.json")
            runs.append({"order": order, "label": label, "repeat": r, "file": f"{label}_r{r}.json", "run_id": rep.run_id})
    manifest = {"sweep": name, "config": {"tiers_hz": hz_list, "deadline_ms": 33.3, "repeats": repeats,
                                          "stopped": [], "no_spin": True, "variants": []},
                "complete": complete, "runs": runs, "git": {"commit": "c" * 40, "dirty": False}}
    (d / f"sweep_{name}.json").write_text(json.dumps(manifest))


def write_profile(d: Path, name: str, board_total_p50: float):
    stages = {s: {"p50_ms": board_total_p50 / 6, "p95_ms": board_total_p50 / 6 * 1.1, "p99_ms": 1, "mean_ms": 1, "max_ms": 1, "n": 300}
              for s in ("decode", "preprocess", "h2d", "inference", "d2h", "postprocess")}
    data = {"stages": stages, "total_per_frame": {"p50_ms": board_total_p50, "p95_ms": board_total_p50 * 1.1, "max_ms": 1},
            "stage_share_of_total_p50": {s: 1 / 6 for s in stages}, "session_run_numpy": {"p50_ms": board_total_p50 / 8},
            "frames": {"n": 300, "dir": "data/f", "set_sha256": "abc" * 20, "decoded_shape": [480, 640, 3]},
            "settings": {"ort_allow_spinning": False}, "platform": {"onnxruntime": "1.24.0", "tensorrt": "10.7.0",
                                                                     "cudnn": "9.3.0", "cuda_runtime": "12.6", "nvpmodel": None},
            "background": {"containers_running": []}, "detections_per_frame": {"mean": 0}}
    (d / f"{name}.json").write_text(json.dumps(data))


@pytest.fixture
def synthetic_case_study(tmp_path):
    cs = tmp_path / "cs"
    runs = cs / "runs"
    base = Variant("m", "onnxruntime", "TensorrtExecutionProvider", "fp32")
    ref = Variant("m", "onnxruntime", "CUDAExecutionProvider", "fp32")
    hz = [10.0, 26.0, 100.0]
    write_sweep(runs, "bench_a", {"trt_fp32": (base, {10.0: 5.0, 26.0: 4.0, 100.0: 3.0}),
                                  "cuda_fp32": (ref, {10.0: 8.0, 26.0: 6.0, 100.0: 5.0})}, hz)
    write_sweep(runs, "bench_a_old", {"trt_fp32": (base, {10.0: 5.5, 26.0: 4.5, 100.0: 3.5})}, hz, repeats=1)
    write_sweep(runs, "field_b", {"trt_fp32": (base, {10.0: 30.0, 26.0: 25.0, 100.0: 5000.0}),
                                  "cuda_fp32": (ref, {10.0: 40.0, 26.0: 28.0, 100.0: 9000.0})}, hz, late_at=[100.0])
    write_profile(runs, "profile_a", 6.0)
    write_profile(runs, "profile_b", 42.0)
    (cs / "budget.yaml").write_text("name: b\ntarget_hz: 30\np99_ms: 25.0\nmax_power_w: 15.0\npower_channel: power_board_w\n")
    (cs / "hz.txt").write_text("# capture\n# configured 30\naverage rate: 26.0\n\tmin: 0.004s\naverage rate: 25.0\naverage rate: 27.0\n\n# ros2 topic hz /scan\naverage rate: 28.5\n")
    (cs / "report.yaml").write_text("""
title: Synthetic
headline_variant: trt_fp32
reference_variant: cuda_fp32
budget: budget.yaml
camera_rate_capture: hz.txt
boards:
  bench-a: {label: Board A, role: bench, baseline: bench_a, references: {bench_a_old: old}}
  field-b: {label: Board B, role: field, baseline: field_b}
profiles:
  profile_a: {board: bench-a, label: same frames}
  profile_b: {board: field-b, label: same frames}
same_frames: [profile_a, profile_b]
headline_profile: profile_b
""")
    return cs


def test_topic_hz_parser_takes_the_first_block_only():
    text = "# header\n# more header\naverage rate: 26.0\n\tmin: 0.1s\naverage rate: 25.0\n\n# second topic\naverage rate: 99.0\n"
    assert parse_topic_hz(text) == [26.0, 25.0]
    assert parse_topic_hz("") == []


def test_load_synthetic_case_study(synthetic_case_study):
    cs = load_case_study(synthetic_case_study)
    assert cs.camera_rate_hz == pytest.approx(26.0) and cs.rate_hz == 26.0 and cs.deadline_ms == 33.3
    assert cs.warnings == []
    a, b = cs.boards["bench-a"], cs.boards["field-b"]
    assert a.baseline.name == "bench_a" and [s.name for s in a.references] == ["bench_a_old"]
    assert a.references[0].status == "reference" and a.baseline.status == "baseline"
    st = a.baseline.tier_stat("trt_fp32", 26.0)
    assert st.n == 3 and st.median == pytest.approx(4.0)
    assert b.baseline.saturated("trt_fp32", 100.0) and not b.baseline.saturated("trt_fp32", 26.0)
    assert b.baseline.misses("trt_fp32", 100.0) == (300, 300)
    assert a.baseline.tier_telemetry("trt_fp32", 26.0, "power_board_w") == pytest.approx(12.6)
    assert cs.profiles["profile_b"].total["p95_ms"] == pytest.approx(46.2)
    assert cs.profiles["profile_b"].host_ms == pytest.approx(21.0)


def test_missing_report_yaml_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="report.yaml"):
        load_case_study(tmp_path)


def test_missing_reference_is_a_warning_not_a_crash(synthetic_case_study):
    cfg = synthetic_case_study / "report.yaml"
    cfg.write_text(cfg.read_text().replace("references: {bench_a_old: old}", "references: {nope: gone}"))
    cs = load_case_study(synthetic_case_study)
    assert cs.boards["bench-a"].references == [] and any("nope" in w for w in cs.warnings)


def test_incomplete_sweep_is_marked_stopped(synthetic_case_study):
    runs = synthetic_case_study / "runs"
    base = Variant("m", "onnxruntime", "TensorrtExecutionProvider", "fp32")
    write_sweep(runs, "bench_a_stopped", {"trt_fp32": (base, {10.0: 5.0, 26.0: 4.0, 100.0: 3.0})}, [10.0, 26.0, 100.0],
                repeats=1, complete=False)
    cfg = synthetic_case_study / "report.yaml"
    cfg.write_text(cfg.read_text().replace("references: {bench_a_old: old}", "references: {bench_a_old: old, bench_a_stopped: s}"))
    cs = load_case_study(synthetic_case_study)
    assert {s.name: s.status for s in cs.boards["bench-a"].references} == {"bench_a_old": "reference", "bench_a_stopped": "stopped"}


def test_render_synthetic_is_self_contained_and_says_the_frame_misses(synthetic_case_study):
    cs = load_case_study(synthetic_case_study)
    page = render(cs)
    assert page.startswith("<!doctype html>") and "http://" not in page and "https://" not in page
    assert "46.2 ms" in page and "does not fit" in page and "25.0 ms" in page
    assert page.count("<svg") == 4 and "✕" in page  # no spin pair configured; the saturated tier is marked
    assert "stopped, not a baseline" not in page and "reference" in page
    assert "NaN" not in page and "None" not in page.replace("none", "")


def test_cli_report_on_the_real_case_study(tmp_path):
    out = tmp_path / "r.html"
    svg = tmp_path / "h.svg"
    assert cli.main(["report", str(CASE_STUDY), "--out", str(out), "--headline-svg", str(svg)]) == 0
    page = out.read_text()
    assert "Case study 01" in page and "does not fit" in page and "http" not in page.replace("http-equiv", "")
    cs = load_case_study(CASE_STUDY)
    assert cs.rate_hz == 26.0 and cs.warnings == []
    assert {k: b.baseline.name for k, b in cs.boards.items()} == {
        "bench-5090": "bench_5090_nospin", "bench-thor": "bench_thor_nospin", "field-orin": "field_orin"}
    assert svg.read_text().startswith("<svg xmlns=") and "var(--" not in svg.read_text()
