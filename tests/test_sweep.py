"""b2f sweep: alternating repeats, one report per run, a manifest of the order."""

import json
import os
from pathlib import Path

import pytest

from bench2field import cli
from bench2field.metrics import latency_stats
from bench2field.schema import RunReport, TierResult, Variant
from bench2field.sweep import (
    SweepConfig,
    SweepVariant,
    plan,
    report_path,
    run_argv,
    run_sweep,
    subprocess_runner,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]


def config(**kw):
    base = dict(name="s", repeats=3, tiers_hz=[30.0], duration_s=1.0,
                variants=[SweepVariant("fp32", "m.onnx", "tensorrt"),
                          SweepVariant("int8", "m.onnx", "tensorrt", "int8", "ptq")])
    return SweepConfig(**(base | kw))


class FakeRunner:
    """Stands in for `b2f run`: writes a report where --out says."""

    def __init__(self, fail_on=None, p95=None):
        self.calls, self.fail_on, self.p95 = [], fail_on, p95 or {}

    def __call__(self, argv):
        self.calls.append(argv)
        label = argv[argv.index("--sweep-label") + 1]
        if len(self.calls) == self.fail_on:
            return 3
        lat = latency_stats([self.p95.get(label, 5.0) + 0.1 * len(self.calls)] * 10)
        hz = float(argv[argv.index("--tiers") + 1])
        RunReport(Variant("m", "onnxruntime", "P", argv[argv.index("--precision") + 1]), "bench-idle",
                  [TierResult(hz, hz, 1.0, 33.3, 0, lat, {}, response=lat)]).save(argv[argv.index("--out") + 1])
        return 0


def test_plan_alternates_variants_within_each_repeat():
    order = [(r, v.label) for r, v in plan(config())]
    assert order == [(1, "fp32"), (1, "int8"), (2, "fp32"), (2, "int8"), (3, "fp32"), (3, "int8")]


def test_sweep_runs_in_planned_order_and_writes_reports_and_manifest(tmp_path):
    runner = FakeRunner()
    manifest = run_sweep(config(), tmp_path, runner)
    assert [c[c.index("--sweep-label") + 1] for c in runner.calls] == ["fp32", "int8"] * 3
    assert [c[c.index("--order") + 1] for c in runner.calls] == ["1", "2", "3", "4", "5", "6"]
    assert sorted(p.name for p in tmp_path.glob("*_r*.json")) == [
        "fp32_r1.json", "fp32_r2.json", "fp32_r3.json", "int8_r1.json", "int8_r2.json", "int8_r3.json"]
    on_disk = json.loads((tmp_path / "sweep_s.json").read_text())
    assert on_disk == manifest and manifest["complete"]
    assert [(r["order"], r["label"], r["repeat"]) for r in manifest["runs"]][:3] == [
        (1, "fp32", 1), (2, "int8", 1), (3, "fp32", 2)]
    assert all(r["run_id"] == RunReport.load(tmp_path / r["file"]).run_id for r in manifest["runs"])
    assert manifest["config"]["repeats"] == 3 and manifest["config"]["variants"][1]["technique"] == "ptq"


def test_run_argv_carries_the_whole_config():
    cfg = config(drop_late=True, cooldown_c=45.0, replay="profiles/rover.json", only="cpu",
                 stopped=["docker container urban-edge-vllm"], no_spin=True,
                 variants=[SweepVariant("a", "det.onnx", "cuda", name="yolo-s")])
    argv = run_argv(cfg, cfg.variants[0], 2, 5, Path("out/a_r2.json"))
    assert argv[argv.index("--stopped") + 1] == "docker container urban-edge-vllm"
    assert argv[:2] == ["run", "det.onnx"]
    for flag, value in (("--provider", "cuda"), ("--tiers", "30"), ("--duration", "1"),
                        ("--deadline-ms", "33.3"), ("--sweep", "s"), ("--repeat", "2"), ("--order", "5"),
                        ("--name", "yolo-s"), ("--cooldown-c", "45"), ("--replay", "profiles/rover.json"),
                        ("--only", "cpu"), ("--out", "out/a_r2.json")):
        assert argv[argv.index(flag) + 1] == value
    assert "--drop-late" in argv and "--no-spin" in argv
    assert "--no-spin" not in run_argv(config(), config().variants[0], 1, 1, Path("o.json"))
    # Every flag is one `b2f run` really has, and lands where the run reads it.
    a = cli.build_parser().parse_args(argv)
    assert (a.cmd, a.model, a.drop_late, a.cooldown_c, a.sweep_label) == ("run", "det.onnx", True, 45.0, "a")
    assert a.no_spin
    assert a.stopped == ["docker container urban-edge-vllm"]


def test_sweep_never_overwrites_reports_and_can_resume(tmp_path):
    run_sweep(config(repeats=1), tmp_path, FakeRunner())
    first = (tmp_path / "fp32_r1.json").read_text()
    with pytest.raises(ValueError, match="already exists"):
        run_sweep(config(), tmp_path, FakeRunner())
    runner = FakeRunner()
    manifest = run_sweep(config(), tmp_path, runner, resume=True)
    assert len(runner.calls) == 4 and (tmp_path / "fp32_r1.json").read_text() == first
    assert [r.get("skipped") is not None for r in manifest["runs"]] == [True, True, False, False, False, False]


def test_failed_run_stops_the_sweep_and_the_manifest_says_so(tmp_path):
    with pytest.raises(RuntimeError, match="int8 repeat 2 failed .*exited 3.* 3 of 6 runs done"):
        run_sweep(config(), tmp_path, FakeRunner(fail_on=4))
    manifest = json.loads((tmp_path / "sweep_s.json").read_text())
    assert not manifest["complete"] and manifest["finished_at"] is None
    assert len(manifest["runs"]) == 4 and manifest["runs"][-1]["failed"] == "b2f run exited 3"
    assert not report_path(tmp_path, config().variants[1], 2).exists()


def test_summary_gives_median_and_range_per_variant(tmp_path):
    cfg = config()
    run_sweep(cfg, tmp_path, FakeRunner(p95={"fp32": 9.0, "int8": 3.0}))
    text = summarize(cfg, tmp_path)
    assert "response_p95_ms across 3 repeats" in text
    fp32 = next(line for line in text.splitlines() if line.startswith("fp32")).split()
    assert [float(x) for x in fp32[1:]] == pytest.approx([30, 9.3, 9.1, 9.5])  # calls 1, 3, 5


@pytest.mark.parametrize("bad, message", [
    (dict(repeats=0), "repeats must be at least 1"),
    (dict(variants=[]), "at least one variant"),
    (dict(variants=[SweepVariant("a", "m"), SweepVariant("a", "n")]), "labels must be unique"),
    (dict(variants=[SweepVariant("a/b", "m")]), "variant label"),
    (dict(name="has space"), "sweep name"),
    (dict(environment="lab"), "environment must be"),
])
def test_bad_configs_are_rejected(bad, message):
    with pytest.raises(ValueError, match=message):
        config(**bad)


def test_unknown_config_key_is_rejected_not_ignored():
    with pytest.raises(ValueError, match="bad sweep config"):
        SweepConfig.from_dict({"name": "s", "variants": [{"label": "a", "model": "m"}], "repeat": 3})


def test_example_config_loads():
    cfg = SweepConfig.from_yaml(ROOT / "configs" / "sweeps" / "bringup_tiny.yaml")
    assert cfg.repeats == 3 and [v.label for v in cfg.variants] == ["cuda_fp32", "trt_fp16"]


def test_real_sweep_on_cpu_records_sweep_position_in_each_report(tmp_path):
    """End to end through real `b2f run` subprocesses."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array(np.eye(8, dtype=np.float32), "W")
    g = helper.make_graph([helper.make_node("MatMul", ["x", "W"], ["y"])], "tiny",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])], [w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, tmp_path / "tiny.onnx")
    cfg = SweepConfig("cpu-check", repeats=2, tiers_hz=[50.0], duration_s=0.2, warmup_s=0.0,
                      variants=[SweepVariant("a", str(tmp_path / "tiny.onnx")),
                                SweepVariant("b", str(tmp_path / "tiny.onnx"), technique="other")])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])])}

    def runner(argv):
        return subprocess_runner([*argv, "--no-telemetry"], env=env)

    manifest = run_sweep(cfg, tmp_path / "out", runner)
    assert manifest["complete"] and len(manifest["runs"]) == 4
    rep = RunReport.load(tmp_path / "out" / "b_r2.json")
    assert rep.platform["sweep"] == {"name": "cpu-check", "label": "b", "repeat": 2, "order": 4}
    assert rep.variant.technique == "other" and rep.tier(50.0).response.n == 10


def test_cli_sweep_stopped_reaches_reports_and_manifest(tmp_path, monkeypatch):
    import json

    from bench2field import sweep as sweep_mod

    cfg_file = tmp_path / "s.yaml"
    cfg_file.write_text("name: s\nrepeats: 1\ntiers_hz: [30]\nstopped: [from-config]\n"
                        "variants:\n  - {label: a, model: m.onnx}\n")
    runner = FakeRunner()
    monkeypatch.setattr(sweep_mod, "subprocess_runner", runner)
    assert cli.main(["sweep", str(cfg_file), "--out-dir", str(tmp_path / "out"),
                     "--stopped", "docker container physical-ai-vllm"]) == 0
    argv = runner.calls[0]
    assert [argv[i + 1] for i, x in enumerate(argv) if x == "--stopped"] == [
        "from-config", "docker container physical-ai-vllm"]
    manifest = json.loads((tmp_path / "out" / "sweep_s.json").read_text())
    assert manifest["config"]["stopped"] == ["from-config", "docker container physical-ai-vllm"]
