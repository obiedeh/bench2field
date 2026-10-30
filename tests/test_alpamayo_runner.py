"""Case study 02's runner: the parts that need no GPU.

The clip list follows the rule written in PLAN.md before any run (the upstream
example clip first, then the first seven ids of the upstream clip list), the
summary statistics are computed the way the README describes them, and the
telemetry window selection matches the sampler's clock. torch and yaml are
optional dependencies of the repository, so the tests skip without them."""

from pathlib import Path

import pytest

CASE = Path(__file__).resolve().parents[1] / "case_studies" / "02_alpamayo_av_reasoning"
EXAMPLE_CLIP = "030c760c-ae38-49aa-9ad8-f5650a545d26"


def _runner():
    pytest.importorskip("torch")
    pytest.importorskip("yaml")
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_decisions", CASE / "run_decisions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_clips_follow_the_rule_in_the_plan():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((CASE / "clips.yaml").read_text())
    assert cfg["clips"][0] == EXAMPLE_CLIP
    assert len(cfg["clips"]) == 8
    assert len(set(cfg["clips"])) == 8
    assert cfg["t0_us"] == 5_100_000 and cfg["num_frames"] == 4


def test_summary_statistics():
    rd = _runner()
    decisions = []
    for i, lat in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        decisions.append({
            "latency_total_s": lat, "ttft_s": 0.5, "rollout_s": lat - 0.4, "trajectory_s": 0.3,
            "generated_tokens": 10 + i, "decode_tokens_per_s": 20.0,
            "peak_memory": {"torch_max_allocated_gib": 21.0 + i * 0.1, "torch_max_reserved_gib": 22.0, "nvml_used_gib": 25.0},
        })
    s = rd.summarize(decisions)
    assert s["n_decisions"] == 5
    assert s["latency_median_s"] == 3.0
    assert s["latency_p95_s"] == 5.0
    assert s["decisions_per_minute"] == 20.0
    assert s["generated_tokens_median"] == 12
    assert s["peak_torch_allocated_gib_max"] == 21.4
    assert rd.summarize([]) == {}


def test_telemetry_window_and_power_summary():
    rd = _runner()
    samples = [{"t": t, "power_gpu_w": 100.0 + t} for t in range(10)]
    win = rd.samples_between(samples, 2.5, 6.0)
    assert [s["t"] for s in win] == [3, 4, 5, 6]
    assert rd.power_summary(win) == {"mean_w": 104.5, "max_w": 106.0, "n": 4, "channel": "power_gpu_w"}
    assert rd.power_summary([{"t": 1, "power_board_w": 40.0}])["channel"] == "power_board_w"
    assert rd.power_summary([]) is None


def test_synthetic_sample_has_the_loader_shapes():
    rd = _runner()
    d = rd.synthetic_sample("abc", 5_100_000, 4)
    assert tuple(d["image_frames"].shape) == (4, 4, 3, 1080, 1920)
    assert tuple(d["ego_history_xyz"].shape) == (1, 1, 16, 3)
    assert tuple(d["ego_history_rot"].shape) == (1, 1, 16, 3, 3)
    assert tuple(d["ego_future_xyz"].shape) == (1, 1, 64, 3)
