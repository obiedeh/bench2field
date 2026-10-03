"""Print the phase 1 tables from the committed run files, so every number in
PHASE1_FINDINGS.md can be regenerated from runs/.

    env -u PYTHONPATH .venv/bin/python case_studies/01_perception_detector/summarize_phase1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bench2field.metrics import repeat_stat  # noqa: E402
from bench2field.schema import RunReport  # noqa: E402
from bench2field.sweep import SweepConfig, report_path  # noqa: E402

RUNS = HERE / "runs"
SWEEPS = {"RTX 5090 (re-run, with the 26 Hz tier)": "bench_5090_rerun",
          "Jetson AGX Thor (re-run, with the 26 Hz tier)": "bench_thor_rerun",
          "RTX 5090 (first sweep)": "bench_5090", "Jetson AGX Thor (first sweep)": "bench_thor"}
PROFILES = sorted(RUNS.glob("profile_*.json"))
STAGES = ("decode", "preprocess", "h2d", "inference", "d2h", "postprocess")


def tel(reports: list[RunReport], hz: float, key: str) -> str:
    vals = [r.tier(hz).telemetry.get(key, {}).get("p50") for r in reports]
    vals = [v for v in vals if v is not None]
    return f"{sorted(vals)[len(vals) // 2]:.1f}" if vals else "-"


def sweep_tables() -> None:
    for machine, sub in SWEEPS.items():
        d = RUNS / sub
        manifests = sorted(d.glob("sweep_*.json"))
        if not manifests:
            print(f"### {machine}: no sweep in {d}\n")
            continue
        m = json.loads(manifests[0].read_text())
        cfg = SweepConfig.from_dict(m["config"])
        first = RunReport.load(report_path(d, cfg.variants[0], 1))
        p = first.platform
        power_key = "power_board_w" if any("power_board_w" in r.tier(cfg.tiers_hz[0]).telemetry
                                           for r in [first]) else "power_gpu_w"
        print(f"### {machine}: `{sub}/`, sweep `{m['sweep']}`, {'complete' if m['complete'] else 'INCOMPLETE'}")
        print(f"onnxruntime {p.get('onnxruntime')}, TensorRT {p.get('tensorrt')}, cuDNN {p.get('cudnn')}, "
              f"CUDA {p.get('cuda_runtime')}, power mode {p.get('nvpmodel') or 'n/a'}, "
              f"stopped: {m['config'].get('stopped') or 'nothing'}\n")
        print(f"| variant | Hz | response p50 (median, range) | response p95 (median, range) | response p99 | "
              f"service p95 | misses / scheduled | {power_key} p50 |")
        print("|---|---|---|---|---|---|---|---|")
        for v in cfg.variants:
            reps = [RunReport.load(report_path(d, v, r)) for r in range(1, cfg.repeats + 1)]
            for hz in cfg.tiers_hz:
                s50 = repeat_stat(reps, hz, "response_p50_ms")
                s95 = repeat_stat(reps, hz, "response_p95_ms")
                s99 = repeat_stat(reps, hz, "response_p99_ms")
                svc = repeat_stat(reps, hz, "p95_ms")
                misses = sum(r.tier(hz).deadline_misses for r in reps)
                sched = sum(r.tier(hz).n_scheduled for r in reps)
                print(f"| {v.label} | {hz:g} | {s50.median:.3f} ({s50.lo:.3f}–{s50.hi:.3f}) | "
                      f"{s95.median:.3f} ({s95.lo:.3f}–{s95.hi:.3f}) | {s99.median:.3f} | {svc.median:.3f} | "
                      f"{misses} / {sched} | {tel(reps, hz, power_key)} |")
        print()


def profile_tables() -> None:
    for path in PROFILES:
        d = json.loads(path.read_text())
        p = d["platform"]
        print(f"### `{path.relative_to(HERE)}`: {d['frames']['n']} frames of "
              f"{d['frames']['decoded_shape'][1]}x{d['frames']['decoded_shape'][0]} from `{d['frames']['dir']}`, "
              f"{p.get('provider_active')} {p.get('precision')}, onnxruntime {p.get('onnxruntime')}, "
              f"TensorRT {p.get('tensorrt')}\n")
        print("| stage | p50 ms | p95 ms | share of p50 total |")
        print("|---|---|---|---|")
        for s in STAGES:
            st = d["stages"][s]
            print(f"| {s} | {st['p50_ms']:.3f} | {st['p95_ms']:.3f} | {d['stage_share_of_total_p50'][s]:.0%} |")
        t = d["total_per_frame"]
        print(f"| **total** | **{t['p50_ms']:.3f}** | **{t['p95_ms']:.3f}** | |")
        print(f"\n`session.run` with NumPy feeds (copies included, what `b2f run` times): "
              f"p50 {d['session_run_numpy']['p50_ms']:.3f} ms. Copies h2d+d2h p50: {d['copies_h2d_plus_d2h_p50_ms']:.3f} ms. "
              f"Detections per frame: mean {d['detections_per_frame']['mean']:.1f}, "
              f"frames with any: {d['detections_per_frame']['frames_with_any']}/{d['frames']['n']}. "
              f"Mean JPEG size {d['frames']['jpeg_bytes_mean'] / 1024:.0f} KB.\n")


if __name__ == "__main__":
    sweep_tables()
    profile_tables()
