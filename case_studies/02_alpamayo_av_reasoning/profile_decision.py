"""Profile one Alpamayo 1 decision: where the time goes, next to the GPU's telemetry.

Runs the same decision as run_decisions.py (same clip, inputs, sampling
parameters and seed) under ``torch.profiler`` with CPU and CUDA activities,
and labels the three stages of the upstream call with ``record_function`` from
the outside, without editing upstream code:

    vision        the Qwen3-VL vision tower (model.vlm.model.visual forward)
    rollout       model.vlm.generate: prefill plus autoregressive reasoning
    trajectory    model.diffusion.sample: the diffusion expert's denoising steps

Everything else inside the call is glue (token fusing, mask building, action
space conversion). The profiled decision follows one unprofiled warm-up on the
same inputs, so it is a steady-state decision, not a first call.

Writes one JSON under runs/ with:

    stages        CUDA time and CPU time per labelled stage, and the wall time of
                  the decision measured outside the profiler
    top_kernels   the 25 device-side kernels with the most device time (name,
                  calls, total and mean device time, share of the kernel table)
    top_ops       the 25 host-side aten operators with the most device time, the
                  sum of the kernels each one launched (share of the op table)
    decode        per-token timestamps of the rollout, so the decode rate can be
                  plotted over the trace
    telemetry     NVML samples (power, utilisation, clocks, temperature, memory)
                  on the 5090, tegrastats samples (VIN board power, GPU load,
                  temperatures) on the Thor, at 0.25 s over the warm-up and the
                  profiled decision, with the window of the profiled decision marked
    memory        peak allocated and reserved GPU memory during the decision

The Chrome trace (tens of MB) goes to runs/scratch/, which git ignores. The
profiler adds overhead, so the wall time here is not the latency result; the
latency result is run_decisions.py's. This file explains the split, not the
total.

    env -u PYTHONPATH .venv/bin/python profile_decision.py --board bench-5090 --variant sdpa
    env -u PYTHONPATH .venv/bin/python profile_decision.py --board bench-5090 --variant sdpa --synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.profiler import ProfilerActivity, profile, record_function

import run_decisions as rd

HERE = Path(__file__).resolve().parent


class StageLabels:
    """record_function wrappers around the vision tower, generate and diffusion.sample."""

    def __init__(self, model: Any) -> None:
        self.visual = model.vlm.model.visual
        self._visual_forward = self.visual.forward
        self._generate = model.vlm.generate
        self._sample = model.diffusion.sample
        self.token_times: list[float] = []
        self.generated_tokens = 0
        self.rollout_t0 = None

        def visual_forward(*a: Any, **k: Any) -> Any:
            with record_function("stage:vision"):
                return self._visual_forward(*a, **k)

        def generate(*a: Any, **k: Any) -> Any:
            clock = rd.TokenClock()
            if k.get("stopping_criteria") is not None:
                k["stopping_criteria"].append(clock)
            n_in = k["input_ids"].shape[1]
            torch.cuda.synchronize()
            self.rollout_t0 = time.perf_counter()
            with record_function("stage:rollout"):
                out = self._generate(*a, **k)
            torch.cuda.synchronize()
            self.token_times = [t - self.rollout_t0 for t in clock.times]
            self.generated_tokens = int(out.sequences.shape[1] - n_in)
            return out

        def sample(*a: Any, **k: Any) -> Any:
            with record_function("stage:trajectory"):
                return self._sample(*a, **k)

        self.visual.forward = visual_forward
        model.vlm.generate = generate
        model.diffusion.sample = sample


def run_once(model: Any, processor: Any, data: dict[str, Any]) -> tuple[float, Any]:
    inputs = rd.build_inputs(model, processor, data)
    torch.cuda.manual_seed_all(rd.SEED)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        with record_function("stage:decision"):
            out = model.sample_trajectories_from_data_with_vlm_rollout(data=inputs, return_extra=True, **rd.SAMPLING)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def table(prof: Any, key: str, n: int) -> list[dict[str, Any]]:
    """Top-n rows by device time. ``kernel`` rows are device-side events (the CUDA
    kernels themselves); ``op`` rows are host-side aten operators, whose device
    time is the sum of the kernels they launched. The stage labels from
    ``record_function`` are reported in ``stages`` and excluded here, so a
    share is a share of one table, not of a total that counts the same
    microseconds twice."""
    events = []
    for e in prof.key_averages():
        if e.key.startswith("stage:"):
            continue
        is_kernel = str(getattr(e, "device_type", "")).endswith("CUDA")
        if (key == "kernel") != is_kernel:
            continue
        if key == "op" and not e.key.startswith("aten::"):
            continue
        if not getattr(e, "device_time_total", 0):
            continue
        events.append(e)
    total_dev = sum(getattr(e, "device_time_total", 0) for e in events) or 1.0
    rows = []
    for e in sorted(events, key=lambda e: getattr(e, "device_time_total", 0), reverse=True)[:n]:
        rows.append({
            "name": e.key[:160],
            "calls": int(e.count),
            "device_time_ms": round(getattr(e, "device_time_total", 0) / 1000, 3),
            "device_time_mean_us": round(getattr(e, "device_time_total", 0) / max(e.count, 1), 1),
            "cpu_time_ms": round(e.cpu_time_total / 1000, 3),
            "share_of_table_device_time_pct": round(100 * getattr(e, "device_time_total", 0) / total_dev, 2),
        })
    return rows


def stage_totals(prof: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for e in prof.key_averages():
        if e.key.startswith("stage:"):
            out[e.key.split(":", 1)[1]] = {
                "calls": int(e.count),
                "cpu_time_ms": round(e.cpu_time_total / 1000, 2),
                "device_time_ms": round(getattr(e, "device_time_total", 0) / 1000, 2),
            }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--board", required=True, choices=rd.BOARDS)
    ap.add_argument("--variant", required=True, choices=sorted(rd.VARIANTS))
    ap.add_argument("--clips", type=Path, default=HERE / "clips.yaml")
    ap.add_argument("--clip-index", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.clips.read_text())
    clip = cfg["clips"][args.clip_index]
    label = "synthetic" if args.synthetic else "dataset"
    out = args.out or (HERE / "runs" / ("scratch" if args.synthetic else "") / f"profile_{args.board}_{args.variant}.json")
    trace_dir = HERE / "runs" / "scratch"
    out.parent.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    sampler = rd.start_sampler(args.board, interval_s=0.25)
    h = rd.nvml_handle()

    model, processor, load = rd.load_model(rd.VARIANTS[args.variant], rd.EXPERT_ATTN[args.variant])
    labels = StageLabels(model)
    data = (rd.synthetic_sample if args.synthetic else rd.dataset_sample)(clip, int(cfg["t0_us"]), int(cfg["num_frames"]))

    warm_s, _ = run_once(model, processor, data)
    print(f"[profile] warm-up {warm_s:.2f} s", flush=True)

    torch.cuda.reset_peak_memory_stats()
    m0 = time.monotonic()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, profile_memory=False) as prof:
        wall_s, result = run_once(model, processor, data)
    m1 = time.monotonic()
    print(f"[profile] profiled decision {wall_s:.2f} s, {labels.generated_tokens} tokens", flush=True)

    trace_path = trace_dir / f"trace_{args.board}_{args.variant}.json"
    prof.export_chrome_trace(str(trace_path))
    telemetry = sampler.stop() if sampler else None
    samples = sampler.samples if sampler else []

    record = {
        "schema": "bench2field.case_study_02.profile/1",
        "label": "lab result" if not args.synthetic else "smoke test, synthetic input, not a result",
        "input": label,
        "board": args.board,
        "variant": args.variant,
        "attn_implementation": rd.VARIANTS[args.variant],
        "attn_implementation_expert": rd.EXPERT_ATTN[args.variant],
        "model_id": rd.MODEL_ID,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clip_id": clip,
        "t0_us": cfg["t0_us"],
        "protocol": {**rd.SAMPLING, "seed": rd.SEED},
        "load": load,
        "warmup_wall_s": round(warm_s, 3),
        "profiled_wall_s": round(wall_s, 3),
        "generated_tokens": labels.generated_tokens,
        "stages": stage_totals(prof),
        "top_kernels": table(prof, "kernel", 25),
        "top_ops": table(prof, "op", 25),
        "decode": {"token_times_s": [round(t, 4) for t in labels.token_times]},
        "memory": rd.gpu_memory(h),
        "window": {"start": m0, "end": m1},
        "telemetry": {"describe": telemetry, "samples": samples},
        "chrome_trace": str(trace_path.relative_to(HERE)) + " (not committed)",
        "provenance": rd.provenance(HERE / "runs" / f"env_{args.board}.json"),
    }
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"[profile] wrote {out}")
    for k, v in record["stages"].items():
        print(f"[profile] {k:11s} device {v['device_time_ms']:9.1f} ms  cpu {v['cpu_time_ms']:9.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
