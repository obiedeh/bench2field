"""Verify and time the fused preprocessing kernel against PyTorch, on one board.

Phase 3b of case study 02. One run does, on real frames from the first clip of
``clips.yaml`` (or random frames with ``--synthetic``, output to runs/scratch):

1. Correctness. Maximum absolute error of the kernel against the PyTorch
   reference on the GPU (``preprocess_ext.reference_torch``) and against the
   real Hugging Face processor on the CPU (``reference_processor``), and of
   the two references against each other. An error of 2/255 = 0.0078 is one
   uint8 step after normalization; anything at that level is a rounding tie.
2. Timing. Median and p95 wall time per batch of 16 frames (4 cameras x 4
   frames, the model's input for one decision) for four paths:
   ``kernel``            the CUDA kernel, frames already on the GPU
   ``kernel+upload``     the kernel plus the host to device copy of the uint8
                         frames, which the processor paths do not need
   ``torch_gpu``         the reference recipe in PyTorch ops on the GPU
   ``processor_cpu``     the Hugging Face processor on the CPU, as Alpamayo
                         calls it today
   GPU paths are timed with CUDA events after synchronisation; CPU paths with
   a monotonic clock. ``--reps`` timed repetitions after ``--warmup`` untimed.

Writes ``runs/preprocess_<board>.json`` with the errors, the timings, the
environment record and the provenance block, and prints a Markdown table.
Measured values only; nothing in the output is estimated.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch

import preprocess_ext as pe
import run_decisions as rd

HERE = Path(__file__).resolve().parent
BOARDS = ("bench-5090", "bench-thor")


def load_frames(clip_id: str, t0_us: int, synthetic: bool) -> torch.Tensor:
    """16 uint8 frames [16, 3, 1080, 1920] on the CPU."""
    if synthetic:
        g = torch.Generator().manual_seed(0)
        return torch.randint(0, 256, (16, 3, pe.IN_H, pe.IN_W), generator=g, dtype=torch.uint8)
    data = rd.dataset_sample(clip_id, t0_us, 4)
    return data["image_frames"].flatten(0, 1).contiguous()


def time_gpu(fn: Callable[[], Any], reps: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / 1000.0)
    return out


def time_cpu(fn: Callable[[], Any], reps: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return out


def stats(xs: list[float]) -> dict[str, float]:
    xs = sorted(xs)
    return {
        "median_s": round(statistics.median(xs), 6),
        "p95_s": round(xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))], 6),
        "min_s": round(xs[0], 6),
        "max_s": round(xs[-1], 6),
        "n": len(xs),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--board", required=True, choices=BOARDS)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--clips", type=Path, default=HERE / "clips.yaml")
    ap.add_argument("--synthetic", action="store_true", help="random frames; output to runs/scratch/")
    ap.add_argument("--skip-processor-cpu", action="store_true", help="skip the CPU processor timing (slow on the Thor)")
    args = ap.parse_args(argv)

    import yaml

    cfg = yaml.safe_load(args.clips.read_text())
    clip_id, t0_us = cfg["clips"][0], int(cfg["t0_us"])
    frames_cpu = load_frames(clip_id, t0_us, args.synthetic)
    frames = frames_cpu.cuda()
    torch.cuda.synchronize()

    t = time.perf_counter()
    pe.load_kernel()
    build_s = time.perf_counter() - t
    tabs = pe.tables(frames.device)

    # Correctness.
    out = pe.preprocess(frames, tabs)
    ref_gpu = pe.reference_torch(frames)
    ref_proc = pe.reference_processor(frames_cpu)
    torch.cuda.synchronize()
    errors = {
        "kernel_vs_torch_gpu": pe.max_abs_error(out, ref_gpu),
        "kernel_vs_processor_cpu": pe.max_abs_error(out, ref_proc),
        "torch_gpu_vs_processor_cpu": pe.max_abs_error(ref_gpu, ref_proc),
        "fraction_kernel_vs_processor_above_1e-3": float(((out.cpu() - ref_proc).abs() > 1e-3).float().mean()),
        "one_uint8_step_after_normalize": 2.0 / 255.0,
    }
    print("[verify] " + ", ".join(f"{k} {v:.6f}" for k, v in errors.items()))

    # Timing.
    timings: dict[str, dict[str, float]] = {}
    timings["kernel"] = stats(time_gpu(lambda: pe.preprocess(frames, tabs), args.reps, args.warmup))
    pinned = frames_cpu.pin_memory()
    timings["kernel+upload"] = stats(time_gpu(lambda: pe.preprocess(pinned.cuda(non_blocking=False), tabs), args.reps, args.warmup))
    timings["torch_gpu"] = stats(time_gpu(lambda: pe.reference_torch(frames), args.reps, args.warmup))
    if not args.skip_processor_cpu:
        reps_cpu = max(5, args.reps // 5)
        timings["processor_cpu"] = stats(time_cpu(lambda: pe.reference_processor(frames_cpu), reps_cpu, 1))
    for k, v in timings.items():
        print(f"[time] {k:16s} median {v['median_s']*1000:8.2f} ms  p95 {v['p95_s']*1000:8.2f} ms  (n={v['n']})")

    env_path = HERE / "runs" / f"env_{args.board}.json"
    rec = {
        "schema": "bench2field.alpamayo.preprocess/1",
        "label": "smoke test, synthetic input, not a result" if args.synthetic else "measured",
        "input": "synthetic" if args.synthetic else "dataset",
        "board": args.board,
        "clip_id": None if args.synthetic else clip_id,
        "t0_us": t0_us,
        "frames": list(frames.shape),
        "output": list(out.shape),
        "kernel": {"source": "kernels/preprocess.cu", "build_s": round(build_s, 2),
                   "cuda_home": __import__("os").environ.get("CUDA_HOME"), "arch_list": __import__("os").environ.get("TORCH_CUDA_ARCH_LIST"),
                   "nvcc_release": nvcc_release(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                   "device": torch.cuda.get_device_name(0), "compute_capability": list(torch.cuda.get_device_capability(0))},
        "errors": errors,
        "timings_per_16_frames": timings,
        "protocol": {"reps": args.reps, "warmup": args.warmup, "gpu_timing": "cuda events, synchronised", "cpu_timing": "perf_counter"},
        "provenance": rd.provenance(env_path),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out_path = HERE / "runs" / ("scratch" if args.synthetic else "") / f"preprocess_{args.board}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"[bench] wrote {out_path.relative_to(HERE)}")
    print(table([rec]))
    return 0


def nvcc_release() -> str | None:
    import os
    import subprocess

    home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    try:
        txt = subprocess.run([f"{home}/bin/nvcc", "--version"], capture_output=True, text=True, check=True).stdout
        return txt.split("release ")[1].split(",")[0]
    except Exception:
        return None


def table(recs: list[dict[str, Any]]) -> str:
    """Markdown table of the timings for the README (one column per board)."""
    cols = [r["board"] for r in recs]
    s = "| Path, 16 frames | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    for key in ("kernel", "kernel+upload", "torch_gpu", "processor_cpu"):
        cells = []
        for r in recs:
            t = r["timings_per_16_frames"].get(key)
            cells.append(f"{t['median_s']*1000:.2f} ms median, {t['p95_s']*1000:.2f} ms p95" if t else "not run")
        s += f"| {key} | " + " | ".join(cells) + " |\n"
    s += "| Max abs error, kernel vs processor on the CPU | " + " | ".join(f"{r['errors']['kernel_vs_processor_cpu']:.6f}" for r in recs) + " |\n"
    s += "| Max abs error, kernel vs torch GPU | " + " | ".join(f"{r['errors']['kernel_vs_torch_gpu']:.6f}" for r in recs) + " |\n"
    return s


if __name__ == "__main__":
    raise SystemExit(main())
