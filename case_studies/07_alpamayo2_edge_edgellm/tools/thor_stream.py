#!/usr/bin/env python3
"""Repeated decisions with the Alpamayo2-Edge Edge-LLM engines on the Jetson AGX Thor.

NVIDIA's thor/run.py times one decision per process. This loads the engines once and runs
N decisions per clip back to back, the way Embedl's model card describes its measurement
(one clip, 180 decisions, inputs already on the GPU), and samples board power with
tegrastats so energy per decision can be reported beside latency.

Rolling mode (the recipe's --rolling): before each timed decision the newest frame of each
camera is dropped from the vision cache, so the ViT encodes six frames per decision, as a
10 Hz stream would present them. Cold mode clears the cache every decision (22 frames).

Usage (after `source $WORK/env.sh`, with RECIPE_THOR pointing at recipes/.../thor):
  python thor_stream.py ENGINES_DIR OUT.json CLIP_DIR [CLIP_DIR ...] --repeats 20 --candidates 6
         --steps 5 [--cold] [--warmup 1] [--seed 0] [--power-interval-ms 100]
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, os.environ["RECIPE_THOR"])
from run import Runtime, load_images  # noqa: E402
from fast_runtime import FastRuntime  # noqa: E402

VIN = re.compile(r"\bVIN (\d+)mW")
GPU = re.compile(r"\bVDD_GPU (\d+)mW")


class Power:
    """tegrastats sampler: (monotonic time, board input mW, GPU rail mW)."""

    def __init__(self, interval_ms: int):
        self.samples = []
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            vin, gpu = VIN.search(line), GPU.search(line)
            if vin:
                self.samples.append(
                    (time.monotonic(), int(vin.group(1)), int(gpu.group(1)) if gpu else None)
                )

    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=5)
        return list(self.samples)

    @staticmethod
    def window(samples, t0, t1):
        """Mean, peak board power (W) and energy (J) between t0 and t1 by trapezoid on the samples."""
        pts = [(t, p / 1000.0) for t, p, _ in samples if t0 <= t <= t1]
        if len(pts) < 2:
            return None
        energy = sum(
            (pts[i + 1][0] - pts[i][0]) * (pts[i + 1][1] + pts[i][1]) / 2 for i in range(len(pts) - 1)
        )
        span = pts[-1][0] - pts[0][0]
        return {
            "mean_w": round(energy / span, 2),
            "peak_w": round(max(p for _, p in pts), 2),
            "samples": len(pts),
            "span_s": round(span, 3),
            "energy_j": round(energy, 2),
        }


def pct(values, q):
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("clips", type=Path, nargs="+")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--candidates", type=int, default=6)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--cold", action="store_true", help="encode all 22 frames every decision")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--power-interval-ms", type=int, default=100)
    ap.add_argument("--idle-s", type=float, default=5.0)
    ap.add_argument("--fast", action="store_true", help="FastRuntime: token stays on the GPU, end checked per chunk")
    ap.add_argument("--chunk", type=int, default=8)
    args = ap.parse_args()

    power = Power(args.power_interval_ms)
    time.sleep(args.idle_s)
    idle_end = time.monotonic()

    t0 = time.time()
    rt = FastRuntime(args.engines.resolve(), args.candidates, args.chunk) if args.fast else Runtime(args.engines.resolve(), args.candidates)
    load_s = time.time() - t0
    print(f"engines loaded in {load_s:.1f} s", flush=True)

    decisions, per_clip = [], {}
    block_start = None
    for clip in args.clips:
        clip = clip.resolve()
        prompt = json.load(open(clip / "prompt.json"))
        images = load_images(clip, prompt)
        current = {
            f["image"] for f in json.load(open(clip / "clip.json"))["frames"] if f["t"] == 0.0
        }
        noise = torch.randn(
            args.candidates, rt.action.tokens, 2, generator=torch.Generator().manual_seed(args.seed)
        )
        for _ in range(args.warmup):
            rt.vision.cache.clear()
            rt.run(prompt, images, steps=args.steps, max_tokens=args.max_tokens, noise=noise)
        if not args.cold:
            rt.vision.cache.clear()
            with torch.cuda.stream(rt.stream):
                rt.vision.encode({k: v for k, v in images.items() if k not in current})
            torch.cuda.synchronize()
        times = []
        if block_start is None:
            block_start = time.monotonic()
        for i in range(args.repeats):
            if args.cold:
                rt.vision.cache.clear()
            else:
                rt.vision.forget(current)
            encoded = sum(1 for k in images if k not in rt.vision.cache)
            ts = time.monotonic()
            r = rt.run(prompt, images, steps=args.steps, max_tokens=args.max_tokens, noise=noise)
            te = time.monotonic()
            times.append(r["total_ms"])
            decisions.append(
                {
                    "clip": clip.name,
                    "i": i,
                    "t_start": ts,
                    "t_end": te,
                    "total_ms": round(r["total_ms"], 2),
                    "stages_ms": {k: round(v, 2) for k, v in r["ms"].items()},
                    "frames_encoded": encoded,
                    "generated_tokens": len(r["generated"]),
                    "nominal": bool(r["text"] and "<NOMINAL>" in r["text"]),
                    "text": r["text"] if i == 0 else None,
                }
            )
        per_clip[clip.name] = {
            "median_ms": round(statistics.median(times), 1),
            "p95_ms": round(pct(times, 0.95), 1),
            "generated_tokens": decisions[-1]["generated_tokens"],
            "nominal": decisions[-1]["nominal"],
            "text": decisions[-len(times)]["text"],
        }
        print(
            f"{clip.name}: median {per_clip[clip.name]['median_ms']} ms, p95 {per_clip[clip.name]['p95_ms']} ms, "
            f"{per_clip[clip.name]['generated_tokens']} tokens, nominal={per_clip[clip.name]['nominal']}",
            flush=True,
        )
    block_end = time.monotonic()
    samples = power.stop()

    totals = [d["total_ms"] for d in decisions]
    timed = Power.window(samples, block_start, block_end)
    idle = Power.window(samples, samples[0][0] if samples else 0, idle_end)
    stage_med = {
        k: round(statistics.median(d["stages_ms"][k] for d in decisions), 1)
        for k in decisions[0]["stages_ms"]
    }
    summary = {
        "platform": "Jetson AGX Thor",
        "label": "bench-thor",
        "engines": str(args.engines.resolve()),
        "mode": "cold" if args.cold else "rolling",
        "runtime": f"fast(chunk={args.chunk})" if args.fast else "recipe",
        "candidates": args.candidates,
        "steps": args.steps,
        "clips": [c.name for c in args.clips],
        "repeats_per_clip": args.repeats,
        "decisions": len(totals),
        "load_s": round(load_s, 1),
        "median_ms": round(statistics.median(totals), 1),
        "mean_ms": round(statistics.mean(totals), 1),
        "p90_ms": round(pct(totals, 0.90), 1),
        "p95_ms": round(pct(totals, 0.95), 1),
        "max_ms": round(max(totals), 1),
        "decisions_per_s_back_to_back": round(len(totals) / (block_end - block_start), 2),
        "stage_median_ms": stage_med,
        "nominal_share": round(sum(d["nominal"] for d in decisions) / len(decisions), 2),
        "power_timed": timed,
        "power_idle": idle,
        "energy_per_decision_j": round(timed["energy_j"] / len(totals), 2) if timed else None,
        "per_clip": per_clip,
        "tensorrt": __import__("tensorrt").__version__,
        "torch": torch.__version__,
        "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "decisions": decisions}, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
