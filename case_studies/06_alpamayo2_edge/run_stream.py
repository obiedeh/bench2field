"""Measure Alpamayo 2 Edge on a 10 Hz decision stream, with and without vision reuse.

In a car the model decides every 0.1 s. Each decision reads 22 camera images
(front cameras at -2, -1, -0.5, -0.1 and 0 s, the others at -1, -0.1 and 0 s),
and every one of those images was already the newest frame of an earlier
decision. Upstream encodes all 22 every time. This script measures what reusing
the vision encoding of frames already seen saves, without changing the model:

- The vision tower of Cosmos3-Edge attends within each image only (packed
  attention with per-image ``cu_seqlens``) and positions are per image, so one
  image's encoding does not depend on the others in the call. The wrapper below
  caches each image's projected features by (camera, capture timestamp) and runs
  the vision tower only on the images it has not seen. Every cache hit is checked
  against a fingerprint of the image's pixels, and one decision per run is also
  computed without the cache and compared, so the record shows the outputs match.
- Stream: for each clip, decisions at t0 = 3.1 s, 3.2 s, ... 6.0 s (30 steps).
  The first 20 fill the cache as two seconds of driving would; the last 10
  (t0 = 5.1 to 6.0 s) are timed. The same steps are run without reuse for the
  baseline. The navigation text is the clip's fixed instruction from nav_clips.

    PYTHONPATH=<bench2field>/src .venv/bin/python run_stream.py --board bench-5090 --variant eager \
        --model <model dir> --nav <nav_futures.json> --sample-cache <dir>
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import run_decisions as rd

HERE = Path(__file__).resolve().parent
STREAM_CLIPS = ("0347d9f9-1493-4954-865d-1d8464e28501",  # turn right, nominal path
                "174407b0-b040-475b-8c1e-512b0bab4675",  # straight, writes reasoning
                "195f1fc6-2a7b-4833-9b7f-753530fda044")  # turn left
STEPS, WARM = 30, 20
T_START_US = 3_100_000


class VisionCache:
    """Cache projected image features by (camera, timestamp) around get_image_features."""

    def __init__(self, vlm_model: Any) -> None:
        self.inner = vlm_model
        self.orig = vlm_model.get_image_features
        self.store: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.keys: list[tuple[int, int]] | None = None
        self.enabled = False
        self.hits = self.misses = self.fingerprint_mismatch = 0

        def get_image_features(pixel_values: torch.Tensor, image_grid_thw: torch.Tensor | None = None, **kw: Any) -> Any:
            if not self.enabled or self.keys is None:
                return self.orig(pixel_values, image_grid_thw, **kw)
            sizes = image_grid_thw.prod(-1).tolist()
            chunks = torch.split(pixel_values, sizes)
            fps = [torch.stack([c.float().sum(), c.float().square().sum(), c.float()[:: max(1, c.shape[0] // 7)].sum()]) for c in chunks]
            out: list[torch.Tensor | None] = [None] * len(chunks)
            missing = []
            for i, key in enumerate(self.keys):
                hit = self.store.get(key)
                if hit is not None and torch.allclose(hit[1], fps[i]):
                    out[i] = hit[0]
                    self.hits += 1
                else:
                    if hit is not None:
                        self.fingerprint_mismatch += 1
                    missing.append(i)
                    self.misses += 1
            if missing:
                res = self.orig(torch.cat([chunks[i] for i in missing]), image_grid_thw[missing], **kw).pooler_output
                for i, feat in zip(missing, res):
                    out[i] = feat
                    self.store[self.keys[i]] = (feat, fps[i])
            return types.SimpleNamespace(pooler_output=tuple(out))

        vlm_model.get_image_features = get_image_features


def image_keys(sample: dict) -> list[tuple[int, int]]:
    """(camera, capture time) per image, in the order the prompt places them (camera, then time)."""
    cams = sample["camera_indices"].reshape(-1).tolist()
    ts = sample["absolute_timestamps"].reshape(-1).tolist()
    return sorted(zip(cams, ts))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--board", required=True, choices=rd.BOARDS)
    ap.add_argument("--variant", required=True, choices=rd.VARIANTS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--nav", type=Path, required=True)
    ap.add_argument("--sample-cache", type=Path, required=True)
    ap.add_argument("--diffusion-steps", type=int, default=rd.SETTINGS["diffusion_steps"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--fp8", action="store_true", help="torchao FP8 on the language model (run_decisions.quantize_fp8)")
    args = ap.parse_args(argv)
    rd.DIFFUSION_STEPS[0] = args.diffusion_steps

    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge.load_physical_aiavdataset import load_physical_aiavdataset

    nav = {s["clip_id"]: s["nav_text"] for s in json.loads(args.nav.read_text())["samples"]}
    model, _ = load_model(args.model, device="cuda", dtype="bfloat16", attn_implementation="sdpa")
    rd.install_prep_mode("cached")
    fp8 = rd.quantize_fp8(model) if args.fp8 else None

    streams = {}
    for clip in STREAM_CLIPS:
        steps = []
        for k in range(STEPS):
            t0 = T_START_US + 100_000 * k
            f = args.sample_cache / f"{clip}_{t0}.pt"
            if f.exists():
                smp = torch.load(f, weights_only=False)
            else:
                smp = load_physical_aiavdataset(clip, t0_us=t0, nav_text=nav[clip])
                f.parent.mkdir(parents=True, exist_ok=True)
                torch.save(smp, f)
            steps.append((t0, image_keys(smp), rd.prepare(smp, model, "cached")))
        streams[clip] = steps
        print(f"[stream] {clip[:8]}: {len(steps)} steps ready", flush=True)

    if args.variant in ("tuned", "compile"):
        from alpamayo2_edge.accel import AccelConfig, enable_acceleration

        enable_acceleration(model, AccelConfig(mode=args.variant))
    cache = VisionCache(model.vlm.model)

    def timed(prepared: dict) -> tuple[float, Any]:
        torch.cuda.synchronize()
        t = time.perf_counter()
        xyz, _r, _e = rd.predict(model, prepared)
        torch.cuda.synchronize()
        return time.perf_counter() - t, xyz

    for steps in streams.values():  # warm-up (and graph capture) for every shape, cache off
        timed(steps[0][2])

    results: dict[str, Any] = {}
    check = None
    for mode in ("baseline", "reuse"):
        cache.enabled = mode == "reuse"
        cache.store.clear()
        lat = []
        for clip, steps in streams.items():
            for k, (t0, keys, prepared) in enumerate(steps):
                cache.keys = keys
                dt, xyz = timed(prepared)
                if k >= WARM:
                    lat.append({"clip_id": clip, "t0_us": t0, "latency_s": round(dt, 4)})
                if mode == "reuse" and check is None and k == WARM:
                    cache.enabled = False
                    _, ref = timed(prepared)
                    cache.enabled = True
                    check = {"clip_id": clip, "t0_us": t0,
                             "waypoints_max_abs_diff_m": round(float((xyz.float() - ref.float()).abs().max()), 5)}
        vals = sorted(x["latency_s"] for x in lat)
        results[mode] = {"n": len(vals), "latency_median_s": round(statistics.median(vals), 4),
                         "latency_p95_s": round(vals[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))], 4),
                         "decisions": lat}
        print(f"[{mode}] {results[mode]['latency_median_s']} s median, {results[mode]['latency_p95_s']} s p95", flush=True)

    record = {
        "schema": "bench2field.case_study_06.stream/1", "label": "lab result",
        "board": args.board, "variant": args.variant, "diffusion_steps": args.diffusion_steps, "fp8": fp8,
        "model": {"id": "nvidia/Alpamayo2-Edge", "revision": rd.MODEL_REVISION},
        "stream": {"clips": list(STREAM_CLIPS), "steps": STEPS, "warm_steps": WARM, "t_start_us": T_START_US, "rate_hz": 10},
        "cache": {"hits": cache.hits, "misses": cache.misses, "fingerprint_mismatch": cache.fingerprint_mismatch},
        "reuse_vs_full_check": check,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": results,
    }
    name = f"stream_{args.board}_{args.variant}" + (f"_{args.tag}" if args.tag else "")
    (HERE / "runs" / f"{name}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "results"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
