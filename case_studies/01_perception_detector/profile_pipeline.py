"""End-to-end pipeline profile for case study 01, phase 1.

Times every stage of one camera frame's journey through the detector, the
way the rover's pipeline would run it, for a set of JPEG frames:

    decode      JPEG bytes -> HxWx3 uint8 BGR            cv2.imdecode
    preprocess  letterbox to 640x640, HWC->CHW, float32  cv2.resize + NumPy, the unfused baseline
    h2d         input copy to the GPU                    onnxruntime OrtValue (IO binding)
    inference   the model, input and output on device    session.run_with_iobinding
    d2h         output copy back                         OrtValue.numpy()
    postprocess objectness x class score, NMS            NumPy (YOLOX's multiclass_nms, copied below)

Each stage is wrapped in an NVTX range, so `nsys profile` on this script
shows them on the timeline. A second pass times `session.run()` with NumPy
feeds, which is what b2f run measures: it includes both copies.

Writes one JSON with per-stage statistics and the provenance of everything
it used (model, frame set, versions, what else the machine was doing).

    env -u PYTHONPATH .venv/bin/python case_studies/01_perception_detector/profile_pipeline.py \\
        --frames data/rover_frames_720p --out case_studies/01_perception_detector/runs/profile_5090_rover720p.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bench2field import background, provenance  # noqa: E402
from bench2field.backends.onnxruntime import OnnxRuntimeBackend, OrtOptions  # noqa: E402
from bench2field.schema import describe_platform  # noqa: E402
from bench2field.telemetry import auto_sampler  # noqa: E402

try:
    import nvtx

    def nvtx_range(name: str):
        return nvtx.annotate(name)
except ImportError:  # the profile still runs; nsys just shows no ranges
    def nvtx_range(name: str):
        return contextlib.nullcontext()

STAGES = ("decode", "preprocess", "h2d", "inference", "d2h", "postprocess")
INPUT_SIZE = 640
PAD = 114
CONF_THRESHOLD, NMS_THRESHOLD = 0.3, 0.45


def preprocess(img: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, float]:
    """YOLOX's preproc as separate library calls: letterbox resize into a
    grey canvas, HWC to CHW, float32, batch axis. No normalisation (YOLOX
    takes 0..255). This is the baseline the phase 3 fused kernel replaces."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(h * r), int(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), PAD, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    x = canvas.transpose(2, 0, 1)[None].astype(np.float32)
    return np.ascontiguousarray(x), r


# NMS as YOLOX's demo code does it, copied from yolox/utils/demo_utils.py
# (Megvii, Apache-2.0) so the profile runs on boards without the yolox
# package or PyTorch. Same code as the 5090 profiles, which imported it.
def nms(boxes: np.ndarray, scores: np.ndarray, nms_thr: float) -> list[int]:
    """Single class NMS implemented in Numpy."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep


def multiclass_nms(boxes: np.ndarray, scores: np.ndarray, nms_thr: float, score_thr: float) -> np.ndarray | None:
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    for cls_ind in range(scores.shape[1]):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        keep = nms(valid_boxes, valid_scores, nms_thr)
        if len(keep) > 0:
            cls_inds = np.ones((len(keep), 1)) * cls_ind
            final_dets.append(np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1))
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def postprocess(out: np.ndarray, ratio: float) -> np.ndarray | None:
    """Decoded YOLOX output (1 x 8400 x 85) -> detections after NMS."""
    pred = out[0]
    boxes = pred[:, :4]
    scores = pred[:, 4:5] * pred[:, 5:]
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    xyxy /= ratio
    return multiclass_nms(xyxy, scores, nms_thr=NMS_THRESHOLD, score_thr=CONF_THRESHOLD)


def stats(samples_ms: list[float]) -> dict[str, float]:
    a = np.asarray(samples_ms)
    p50, p95, p99 = np.percentile(a, [50, 95, 99])
    return {"n": int(a.size), "p50_ms": float(p50), "p95_ms": float(p95), "p99_ms": float(p99),
            "mean_ms": float(a.mean()), "max_ms": float(a.max())}


def load_frames(frames_dir: Path) -> tuple[list[bytes], dict]:
    manifest = json.loads((frames_dir / "manifest.json").read_text())
    frames = [(frames_dir / f["file"]).read_bytes() for f in manifest["files"]]
    for f, data in zip(manifest["files"], frames):
        if hashlib.sha256(data).hexdigest() != f["sha256"]:
            raise SystemExit(f"{f['file']} does not match its manifest hash")
    return frames, manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(ROOT / "models" / "yolox_s.onnx"))
    p.add_argument("--frames", required=True, help="directory from tools/capture_frames.py or frames_from_video.py")
    p.add_argument("--provider", default="tensorrt")
    p.add_argument("--precision", default="fp32")
    p.add_argument("--no-spin", action="store_true",
                   help="onnxruntime intra-op threads sleep instead of spin-waiting between runs "
                        "(the same knob as b2f run --no-spin)")
    p.add_argument("--cv-threads", type=int, help="cv2.setNumThreads for decode and preprocess")
    p.add_argument("--warmup", type=int, default=30, help="frames run before timing starts")
    p.add_argument("--limit", type=int, help="time at most this many frames")
    p.add_argument("--out", required=True)
    p.add_argument("--expect-commit", metavar="HASH",
                   help="refuse to run unless this checkout is at HASH (prefix ok) and clean")
    a = p.parse_args()
    if a.expect_commit:
        provenance.check_expected_commit(a.expect_commit)

    frames, manifest = load_frames(Path(a.frames))
    if a.limit:
        frames = frames[:a.limit]
    import onnxruntime as ort

    if a.cv_threads is not None:
        cv2.setNumThreads(a.cv_threads)
    be = OnnxRuntimeBackend(a.model, OrtOptions(provider=a.provider, precision=a.precision,
                                                allow_spinning=not a.no_spin))
    sess = be.session
    in_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    sampler = auto_sampler()
    bg = background.snapshot()

    def one_frame(jpeg: bytes, timings: dict[str, list[float]] | None) -> int:
        t = {}
        with nvtx_range("decode"):
            t0 = time.perf_counter()
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            t["decode"] = time.perf_counter() - t0
        with nvtx_range("preprocess"):
            t0 = time.perf_counter()
            x, ratio = preprocess(img)
            t["preprocess"] = time.perf_counter() - t0
        with nvtx_range("h2d"):
            t0 = time.perf_counter()
            x_dev = ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0)
            t["h2d"] = time.perf_counter() - t0
        with nvtx_range("inference"):
            t0 = time.perf_counter()
            binding = sess.io_binding()
            binding.bind_ortvalue_input(in_name, x_dev)
            binding.bind_output(out_name, "cuda", 0)
            sess.run_with_iobinding(binding)
            t["inference"] = time.perf_counter() - t0
        with nvtx_range("d2h"):
            t0 = time.perf_counter()
            out = binding.get_outputs()[0].numpy()
            t["d2h"] = time.perf_counter() - t0
        with nvtx_range("postprocess"):
            t0 = time.perf_counter()
            dets = postprocess(out, ratio)
            t["postprocess"] = time.perf_counter() - t0
        if timings is not None:
            for k, v in t.items():
                timings[k].append(v * 1000.0)
        return 0 if dets is None else len(dets)

    for jpeg in frames[:a.warmup]:
        one_frame(jpeg, None)

    timings: dict[str, list[float]] = {s: [] for s in STAGES}
    n_dets: list[int] = []
    sampler.start()
    t_start = time.perf_counter()
    with nvtx_range("timed_frames"):
        for jpeg in frames:
            n_dets.append(one_frame(jpeg, timings))
    wall = time.perf_counter() - t_start
    telemetry = sampler.stop()

    # What b2f run measures: session.run with NumPy feeds, copies included.
    plain: list[float] = []
    with nvtx_range("session_run_numpy"):
        for jpeg in frames:
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            x, _ = preprocess(img)
            t0 = time.perf_counter()
            sess.run(None, {in_name: x})
            plain.append((time.perf_counter() - t0) * 1000.0)
    sampler.close()

    per_frame_total = [sum(timings[s][i] for s in STAGES) for i in range(len(frames))]
    stage_stats = {s: stats(v) for s, v in timings.items()}
    result = {
        "what": "end-to-end pipeline profile, one frame at a time, stages timed on the host clock",
        "model": {"file": str(Path(a.model).relative_to(ROOT)) if Path(a.model).is_relative_to(ROOT) else a.model,
                  "sha256": hashlib.sha256(Path(a.model).read_bytes()).hexdigest()},
        "frames": {"dir": a.frames, "n": len(frames), "warmup": a.warmup, "set_sha256": manifest["set_sha256"],
                   "decoded_shape": manifest["decoded_shape"],
                   "jpeg_bytes_mean": float(np.mean([len(f) for f in frames]))},
        "stages": stage_stats,
        "stage_share_of_total_p50": {s: stage_stats[s]["p50_ms"] / sum(stage_stats[k]["p50_ms"] for k in STAGES)
                                     for s in STAGES},
        "total_per_frame": stats(per_frame_total),
        "session_run_numpy": stats(plain),
        "copies_h2d_plus_d2h_p50_ms": stage_stats["h2d"]["p50_ms"] + stage_stats["d2h"]["p50_ms"],
        "throughput_fps_sequential": len(frames) / wall,
        "detections_per_frame": {"mean": float(np.mean(n_dets)), "max": int(max(n_dets)),
                                 "frames_with_any": int(sum(1 for n in n_dets if n))},
        "thresholds": {"conf": CONF_THRESHOLD, "nms": NMS_THRESHOLD},
        "settings": {"ort_allow_spinning": not a.no_spin, "cv2_threads": cv2.getNumThreads()},
        "telemetry": telemetry,
        "platform": describe_platform() | sampler.describe() | be.describe() | {"git": provenance.git_state()},
        "background": bg,
        "nvtx": "nvtx" in sys.modules,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"{'stage':<12}{'p50 ms':>9}{'p95 ms':>9}{'share':>8}")
    for s in STAGES:
        print(f"{s:<12}{stage_stats[s]['p50_ms']:>9.3f}{stage_stats[s]['p95_ms']:>9.3f}"
              f"{result['stage_share_of_total_p50'][s]:>8.0%}")
    print(f"{'total':<12}{result['total_per_frame']['p50_ms']:>9.3f}{result['total_per_frame']['p95_ms']:>9.3f}")
    print(f"session.run with numpy feeds (what b2f run times): p50 {result['session_run_numpy']['p50_ms']:.3f} ms")
    print(f"detections per frame: mean {result['detections_per_frame']['mean']:.1f}; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
