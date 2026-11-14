"""Image prep alone, with no ONNX Runtime session in the process.

Decodes and preprocesses the same frames the pipeline profiler uses, with the
same preprocess steps, and records per-frame times. Usage:
    prep_alone.py FRAMES_DIR OUT.json [--cv-threads N] [--warmup 30]
"""
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np

INPUT_SIZE, PAD = 640, 114

def preprocess(img, size=INPUT_SIZE):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(h * r), int(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), PAD, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    x = canvas.transpose(2, 0, 1)[None].astype(np.float32)
    return np.ascontiguousarray(x), r

p = argparse.ArgumentParser()
p.add_argument("frames"); p.add_argument("out")
p.add_argument("--cv-threads", type=int); p.add_argument("--warmup", type=int, default=30)
a = p.parse_args()
if a.cv_threads is not None:
    cv2.setNumThreads(a.cv_threads)
d = Path(a.frames)
frames = [(d / f["file"]).read_bytes() for f in json.loads((d / "manifest.json").read_text())["files"]]
assert "onnxruntime" not in sys.modules
dec, pre = [], []
for i, jpeg in enumerate(frames[:a.warmup] + frames):
    t0 = time.perf_counter(); img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    t1 = time.perf_counter(); preprocess(img); t2 = time.perf_counter()
    if i >= a.warmup:
        dec.append((t1 - t0) * 1e3); pre.append((t2 - t1) * 1e3)
def s(v):
    v = np.array(v); return {"n": len(v), "p50_ms": float(np.percentile(v, 50)), "p95_ms": float(np.percentile(v, 95)),
                             "p99_ms": float(np.percentile(v, 99)), "max_ms": float(v.max())}
json.dump({"what": "decode and preprocess alone, no onnxruntime session", "cv2_threads": cv2.getNumThreads(),
           "decode": s(dec), "preprocess": s(pre), "preprocess_per_frame_ms": [round(x, 4) for x in pre]},
          open(a.out, "w"), indent=1)
print(a.out, s(pre))
