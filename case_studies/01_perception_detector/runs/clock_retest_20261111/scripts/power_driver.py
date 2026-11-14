"""Run the case study 01 frame pipeline in three phases for a power reading:
idle (session loaded, no frames), paced at 30 frames per second, and back to back.
Uses the profiler's own preprocess, postprocess and backend; spinning off."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "case_studies/01_perception_detector")
import profile_pipeline as pp
import numpy as np, cv2, onnxruntime as ort

out, idle_s, run_s = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
frames, _ = pp.load_frames(Path("data/rover_frames_480p"))
be = pp.OnnxRuntimeBackend(str(pp.ROOT / "models" / "yolox_s.onnx"),
                           pp.OrtOptions(provider="tensorrt", precision="fp32", allow_spinning=False))
sess = be.session
i_name, o_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name

def one(jpeg):
    t0 = time.perf_counter()
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    x, ratio = pp.preprocess(img)
    xd = ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0)
    b = sess.io_binding(); b.bind_ortvalue_input(i_name, xd); b.bind_output(o_name, "cuda", 0)
    sess.run_with_iobinding(b)
    pp.postprocess(b.get_outputs()[0].numpy(), ratio)
    return (time.perf_counter() - t0) * 1000.0

for j in frames[:30]:
    one(j)
rec = {"phases": {}}
def phase(name, seconds, period=None):
    start = time.time(); t_end = time.perf_counter() + seconds; ms = []; k = 0
    nxt = time.perf_counter()
    while time.perf_counter() < t_end:
        if name == "idle":
            time.sleep(0.5); continue
        ms.append(one(frames[k % len(frames)])); k += 1
        if period:
            nxt += period
            d = nxt - time.perf_counter()
            if d > 0: time.sleep(d)
            else: nxt = time.perf_counter()
    rec["phases"][name] = {"start": start, "end": time.time(), "frames": k,
                           "p50_ms": float(np.percentile(ms, 50)) if ms else None,
                           "p95_ms": float(np.percentile(ms, 95)) if ms else None,
                           "max_ms": float(max(ms)) if ms else None}
phase("idle", idle_s)
for rate in (sys.argv[4].split(",") if len(sys.argv) > 4 else ["30"]):
    phase(f"paced{rate}", run_s, 1 / float(rate))
phase("back_to_back", run_s)
out.write_text(json.dumps(rec, indent=1) + "\n")
