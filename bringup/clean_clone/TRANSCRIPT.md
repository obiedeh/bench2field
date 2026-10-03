# Clean-clone test, RTX 5090 host, 2026-10-03

A fresh clone into a new directory, a new venv, following only README.md.
Two passes: the first found two gaps and the README was fixed; the second,
from the fixed README, passed with no step needing knowledge outside it.
The v1 work was on branch `v1-release` at the time, so `git checkout
v1-release` followed the clone; after the merge the README's clone line
alone is enough. The host's shell sources ROS 2, so every command carried
the `env -u PYTHONPATH` prefix the README prescribes.

## Pass 1 (README at b9e6256)

# step 1: clone (README)
cloned ok
# deviation: the v1 work is on branch v1-release until the PR merges
error: pathspec 'v1-release' did not match any file(s) known to git
# step 2: venv + install (README)
install ok
# (v1-release pushed, checked out: b9e6256 Scrub hostnames, the LAN address and home paths from tracked files)
# step 3: ONNX Runtime GPU + TensorRT (README)
ort 1.30.0 ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] trt 11.3.0.99
# step 4: pytest (README)
FAILED tests/test_onnxruntime_backend.py::test_tensorrt_provider_runs_a_conv_in_fp16
1 failed, 144 passed, 1 skipped in 3.34s
# FINDING 1: unpinned tensorrt-cu13 installed TensorRT 11.3, but onnxruntime-gpu 1.30.0's TensorRT provider links libnvinfer.so.10; the fp16 TensorRT test failed. README fix: pin onnxruntime-gpu==1.30.0 and tensorrt-cu13==10.16.1.11 (the pair validated in bringup/rtx5090/versions.txt).
# step 3 retry with the pinned pair
trt 10.16.1.11
# step 4 retry: pytest
145 passed, 1 skipped in 5.65s
# step 5: export tooling (README): pip install -r requirements-export.txt --extra-index-url .../cu130
× Getting requirements to build wheel did not run successfully.
│ exit code: 1
╰─> See above for output.
note: This error originates from a subprocess, and is likely not a problem with pip.
exit=1
# FINDING 2: pip -r requirements-export.txt failed building yolox: its setup.py imports torch, which pip's isolated build env does not have. Fix: requirements-export.txt now lists torch and the export imports only; README adds a second command installing YOLOX with --no-deps --no-build-isolation after torch.
# step 5 retry (fixed files copied into the clone)
torch install exit=0
yolox install exit=0
torch 2.11.0+cu130 True
# step 6: export YOLOX-s (README)
downloading https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
wrote ~/bench2field_clean_clone/bench2field/models/yolox_s.onnx (34.3 MB, sha256 00b70c44385b...)
max abs diff vs PyTorch: 4.27e-03
recorded in ~/bench2field_clean_clone/bench2field/case_studies/01_perception_detector/exports.json
00b70c44385bb556
# (hash in the committed exports.json: 00b70c44385bb556)
 M case_studies/01_perception_detector/exports.json
 M case_studies/01_perception_detector/requirements-export.txt
# note: the export rewrote case_studies/01_perception_detector/exports.json (host, timestamp, hashes); it is provenance for the committed artifact, so a reader should discard that change: git checkout -- case_studies/01_perception_detector/exports.json (README now says so)
# step 7: one tier at 26 Hz (README)
     26 Hz  response p50 1.839  p95 2.026  p99 2.135 ms  (service p95 1.855 ms)  misses 0  dropped 0
wrote runs/my_first.json
run exit=0
26 Hz: response p50 1.84 p95 2.03 ms, misses 0/1560, spinning False, provider TensorrtExecutionProvider, trt 10.16.1, commit b9e6256e dirty=True
# step 8: b2f report (README)
wrote /tmp/report.html (34 KB)
report exit=0

Findings and fixes (commit b3c738a):

1. `pip install tensorrt-cu13` unpinned installed TensorRT 11.3.0, whose `libnvinfer.so.11` onnxruntime-gpu 1.30.0's TensorRT provider cannot use (it links `.so.10`); the fp16 TensorRT test failed. README now pins `onnxruntime-gpu[cuda,cudnn]==1.30.0` and `tensorrt-cu13==10.16.1.11`.
2. `pip install -r requirements-export.txt` failed building YOLOX: its `setup.py` imports torch, which pip's isolated build environment does not have. The requirements file now holds torch and the export's imports, and the README installs YOLOX in a second step with `--no-deps --no-build-isolation`.
3. The export rewrites `exports.json` (provenance of the committed artifact). README now says to `git checkout` it afterwards.

## Pass 2 (README at b3c738a)

# pass 2: fresh clone, README commands verbatim (branch v1-release checked out after cloning, since master is not yet merged); every command prefixed with env -u PYTHONPATH because this shell has ROS 2 sourced, as the README says
clone: b3c738a README: pin the ONNX Runtime and TensorRT pair, install YOLOX in two steps
install: ok
ort+trt: ok
pytest: 145 passed, 1 skipped in 6.04s
export deps: ok
yolox: ok
wrote ~/bench2field_clean_clone2/bench2field/models/yolox_s.onnx (34.3 MB, sha256 00b70c44385b...)
max abs diff vs PyTorch: 4.27e-03
exports.json restored: 0 files modified
     26 Hz  response p50 1.863  p95 2.095  p99 2.228 ms  (service p95 1.919 ms)  misses 0  dropped 0
wrote runs/my_first.json
wrote /tmp/report.html (34 KB)
PASS 2 RESULT: clean, no step needed knowledge outside the README

Outcome: the exported ONNX has the committed SHA-256 (`00b70c44385b…`) on both passes; the one-tier run at 26 Hz gave response p95 2.03 ms and 2.10 ms (the committed baseline: 2.04 ms, range 2.00–2.14); `b2f report` built the page from the committed runs. The clone directories were left in place outside the repo.
