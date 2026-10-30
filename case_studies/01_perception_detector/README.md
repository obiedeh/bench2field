# Case study 01: perception detector on the ROSMASTER rover

Plan and phases: [PLAN.md](PLAN.md). Findings: `PHASE1_FINDINGS.md` (pending).

## Detector

**YOLOX** ([Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), Apache-2.0 code and COCO weights), pinned to commit `6ddff482`. YOLOX-s is the model under study and the distillation student; YOLOX-l is the teacher for phase 2. Chosen over RT-DETRv2 because it is a plain CNN (INT8 PTQ, 2:4 sparsity and distillation are all well-trodden on TensorRT), its decoding and NMS can be kept outside the graph where the end-to-end profile can see them, and opset 11 is the safer bet for the MIGraphX run in phase 4.

## ONNX export

`export_yolox.py` produces the one artifact every run uses, and records what it made in `exports.json` (committed; the `.onnx` and `.pth` files are not). To reproduce on the 5090 host:

```bash
env -u PYTHONPATH .venv/bin/pip install -r case_studies/01_perception_detector/requirements-export.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130
env -u PYTHONPATH .venv/bin/pip install --no-deps --no-build-isolation \
    "yolox @ git+https://github.com/Megvii-BaseDetection/YOLOX.git@6ddff4824372906469a7fae2dc3206c7aa4bbaee"
env -u PYTHONPATH .venv/bin/python case_studies/01_perception_detector/export_yolox.py s
```

| Choice | Value | Why |
|---|---|---|
| Input | `images`, static `1x3x640x640`, float32, BGR, raw 0..255 | One camera frame at a time, as on the rover; a static shape lets TensorRT build its best engine; YOLOX takes unnormalised pixels |
| Output | `output`, `1x8400x85`: cx, cy, w, h, objectness, 80 class scores, in input pixels | `decode_in_inference=True`, so decoding is in the graph |
| NMS | outside the graph | It is the pipeline's postprocessing stage, which phase 1 measures |
| Opset | 11 | YOLOX's own default; widest provider support |
| Exporter | TorchScript (`dynamo=False`) | The path YOLOX's `tools/export_onnx.py` takes |
| Simplifier | none | The graph is reproducible from the pinned commit alone |
| Dynamic axes | none | See input |

### yolox_s (exported 2026-10-03)

- Weights: `yolox_s.pth` from the YOLOX 0.1.1rc0 release, SHA-256 `f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30`
- ONNX: `models/yolox_s.onnx`, 34.3 MB, SHA-256 `00b70c44385bb5567e784b0c1efae213c4f2ec06efacfc8e3f52eca333356205`
- Check against PyTorch on one random input: max abs diff 4.27e-03 on outputs in pixel units
- torch 2.11.0+cu130, onnx 1.23.1, onnxruntime 1.30.0, Python 3.12.3

The same file (same SHA-256) is on the 5090 host, the Thor and the Orin under `models/`.

## Accuracy

Measured on a labelled COCO val2017 subset with the repository's saved-output
scorer (`b2f accuracy`, docs/ACCURACY.md): the first 500 images by id that
carry no crowd annotation, 2,918 boxes over 80 classes, every image hashed
into the truth manifest (`accuracy/val2017_subset500/truth.json`). The
detections come from `eval_coco_subset.py`, the same letterbox, BGR 0..255
input and YOLOX multiclass NMS at 0.45 as `profile_pipeline.py`, kept down
to score 0.001 so the precision and recall curve is complete, and scored
twice: at threshold 0 for average precision at IoU 0.5 (101-point
interpolation, one-to-one same-class matching, one IoU, not the full COCO
metric), and at the deployment threshold the case study runs at (0.3) for
the precision and recall an operator sees. RTX 5090, 2026-10-29, the ONNX
export at the SHA-256 in `exports.json`, from
`accuracy/val2017_subset500/bench-5090_<provider>_fp32/accuracy_*.json`:

| Provider, fp32 | Mean AP at IoU 0.5 (80 classes) | Person AP at IoU 0.5 | Recall at score 0.001 | Precision / recall at score 0.3 | Inference per image, median |
|---|---|---|---|---|---|
| CUDA | 0.6521 | 0.7964 | 0.846 | 0.703 / 0.617 | 5.91 ms |
| TensorRT | 0.6521 | 0.7964 | 0.846 | 0.703 / 0.617 | 3.79 ms |

- The TensorRT path runs fp32 as TF32 inside (PHASE1_FINDINGS.md, section
  3). On this subset that changes 17 boxes out of 57,834 and the mean AP by
  0.00004: the precision that the setting silently changed is not one the
  detector's output depends on. That is a measured answer, for this model,
  to the question the phase 1 findings left open.
- The per-image inference time here is the whole `session.run` on a shared
  GPU (an unrelated simulation process was resident), not the case study's
  paced measurement; the latency numbers are in the sections above.
- The same table for the fp16 and int8 variants on the Thor and the Orin NX
  is in the phase 2 section below. Still pending: accuracy on labelled
  rover frames (no labels exist yet). The earlier
  attempt on Isaac-rendered frames (`reports/accuracy/2026-10-20-retained-simulation/`)
  found no person above score 0.004 on any of 60 frames; with the detector
  now measured at person AP 0.80 on photographs through the same pipeline,
  that result reads as the stylised renders being outside what the detector
  recognises, not as a pipeline fault. Labelled real frames remain the
  missing piece.

## Phase 2: precision ladder (fp32, fp16, int8), Thor and Orin NX, 2026-10-30

The same ONNX export (SHA-256 above) through the TensorRT execution provider
at three precisions, with accuracy and speed measured together so no speed
number stands without its accuracy (docs/METHODOLOGY.md, rule 9). The
tables below are printed by `summarize_precision.py` from the committed
files, the accuracy from `accuracy/val2017_subset500/<board>_tensorrt_<precision>/`
and the speed from `runs/bench_thor_precision/` and `runs/bench_orin_precision/`.

- fp16: TensorRT's fp16 flag, nothing else changed.
- int8: post-training quantization with a calibration table from
  onnxruntime's min-max calibrator over 100 COCO val2017 images disjoint
  from the 500-image accuracy subset (`calibrate_int8.py`, record in
  `accuracy/int8_calibration/calibration_record.json`, 293 tensors). The
  entropy calibrator exhausted host memory on this model and was not used.
  Calibration on rover frames waits for labelled frames.
- One engine cache per precision (`.trt_cache/<precision>/`), built once by
  the accuracy pass and reused by the sweep, so the sweep never includes an
  engine build. The Orin NX runs with a 1024 MB TensorRT workspace.
- Speed protocol as in phase 1: `b2f sweep` from
  `sweeps/phase2_precision.yaml` (Thor) and `sweeps/phase2_precision_orin.yaml`
  (Orin NX), one process per run, tiers 10, 26, 30 and 100 Hz, 60 s per tier,
  three repeats, `--no-spin`, 33.3 ms deadline, bench-idle environment.
  Thor: onnxruntime 1.24.0, TensorRT 10.13.3, CUDA 13.2, 120 W mode. Orin NX:
  onnxruntime 1.24.0, TensorRT 10.7.0, CUDA 12.6, MAXN_SUPER.

### What it shows

- fp16 costs nothing in accuracy on either board: mean AP 0.6530 (Thor) and
  0.6532 (Orin NX) against 0.6522 for fp32, person AP within 0.0003, the same
  precision and recall at the 0.3 deployment threshold. On the Thor it takes
  the 10 Hz response from 10.4 to 6.8 ms; on the Orin NX from 34.3 to 19.8 ms.
- int8 (min-max PTQ, 100 images) gives up 1.5 AP points (0.6374 Thor, 0.6361
  Orin NX), 0.6 of a point on person, and raises the box count at score 0.001
  by 11 to 15 percent: at the 0.3 threshold precision falls from 0.70 to 0.69
  (Thor) and 0.68 (Orin NX) at the same recall. Whether that is acceptable
  depends on the deployment; the number is now measured rather than assumed.
- int8 is not faster than fp16 on the Thor: 7.3 against 6.8 ms at 10 Hz,
  6.6 against 6.4 ms at 26 and 30 Hz, both about 3.9 ms at 100 Hz. The GPU
  rail is the only place it shows: 4.7 W against 5.9 W (fp16) and 11.0 W
  (fp32) at 100 Hz, board 23.4 against 25.6 and 33.0 W.
- On the Orin NX int8 is faster than fp16, 15.6 against 19.8 ms at 10 Hz,
  and draws less: board 5.9 / 6.5 / 9.7 W at 10 / 26 / 100 Hz against
  6.3 / 7.3 / 14.6 W for fp16 and 7.0 / 9.2 / 23.7 W for fp32.
- fp32 on the Orin NX misses the 33.3 ms deadline at 10 Hz (service 34.2 ms,
  about 580 of 600 frames late in every repeat) and meets it at 26 and 30 Hz
  (23.2 and 21.0 ms): the GPU clock stays low at light load and rises with
  it, the same effect phase 1 saw on the 26 Hz repeats. At 100 Hz fp32
  saturates the Orin NX (82 Hz achieved, response p50 above 6 s from queueing);
  fp16 and int8 hold 100 Hz at 8.5 and 8.1 ms with no misses. On the Thor no
  tier at any precision missed a deadline.
- The Thor was not idle for the whole sweep: another session's model
  profile ran during fp16 repeat 1 (its 10 and 26 Hz tiers show 26 W board
  power, 10 percent CPU and 9 GB more RAM than the other repeats), and about
  8 GB of RAM stayed resident from then on. The fp16 latency medians of that
  repeat sit inside the range of the other two, so the tables keep it; the
  power columns for those two tiers are the ones to read with care.
- The deployment call for the rover's Orin NX from these numbers: fp16 for
  accuracy parity at the camera's 26 Hz with 7.3 W board power; int8 only if
  the 1.5 AP points and the extra false positives are acceptable in exchange
  for 0.8 W and 4 ms. Accuracy on labelled rover frames is still the missing
  measurement for both.

### Accuracy, 500-image COCO val2017 subset, TensorRT provider

| board | precision | mean AP at IoU 0.5 | person AP | recall at 0.001 | precision / recall at 0.3 | boxes kept | session.run median |
|---|---|---|---|---|---|---|---|
| Jetson AGX Thor | fp32 | 0.6522 | 0.7964 | 0.846 | 0.702 / 0.617 | 57854 | 10.48 ms |
| Jetson AGX Thor | fp16 | 0.6530 | 0.7962 | 0.847 | 0.702 / 0.617 | 57823 | 6.21 ms |
| Jetson AGX Thor | int8 | 0.6374 | 0.7947 | 0.844 | 0.692 / 0.603 | 64061 | 6.72 ms |
| Jetson Orin NX | fp32 | 0.6522 | 0.7964 | 0.846 | 0.703 / 0.617 | 57837 | 33.66 ms |
| Jetson Orin NX | fp16 | 0.6532 | 0.7961 | 0.847 | 0.702 / 0.617 | 57832 | 19.65 ms |
| Jetson Orin NX | int8 | 0.6361 | 0.7903 | 0.844 | 0.678 / 0.613 | 66707 | 15.39 ms |
| RTX 5090 | fp32 | 0.6521 | 0.7964 | 0.846 | 0.703 / 0.617 | 57851 | 3.79 ms |

### Jetson AGX Thor: `bench_thor_precision/`, sweep `phase2-precision-ladder`, complete
onnxruntime 1.24.0, TensorRT 10.13.3, CUDA 13.2, power mode NV Power Mode: 120W

| variant | Hz | response p50 (median, range) | response p95 (median, range) | response p99 | service p95 | misses / scheduled | power_board_w p50 |
|---|---|---|---|---|---|---|---|
| trt_fp32 | 10 | 10.363 (10.131 to 10.766) | 11.079 (10.575 to 11.275) | 11.747 | 10.883 | 0 / 1800 | 19.9 |
| trt_fp32 | 26 | 10.412 (10.204 to 10.421) | 11.018 (10.855 to 11.023) | 11.350 | 10.951 | 0 / 4680 | 20.2 |
| trt_fp32 | 30 | 10.301 (10.106 to 10.470) | 10.872 (10.852 to 10.982) | 11.190 | 10.802 | 0 / 5400 | 20.2 |
| trt_fp32 | 100 | 3.963 (3.874 to 4.027) | 4.498 (4.437 to 4.581) | 4.866 | 4.437 | 0 / 18000 | 33.0 |
| trt_fp16 | 10 | 6.786 (6.730 to 6.964) | 7.256 (6.936 to 7.293) | 7.570 | 7.014 | 0 / 1800 | 20.1 |
| trt_fp16 | 26 | 6.430 (6.400 to 6.495) | 6.999 (6.753 to 7.041) | 7.338 | 6.937 | 0 / 4680 | 20.1 |
| trt_fp16 | 30 | 6.386 (6.027 to 6.514) | 7.005 (6.898 to 7.054) | 7.298 | 6.937 | 0 / 5400 | 20.2 |
| trt_fp16 | 100 | 4.006 (3.891 to 4.066) | 4.762 (4.452 to 5.224) | 5.290 | 4.665 | 0 / 18000 | 25.6 |
| trt_int8 | 10 | 7.257 (6.956 to 7.506) | 7.546 (7.542 to 8.033) | 7.880 | 7.311 | 0 / 1800 | 20.0 |
| trt_int8 | 26 | 6.589 (6.569 to 6.612) | 7.279 (7.266 to 7.457) | 7.533 | 7.217 | 0 / 4680 | 19.8 |
| trt_int8 | 30 | 6.604 (6.559 to 6.767) | 7.378 (7.228 to 7.481) | 7.788 | 7.316 | 0 / 5400 | 20.0 |
| trt_int8 | 100 | 3.931 (3.874 to 4.076) | 4.647 (4.470 to 4.740) | 5.190 | 4.587 | 0 / 18000 | 23.4 |

### Jetson Orin NX: `bench_orin_precision/`, sweep `phase2-precision-ladder-orin`, complete
onnxruntime 1.24.0, TensorRT 10.7.0, CUDA 12.6, power mode NV Power Mode: MAXN_SUPER

| variant | Hz | response p50 (median, range) | response p95 (median, range) | response p99 | service p95 | misses / scheduled | power_board_w p50 |
|---|---|---|---|---|---|---|---|
| trt_fp32 | 10 | 34.290 (34.288 to 34.311) | 34.577 (34.565 to 34.597) | 34.690 | 34.471 | 1738 / 1800 | 7.0 |
| trt_fp32 | 26 | 23.240 (23.230 to 23.240) | 23.452 (23.451 to 23.485) | 23.573 | 23.361 | 0 / 4680 | 9.2 |
| trt_fp32 | 30 | 21.044 (21.031 to 23.219) | 21.211 (21.190 to 23.424) | 21.346 | 21.124 | 0 / 5400 | 9.8 |
| trt_fp32 | 100 | 6392.569 (6347.104 to 6435.946) | 12144.426 (12104.613 to 12234.112) | 12657.394 | 12.228 | 17969 / 18000 | 23.7 |
| trt_fp16 | 10 | 19.825 (19.794 to 19.879) | 20.084 (20.071 to 20.115) | 20.214 | 19.949 | 0 / 1800 | 6.3 |
| trt_fp16 | 26 | 19.638 (19.618 to 19.644) | 19.881 (19.877 to 19.920) | 19.987 | 19.786 | 0 / 4680 | 7.3 |
| trt_fp16 | 30 | 19.608 (19.606 to 19.659) | 19.808 (19.778 to 19.843) | 19.945 | 19.708 | 0 / 5400 | 7.5 |
| trt_fp16 | 100 | 8.515 (7.918 to 8.550) | 8.573 (8.003 to 8.664) | 8.732 | 8.487 | 0 / 18000 | 14.6 |
| trt_int8 | 10 | 15.588 (15.543 to 15.636) | 15.866 (15.809 to 15.905) | 15.982 | 15.734 | 0 / 1800 | 5.9 |
| trt_int8 | 26 | 15.360 (15.359 to 15.419) | 15.612 (15.607 to 15.654) | 15.728 | 15.525 | 0 / 4680 | 6.5 |
| trt_int8 | 30 | 15.325 (15.322 to 15.394) | 15.507 (15.507 to 15.568) | 15.686 | 15.411 | 0 / 5400 | 6.7 |
| trt_int8 | 100 | 8.064 (8.054 to 8.107) | 8.180 (8.178 to 8.211) | 8.320 | 8.089 | 0 / 18000 | 9.7 |

## Runs

`runs/` (pending): one report per run, produced by `b2f sweep`. Every number in the findings comes from a file there.
