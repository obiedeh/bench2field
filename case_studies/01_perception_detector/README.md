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
- Still pending: accuracy on labelled rover frames (no labels exist yet),
  and the same table for every optimized variant from phase 2 on
  (docs/METHODOLOGY.md, rule 9), which this baseline is for. The earlier
  attempt on Isaac-rendered frames (`reports/accuracy/2026-10-20-retained-simulation/`)
  found no person above score 0.004 on any of 60 frames; with the detector
  now measured at person AP 0.80 on photographs through the same pipeline,
  that result reads as the stylised renders being outside what the detector
  recognises, not as a pipeline fault. Labelled real frames remain the
  missing piece.

## Runs

`runs/` (pending): one report per run, produced by `b2f sweep`. Every number in the findings comes from a file there.
