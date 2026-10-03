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

Pending: mAP on the COCO validation subset and on labelled rover frames, reported with every variant from phase 2 on (docs/METHODOLOGY.md, rule 9).

## Runs

`runs/` (pending): one report per run, produced by `b2f sweep`. Every number in the findings comes from a file there.
