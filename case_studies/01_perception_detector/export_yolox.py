"""Export a YOLOX detector to the ONNX artifact every Bench2Field run of case
study 01 uses (docs/METHODOLOGY.md, rule 1: same artifact everywhere).

    env -u PYTHONPATH .venv/bin/python case_studies/01_perception_detector/export_yolox.py s
    env -u PYTHONPATH .venv/bin/python case_studies/01_perception_detector/export_yolox.py l

Choices, and why (also in README.md next to this file):

* Static input `images` of shape 1x3x640x640, float32, BGR, no normalisation
  (YOLOX expects raw 0..255 pixel values). One camera frame at a time is
  what the rover does, and a static shape lets TensorRT build its best engine.
* `decode_in_inference=True`: the graph outputs decoded boxes, so the ONNX
  output `output` is 1x8400x85 = (cx, cy, w, h, objectness, 80 class scores)
  in input pixels. Non-maximum suppression stays outside the graph, in the
  pipeline's postprocessing stage, where phase 1 measures it.
* Opset 11, YOLOX's own default, through the TorchScript exporter
  (`dynamo=False`), the path YOLOX's tools/export_onnx.py takes. No
  onnx-simplifier pass: the graph is left as exported so it is reproducible
  from the pinned YOLOX commit alone.
* Weights are the COCO checkpoints from the YOLOX 0.1.1rc0 GitHub release,
  Apache-2.0 like the code. Their SHA-256 and the ONNX file's are written to
  exports.json next to this script, which is committed; the models are not.

Needs requirements-export.txt (torch, torchvision, yolox) on top of the venv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
SIZES = {"s": "yolox-s", "l": "yolox-l", "m": "yolox-m", "x": "yolox-x"}
OPSET = 11
INPUT, OUTPUT = "images", "output"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_weights(size: str, weights_dir: Path) -> Path:
    path = weights_dir / f"yolox_{size}.pth"
    if not path.exists():
        weights_dir.mkdir(parents=True, exist_ok=True)
        url = f"{RELEASE}/yolox_{size}.pth"
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def export(size: str, weights_dir: Path, out: Path) -> dict:
    import onnx
    import torch
    from yolox.exp import get_exp

    weights = fetch_weights(size, weights_dir)
    exp = get_exp(exp_name=SIZES[size])
    model = exp.get_model()
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.head.decode_in_inference = True

    h, w = exp.test_size
    dummy = torch.randn(1, 3, h, w)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out), input_names=[INPUT], output_names=[OUTPUT],
        opset_version=OPSET, do_constant_folding=True, dynamic_axes=None, dynamo=False,
    )
    m = onnx.load(str(out))
    onnx.checker.check_model(m)

    # The exported graph must agree with the PyTorch model it came from.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = (np.random.default_rng(0).random((1, 3, h, w), dtype=np.float32) * 255.0)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    got = sess.run(None, {INPUT: x})[0]
    max_abs = float(np.abs(got - ref).max())
    if got.shape != ref.shape or max_abs > 1e-2:
        raise RuntimeError(f"ONNX output disagrees with PyTorch: shape {got.shape} vs {ref.shape}, "
                           f"max abs diff {max_abs}")

    yolox_commit = subprocess.run(
        [sys.executable, "-m", "pip", "show", "yolox"], capture_output=True, text=True
    ).stdout
    return {
        "size": size, "exp": SIZES[size], "depth": exp.depth, "width": exp.width,
        "num_classes": exp.num_classes,
        "weights": {"file": weights.name, "url": f"{RELEASE}/{weights.name}", "sha256": sha256(weights)},
        "onnx": {"file": str(out.relative_to(ROOT)), "sha256": sha256(out), "bytes": out.stat().st_size,
                 "opset": OPSET, "ir_version": m.ir_version, "exporter": "torchscript (dynamo=False)",
                 "input": {INPUT: [1, 3, h, w], "dtype": "float32", "layout": "NCHW BGR 0..255"},
                 "output": {OUTPUT: list(got.shape), "fields": "cx, cy, w, h, objectness, 80 class scores"},
                 "decode_in_inference": True, "nms_in_graph": False, "onnxsim": False},
        "check": {"max_abs_diff_vs_pytorch": max_abs, "input_seed": 0},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "onnx": onnx.__version__,
                     "onnxruntime": ort.__version__,
                     "yolox": next((ln.split(":", 1)[1].strip() for ln in yolox_commit.splitlines()
                                    if ln.startswith("Version")), None)},
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("size", choices=sorted(SIZES))
    p.add_argument("--weights-dir", default=str(ROOT / "models" / "weights"))
    p.add_argument("--out", help="default models/yolox_<size>.onnx")
    a = p.parse_args()
    out = Path(a.out) if a.out else ROOT / "models" / f"yolox_{a.size}.onnx"
    record = export(a.size, Path(a.weights_dir), out)

    index = HERE / "exports.json"
    exports = json.loads(index.read_text()) if index.exists() else {}
    exports[f"yolox_{a.size}"] = record
    index.write_text(json.dumps(exports, indent=2) + "\n")
    print(f"wrote {out} ({record['onnx']['bytes'] / 2**20:.1f} MB, sha256 {record['onnx']['sha256'][:12]}...)")
    print(f"max abs diff vs PyTorch: {record['check']['max_abs_diff_vs_pytorch']:.2e}")
    print(f"recorded in {index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
