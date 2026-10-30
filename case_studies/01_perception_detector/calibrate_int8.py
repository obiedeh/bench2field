"""INT8 calibration table for the YOLOX-s export, for ONNX Runtime's TensorRT provider.

TensorRT INT8 without QDQ nodes needs per-tensor activation ranges. ONNX
Runtime's quantization tooling computes them by running an augmented copy of
the model (every activation wrapped in ReduceMin and ReduceMax, or binned for
the entropy method) over calibration images, and writes the result in the
three formats the TensorRT provider accepts (``calibration.flatbuffers``,
``calibration.cache``, ``calibration.json``). The provider reads the table
from its engine cache directory when ``trt_int8_enable`` is set and
``trt_int8_calibration_table_name`` names the file.

Calibration images are COCO val2017 images that are *not* in the accuracy
subset: the images that follow the 500-image subset in id order, again with
no crowd annotations, so the scored set never calibrates the scales. The
preprocessing is the case study's own (letterbox to 640, BGR, 0..255 float),
loaded from ``profile_pipeline.py`` like ``eval_coco_subset.py`` does.

Steps::

    --fetch N      download the N calibration images (default 100) and write calib manifest
    --compute      run the calibrator on the chosen provider (CUDA by default) and write the table

The table is deterministic for a given image set, method and model, and is
committed next to the accuracy results with the manifest (image ids and
SHA-256) so a board can re-check what it calibrated on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from eval_coco_subset import COCO_IMAGE_URL, _pipeline, git_commit, sha256  # noqa: E402

SUBSET_SKIP = 500      # the accuracy subset is the first 500 by id; calibration starts after it


def fetch(data_dir: Path, n: int, name: str) -> Path:
    ann_path = data_dir / "annotations" / "instances_val2017.json"
    coco = json.loads(ann_path.read_text())
    crowd_images = {a["image_id"] for a in coco["annotations"] if a.get("iscrowd", 0)}
    images = sorted((im for im in coco["images"] if im["id"] not in crowd_images), key=lambda im: im["id"])[SUBSET_SKIP:SUBSET_SKIP + n]
    out_dir = data_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, im in enumerate(images):
        dst = out_dir / im["file_name"]
        if not dst.exists():
            urllib.request.urlretrieve(COCO_IMAGE_URL.format(file_name=im["file_name"]), dst)
        manifest.append({"id": im["id"], "file_name": im["file_name"], "sha256": sha256(dst)})
        if (i + 1) % 25 == 0:
            print(f"[fetch] {i + 1}/{len(images)}")
    doc = {"source": f"COCO val2017, images {SUBSET_SKIP + 1} to {SUBSET_SKIP + len(images)} by id without crowd annotations (disjoint from the accuracy subset)",
           "annotations_sha256": sha256(ann_path), "images": manifest}
    path = out_dir / "calib_manifest.json"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[fetch] wrote {path}: {len(manifest)} images")
    return path


class ImageReader:
    """onnxruntime.quantization.CalibrationDataReader over the manifest."""

    def __init__(self, data_dir: Path, name: str, input_name: str) -> None:
        import cv2

        self.cv2 = cv2
        self.pp = _pipeline()
        self.input_name = input_name
        self.dir = data_dir / name
        self.items = json.loads((self.dir / "calib_manifest.json").read_text())["images"]
        self.i = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.i >= len(self.items):
            return None
        im = self.items[self.i]
        self.i += 1
        path = self.dir / im["file_name"]
        if sha256(path) != im["sha256"]:
            raise RuntimeError(f"{path.name} does not match the calibration manifest")
        x, _ = self.pp.preprocess(self.cv2.imread(str(path), self.cv2.IMREAD_COLOR))
        return {self.input_name: np.ascontiguousarray(x, dtype=np.float32)}

    def rewind(self) -> None:
        self.i = 0


def compute(data_dir: Path, name: str, model: Path, out_dir: Path, method: str, provider: str) -> None:
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationMethod, create_calibrator, write_calibration_table

    out_dir.mkdir(parents=True, exist_ok=True)
    methods = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy, "percentile": CalibrationMethod.Percentile}
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    del sess
    augmented = out_dir / "augmented_model.onnx"
    t0 = time.time()
    calibrator = create_calibrator(str(model), [], augmented_model_path=str(augmented), calibrate_method=methods[method])
    prov = {"cuda": "CUDAExecutionProvider", "cpu": "CPUExecutionProvider", "tensorrt": "TensorrtExecutionProvider"}[provider]
    if prov != "CPUExecutionProvider":
        from bench2field.backends.onnxruntime import preload_gpu_libraries

        preload_gpu_libraries(ort, prov)      # cuDNN and cuBLAS from the pip wheels, as the backend does
    calibrator.set_execution_providers([prov] + (["CPUExecutionProvider"] if prov != "CPUExecutionProvider" else []))
    reader = ImageReader(data_dir, name, input_name)
    calibrator.collect_data(reader)
    data = calibrator.compute_data()
    write_calibration_table(data, dir=str(out_dir))
    augmented.unlink(missing_ok=True)
    for extra in ("augmented_model.onnx.data",):
        (out_dir / extra).unlink(missing_ok=True)
    table = out_dir / "calibration.flatbuffers"
    table_json = json.loads((out_dir / "calibration.json").read_text())
    ranges = table_json.get("data", table_json)      # newer onnxruntime wraps the per-tensor ranges under "data"
    record: dict[str, Any] = {
        "model": model.name, "model_sha256": sha256(model), "input_name": input_name,
        "images": len(reader.items), "calib_manifest_sha256": sha256(data_dir / name / "calib_manifest.json"),
        "method": method, "provider": prov, "onnxruntime": ort.__version__,
        "tensors_calibrated": len(ranges), "wall_s": round(time.time() - t0, 1),
        "outputs": {p.name: sha256(p) for p in sorted(out_dir.glob("calibration.*"))},
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"), "git": git_commit(),
        "source": "case_studies/01_perception_detector/calibrate_int8.py", "source_sha256": sha256(Path(__file__).resolve()),
    }
    (out_dir / "calibration_record.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"[compute] {len(ranges)} tensors calibrated from {len(reader.items)} images in {record['wall_s']} s; table {table} ({table.stat().st_size} bytes)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--name", default="calib100", help="directory under --data-dir for the calibration images")
    ap.add_argument("--fetch", type=int, default=0, metavar="N")
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "yolox_s.onnx")
    ap.add_argument("--method", default="entropy", choices=["minmax", "entropy", "percentile"])
    ap.add_argument("--provider", default="cuda", choices=["cuda", "cpu", "tensorrt"])
    ap.add_argument("--out-dir", type=Path, default=HERE / "accuracy" / "int8_calibration")
    args = ap.parse_args(argv)
    if args.fetch:
        fetch(args.data_dir, args.fetch, args.name)
    if args.compute:
        compute(args.data_dir, args.name, args.model, args.out_dir, args.method, args.provider)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
