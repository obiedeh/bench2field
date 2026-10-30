"""Labelled accuracy of the YOLOX-s export on a COCO val2017 subset, on one board.

Case study 01 promised mAP on the COCO validation subset for every variant
(README, "Accuracy": pending). This script produces it with the repository's
own saved-output scorer (``b2f accuracy``, docs/ACCURACY.md), in three steps
that can run on different machines:

``--fetch N``
    Selects the first N images of val2017 by image id that carry no crowd
    annotation (the scorer rejects crowd boxes rather than dropping them),
    downloads each image from the COCO image server into ``--data-dir``,
    hashes it, and writes the truth manifest (COCO-style images with
    SHA-256, the 80 categories, the annotations of those images) next to
    them. The annotation file ``instances_val2017.json`` must already be in
    ``--data-dir/annotations/``.

``--predict``
    Runs the export through the same preprocessing and post-processing as
    ``profile_pipeline.py`` (letterbox to 640, BGR 0..255, YOLOX multiclass
    NMS at 0.45) on the chosen ONNX Runtime provider and precision, keeps
    every detection above ``--conf`` (0.001 by default, so the precision and
    recall curve is complete), maps YOLOX's class index to the COCO category
    id, clips boxes to the image, and writes the prediction file the scorer
    requires: the truth file's SHA-256, the model's SHA-256, this script's
    SHA-256, the device, the provider and library versions, a timestamp.

``--score``
    Calls the scorer twice: at score threshold 0 for AP at IoU 0.5, and at
    the deployment threshold the case study runs at (``CONF_THRESHOLD`` in
    ``profile_pipeline.py``, 0.3) for the precision and recall an operator
    would see. Both reports go under ``accuracy/<subset>/<board>_<variant>/``.

The metric is the scorer's ``mean_ap_at_declared_iou`` at a single IoU of
0.5 with 101-point interpolation and one-to-one same-class matching: not
the full COCO metric (no IoU sweep, no size buckets, crowd boxes excluded by
selection). It is the same number for every variant, which is what the
quantization ladder needs. Measured values only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/{file_name}"


def _pipeline() -> Any:
    spec = importlib.util.spec_from_file_location("profile_pipeline", HERE / "profile_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def fetch(data_dir: Path, n: int, subset: str) -> Path:
    ann_path = data_dir / "annotations" / "instances_val2017.json"
    coco = json.loads(ann_path.read_text())
    crowd_images = {a["image_id"] for a in coco["annotations"] if a.get("iscrowd", 0)}
    images = sorted((im for im in coco["images"] if im["id"] not in crowd_images), key=lambda im: im["id"])[:n]
    keep = {im["id"] for im in images}
    out_dir = data_dir / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, im in enumerate(images):
        dst = out_dir / im["file_name"]
        if not dst.exists():
            urllib.request.urlretrieve(COCO_IMAGE_URL.format(file_name=im["file_name"]), dst)
        manifest.append({"id": im["id"], "file_name": im["file_name"], "width": im["width"], "height": im["height"], "sha256": sha256(dst)})
        if (i + 1) % 50 == 0:
            print(f"[fetch] {i + 1}/{len(images)}")
    truth = {
        "info": {"source": "COCO val2017 (instances_val2017.json), first images by id without crowd annotations",
                 "annotations_sha256": sha256(ann_path), "images": len(manifest)},
        "images": manifest,
        "categories": sorted(coco["categories"], key=lambda c: c["id"]),
        "annotations": [a for a in coco["annotations"] if a["image_id"] in keep],
    }
    truth_path = out_dir / "truth.json"
    truth_path.write_text(json.dumps(truth, indent=1) + "\n")
    print(f"[fetch] wrote {truth_path}: {len(manifest)} images, {len(truth['annotations'])} boxes")
    return truth_path


def predict(data_dir: Path, subset: str, model: Path, board: str, provider: str, precision: str, conf: float, out_path: Path) -> None:
    import cv2
    import onnxruntime as ort

    from bench2field.backends.onnxruntime import OnnxRuntimeBackend, OrtOptions

    pp = _pipeline()
    truth_path = data_dir / subset / "truth.json"
    truth = json.loads(truth_path.read_text())
    cat_ids = [c["id"] for c in truth["categories"]]            # YOLOX class index -> COCO category id
    be = OnnxRuntimeBackend(str(model), OrtOptions(provider=provider, precision=precision))
    in_name = be.input_specs()[0][0]
    rows: list[dict[str, Any]] = []
    started = time.time()
    t_infer: list[float] = []
    for i, im in enumerate(truth["images"]):
        path = data_dir / subset / im["file_name"]
        if sha256(path) != im["sha256"]:
            raise RuntimeError(f"{path.name} does not match the manifest hash")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        x, ratio = pp.preprocess(img)
        t0 = time.perf_counter()
        out = be.infer({in_name: x})[0]
        t_infer.append(time.perf_counter() - t0)
        pred = out[0]
        boxes = pred[:, :4]
        scores = pred[:, 4:5] * pred[:, 5:]
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        xyxy /= ratio
        dets = pp.multiclass_nms(xyxy, scores, nms_thr=pp.NMS_THRESHOLD, score_thr=conf)
        if dets is None:
            continue
        w, h = im["width"], im["height"]
        for x1, y1, x2, y2, score, cls in dets.tolist():
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w), x2), min(float(h), y2)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            rows.append({"image_id": im["id"], "category_id": cat_ids[int(cls)],
                         "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                         "score": round(min(1.0, max(0.0, float(score))), 5)})
        if (i + 1) % 100 == 0:
            print(f"[predict] {i + 1}/{len(truth['images'])}, {len(rows)} boxes so far")
    record = {
        "dataset_sha256": sha256(truth_path),
        "prediction_kind": "measured",
        "model_sha256": sha256(model),
        "configuration": {"model": model.name, "input_size": pp.INPUT_SIZE, "letterbox_pad": pp.PAD, "layout": "NCHW BGR 0..255",
                          "nms_threshold": pp.NMS_THRESHOLD, "score_threshold_kept": conf,
                          "deployment_score_threshold": pp.CONF_THRESHOLD, "class_map": "YOLOX index to COCO category id in id order"},
        "inference": {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "device": board,
                      "source_sha256": sha256(Path(__file__).resolve()), "source": "case_studies/01_perception_detector/eval_coco_subset.py",
                      "provider_requested": provider, "provider": be.provider(), "precision": precision,
                      "onnxruntime": ort.__version__, "backend": be.describe(),
                      "images": len(truth["images"]), "wall_s": round(time.time() - started, 1),
                      "inference_ms_median": round(1000 * float(np.median(t_infer)), 3) if t_infer else None,
                      "git": git_commit(), "data_dir": str(data_dir)},
        "predictions": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"[predict] wrote {out_path}: {len(rows)} boxes on {len(truth['images'])} images, provider {be.provider()}")


def score(data_dir: Path, subset: str, predictions: Path, board: str, out_dir: Path) -> None:
    from bench2field.accuracy import run

    pp = _pipeline()
    truth_path = data_dir / subset / "truth.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, thr in (("ap_iou0.5", 0.0), (f"deploy_conf{pp.CONF_THRESHOLD}", pp.CONF_THRESHOLD)):
        out = out_dir / f"accuracy_{name}.json"
        rep = run(truth_path, predictions, out, device=board, iou=0.5, score_threshold=thr, max_detections=100)
        print(f"[score] {out.name}: images {rep['images']}, tp {rep['tp']} fp {rep['fp']} fn {rep['fn']}, "
              f"precision {rep['precision']}, recall {rep['recall']}, mean_ap_at_declared_iou {rep['mean_ap_at_declared_iou']}")
    # the truth manifest travels with the reports so the hashes can be re-checked without the data directory
    (out_dir.parent / "truth.json").write_bytes(truth_path.read_bytes())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, required=True, help="holds annotations/instances_val2017.json and the image subset")
    ap.add_argument("--subset", default="val2017_subset500")
    ap.add_argument("--fetch", type=int, default=0, metavar="N")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "yolox_s.onnx")
    ap.add_argument("--board", default="bench-5090")
    ap.add_argument("--provider", default="cuda")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--out-dir", type=Path, default=None, help="default accuracy/<subset>/<board>_<provider>_<precision>/")
    args = ap.parse_args(argv)
    out_dir = args.out_dir or HERE / "accuracy" / args.subset / f"{args.board}_{args.provider}_{args.precision}"
    if args.fetch:
        fetch(args.data_dir, args.fetch, args.subset)
    if args.predict:
        predict(args.data_dir, args.subset, args.model, args.board, args.provider, args.precision, args.conf, out_dir / "predictions.json")
    if args.score:
        score(args.data_dir, args.subset, out_dir / "predictions.json", args.board, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
