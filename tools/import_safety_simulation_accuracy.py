"""Adapt existing semantic simulation boxes and frozen YOLOX outputs, never infer labels.

Raw images/arrays stay in the source repository. Output is a local evaluation
bundle, not a reviewed field dataset or a speed promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from bench2field.accuracy import run


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def annotation(sidecar: dict, record: dict, image_id: int) -> dict | None:
    """Use the renderer's recorded tight box; presence alone cannot supply a box."""
    if sidecar["frame_id"] != record["input"]["frame_id"]:
        raise ValueError("source frame identity mismatch")
    width, height = sidecar["resolution"]
    if [record["preprocessing"]["width"], record["preprocessing"]["height"]] != [
        width,
        height,
    ]:
        raise ValueError("source label/image dimensions differ")
    box = sidecar["render_validation"]["person_bbox"]
    present = sidecar["ground_truth"]["person_present"]
    if not present:
        if box is not None:
            raise ValueError("negative image unexpectedly has a person box")
        return None
    if box is None or len(box) != 4:
        raise ValueError("positive image requires a recorded semantic box")
    x1, y1, x2, y2 = box
    if not all(np.isfinite(box)) or not (
        0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
    ):
        raise ValueError("invalid recorded semantic box")
    return {
        "id": image_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "iscrowd": 0,
    }


def convert(source: Path, output: Path, model: Path) -> dict:
    """Verify retained bytes/source and rescore saved arrays without a model session."""
    raw = source / "reports/thor/person_backend_raw_v1_2026-10-18"
    labels = source / "reports/accuracy/simulation_2026-10-17"
    measured = json.loads((raw / "run.json").read_text())
    commit = measured["measured_source_commit"]
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("recorded inference commit required")
    if measured["status"] != "complete" or digest(model) != measured["model_sha256"]:
        raise ValueError("incomplete inference or changed model artifact")
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for filename in ["infer_person_diagnostic.py", "person_detector_contract.py"]:
        data = subprocess.check_output(
            ["git", "-C", str(source), "show", f"{commit}:scripts/{filename}"]
        )
        sources[filename] = hashlib.sha256(data).hexdigest()
        (output / filename).write_bytes(data)
    spec = importlib.util.spec_from_file_location(
        "frozen_person_contract", output / "person_detector_contract.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load frozen postprocessing source")
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    images: list[dict] = []
    boxes: list[dict] = []
    predictions: list[dict] = []
    chain: list[dict] = []
    seen = set()
    for item in measured["frames"]:
        path = raw / item["record"]
        if digest(path) != item["record_sha256"]:
            raise ValueError("changed raw frame record")
        record = json.loads(path.read_text())
        if record["input"]["suite"] != "original60":
            continue
        frame_id = record["input"]["frame_id"]
        if frame_id in seen:
            raise ValueError("duplicate source image")
        seen.add(frame_id)
        sidecar_path = labels / (frame_id + ".json")
        sidecar = json.loads(sidecar_path.read_text())
        rendered_image = labels / sidecar["image_path"]
        if digest(rendered_image) != sidecar["frame_sha256"]:
            raise ValueError("changed source rendering referenced by label")
        image = source / record["input"]["image"]
        if digest(image) != record["input"]["image_sha256"]:
            raise ValueError("changed inference image")
        array = raw / record["raw_array"]
        if digest(array) != record["raw_array_sha256"]:
            raise ValueError("changed raw model head")
        with np.load(array, allow_pickle=False) as arrays:
            head = arrays["output"]
            if (
                head.shape != (1, 8400, 85)
                or head.dtype != np.float32
                or not np.isfinite(head).all()
            ):
                raise ValueError("wrong YOLOX output contract")
            decoded = contract.postprocess_person(
                head[0], record["preprocessing"]["scale"]
            )
        if decoded != record["postprocessing"]:
            raise ValueError("saved postprocessing does not reproduce")
        identity = len(images) + 1
        images.append(
            {
                "id": identity,
                "file_name": image.name,
                "width": sidecar["resolution"][0],
                "height": sidecar["resolution"][1],
                "sha256": digest(image),
            }
        )
        box = annotation(sidecar, record, identity)
        if box is not None:
            boxes.append(box)
        for detection in decoded["detections"]:
            x1, y1, x2, y2 = detection["bbox"]
            predictions.append(
                {
                    "image_id": identity,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": detection["confidence"],
                }
            )
        chain.append(
            {
                "frame_id": frame_id,
                "record_sha256": digest(path),
                "sidecar_sha256": digest(sidecar_path),
                "rendered_image_sha256": digest(rendered_image),
                "inference_image_sha256": digest(image),
                "raw_array_sha256": digest(array),
            }
        )
    if len(images) != 60:
        raise ValueError("require the complete frozen original60 suite")
    truth = {
        "images": images,
        "categories": [{"id": 1, "name": "person"}],
        "annotations": boxes,
        "review": {
            "status": "generated tight semantic boxes; human review unsigned",
            "scope": "exposed correlated simulation,not field/held-out evidence",
        },
    }
    truth_path = output / "truth.json"
    truth_path.write_text(json.dumps(truth, indent=2) + "\n")
    source_hash = hashlib.sha256(
        json.dumps(sources, sort_keys=True).encode()
    ).hexdigest()
    pred = {
        "dataset_sha256": digest(truth_path),
        "prediction_kind": "measured",
        "model_sha256": measured["model_sha256"],
        "inference": {
            "recorded_at": measured["finished_at"],
            "device": "bench-thor",
            "source_sha256": source_hash,
            "source_commit": commit,
        },
        "configuration": {
            "task": "person only",
            "model": "YOLOX-s FP32",
            "preprocessing": "top-left letterbox,BGR0..255,640square",
            "confidence_strict_gt": 0.3,
            "nms_strict_gt": 0.45,
            "raw_run_sha256": digest(raw / "run.json"),
        },
        "predictions": predictions,
    }
    pred_path = output / "predictions.json"
    pred_path.write_text(json.dumps(pred, indent=2) + "\n")
    report = run(
        truth_path, pred_path, output / "accuracy.json", device="bench-5090", iou=0.5
    )
    manifest = {
        "source_files": sources,
        "source_bundle_hash_definition": "SHA256 of sorted JSON file-to-source-hash map",
        "adapter_sha256": digest(Path(__file__)),
        "source_records": chain,
        "truth_sha256": digest(truth_path),
        "prediction_sha256": digest(pred_path),
        "accuracy_sha256": digest(output / "accuracy.json"),
        "new_model_calls": 0,
        "image_derivation": "frozen original60 frame ID and dimensions bind JPEG inference inputs to PNG rendering sidecars; both byte hashes verified; lossy encoding is not pixel identity",
        "label_conversion": "recorded semantic xyxy to xywh by subtraction; no clipping or padding",
        "license_scope": "existing owner-generated local simulation; generator repository MIT; background asset redistribution not separately established; no images redistributed",
    }
    (output / "source-chain.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    result = convert(args.source, args.output, args.model)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "images",
                    "tp",
                    "fp",
                    "fn",
                    "precision",
                    "recall",
                    "mean_ap_at_declared_iou",
                ]
            }
        )
    )
