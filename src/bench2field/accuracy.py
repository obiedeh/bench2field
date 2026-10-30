"""Offline fixed-IoU detector accuracy; never infer labels or promote speed results.

COCO-style boxes are xywh pixels. This reports AP at one declared IoU using
101-point interpolation, not full COCO mAP across IoUs, sizes or crowd rules.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite numeric values required")
    return float(value)


def _id(value: Any) -> int:
    if type(value) is not int:
        raise ValueError("integer IDs required")
    return value


def _unique(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {}
    for row in rows:
        key = _id(row["id"])
        if key in result:
            raise ValueError("duplicate ID")
        result[key] = row
    return result


def _box(row: dict[str, Any], image: dict[str, Any]) -> list[float]:
    if len(row["bbox"]) != 4:
        raise ValueError("xywh box must contain four numbers")
    x, y, w, h = map(_number, row["bbox"])
    width, height = _number(image["width"]), _number(image["height"])
    if width <= 0 or height <= 0 or x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError("positive dimensions and nonnegative box coordinates required")
    if x + w > width or y + h > height:
        raise ValueError("box outside image; clip explicitly before evaluation")
    return [x, y, w, h]


def _iou(a: list[float], b: list[float]) -> float:
    intersection = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(
        0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return intersection / (a[2]*a[3]+b[2]*b[3]-intersection)


def evaluate(
    truth: dict[str, Any], predictions: dict[str, Any], *, iou: float = .5,
    score_threshold: float = 0., max_detections: int = 100,
) -> dict[str, Any]:
    """Score all declared images, including missed frames and negative images."""
    if not 0 < _number(iou) <= 1 or not 0 <= _number(score_threshold) <= 1:
        raise ValueError("invalid matching or score threshold")
    if type(max_detections) is not int or max_detections <= 0:
        raise ValueError("positive per-image detection limit required")
    images, categories = _unique(truth["images"]), _unique(truth["categories"])
    if not images or not categories:
        raise ValueError("nonempty image and category manifests required")
    for image_row in images.values():
        if _number(image_row["width"]) <= 0 or _number(image_row["height"]) <= 0:
            raise ValueError("positive image dimensions required")
    annotations = _unique(truth["annotations"])
    targets: dict[tuple[int, int], list[list[float]]] = {}
    positives = {c: 0 for c in categories}
    for row in annotations.values():
        if row.get("iscrowd", 0) or row.get("ignore", 0):
            raise ValueError("crowd/ignored annotations unsupported; never silently drop")
        image, category = _id(row["image_id"]), _id(row["category_id"])
        if image not in images or category not in categories:
            raise ValueError("annotation references unknown image or category")
        targets.setdefault((image, category), []).append(_box(row, images[image]))
        positives[category] += 1
    by_image: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(predictions["predictions"]):
        image, category = _id(row["image_id"]), _id(row["category_id"])
        if image not in images or category not in categories:
            raise ValueError("prediction references unknown image or category")
        score = _number(row["score"])
        if not 0 <= score <= 1:
            raise ValueError("score outside [0,1]")
        box = _box(row, images[image])
        if score >= score_threshold:
            by_image.setdefault(image, []).append(
                {"image": image, "category": category, "score": score,
                 "box": box, "index": index})
    selected = [p for rows in by_image.values() for p in
                sorted(rows, key=lambda p: (-p["score"], p["index"]))[:max_detections]]
    per_class: list[dict[str, Any]] = []
    for category in categories:
        ranked = sorted((p for p in selected if p["category"] == category),
                        key=lambda p: (-p["score"], p["index"]))
        used: set[tuple[int, int]] = set()
        tp = fp = 0
        curve = []
        for row in ranked:
            candidates = targets.get((row["image"], category), [])
            overlaps = [(j, _iou(row["box"], b)) for j, b in enumerate(candidates)
                        if (row["image"], j) not in used]
            best = max(overlaps, key=lambda x: x[1], default=(-1, 0.))
            if best[1] >= iou:
                tp += 1
                used.add((row["image"], best[0]))
            else:
                fp += 1
            curve.append((tp / positives[category] if positives[category] else 0.,
                          tp / (tp + fp)))
        ap = (sum(max((p for r, p in curve if r >= k/100), default=0.)
                  for k in range(101)) / 101 if positives[category] else None)
        per_class.append({"category_id": category, "positives": positives[category],
                          "tp": tp, "fp": fp, "fn": positives[category]-tp, "ap": ap})
    tp, fp, fn = (sum(c[k] for c in per_class) for k in ("tp", "fp", "fn"))
    aps = [c["ap"] for c in per_class if c["ap"] is not None]
    return {"protocol": {"iou": iou, "score_threshold": score_threshold,
                         "max_detections_per_image": max_detections,
                         "interpolation_points": 101, "tie_order": "input order",
                         "crowd_policy": "reject"},
            "images": len(images), "images_with_predictions": len(by_image),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "mean_ap_at_declared_iou": sum(aps)/len(aps) if aps else None,
            "per_class": per_class}


def run(truth_path: Path, prediction_path: Path, output: Path, *, device: str,
        iou: float = .5, score_threshold: float = 0., max_detections: int = 100) -> dict[str, Any]:
    """Hash immutable evaluation inputs, record scope, and refuse report overwrite."""
    truth_bytes, prediction_bytes = truth_path.read_bytes(), prediction_path.read_bytes()
    truth, predictions = json.loads(truth_bytes), json.loads(prediction_bytes)
    truth_hash = hashlib.sha256(truth_bytes).hexdigest()
    if predictions.get("dataset_sha256") != truth_hash:
        raise ValueError("predictions do not bind the exact truth manifest")
    kind = predictions.get("prediction_kind")
    if kind not in {"measured", "synthetic"}:
        raise ValueError("declare measured or synthetic predictions")
    if kind == "measured":
        checksums = [predictions.get("model_sha256"),
                     predictions.get("inference", {}).get("source_sha256")]
        checksums.extend(image.get("sha256") for image in truth["images"])
        if any(not isinstance(s, str) or not re.fullmatch(r"[0-9a-f]{64}", s)
               for s in checksums):
            raise ValueError("measured predictions require model, inference and image SHA256")
        inference = predictions.get("inference", {})
        if not inference.get("device"):
            raise ValueError("measured inference device required")
        recorded_at = inference.get("recorded_at")
        if not isinstance(recorded_at, str):
            raise ValueError("inference timestamp required")
        timestamp = datetime.fromisoformat(
            recorded_at[:-1]+"+00:00" if recorded_at.endswith("Z") else recorded_at)
        if timestamp.tzinfo is None:
            raise ValueError("timezone-aware inference timestamp required")
    if not predictions.get("configuration"):
        raise ValueError("prediction configuration required")
    report = evaluate(truth, predictions, iou=iou, score_threshold=score_threshold,
                      max_detections=max_detections)
    try:
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except subprocess.CalledProcessError:
        source, dirty = None, None
    report.update(schema_version="detection-accuracy-1", created_at=datetime.now(timezone.utc).isoformat(),
                  device=device, machine=platform.machine(), python=platform.python_version(),
                  evaluation_source_commit=source, evaluation_source_dirty=dirty,
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dataset_sha256=truth_hash,
                  predictions_sha256=hashlib.sha256(prediction_bytes).hexdigest(),
                  model_sha256=predictions.get("model_sha256"),
                  configuration=predictions["configuration"], prediction_kind=kind,
                  inference_provenance=predictions.get("inference"),
                  reference_review=truth.get("review", {"status": "unverified"}),
                  scope="fixed-IoU saved-output evaluation; no inference or speed promotion",
                  full_coco_map=False, promotion_accepted=False)
    if truth_path.read_bytes() != truth_bytes or prediction_path.read_bytes() != prediction_bytes:
        raise RuntimeError("evaluation inputs changed")
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report
