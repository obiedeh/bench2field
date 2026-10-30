"""Handwritten unit fixtures validate conversion, not detector performance."""

import copy
import runpy
from pathlib import Path

import pytest

annotation = runpy.run_path(
    str(Path(__file__).parents[1] / "tools/import_safety_simulation_accuracy.py")
)["annotation"]


def fixture():
    return (
        {
            "frame_id": "unit",
            "resolution": [100, 80],
            "render_validation": {"person_bbox": [10, 5, 30, 45]},
            "ground_truth": {"person_present": True},
        },
        {"input": {"frame_id": "unit"}, "preprocessing": {"width": 100, "height": 80}},
    )


def test_recorded_xyxy_converts_without_geometry_adjustment():
    sidecar, record = fixture()
    original = copy.deepcopy(sidecar)
    assert annotation(sidecar, record, 7) == {
        "id": 7,
        "image_id": 7,
        "category_id": 1,
        "bbox": [10, 5, 20, 40],
        "iscrowd": 0,
    }
    assert sidecar == original


@pytest.mark.parametrize(
    "change",
    [
        "identity",
        "dimensions",
        "missing",
        "negative",
        "inverted",
        "outside",
        "nonfinite",
    ],
)
def test_incompatible_recorded_labels_reject(change):
    sidecar, record = fixture()
    if change == "identity":
        record["input"]["frame_id"] = "different"
    elif change == "dimensions":
        record["preprocessing"]["width"] = 101
    elif change == "missing":
        sidecar["render_validation"]["person_bbox"] = None
    elif change == "negative":
        sidecar["ground_truth"]["person_present"] = False
    elif change == "inverted":
        sidecar["render_validation"]["person_bbox"] = [30, 5, 10, 45]
    elif change == "outside":
        sidecar["render_validation"]["person_bbox"] = [10, 5, 101, 45]
    else:
        sidecar["render_validation"]["person_bbox"][0] = float("nan")
    with pytest.raises(ValueError):
        annotation(sidecar, record, 7)


def test_recorded_negative_has_no_fabricated_box():
    sidecar, record = fixture()
    sidecar["ground_truth"]["person_present"] = False
    sidecar["render_validation"]["person_bbox"] = None
    assert annotation(sidecar, record, 7) is None
