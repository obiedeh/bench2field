"""Deterministic handcrafted boxes exercise scorer logic, not real model quality."""
import copy
import hashlib
import json

import pytest

from bench2field.accuracy import evaluate, run
from bench2field.cli import build_parser


def fixture():
    truth = {"images": [{"id": i, "width": 100, "height": 100} for i in [1, 2]],
             "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "vehicle"}],
             "annotations": [{"id": i, "image_id": i, "category_id": 1,
                              "bbox": [10, 10, 20, 20]} for i in [1, 2]],
             "review": {"status": "synthetic fixture only"}}
    row = {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": .9}
    predictions = {"prediction_kind": "synthetic", "configuration": {"fixture": True},
                   "predictions": [row, {**row, "score": .8},
                                   {**row, "category_id": 2, "score": .7}]}
    return truth, predictions


def test_missing_frames_duplicates_and_negative_class_all_count():
    truth, pred = fixture()
    score = evaluate(truth, pred)
    assert (score["tp"], score["fp"], score["fn"]) == (1, 2, 1)
    assert score["precision"] == pytest.approx(1/3)
    assert score["recall"] == .5
    assert score["mean_ap_at_declared_iou"] == pytest.approx(51/101)
    assert score["per_class"][1]["ap"] is None
    assert score["images_with_predictions"] == 1


def test_ranking_threshold_and_max_detection_protocol_are_explicit():
    truth, pred = fixture()
    score = evaluate(truth, pred, score_threshold=.85, max_detections=1)
    assert (score["tp"], score["fp"], score["fn"]) == (1, 0, 1)
    pred["predictions"][0] = {**pred["predictions"][0], "bbox": [50, 50, 10, 10]}
    score = evaluate(truth, pred)
    assert score["per_class"][0]["ap"] == pytest.approx(51/202)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1, 1.1])
def test_invalid_scores_never_silently_pass(value):
    truth, pred = fixture()
    pred["predictions"][0]["score"] = value
    with pytest.raises(ValueError):
        evaluate(truth, pred)


def test_unsupported_labels_crowds_and_duplicate_ids_reject():
    truth, pred = fixture()
    for change in ['unknown', 'crowd', 'duplicate', 'badbox']:
        t, p = copy.deepcopy(truth), copy.deepcopy(pred)
        if change == 'unknown':
            p['predictions'][0]['image_id'] = 999
        elif change == 'crowd':
            t['annotations'][0]['iscrowd'] = 1
        elif change == 'duplicate':
            t['images'].append(t['images'][0])
        else:
            p['predictions'][0]['bbox'] = [90, 0, 20, 10]
        with pytest.raises(ValueError):
            evaluate(t, p)


def test_empty_predictions_and_no_positive_classes_are_not_perfect():
    truth, pred = fixture()
    pred['predictions'] = []
    score = evaluate(truth, pred)
    assert score['precision'] is None and score['recall'] == 0
    assert score['mean_ap_at_declared_iou'] == 0
    truth['annotations'] = []
    score = evaluate(truth, pred)
    assert score['mean_ap_at_declared_iou'] is None
    assert score['recall'] is None


def test_exact_dataset_binding_and_overwrite_refusal(tmp_path):
    truth, pred = fixture()
    t, p, out = (tmp_path / n for n in ['truth.json', 'pred.json', 'score.json'])
    t.write_text(json.dumps(truth))
    pred['dataset_sha256'] = hashlib.sha256(t.read_bytes()).hexdigest()
    p.write_text(json.dumps(pred))
    result = run(t, p, out, device='bench-5090')
    assert result['prediction_kind'] == 'synthetic'
    assert result['promotion_accepted'] is False
    assert result['full_coco_map'] is False
    assert result['dataset_sha256'] == pred['dataset_sha256']
    with pytest.raises(FileExistsError):
        run(t, p, out, device='bench-5090')
    t.write_text(t.read_text()+' ')
    with pytest.raises(ValueError, match='exact truth'):
        run(t, p, tmp_path/'other.json', device='bench-5090')


def test_measured_declaration_cannot_omit_provenance(tmp_path):
    truth, pred = fixture()
    t, p = tmp_path/'t.json', tmp_path/'p.json'
    t.write_text(json.dumps(truth))
    pred.update(prediction_kind='measured', dataset_sha256=hashlib.sha256(t.read_bytes()).hexdigest())
    p.write_text(json.dumps(pred))
    with pytest.raises(ValueError, match='SHA256'):
        run(t, p, tmp_path/'result.json', device='bench-5090')


def test_cli_exposes_protocol_without_altering_run_contract():
    args = build_parser().parse_args(['accuracy', '--truth', 't', '--predictions', 'p',
                                      '--out', 'o', '--device', 'bench-5090'])
    assert args.iou == .5 and args.max_detections == 100


def test_exact_iou_boundary_is_inclusive_and_not_rounded():
    truth, pred = fixture()
    truth['annotations'] = truth['annotations'][:1]
    row = pred['predictions'][0]
    pred['predictions'] = [{**row, 'bbox': [10, 10, 10, 20]}]
    assert evaluate(truth, pred)['tp'] == 1
    pred['predictions'][0]['bbox'][2] = 9.999
    assert evaluate(truth, pred)['tp'] == 0


def test_measured_metadata_accepts_utc_z_but_rejects_naive_time(tmp_path):
    # Dummy hashes exercise the metadata contract only; no model inference.
    truth, pred = fixture()
    for image in truth['images']:
        image['sha256'] = 'c'*64
    t, p = tmp_path/'t.json', tmp_path/'p.json'
    t.write_text(json.dumps(truth))
    pred.update(prediction_kind='measured', model_sha256='a'*64,
                dataset_sha256=hashlib.sha256(t.read_bytes()).hexdigest(),
                inference={'recorded_at': '2026-10-20T00:00:00Z',
                           'device': 'synthetic-test', 'source_sha256': 'b'*64})
    p.write_text(json.dumps(pred))
    assert run(t, p, tmp_path/'ok.json', device='synthetic-test')['promotion_accepted'] is False
    pred['inference']['recorded_at'] = '2026-10-20T00:00:00'
    p.write_text(json.dumps(pred))
    with pytest.raises(ValueError, match='timezone-aware'):
        run(t, p, tmp_path/'bad.json', device='synthetic-test')
