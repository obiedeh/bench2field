# Saved detector accuracy

`b2f accuracy` evaluates COCO-style `images`, `categories` and `annotations`
against saved predictions. Boxes are pixel `[x,y,width,height]`, with positive
size and within declared image dimensions. Labels are never inferred. Every
image in the truth manifest contributes, even when predictions omit that frame.
Crowd/ignored annotations, duplicate IDs and unknown images/classes reject.

```bash
b2f accuracy --truth annotations.json --predictions predictions.json \
  --device bench-5090 --iou 0.5 --score-threshold 0 --max-detections 100 \
  --out /new/path/accuracy.json
```

The output must not exist. Set the protocol before evaluation. Prediction JSON
contains `dataset_sha256` of the exact annotation-file bytes,
`prediction_kind` (`measured` or `synthetic`), `configuration`, and `predictions`
(COCO-style image_id/category_id/bbox/score rows). Measured inputs additionally
require a model SHA256, each image's SHA256, and `inference` with recorded_at
(timezone-aware), device and source_sha256. Hashes in the image manifest are
declared provenance; this saved-output scorer does not reopen image files or
verify that the producer actually used those pixels. Review that chain separately.

Reports include one-to-one same-class matching, global precision/recall and
101-point interpolated AP averaged over classes with positive labels at the
single declared IoU. Negative-only classes have undefined AP and their false
positives still count globally. Empty predictions do not become perfect results.
Detection limits apply per image across categories. This is not the full COCO
metric (IoU sweep, crowd, size and max-det conventions). Keep the explicit metric
name `mean_ap_at_declared_iou`; do not silently relabel it full COCO mAP.

Fixtures under `tests/fixtures/accuracy/` are handcrafted scorer tests. The
committed synthetic run validates implementation, not a model or deployment.
No speed report, budget, prior run or schema was modified. Existing accuracy
floors remain pending until a real baseline is measured. Before promotion,
independently review truth, bind actual model/input/config artifacts, evaluate
baseline and candidate on the same held-out data/protocol, and associate the
accepted evidence with the speed variant. This command does not automate or
assert that promotion; `promotion_accepted` stays false.
