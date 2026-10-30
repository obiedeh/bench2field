# Why the retained simulation run found no person: diagnosis, 2026-10-29

`accuracy.json` in this directory records recall 0 and person AP 0 for the
YOLOX-s fp32 export on 60 Isaac-rendered frames with 45 labelled people.
The README said the cause was not identified. Two measurements now place it.

**The pipeline is not the cause.** The same export, through the same
letterbox, BGR 0..255 input and YOLOX NMS, scores person AP 0.796 at IoU
0.5 on 500 labelled COCO val2017 photographs (case study 01, Accuracy;
`case_studies/01_perception_detector/accuracy/val2017_subset500/`). A
broken preprocessing, class map or decode would not produce that.

**The renders are outside what the detector recognises as a person.**
`diagnosis_2026-10-29.json` reruns the export on the 60 PNG originals (not
the JPEG inference copies of the retained run) with no score threshold, on
the RTX 5090, CUDA provider, and records per frame the highest person score,
the highest objectness and the class that scored highest:

| Quantity, over the 45 frames with a labelled person | Value |
|---|---|
| Highest person score on any frame | 0.007 |
| Median of the per-frame highest person score | 0.00076 |
| Highest person score on the 15 empty frames | 0.00061 |
| Highest objectness on a person frame | 0.599 |
| Highest class score on a person frame, any class | 0.567 |
| Classes that score highest on person frames (COCO index) | 32 sports ball on 15 frames, 9 traffic light on 13, 72 refrigerator on 9, 75 vase on 4, 61 toilet on 3 |
| Labelled person box, median size | 251 x 425 px of 1280 x 720 |

The detector does see an object where the labelled person is (objectness up
to 0.6, and the figure fills a third of the image height, so size is not the
problem) and calls it a ball, a traffic light or a refrigerator. The
stylised construction-worker asset at this rendering is not within the
detector's notion of a person, and no threshold, NMS setting or matching
rule recovers a class the network did not assign. The person scores on
person frames sit at the level of the empty frames.

**What this means for the accuracy harness.** The scorer, the label
conversion and the hash binding were right; the measurement was of the
wrong pair of detector and scene. Closing the rover-frame accuracy gap
needs labelled real frames from the rover's camera, which do not exist yet,
or a detector trained on renders of this kind, which is not this case
study's model. The COCO subset baseline is the labelled reference for the
quantization ladder until real labels exist.

Inputs: the 60 PNG frames and `ground_truth.json` (SHA-256
`6fe444e7b3d64b61f6833ce8c730f2e70567424d47550e00fdffb4490836b42f`) live in
the private safety observability repository; only the scores are recorded
here. Model SHA-256 as in `exports.json`.
