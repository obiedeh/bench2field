"""Print the phase 2 precision-ladder tables from the committed files, so every
number in the README's precision section can be regenerated from runs/ and
accuracy/.

    env -u PYTHONPATH .venv/bin/python case_studies/01_perception_detector/summarize_precision.py

Speed comes from the `b2f sweep` reports (phase2_precision*.yaml, one engine
cache per precision); accuracy from eval_coco_subset.py on the 500-image COCO
subset. The two halves are joined by variant so no speed number appears
without its accuracy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bench2field.metrics import repeat_stat  # noqa: E402
from bench2field.schema import RunReport  # noqa: E402
from bench2field.sweep import SweepConfig, report_path  # noqa: E402

RUNS = HERE / "runs"
ACC = HERE / "accuracy" / "val2017_subset500"
BOARDS = {
    "Jetson AGX Thor": ("bench_thor_precision", "bench-thor"),
    "Jetson Orin NX": ("bench_orin_precision", "field-orin"),
    "RTX 5090": ("bench_5090_precision", "bench-5090"),
}
PRECISIONS = ("fp32", "fp16", "int8")


def tel(reports: list[RunReport], hz: float, key: str) -> str:
    vals = [r.tier(hz).telemetry.get(key, {}).get("p50") for r in reports]
    vals = [v for v in vals if v is not None]
    return f"{sorted(vals)[len(vals) // 2]:.1f}" if vals else "-"


def accuracy_row(board: str, precision: str) -> dict | None:
    d = ACC / f"{board}_tensorrt_{precision}"
    ap = d / "accuracy_ap_iou0.5.json"
    dep = d / "accuracy_deploy_conf0.3.json"
    pred = d / "predictions.json"
    if not (ap.exists() and dep.exists()):
        return None
    a = json.loads(ap.read_text())
    b = json.loads(dep.read_text())
    n_boxes = None
    infer_ms = None
    if pred.exists():
        p = json.loads(pred.read_text())
        n_boxes = len(p["predictions"])
        infer_ms = p["inference"].get("inference_ms_median")
    person = next(
        (c.get("ap") for c in a.get("per_class", []) if c.get("category_id") == 1), None
    )  # COCO category 1 is person
    return {
        "map": a["mean_ap_at_declared_iou"],
        "recall": a["recall"],
        "precision_dep": b["precision"],
        "recall_dep": b["recall"],
        "boxes": n_boxes,
        "infer_ms": infer_ms,
        "person_ap": person,
    }


def accuracy_tables() -> None:
    print("## Accuracy, 500-image COCO val2017 subset, TensorRT provider\n")
    print(
        "| board | precision | mean AP at IoU 0.5 | person AP | recall at 0.001 | precision / recall at 0.3 | boxes kept | session.run median |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for name, (_sub, board) in BOARDS.items():
        for prec in PRECISIONS:
            r = accuracy_row(board, prec)
            if r is None:
                continue
            person = f"{r['person_ap']:.4f}" if r["person_ap"] is not None else "-"
            boxes = r["boxes"] if r["boxes"] is not None else "-"
            infer = f"{r['infer_ms']:.2f} ms" if r["infer_ms"] else "-"
            print(
                f"| {name} | {prec} | {r['map']:.4f} | {person} | {r['recall']:.3f} | "
                f"{r['precision_dep']:.3f} / {r['recall_dep']:.3f} | {boxes} | {infer} |"
            )
    print()


def sweep_tables() -> None:
    for name, (sub, _board) in BOARDS.items():
        d = RUNS / sub
        manifests = sorted(d.glob("sweep_*.json")) if d.exists() else []
        if not manifests:
            print(f"### {name}: no precision sweep in `{sub}/`\n")
            continue
        m = json.loads(manifests[0].read_text())
        cfg = SweepConfig.from_dict(m["config"])
        first = RunReport.load(report_path(d, cfg.variants[0], 1))
        p = first.platform
        power_key = (
            "power_board_w"
            if "power_board_w" in first.tier(cfg.tiers_hz[0]).telemetry
            else "power_gpu_w"
        )
        print(
            f"### {name}: `{sub}/`, sweep `{m['sweep']}`, {'complete' if m['complete'] else 'INCOMPLETE'}"
        )
        print(
            f"onnxruntime {p.get('onnxruntime')}, TensorRT {p.get('tensorrt')}, CUDA {p.get('cuda_runtime')}, "
            f"power mode {p.get('nvpmodel') or 'n/a'}\n"
        )
        print(
            f"| variant | Hz | response p50 (median, range) | response p95 (median, range) | response p99 | "
            f"service p95 | misses / scheduled | {power_key} p50 |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for v in cfg.variants:
            reps = [
                RunReport.load(report_path(d, v, r))
                for r in range(1, cfg.repeats + 1)
                if report_path(d, v, r).exists()
            ]
            if not reps:
                continue
            for hz in cfg.tiers_hz:
                s50 = repeat_stat(reps, hz, "response_p50_ms")
                s95 = repeat_stat(reps, hz, "response_p95_ms")
                s99 = repeat_stat(reps, hz, "response_p99_ms")
                svc = repeat_stat(reps, hz, "p95_ms")
                misses = sum(r.tier(hz).deadline_misses for r in reps)
                sched = sum(r.tier(hz).n_scheduled for r in reps)
                print(
                    f"| {v.label} | {hz:g} | {s50.median:.3f} ({s50.lo:.3f} to {s50.hi:.3f}) | "
                    f"{s95.median:.3f} ({s95.lo:.3f} to {s95.hi:.3f}) | {s99.median:.3f} | {svc.median:.3f} | "
                    f"{misses} / {sched} | {tel(reps, hz, power_key)} |"
                )
        print()


if __name__ == "__main__":
    accuracy_tables()
    sweep_tables()
