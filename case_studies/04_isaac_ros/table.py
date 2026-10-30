"""Print the Isaac ROS quickstart table from the committed JSON records under logs/.

    env -u PYTHONPATH .venv/bin/python case_studies/04_isaac_ros/table.py
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def gpu_from_csv(path: Path) -> str:
    if not path.exists():
        return "-"
    util: list[float] = []
    mem: list[float] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                util.append(float(next(v for k, v in row.items() if "utilization" in k).strip().rstrip("%")))
                mem.append(float(next(v for k, v in row.items() if "memory" in k).strip().split()[0]))
            except (StopIteration, ValueError, AttributeError):
                continue
    if not util:
        return "-"
    return f"{statistics.median(util):.0f}% util, {statistics.median(mem) / 1024:.1f} GiB used (host, all processes)"


def main() -> None:
    print("| run | bag rate | input frames (Hz) | detections out (Hz) | out / in | per frame | latency p50 / p95 (ms) | host GPU during the window |")
    print("|---|---|---|---|---|---|---|---|")
    for rec_path in sorted(HERE.glob("logs/*/isaac_ros_yolov8_quickstart-*.json")):
        r = json.loads(rec_path.read_text())
        i, o, lat = r["input"], r["output"], r["latency_ms_image_stamp_to_detections"]
        ratio = o["messages"] / i["messages"] if i["messages"] else float("nan")
        rate = rec_path.parent.name.split("rate")[-1] if "rate" in rec_path.parent.name else ("single pass" if "single" in rec_path.parent.name else "1")
        print(f"| `{rec_path.parent.name}` | {rate} | {i['messages']} ({i['hz']}) | {o['messages']} ({o['hz']}) | {ratio:.2f} | "
              f"{o['detections_per_frame_mean']} | {lat['p50']} / {lat['p95']} | {gpu_from_csv(rec_path.parent / 'nvidia_smi_host.csv')} |")


if __name__ == "__main__":
    main()
