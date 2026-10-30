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


def power_from_tegrastats(path: Path) -> str:
    """Median of the instantaneous VIN (whole board) and VDD_GPU rails over the tegrastats log."""
    if not path.exists():
        return "-"
    vin: list[float] = []
    gpu: list[float] = []
    for line in path.read_text().splitlines():
        for key, out in (("VIN ", vin), ("VDD_GPU ", gpu)):
            if key in line:
                try:
                    out.append(float(line.split(key, 1)[1].split("mW", 1)[0]) / 1000.0)
                except ValueError:
                    continue
    if not vin:
        return "-"
    gpu_s = f", GPU rail {statistics.median(gpu):.1f} W" if gpu else ""
    return f"board {statistics.median(vin):.1f} W{gpu_s} (tegrastats, whole run incl. engine build)"


def host_column(run_dir: Path) -> str:
    csv_path = run_dir / "nvidia_smi_host.csv"
    if csv_path.exists():
        return gpu_from_csv(csv_path)
    return power_from_tegrastats(run_dir / "tegrastats_host.log")


def main() -> None:
    print("| run | bag rate | input frames (Hz) | detections out (Hz) | out / in | per frame | latency p50 / p95 (ms) | host GPU or board power |")
    print("|---|---|---|---|---|---|---|---|")
    for rec_path in sorted(HERE.glob("logs/*/isaac_ros_yolov8_quickstart-*.json")):
        r = json.loads(rec_path.read_text())
        i, o, lat = r["input"], r["output"], r["latency_ms_image_stamp_to_detections"]
        ratio = o["messages"] / i["messages"] if i["messages"] else float("nan")
        rate = rec_path.parent.name.split("rate")[-1] if "rate" in rec_path.parent.name else ("single pass" if "single" in rec_path.parent.name else "1")
        print(f"| `{rec_path.parent.name}` | {rate} | {i['messages']} ({i['hz']}) | {o['messages']} ({o['hz']}) | {ratio:.2f} | "
              f"{o['detections_per_frame_mean']} | {lat['p50']} / {lat['p95']} | {host_column(rec_path.parent)} |")


if __name__ == "__main__":
    main()
