"""Turn a deepstream-app run and its tegrastats capture into one measured record.

Case study 03 runs NVIDIA's `deepstream-app` on the rover's Jetson Orin NX
with a YOLOv8n TensorRT engine and the pipeline's own performance counter
switched on (`enable-perf-measurement=1`, one line per second). Alongside,
`tegrastats --interval 1000` samples the board rails. This script reads both
logs and writes a JSON record with the frame rate statistics (median, p5,
min, max of the per-second samples after the first ``--settle`` seconds),
the board power (`VDD_IN`, median and peak, idle median from a separate
capture), GPU load and temperatures, the engine build and the versions, so
the README can quote numbers that trace to a file. Measured values only;
nothing is estimated.

Usage:
    summarize_run.py --run-dir <dir with app_<name>.log and tegrastats_<name>.log> --name file \\
        --board field-orin --idle tegrastats_idle.log --trtexec trtexec_timing.log --out runs/deepstream_field-orin_file.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

PERF_ROW = re.compile(r"\*\*PERF:\s+([0-9.]+)\s+\(([0-9.]+)\)")
VDD_IN = re.compile(r"VDD_IN (\d+)mW/(\d+)mW")
VDD_CPU_GPU = re.compile(r"VDD_CPU_GPU_CV (\d+)mW/(\d+)mW")
GR3D = re.compile(r"GR3D_FREQ (\d+)%")
TEMP = re.compile(r"(\w+)@([0-9.]+)C")
TRT_LAT = re.compile(r"Latency: min = ([0-9.]+) ms, max = ([0-9.]+) ms, mean = ([0-9.]+) ms, median = ([0-9.]+) ms")
TRT_THR = re.compile(r"Throughput: ([0-9.]+) qps")


def stats(xs: list[float]) -> dict[str, float] | None:
    if not xs:
        return None
    s = sorted(xs)
    return {"median": round(statistics.median(s), 3), "p5": round(s[int(0.05 * (len(s) - 1))], 3),
            "p95": round(s[int(round(0.95 * (len(s) - 1)))], 3), "min": round(s[0], 3), "max": round(s[-1], 3), "n": len(s)}


def perf(path: Path, settle: int) -> dict[str, Any]:
    rows = [float(m.group(1)) for m in (PERF_ROW.search(line) for line in path.read_text(errors="replace").splitlines()) if m]
    return {"samples_total": len(rows), "settle_s": settle, "fps": stats(rows[settle:]), "fps_all": stats(rows)}


def tegra(path: Path, settle: int = 0) -> dict[str, Any]:
    vin, cpu_gpu, gpu, temps = [], [], [], {}
    for line in path.read_text(errors="replace").splitlines()[settle:]:
        m = VDD_IN.search(line)
        if m:
            vin.append(int(m.group(1)))
        m = VDD_CPU_GPU.search(line)
        if m:
            cpu_gpu.append(int(m.group(1)))
        m = GR3D.search(line)
        if m:
            gpu.append(int(m.group(1)))
        for name, val in TEMP.findall(line):
            temps.setdefault(name, []).append(float(val))
    return {"vdd_in_mw": stats([float(v) for v in vin]), "vdd_cpu_gpu_cv_mw": stats([float(v) for v in cpu_gpu]),
            "gr3d_pct": stats([float(v) for v in gpu]),
            "temps_c_max": {k: max(v) for k, v in temps.items() if k in ("gpu", "cpu", "soc0", "tj")}}


def trtexec(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    text = path.read_text(errors="replace")
    lat = TRT_LAT.search(text)
    thr = TRT_THR.search(text)
    return {"latency_ms": {"min": float(lat.group(1)), "max": float(lat.group(2)), "mean": float(lat.group(3)), "median": float(lat.group(4))} if lat else None,
            "throughput_qps": float(thr.group(1)) if thr else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--name", required=True, help="file or camera")
    ap.add_argument("--board", default="field-orin")
    ap.add_argument("--settle", type=int, default=10, help="per-second samples dropped at the start")
    ap.add_argument("--idle", type=Path, default=None)
    ap.add_argument("--trtexec", type=Path, default=None)
    ap.add_argument("--versions", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    app_log = args.run_dir / f"app_{args.name}.log"
    ts_log = args.run_dir / f"tegrastats_{args.name}.log"
    rec = {
        "schema": "bench2field.deepstream/1",
        "label": "measured",
        "board": args.board,
        "source": args.name,
        "model": "yolov8n, Ultralytics ONNX export, TensorRT fp16 engine built by trtexec",
        "pipeline": "deepstream-app: source, nvstreammux 640x480, nvinfer (batch 1, interval 0), fakesink, no display",
        "perf": perf(app_log, args.settle),
        "tegrastats": tegra(ts_log, args.settle),
        "idle_tegrastats": tegra(args.idle) if args.idle and args.idle.exists() else None,
        "engine_alone_trtexec": trtexec(args.trtexec),
        "versions": args.versions.read_text().strip().splitlines() if args.versions and args.versions.exists() else None,
        "logs": {"app": app_log.name, "tegrastats": ts_log.name},
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: rec[k] for k in ("perf", "tegrastats", "engine_alone_trtexec")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
