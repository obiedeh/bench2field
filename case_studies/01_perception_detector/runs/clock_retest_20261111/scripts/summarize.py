"""Summarize the 2026-11-11 clock and thread-spinning retest from its records.

Reads the profile JSON files, the one-second clock samples and the tegrastats
logs in this directory and prints the figures quoted in README.md, so every
number in the write-up can be regenerated from the committed records:

    python3 scripts/summarize.py

Frame times are the profiler's whole-frame totals (300 frames after 30 of
warmup). Power is the mean of the one-second board input rail (VDD_IN on the
Orin NX, VIN on the Thor) inside each phase, with the first two seconds of
the phase dropped so the previous phase does not leak in.
"""

from __future__ import annotations

import collections
import glob
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RAIL = {"orin": "VDD_IN", "thor": "VIN"}


def span(values: list[float]) -> str:
    """Format the lowest and highest of a set of runs."""
    return f"{min(values):.1f} to {max(values):.1f}"


def frame_runs(pattern: str) -> list[dict]:
    """Load the profile records matching a glob, in name order."""
    return [json.loads(Path(f).read_text()) | {"_file": f} for f in sorted(glob.glob(str(HERE / pattern)))]


def clocks() -> None:
    """Whole-frame time and sampled GPU clock, default against pinned, per board."""
    for board in ("orin", "thor"):
        for cond in ("default", "pinned"):
            runs = frame_runs(f"{board}/pinned_480p/*_{cond}.json")
            gpu: collections.Counter[int] = collections.Counter()
            for run in runs:
                for line in Path(run["_file"][:-5] + ".freq").read_text().splitlines():
                    fields = line.split("|")[1].split()
                    if fields:
                        gpu[int(fields[0]) // 1_000_000] += 1
            total = [r["total_per_frame"] for r in runs]
            infer = [r["stages"]["inference"]["p50_ms"] for r in runs]
            print(f"clocks {board} {cond}: runs {len(runs)}, frame p50 {span([t['p50_ms'] for t in total])} ms, "
                  f"p95 {span([t['p95_ms'] for t in total])} ms, max {max(t['max_ms'] for t in total):.1f} ms, "
                  f"inference p50 {span(infer)} ms, GPU clock samples MHz {dict(gpu.most_common(3))}")


def power() -> None:
    """Board input power and frames completed per phase, default against pinned."""
    for board in ("orin", "thor"):
        for cond in ("default", "pinned"):
            phases: dict[str, list[tuple[float, int, float | None]]] = collections.defaultdict(list)
            for f in sorted(glob.glob(str(HERE / f"{board}/power/*_{cond}.json"))):
                rows = []
                for line in Path(f[:-5] + ".tegra").read_text().splitlines():
                    m = re.search(RAIL[board] + r" (\d+)mW", line)
                    if m:
                        rows.append((float(line.split()[0]), int(m.group(1)) / 1000))
                for name, p in json.loads(Path(f).read_text())["phases"].items():
                    watts = [w for t, w in rows if p["start"] + 2 <= t <= p["end"]]
                    phases[name].append((statistics.fmean(watts), p["frames"], p["p95_ms"]))
            for name, runs in phases.items():
                p95 = [r[2] for r in runs if r[2] is not None]
                print(f"power {board} {cond} {name}: {span([r[0] for r in runs])} W, frames {[r[1] for r in runs]}"
                      + (f", frame p95 {span(p95)} ms" if p95 else ""))


def spinning() -> None:
    """Whole-frame and preprocess tails with thread spinning on and off, every condition."""
    for d in ("workstation", "orin/rover_frames_480p", "orin/rover_frames_720p", "orin/busy4_480p",
              "thor/rover_frames_480p", "thor/busy10_480p"):
        groups: dict[str, list[dict]] = collections.defaultdict(list)
        for run in frame_runs(f"{d}/*spin_*.json"):
            groups[re.sub(r"_\d+$", "", Path(run["_file"]).stem)].append(run)
        for name, runs in sorted(groups.items()):
            print(f"spinning {d} {name}: runs {len(runs)}, frame p95 {span([r['total_per_frame']['p95_ms'] for r in runs])} ms, "
                  f"frame max {max(r['total_per_frame']['max_ms'] for r in runs):.1f} ms, "
                  f"preprocess max {max(r['stages']['preprocess']['max_ms'] for r in runs):.1f} ms")


if __name__ == "__main__":
    clocks()
    power()
    spinning()
