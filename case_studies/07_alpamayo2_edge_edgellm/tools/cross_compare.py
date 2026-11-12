#!/usr/bin/env python3
"""ADE between builds for one clip, and against the PyTorch BF16 waypoints of case study 06.

Usage: python cross_compare.py CLIP_SHORT NAME=COMPARE_JSON [NAME=COMPARE_JSON ...] [--pytorch DECISIONS_JSON]
  COMPARE_JSON   host/compare.py output of a thor/run.py result (trajectory_xyz: candidates x 64 x 3, t0 rig frame)
  DECISIONS_JSON case study 06 run record (records with clip_id and waypoints_xyz, the upstream decode of the same
                 action space; several repeats per clip)
Prints, per pair of builds, the ADE (mean 2-D distance over the 64 points, metres) between first candidates
(same seed, same noise) and the smallest ADE over all candidate pairs; against PyTorch, the ADE of each build's
first candidate to each PyTorch repeat, smallest and mean.
"""
import json
import sys
from itertools import combinations

import numpy as np


def ade(a, b):
    a, b = np.asarray(a)[:, :2], np.asarray(b)[:, :2]
    n = min(len(a), len(b))
    return float(np.linalg.norm(a[:n] - b[:n], axis=-1).mean())


def main():
    args = sys.argv[1:]
    clip = args.pop(0)
    pytorch = None
    if "--pytorch" in args:
        i = args.index("--pytorch")
        pytorch = args[i + 1]
        del args[i : i + 2]
    builds = {}
    for a in args:
        name, path = a.split("=", 1)
        builds[name] = json.load(open(path))["trajectory_xyz"]
    out = {"clip": clip, "pairs": {}, "pytorch": {}}
    for (n1, t1), (n2, t2) in combinations(builds.items(), 2):
        out["pairs"][f"{n1} vs {n2}"] = {
            "first_candidates_m": round(ade(t1[0], t2[0]), 3),
            "closest_candidates_m": round(min(ade(a, b) for a in t1 for b in t2), 3),
        }
    if pytorch:
        recs = json.load(open(pytorch))
        recs = recs.get("decisions") or recs.get("records") or recs.get("results") or []
        ref = [r["waypoints_xyz"] for r in recs if str(r.get("clip_id", "")).startswith(clip) and r.get("waypoints_xyz")]
        for name, t in builds.items():
            if ref:
                ds = [ade(t[0], w) for w in ref]
                out["pytorch"][name] = {
                    "repeats": len(ref),
                    "closest_m": round(min(ds), 3),
                    "mean_m": round(float(np.mean(ds)), 3),
                }
        if len(ref) > 1:
            out["pytorch"]["pytorch_repeat_spread_m"] = round(
                float(np.mean([ade(a, b) for a, b in combinations(ref, 2)])), 3
            )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
